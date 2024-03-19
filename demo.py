import os
import sys
import time
import datetime
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from torch import nn
from torch.utils.data import RandomSampler, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
from vis import *
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T


# discontinuous network
class Trunk_Net_Discon(nn.Module):
    def __init__(self):
        super(Trunk_Net_Discon, self).__init__()
        self.values = torch.tensor([0.]).to('cuda')
        self.layer1 = nn.Linear(2, 256)
        self.layer2 = nn.Linear(256, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 256)
        self.layer5 = nn.Linear(256, 1)
        # self.leakrelu = nn.LeakyReLU(0.2)

        # 连续激活函数 + epsilon * heaviside
        # epsilon 是可学习的向量
        self.epsilon1 = nn.Parameter(torch.randn(1, 8))   # 8为不连续层的节点个数
        self.epsilon2 = nn.Parameter(torch.randn(1, 8))

        nn.init.xavier_normal_(self.epsilon1, gain=1)
        nn.init.xavier_normal_(self.epsilon2, gain=1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.constant_(self.layer1.bias, 0.)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.constant_(self.layer2.bias, 0.)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)
        nn.init.constant_(self.layer3.bias, 0.)
        nn.init.xavier_normal_(self.layer4.weight, gain=1)
        nn.init.constant_(self.layer4.bias, 0.)
        nn.init.xavier_normal_(self.layer5.weight, gain=1)
        nn.init.constant_(self.layer5.bias, 0.)


    def forward(self, x):
        X = x
        H = torch.relu(self.layer1(X))
        H = torch.relu(self.layer2(H)) + self.epsilon1 * torch.heaviside(self.layer2(H), values=self.values)
        H = torch.relu(self.layer3(H)) + self.epsilon2 * torch.heaviside(self.layer3(H), values=self.values)
        H = torch.relu(self.layer4(H))
        H = self.layer5(H)
        return H







device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

with open('dataset_config.json') as f:
    try:
        ctx = json.load(f)['flatvel-a']
    except KeyError:
        print('Unsupported dataset.')
        sys.exit()

ctx['file_size'] = 500

# Normalize data and label to [-1, 1]
transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=1), T.log_transform(ctx['data_max'], k=1))
])
transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])


dataset_train = FWIDataset(
    './split_files/flatvel_a_train.txt',
    preload=True,
    sample_ratio=1,
    file_size=ctx['file_size'],
    transform_data=transform_data,
    transform_label=transform_label
)

dataset_valid = FWIDataset(
    './split_files/flatvel_a_val.txt',
    preload=True,
    sample_ratio=1,
    file_size=ctx['file_size'],
    transform_data=transform_data,
    transform_label=transform_label
)

train_sampler = RandomSampler(dataset_train)
valid_sampler = RandomSampler(dataset_valid)


dataloader_train = DataLoader(
    dataset_train, batch_size=64,
    sampler=train_sampler, num_workers=1,
    pin_memory=True, drop_last=True, collate_fn=default_collate)

dataloader_valid = DataLoader(
    dataset_valid, batch_size=64,
    sampler=valid_sampler, num_workers=1,
    pin_memory=True, collate_fn=default_collate)

model = network.model_dict['FWIDeeponet'](
        sample_spatial=1.0, sample_temporal=1).to(device)

# Define loss function
l1loss = nn.L1Loss()  # MAE
l2loss = nn.MSELoss()  # MSE
l3loss = nn.CrossEntropyLoss()  # 交叉熵


def criterion(pred, gt):
    loss_g1v = l1loss(pred, gt)
    loss_g2v = l2loss(pred, gt)
    loss_g3v = l3loss(pred, )

    loss = loss_g1v + 0.01 * loss_g3v
    return loss, loss_g1v, loss_g2v, loss_g3v

lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

for data, label in metric_logger.log_every(dataloader, print_freq, header):
    start_time = time.time()
    optimizer.zero_grad()
    data, label = data.to(device), label.to(device)
    contours_label = utils.extract_contours(label)
    print()
    output = model(data)
    loss, loss_g1v, loss_g2v = criterion(output, label)
    loss.backward()
    optimizer.step()

    loss_val = loss.item()
    loss_g1v_val = loss_g1v.item()
    loss_g2v_val = loss_g2v.item()
    batch_size = data.shape[0]
    metric_logger.update(loss=loss_val, loss_g1v=loss_g1v_val,
                         loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
    metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
    if writer:
        writer.add_scalar('loss', loss_val, step)
        writer.add_scalar('loss_g1v', loss_g1v_val, step)
        writer.add_scalar('loss_g2v', loss_g2v_val, step)
    step += 1
    lr_scheduler.step()


data_train, data_label = dataset_train[6]   # 测试第一张velocity图
data_label = torch.tensor(data_label).reshape(-1, 1)  # (4900,1)

# Coordinate
xc = torch.arange(1, 70+1)
yc = torch.arange(1, 70+1)
xm, ym = torch.meshgrid(xc, yc)
x = xm.reshape(-1, 1)
y = ym.reshape(-1, 1)
xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)

# 设置超参数
epochs = 50
learning_rate = 1e-3
batch_size = 256
lambda_g1v = 1
lambda_g2v = 0
layer_sizes = [2, 256, 512, 256, 1]



l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

model = Trunk_Net_Discon().to(device)  # 不连续神经网络
#model = Trunk_Net_Fcl(layer_sizes).to(device)  # 全连接神经网络
#model = DeLU(layer_sizes).to(device)
# model = MyNet(layer_sizes).to(device)
dataLoader = DataLoader(TensorDataset(xy_coordinate, data_label), batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)


loss_history = []
for i in range(1, epochs + 1):
    model.train()
    for _, (coordinate, label) in enumerate(dataLoader):
        coordinate = coordinate.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss_g1v = l1loss(model(coordinate), label)
        loss_g2v = l2loss(model(coordinate), label)
        loss = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())
    print('epoch[' + str(i) + '] loss:' + str(loss.item()) + ' L1loss:' + str(loss_g1v.item()) + ' L2loss:' + str(loss_g2v.item()))


def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity model (numpy)
    :return:            Binary contour structure of the velocity model (numpy)
    '''

    image = para_image

    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny

# Visualization
utils.mkdir('output')
with torch.no_grad():
    pred = model(xy_coordinate.to(device)).cpu()
    # label_np = T.tonumpy_denormalize(data_label, ctx['label_min'], ctx['label_max'], exp=False).reshape(70, 70)
    # pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False).reshape(70, 70)
    pred_np = pred.numpy().reshape(70, 70)
    label_np = data_label.numpy().reshape(70, 70)
    plot_velocity(pred_np, label_np, f'./output/vis.png')

print('--------------------pred_np-------------------')
print(pred_np)
print('--------------------con_pred-------------------')
con_pred = extract_contours(pred_np) * 255
con_label = extract_contours(label_np) * 255
print(con_pred)

img1 = Image.fromarray(con_pred)
img2 = Image.fromarray(con_label)
img1.show()
img2.show()


fig_1 = plt.figure(1)
plt.subplot(1, 1, 1)
plt.plot(loss_history, 'r', label='loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.savefig('./output/loss.png')