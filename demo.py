import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter
from torch.autograd import Variable

import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import scipy.io
import scipy

font18 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 18,
}
font21 = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 21,
}

def pain_openfwi_seismic_data(para_seismic_data,  path, is_colorbar = 0):
    '''
    Plotting seismic data images of openfwi dataset

    :param para_seismic_data:   Seismic data (1000 x 70) (numpy)
    :param is_colorbar:         Whether to add a color bar (1 means add, 0 is the default, means don't add)
    '''

    # The size of 1000 x 70 is not easy to display, we compressed it to a similar size of 400 x 301 as the SEG dataset.
    data = cv2.resize(para_seismic_data, dsize=(400, 301), interpolation=cv2.INTER_CUBIC)   #

    if is_colorbar == 0:
        fig, ax = plt.subplots(figsize=(6.5, 8), dpi = 120)
    else:
        fig, ax = plt.subplots(figsize=(6.1, 8), dpi = 120)

    im = ax.imshow(data, extent=[0, 0.7, 1.0, 0], cmap=plt.cm.seismic, vmin=-18, vmax=19)
    ax.set_xlabel('Position (km)', font21)
    ax.set_ylabel('Time (s)', font21)
    #plt.xticks([])
    #plt.yticks([])

    #ax.set_xticks([])
    #ax.set_yticks([])

    if is_colorbar == 0:
        plt.subplots_adjust(bottom=0.11, top=0.95, left=0.11, right=0.99)
    else:
        plt.rcParams['font.size'] = 14      # Set colorbar font size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.3)
        plt.colorbar(im, ax=ax, cax=cax, orientation='horizontal')

        plt.subplots_adjust(bottom=0.08, top=0.98, left=0.11, right=0.99)
    plt.savefig(path)
    plt.show()



def plot_single_seismic(data, path):
    colors = [(0, 0, 1),  # Red
              (1, 1, 1),  # White
              (1, 0, 0)]  # Blue

    # Create the custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('red_white_blue', colors)

    nz, nx = data.shape
    plt.rcParams.update({'font.size': 18})
    vmin, vmax = np.min(data), np.max(data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(data, aspect='auto', cmap=cmap, vmin=vmin * 0.01, vmax=vmax * 0.01)
    ax.set_aspect(aspect=nx/nz)
    plt.xticks([])
    plt.yticks([])

    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xticks(range(0, nx, int(300//(1050/nx)))[:5])
    #ax.set_xticklabels(range(0, 1050, 300))
    #ax.set_title('Offset (m)', y=1.08)
    #ax.set_yticks(range(0, nz, int(200//(1000/nz)))[:5])
    #ax.set_yticklabels(range(0, 1000, 200))
    #ax.set_ylabel('Time (ms)', fontsize=18)
    #fig.colorbar(im, ax=ax, shrink=1.0, pad=0.01, label='Amplitude')
    plt.savefig(path)
    plt.close('all')




data  = np.load(r'E:\FWI\FWI-F\DATA\CurveVel_A\data\data1.npy')[0]



for i in range(5):
    pain_openfwi_seismic_data(data[i], f'.\sesimic_{i}.png')



#-------------------------------------------------------------------------------
# 创建二维矩阵
# matrix = np.full((500, 5), 15.0, dtype=np.float32)  # n 表示矩阵的行数
# for i in range(108):
#     # 保存为.npy文件
#     np.save(f'model{i+1}.npy', matrix)
#



# ----------------------------------------------------------------------------
#

# def l2_relative_loss(input, target):
#     diff = input - target
#     l2_norm_diff = torch.norm(diff, p=2, dim=(1, 2, 3))
#     l2_norm_target = torch.norm(target, p=2, dim=(1, 2, 3))
#     loss = l2_norm_diff / l2_norm_target
#     return loss
#
# # 创建输入张量和目标张量
# input = torch.randn(2, 3, 3, 4)
# target = torch.randn(2, 3, 3, 4)
#
# # 计算L2相对损失
# loss = l2_relative_loss(input, target)
#
# print(loss)

# data  = np.load(r'E:\FWI\FWI-F\DATA\FlatVel_B\source\source9.npy')
# print(np.min(data))
# print(np.max(data))

# ----------------------------------------------------------------------------
#
# #测试data中最大值最小值
#min = 0
#max = 0
#for i in range(1, 109):
#    data  = np.load('D:\FWI\FWI-L\DATA\FlatFault_A\data\data{}.npy'.format(i))
#    print('max:{} min:{}'.format(np.max(data), np.min(data)))
#    min = min if min < np.min(data) else np.min(data)
#    max = max if max > np.max(data) else np.max(data)
#print('Finnal max:{} min:{}'.format(max, min))


# ---------------------------------------------------------------------------

# import torch
# import torch.nn.functional as F
# from torchvision import models
# # 加载预训练的VGG模型
#
# # vgg = models.vgg16(pretrained=True).features
# #
# # torch.save(vgg.state_dict(), "1.pth")
# # print("--------------------------------------")
# # print(torch.load('1.pth'))
# # print("--------------------------------------")
# # vgg[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
# # torch.save(vgg.state_dict(), "2.pth")
# # print("--------------------------------------")
# # print(torch.load('2.pth'))
# # print("--------------------------------------")
# # print(vgg)
#
#
#
# # torch.save(vgg.state_dict(), "vgg16_method2.pth")
#
#
# vgg = models.vgg16(pretrained=False).features
# vgg[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
#
# vgg.load_state_dict(torch.load("vgg16-2.pth"))
# # vgg = vgg.load_state_dict(torch.load(r'.\pretrained\vgg16\vgg16.pth'))
# # 将模型设置为评估模式
#
# def perceptual_loss(input, target):
#     # 计算输入和目标图像在VGG特征图上的差异
#     input_features = vgg()(input)
#     target_features = vgg()(target)
#     loss = F.l1_loss(input_features, target_features)
#     return loss
#
# model = np.load('E:\FWI\FWI-F\DATA\CurveFault_A\model\model101.npy')
# model = torch.from_numpy(model)
#
# i = vgg(model)  # (500, 512, 2, 2)
# print(i)
# print("11111111111111111111111111111111111111111")
# print(torch.max(i))
# vgg2 = models.vgg16(pretrained=True).features
# vgg2[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
# print(vgg2(model))

# --------------------------------------------------------------------------
# 数据集随机分割

# import torch
# from torch.utils.data import random_split
# dataset = range(10)
# train_dataset, test_dataset = random_split(
#     dataset=dataset,
#     lengths=[7, 3],
#     generator=torch.Generator().manual_seed(0)
# )
# print(list(train_dataset))
# print(list(test_dataset))

# -------------------------------------------------------------------
# import torch
# import torch.nn.functional as F
#
#
# class model(torch.nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.linear = torch.nn.Linear(3,3)
#
#     def forward(self, x):
#         return self.linear(x)
#
# class model2(torch.nn.Module):
#     def __init__(self):
#         super(model2, self).__init__()
#         self.linear = torch.nn.Linear(3,3)
#
#     def forward(self, x):
#         return self.linear(x)
#
# class model3(torch.nn.Module):
#     def __init__(self):
#         super(model3, self).__init__()
#         self.linear = model2()
#
#     def forward(self, x):
#         return self.linear(x)
#
#
# x = torch.rand(2, 3)
# y = torch.rand(2, 3)
# model11 = model()
# model22 = model2()
# model33 = model3()
# l1Loss = torch.nn.L1Loss()
# lr = 0.001
# optimizer = torch.optim.AdamW(model11.parameters(), lr= lr)
# for i in range(10):
#     model11.train()
#     optimizer.zero_grad()
#     output = model11(x)
#     loss = l1Loss(output, y) + l1Loss(model33(output), y)
#     loss.backward()
#     optimizer.step()
#     print(model22.parameters())

