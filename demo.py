import numpy as np
import matplotlib.pyplot as plt


data  = np.load(r'E:\FWI\FWI-FL\DATA\FlatVel_B\frequency\frequency49.npy')
print(np.min(data))
print(np.max(data))

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

