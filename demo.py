import numpy as np
import matplotlib.pyplot as plt


data  = np.load(r'E:\FWI\FWI-FL\DATA\CurveVel_A\frequency\frequency60.npy')
print(np.min(data))
print(np.max(data))

# 测试data中最大值最小值
# min = 0
# max = 0
# for i in range(1, 61):
#     data  = np.load('E:\FWI\FWI-FL\DATA\Style_A\data\data{}.npy'.format(i))
#     print('max:{} min:{}'.format(np.max(data), np.min(data)))
#     min = min if min < np.min(data) else np.min(data)
#     max = max if max > np.max(data) else np.max(data)
# print('Finnal max:{} min:{}'.format(max, min))

# loss可视化
# 绘制loss曲线图
# x = np.arange(1, 121)
# plt.figure(1)
# ax11 = plt.subplot(1, 2, 1)
# ax11.semilogy(x, np.log10(x), label='train')
# ax11.semilogy(x, 0.1 * np.log10(x), label='test')
# ax11.set_xlabel('Epochs')
# ax11.set_ylabel('Loss')
# ax11.set_title('MAELoss')
# ax11.legend()
#
# ax12 = plt.subplot(1, 2, 2)
# ax12.plot(x, np.log10(x), label='train')
# ax12.plot(x, np.log10(x), label='test')
# ax12.set_yscale('log')
# ax12.set_xlabel('Epochs')
# ax12.set_ylabel('Loss')
# ax12.set_title('MAELoss')
# ax12.legend()
#
# plt.show()




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