# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:09:22 2023

@author: Ye Li
"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from numpy import pi
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 1)
        
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.constant_(self.layer1.bias, 0.)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.constant_(self.layer2.bias, 0.)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)
        nn.init.constant_(self.layer3.bias, 0.)
        # nn.init.xavier_normal_(self.layer4.weight, gain=1)
        # nn.init.constant_(self.layer4.bias, 0.)
        nn.init.xavier_normal_(self.layer5.weight, gain=1)
        nn.init.constant_(self.layer5.bias, 0.)

    def forward(self, x):
        X = x
        H = torch.relu(self.layer1(X))
        H = torch.relu(self.layer2(H))
        H = torch.relu(self.layer3(H))
        # H = torch.relu(self.layer4(H))
        H = self.layer5(H)
        return H

# discontinuous network
class Net_Discon(nn.Module):
    def __init__(self):
        super(Net_Discon, self).__init__()
        self.values = torch.tensor([0.])
        self.layer1 = nn.Linear(1, 256)
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

class Fcl_Net(nn.Module):
    def __init__(self, layer_sizes, is_last_bias):
        super(Fcl_Net,self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            m = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
        last_layer = nn.Linear(layer_sizes[len(layer_sizes) - 2], layer_sizes[len(layer_sizes) - 1], bias=is_last_bias)
        nn.init.xavier_normal_(last_layer.weight, gain=1)
        self.Wz.append(last_layer)

    def forward(self, x):
        X = x
        H = torch.relu(self.Wz[0](X))
        ones = torch.ones_like(H)
        zero = torch.zeros_like(H)
        R = torch.where(H > 0, ones, zero)
        for linear in self.Wz[1:-1]:
            H = torch.relu(linear(H))
            ones = torch.ones_like(H)
            zero = torch.zeros_like(H)
            r = torch.where(H > 0, ones, zero)
            R = torch.cat((R, r), 1)      # 将每一层的0,1向量 按列拼接
        H = self.Wz[-1](H)
        return H, R


class Bias_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Bias_Net, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 512)
        self.last_layer = nn.Linear(512, output_size)

    def forward(self, x):
        X = x;
        H = torch.tanh(self.hidden_layer(X))
        H = self.last_layer(H)
        return H


# DeLU
class DeLU(nn.Module):
    def __init__(self, layer_sizes):
        super(DeLU, self).__init__()
        self.fcl_net = Fcl_Net(layer_sizes, False)  # false为最后一层不加bias
        bias_input_size = sum(layer_sizes) - layer_sizes[0] - layer_sizes[-1]   # 所有隐藏层节点大小和
        bias_output_size = layer_sizes[-1]
        self.bias_net = Bias_Net(bias_input_size, bias_output_size)


    def forward(self, x):
        X = x
        H1, R = self.fcl_net(X)
        H2 = self.bias_net(R)
        return H1 + H2


class MyNet(nn.Module):
    def __init__(self, layer_sizes):
        super(MyNet, self).__init__()
        self.values = torch.tensor([0.])
        self.fcl_net = Fcl_Net(layer_sizes, False)
        self.bias_layer1 = nn.Linear(layer_sizes[0], 16)
        self.bias_layer2 = nn.Linear(16, layer_sizes[-1])
        self.epsilon1 = nn.Parameter(torch.randn(1, 16))  # 8为不连续层的节点个数
        # self.epsilon2 = nn.Parameter(torch.randn(1, 8))  # 8为不连续层的节点个数

        nn.init.xavier_normal_(self.epsilon1, gain=1)
        # nn.init.xavier_normal_(self.epsilon2, gain=1)
        nn.init.xavier_normal_(self.bias_layer1.weight, gain=1)
        nn.init.constant_(self.bias_layer1.bias, 0.)
        nn.init.xavier_normal_(self.bias_layer2.weight, gain=1)
        nn.init.constant_(self.bias_layer2.bias, 0.)



    def forward(self, x):
        X = x
        H1,_ = self.fcl_net(X)
        H2 = torch.relu(self.bias_layer1(X)) + self.epsilon1 * torch.heaviside(self.bias_layer1(X), values=self.values)
        # H2 = torch.relu(self.bias_layer2(H2)) + self.epsilon2 * torch.heaviside(self.bias_layer2(H2), values=self.values)
        H2 = self.bias_layer2(H2)
        return H1 + H2


def u(x):
    a1 = x < 1/3
    y1 = a1 * torch.sin(x)
    a2 = (x>=1/3) & (x<=2/3)
    y2 = a2 * torch.ones_like(x)
    a3 = (x>2/3)
    y3 = a3 * torch.sin(x)
    y = y1 + y2 + y3
    return y

def loss_fn(net, x_data):
    return (net(x_data) - u(x_data)).pow(2).mean() #测试拟合函数

#训练
epochs = 1000
learning_rate = 1e-3
layer_sizes = [1, 256, 256, 256, 1]
x_train = torch.unsqueeze(torch.linspace(0,1,64),dim=1)
net = MyNet(layer_sizes)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_history = []

for i in range(epochs):
    x = x_train        #full batch
    loss = loss_fn(net, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print('\repoch {:d} MSE loss = {:.6f}'.format(i+1,loss.item()), end='', flush=True)

#测试
x_test = torch.unsqueeze(torch.linspace(0,1,100),dim=1)
y_test = u(x_test)
y_predict = net(x_test)
plt.figure(1)
plt.plot(x_test, y_test, label='true solution')
plt.plot(x_test, y_predict.detach(), 'bo:', label='fitted solution')
plt.xlabel('x')
plt.ylabel('solution u(x)')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(loss_history,label='training loss')
plt.xlabel('epochs')
plt.yscale("log")
plt.legend()
plt.show()
