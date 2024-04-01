import os
import sys
import time
import datetime
import json

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T
import random
import numpy as np

# Coordinate
# length = 70
# deepth = 70
# xc = np.arange(0, length)
# yc = np.arange(0, deepth)
# xm, ym = np.meshgrid(xc, yc)
# x = xm.reshape(-1, 1)
# y = ym.reshape(-1, 1)
# xy_loc = np.concatenate((y, x), axis=1)
# xy_loc = xy_loc / length  # 输入转换为（0，1）之间
#
# N_sample = 500
# xy_sample = random.sample(xy_loc, N_sample)
#
batch = 64

xc = torch.arange(0, 70)
yc = torch.arange(0, 70)
xm, ym = torch.meshgrid(xc, yc)
x = xm.reshape(-1, 1)
y = ym.reshape(-1, 1)
xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
xy_coordinate = xy_coordinate / 70  # 输入转换为（0，1）之间
# xy_coordinate = xy_coordinate.to('cuda')
N_sample = 500

test = torch.tensor([[1,2,3,4,5,6],[1,2,3,4,5,6]])
test[0] = test[0][[0,3,5],:]



arr = [i for i in range(4900)]
loc_samples = []
for i in range(batch):
    arr_sample = random.sample(arr, N_sample)
    loc_sample = xy_coordinate[arr_sample, :]
    loc_samples.append(loc_sample)


print()