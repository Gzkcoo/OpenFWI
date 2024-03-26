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


# Coordinate
xc = torch.arange(1, 70 + 1)
yc = torch.arange(1, 70 + 1)
xm, ym = torch.meshgrid(xc, yc)
x = xm.reshape(-1, 1)
y = ym.reshape(-1, 1)
xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
xy_coordinate = xy_coordinate / 70  # 输入转换为（0，1）之间
# xy_coordinate = xy_coordinate.to('cuda')

N_loc = 64

print([i for i in sampler])