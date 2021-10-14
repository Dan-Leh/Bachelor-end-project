#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:25:41 2021

@author: jsun
"""
import torch
from torch import nn
import sys
sys.path.append('/home/jsun/Project/HeatNet-master/')

from model.discriminator import FCDiscriminator
from model.Loss import MSE_loss, CE_Loss, CrossEntropy2d

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda(0)
    criterion = CrossEntropy2d().cuda(0)

    return criterion(pred, label)
net = nn.Sequential(
            nn.Conv2d(512,256,1,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
net = net.cuda(0)
input = torch.Tensor(8,512,119,119)
input = input.cuda(0)
label = torch.Tensor(8,119,119)
out = net(input)
loss = loss_calc(out, label)
loss.backward()
#%%
from PIL import Image
import numpy as np
path = '/home/jsun/Project/Freiburg/train/day/label/gray/'
name = 'fl_rgb_1570722163_2325205760.png'
im = Image.open(path+name)
nim = im.resize((960,320),Image.BICUBIC)
nim = np.array(nim)
im = np.array(im)
#%%
import os
m = []
lists = os.listdir(path)
maximum, minimum = 0, 0
for l in lists:
    img = Image.open(path+l)
    img = np.array(img)
    maxi = np.max(img)
    mini = np.min(img)
    maximum = max(maximum, maxi)
    minimum = min(minimum, mini)
#%%
import torch
checkpoint = torch.load('/home/jsun/Project/HeatNet-master/snapshots/train_epoch_50.pth')
# optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
# print(optimizer_D1)
#%%
import torch  
checkpoint = torch.load('/home/jsun/Project/semseg-master/exp/mapillary_vistas/pspnet50/model/train_epoch_400.pth')
torch.save(checkpoint,'/home/jsun/Project/train400.pth',_use_new_zipfile_serialization= False)