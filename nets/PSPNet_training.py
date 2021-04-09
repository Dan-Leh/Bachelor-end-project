# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:40:07 2021

@author: sonne
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from random import shuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2

def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)
    
    temp_inputs = inputs.permute(0,2,3,1).contiguous().view(-1,c)
    temp_target = target.view(-1)
    
    CE_Loss = nn.NLLLoss(F.log_softmax(temp_inputs, dim=-1), temp_target, ignore_index=num_classes)
    
    return CE_Loss

def Dice_Loss(inputs, target, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.permute(0,2,3,1).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
        
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    
    dice_coff = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    return 1 - torch.mean(dice_coff)
