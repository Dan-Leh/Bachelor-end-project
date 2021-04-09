# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:12:14 2021

@author: sonne
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from resnet import resnet50
from functools import partial

    
class ResNet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(ResNet, self).__init__()

        model = resnet50(pretrained)
        
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))
            
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2,2):
                m.stride = (1,1)
                if m.kernel_size == (3,3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3,3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        
        return x_aux, x

class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size) \
                                    for pool_size in pool_sizes])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, \
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))
    
    def _make_stages(self, in_channels, out_channels, bin_sz):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(bin_sz),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    
    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        pyramids = [x]
        # pyramids.extend([F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        # output = self.bottleneck(torch.cat(pyramids, dim=1))
        for stage in self.stages:
            feat = stage(x)
            up_feat = F.interpolate(feat, size=(h,w), mode='bilinear', align_corners=True)
            pyramids.append(up_feat)
        final_feat = torch.cat(pyramids, dim=1)
        output = self.bottleneck(final_feat)
        
        return output

class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone='resnet50', pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = ResNet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        else:
            raise ValueError('Unsupported backbone')
        
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1,2,3,6]),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1))
        
        self.aux_branch = aux_branch
        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1))
            
        self.init_weights(self.master_branch)
    
    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output
        
    def init_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.001)
                    m.bias.data.zero_()
#%%
x = torch.Tensor(2,3,480,480)
model = PSPNet(10, 16, backbone='resnet50', pretrained=False, aux_branch=False)
# model.train()
x = model(x)
