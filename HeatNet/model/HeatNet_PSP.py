# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:34:26 2021

@author: sonne
"""
import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('/home/jsun/Project/HeatNet-master/')
import net.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class HeatNet_PSP(nn.Module):
    def __init__(self, layers=50, bins=(1,2,3,6), dropout=0.1, classes=20, zoom_factor=8, \
                  use_ppm=True, pretrained=True):
        super(HeatNet_PSP, self).__init__()
        assert layers in [50,101,152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1,2,4,8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.pretrained = pretrained
        self.bins = bins
        
        if layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=self.pretrained)
            resnet_raw_model2 = models.resnet50(pretrained=self.pretrained)
        elif layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=self.pretrained)
            resnet_raw_model2 = models.resnet101(pretrained=self.pretrained)
        else:
            resnet_raw_model1 = models.resnet152(pretrained=self.pretrained)
            resnet_raw_model2 = models.resnet152(pretrained=self.pretrained)
            
        ##### RGB encoder #####
        self.encoder_rgb_layer0 = nn.Sequential(resnet_raw_model1.conv1, resnet_raw_model1.bn1, \
                                                resnet_raw_model1.relu, resnet_raw_model1.conv2, \
                                                resnet_raw_model1.bn2, resnet_raw_model1.relu, \
                                                resnet_raw_model1.conv3, resnet_raw_model1.bn3, \
                                                resnet_raw_model1.relu, resnet_raw_model1.maxpool)

        self.encoder_rgb_layer1 = resnet_raw_model1.layer1
        
        ##### Thermal encoder #####
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model2.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model2.bn1
        self.encoder_thermal_relu1 = resnet_raw_model2.relu
        self.encoder_thermal_conv2 = resnet_raw_model2.conv2
        self.encoder_thermal_bn2 = resnet_raw_model2.bn2
        self.encoder_thermal_relu2 = resnet_raw_model2.relu
        self.encoder_thermal_conv3 = resnet_raw_model2.conv3 
        self.encoder_thermal_bn3 = resnet_raw_model2.bn3
        self.encoder_thermal_relu3 = resnet_raw_model2.relu
        self.encoder_thermal_maxpool = resnet_raw_model2.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model2.layer1

        
        ##### bottelneck block #####
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512,256,1,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        ##### merged encoder #####      
        self.encoder_layer2 = resnet_raw_model1.layer2
        self.encoder_layer3 = resnet_raw_model1.layer3
        self.encoder_layer4 = resnet_raw_model1.layer4 
        for n, m in self.encoder_layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.encoder_layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        ##### decoder #####
        if self.use_ppm:
            self.ppm = PPM(2048, int(2048/len(self.bins)), self.bins)
        # if self.multi_strategy:
        #     self.ppm2 = PPM(1024, int(1024/len(self.bins)), self.bins)
            
        self.cls = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
            
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )    
        
    def forward(self, x):
        rgb = x[:,:3,...]
        thermal = x[:,3:,...]
        
        rgb = self.encoder_rgb_layer0(rgb)
        rgb = self.encoder_rgb_layer1(rgb)
        
        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu1(thermal)
        thermal = self.encoder_thermal_conv2(thermal)
        thermal = self.encoder_thermal_bn2(thermal)
        thermal = self.encoder_thermal_relu2(thermal) 
        thermal = self.encoder_thermal_conv3(thermal)
        thermal = self.encoder_thermal_bn3(thermal)
        thermal = self.encoder_thermal_relu3(thermal)
        thermal = self.encoder_thermal_maxpool(thermal)
        thermal = self.encoder_thermal_layer1(thermal)

        feat = torch.cat([rgb, thermal], dim=1)
        feat = self.bottleneck(feat)
        feat = self.encoder_layer2(feat)
        # print(feat.shape)
        
        feat_tmp = self.encoder_layer3(feat)              # 1024, 60, 80
        # print(feat_tmp.shape)
        feat = self.encoder_layer4(feat_tmp)              # 2048, 60, 80
        # print(feat.shape)
        if self.use_ppm:
            feat = self.ppm(feat)
        feat = self.cls(feat)
        # print(feat.shape)
        if self.training:
            aux = self.aux(feat_tmp)
            # print(feat.shape,aux.shape)
            return feat, aux
        return feat
    
    def get_1x_lr_params_NOscale(self):
        
        b = []
        
        b.append(self.encoder_rgb_layer0)
        b.append(self.encoder_rgb_layer1)
        b.append(self.encoder_thermal_conv1)
        b.append(self.encoder_thermal_bn1)
        b.append(self.encoder_thermal_conv2)
        b.append(self.encoder_thermal_bn2)
        b.append(self.encoder_thermal_conv3)
        b.append(self.encoder_thermal_bn3)
        b.append(self.encoder_thermal_layer1)
        b.append(self.bottleneck)
        b.append(self.encoder_layer2)
        b.append(self.encoder_layer3)
        b.append(self.encoder_layer4)
        
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    
    def get_10x_lr_params(self):
        
        b = []
        
        b.append(self.ppm.parameters())
        b.append(self.cls.parameters())
        b.append(self.aux.parameters())
        
        for j in range(len(b)):
            for i in b[j]:
                yield i
                
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr':10*args.learning_rate}]

#%%
# from PIL import Image
# p = r'E:\graduation project\Freiburg\train_set\seq_00_day\00\fl_ir_aligned\fl_ir_aligned_1570722183_4564325440.png'
# image1 = Image.open(p)
# image1 = np.array(image1)
# image2 = cv2.imread(p,cv2.IMREAD_GRAYSCALE)
       
# m1 = HeatNet_PSP(pretrained=False)
# m1 = m1.cuda()
# input = torch.Tensor(2,4,650,1920).cuda()
# out = m1(input)