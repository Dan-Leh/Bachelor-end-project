# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:54:31 2021

@author: sonne
"""

import torch
import torch.nn as nn 
import torchvision.models as models
import torch.nn.functional as F

mean, std = [0.425, 0.256]

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

class Siamese_net(nn.Module):
    def __init__(self, n_class, num_resnet_layers=50, pretrained=True):
        super(Siamese_net, self).__init__()
        self.n_class = n_class
        self.num_resnet_layers = num_resnet_layers
        self.pretrained = pretrained
        if self.num_resnet_layers == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.num_resnet_layers == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.num_resnet_layers == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.num_resnet_layers == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.num_resnet_layers == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        
        encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet.conv1.weight.data, dim=1), dim=1)
        encoder_bn1 = resnet.bn1
        encoder_relu = resnet.relu
        encoder_maxpool = resnet.maxpool
        self.encoder_layer0 = nn.Sequential(encoder_conv1,
                                            encoder_bn1,
                                            encoder_relu,
                                            encoder_maxpool)
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
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
        
        fea_dim = 2048
        bins = [1,2,3,6]
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.n_class, kernel_size=1)
            )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, self.n_class, kernel_size=1)
            )
            
    def forward(self, x):
        x = self.encoder_layer0(x)       
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)

        x_tmp = self.encoder_layer3(x)   
        x = self.encoder_layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        if self.training:
            x_aux = self.aux(x_tmp)
            return x, x_aux
        return x


# class Siamese_net(nn.Module):
#     def __init__(self, n_class, num_resnet_layers=50, pretrained=True):
#         super(Siamese_net, self).__init__()
#         self.n_class = n_class
#         self.num_resnet_layers = num_resnet_layers
#         self.pretrained = pretrained
#         if self.num_resnet_layers == 18:
#             resnet = models.resnet18(pretrained=self.pretrained)
#         elif self.num_resnet_layers == 34:
#             resnet = models.resnet34(pretrained=self.pretrained)
#         elif self.num_resnet_layers == 50:
#             resnet = models.resnet50(pretrained=self.pretrained)
#         elif self.num_resnet_layers == 101:
#             resnet = models.resnet101(pretrained=self.pretrained)
#         elif self.num_resnet_layers == 152:
#             resnet = models.resnet152(pretrained=self.pretrained)
#         encoder_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         encoder_conv1.weight.data = torch.unsqueeze(torch.mean(resnet.conv1.weight.data, dim=1), dim=1)
#         encoder_bn1 = resnet.bn1
#         encoder_relu = resnet.relu
#         encoder_maxpool = resnet.maxpool
#         self.encoder_layer0 = nn.Sequential(encoder_conv1,
#                                             encoder_bn1,
#                                             encoder_relu,
#                                             encoder_maxpool)
#         self.encoder_layer1 = resnet.layer1
#         self.encoder_layer2 = resnet.layer2
#         self.encoder_layer3 = resnet.layer3
#         self.encoder_layer4 = resnet.layer4
#         for n, m in self.encoder_layer3.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
#         for n, m in self.encoder_layer4.named_modules():
#             if 'conv2' in n:
#                 m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
#             elif 'downsample.0' in n:
#                 m.stride = (1, 1)
#         fea_dim = 2048
#         bins = [1,2,3,6]
#         self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
#         fea_dim *= 2
#         self.cls = nn.Sequential(
#             nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(512, self.n_class, kernel_size=1)
#             )
#         self.feat_adapt = nn.Sequential(
#             nn.Conv2d(2048, 512, 3, 2, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 128, 3, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 32, 3, 2, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Flatten(),
#             nn.Linear(32*640//64*320//64, 1000, bias=False),
#             nn.BatchNorm1d(1000),
#             nn.ReLU(inplace=True),
#             )
#         if self.training:
#             self.aux = nn.Sequential(
#                 nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout2d(p=0.1),
#                 nn.Conv2d(256, self.n_class, kernel_size=1)
#             )
            
#     def forward(self, x1, x2):
#         x1 = self.encoder_layer0(x1)
#         x2 = self.encoder_layer0(x2)
        
#         x1 = self.encoder_layer1(x1)
#         x2 = self.encoder_layer1(x2)
        
#         x1 = self.encoder_layer2(x1)
#         x2 = self.encoder_layer2(x2)
        
#         x1_tmp = self.encoder_layer3(x1)
#         x2 = self.encoder_layer3(x2)
        
#         x1 = self.encoder_layer4(x1_tmp)
#         x2 = self.encoder_layer4(x2)
#         #  print(x1.shape, x2.shape)
        
#         x1_domain = self.feat_adapt(x1)
#         x2_domain = self.feat_adapt(x2)
        
#         x1_feat = self.ppm(x1)
#         x1_feat = self.cls(x1_feat)
#         if self.training:
#             x1_aux = self.aux(x1_tmp)
#             return x1_domain, x2_domain, x1_feat, x1_aux
#         return x1_feat