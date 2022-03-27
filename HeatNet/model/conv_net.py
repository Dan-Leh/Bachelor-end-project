# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 22:52:43 2021

@author: sonne
"""

"""
HeatNet with the discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#from model.discriminator import FCDiscriminator
# import utils
# from models import trgb_segnet as models
# from utils import weights_init_normal
# from models import critic_resnet
# from models import downscale_network
# from model.HeatNet_PSP import PSPNet
import sys
sys.path.append('/home/jsun/Project/HeatNet-master/')
from model.HeatNet import PSPNet

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x) 
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x

def create_discriminator(input_num):
    return FCDiscriminator(input_num)

class HeatNet(nn.Module):
    def __init__(self, num_classes=14, num_dis=6):
        super(HeatNet, self).__init__()
        self.num_classes = num_classes
        self.net = PSPNet(n_classes=self.num_classes, pretrained=False, late_fusion=True)
        self.net.apply(weights_init_normal)
        
        critic_num = [14, 2048, 1024, 512*2, 256*2, 64*2]
        critic_num = critic_num[0:num_dis]
        self.critics = torch.nn.ModuleList()
        for i in range(len(critic_num)):
            self.critics.append(create_discriminator(critic_num[i]))
            
        self.phase = 'train_seg'
        
    def setLearningModel(self, module, val):
        for p in module.parameters():
            p.requires_grad = val

    def setPhase(self, phase):
        self.phase = phase
        # print("Switching to phase: %s" % self.phase)
        if self.phase == "train_seg":
            for c in self.critics:
                self.setLearningModel(c, False)
            self.setLearningModel(self.net, True)
        elif self.phase == "train_critic":
            for c in self.critics:
                self.setLearningModel(c, True)
            self.setLearningModel(self.net, False)
    
    def forward(self, input_a, input_b=None):
        output = {}
        pred_label_day, inter_f_a, cert_a = self.net(*input_a)
                
        output['critics_a'] = []
                
        for i, c in enumerate(self.critics):
            output['critics_a'].append(c(inter_f_a[i]))

        output['pred_label_a'] = pred_label_day      
        output['cert_a'] = cert_a
        
        if input_b is not None:       
            pred_label_night, inter_f_b, cert_b = self.net(*input_b)
            output['critics_b'] = []
            for i, c in enumerate(self.critics):
                output['critics_b'].append(c(inter_f_b[i]))
            output['pred_label_b'] = pred_label_night
            output['cert_b'] = cert_b
            output['inter_f_b'] = inter_f_b
        
        return output

# model = HeatNet()
# model = model.cuda()
# in1_1 = torch.Tensor(2,3,320,640).cuda()
# in1_2 = torch.Tensor(2,1,320,640).cuda()
# in2_1 = torch.Tensor(2,3,160,320).cuda()
# in2_2 = torch.Tensor(2,1,160,320).cuda()
# out = model([in1_1,in1_2])
