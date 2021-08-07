# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:02:21 2021

@author: sonne
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.append(r'E:\graduation project\HeatNet-master')
from net.RTFNet import RTFNet
from net.PSPNet.util.test_demo import create_model, test

# #%%
# num_minibatch = 1
# w, h = 650, 1920
# rgb = torch.randn(num_minibatch, 3, h, w).cuda(0)
# thermal = torch.randn(num_minibatch, 1, h, w).cuda(0)
# rtf_net = RTFNet(66,18).cuda(0)
# input1 = torch.cat((rgb, thermal), dim=1)
# out1 = rtf_net(input1)
# #%%
# input2 = torch.randn(2,3,480,640)
# pspnet = pspnet(num_classes=10, downsample_factor=16,pretrained=False,aux_branch=False)
# out2 = pspnet(input2)
import time
class HeatNet_RTF(nn.Module):
    def __init__(self, num_classes, RTF_layers):
        super(HeatNet_RTF, self).__init__()
        self.num_classes = num_classes
        self.RTF_layers = RTF_layers
        
        self.rtf_net = RTFNet(self.num_classes, self.RTF_layers)
        self.psp_net = create_model()
        
    def forward(self, x):
        psp_input = x[:,:3,:,:]
        rtf_input = x / 255
        rtf_out = self.rtf_net(rtf_input)
        gray = test(self.psp_net.eval(), psp_input[0])
        
        return rtf_out, gray

#%%
from PIL import Image
import numpy as np
path = 'E:/graduation project/Freiburg/train_set/seq_01_day/00/'
rgb = Image.open(path+'fl_rgb/fl_rgb_1578919271_9487361180.png')
rgb = np.array(rgb)
thermal = Image.open(path+'fl_ir_aligned/fl_ir_aligned_1578919271_9487361180.png')
thermal = np.array(thermal)
rgb = torch.from_numpy(rgb).permute(2,0,1).float()
thermal = torch.from_numpy(thermal)
thermal = thermal.unsqueeze(2).float()
thermal = thermal.permute(2,0,1)
input = torch.cat((rgb,thermal), 0).unsqueeze(0)
#%%
heatnet = HeatNet_RTF(66,18).cuda()
#%%
#input = torch.Tensor(1,4,480,640)
input = input.cuda()
import time
start = time.time()
out = heatnet(input)
end = time.time()
print(end-start)
#%%
import numpy as np
image = np.array(out[2])
out[2].show()
