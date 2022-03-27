# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:33:58 2021

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

def MSE_loss(inputs, targets):
    n, c, h, w = inputs.size()
    nt, ct, ht, wt = targets.size()
    assert n == nt
    assert c == ct
    assert h == ht
    assert w == wt
    loss_fn = nn.MSELoss(reduction='sum')
    return loss_fn(inputs, targets) / (h*w*c)

# inputs = torch.ones(2,3,2,2)
# targets = torch.ones(2,3,2,2) * 2
# loss = MSE_loss(inputs, targets)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio
    
    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long)*self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(), 
                            bins=self.num_class+1, min=-1,
                            max=self.num_class-1).float()
            hist = hist[1:]
            weight = (1/torch.max(torch.pow(hist, self.ratio)*\
                    torch.pow(hist.sum(), 1-self.ratio), torch.ones(1))).\
                    to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2,3), True).detach()
        weights = weights.unsqueeze(1).expand_as(prob)
        loss = -torch.sum((torch.pow(prob, 2)*weights)[mask]) / (batch_size*self.num_class)
        return loss

class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
    
    def forward(self, pred, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss
    
    
# prob = torch.rand(2, 19, 10, 10)
# pred = F.softmax(prob, dim=1)
# Loss = MaxSquareloss()
# loss = Loss(pred, prob)
# print(loss)
