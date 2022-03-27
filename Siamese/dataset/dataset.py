# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:18:38 2021

@author: sonne
"""

import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
import cv2
from PIL import Image

# import sys
# sys.path.append(r'E:\graduation project\semseg-master')

# from util import dataset, transform, config

def train_val_split(path, train_percent=0.8):
    file_list = path + 'file_list.txt'
    images = []
    with open(file_list, 'r') as f:
        for i in f.readlines():
            images.append(i.strip())
    f.close()
    
    num = len(images)
    list=range(num)  
    tr=int(num*train_percent)  
    train=random.sample(list,tr) 
    
    train_list = open(path+'train.txt', 'w')
    val_list = open(path+'val.txt', 'w')
    for i in range(num):
        if i not in train:
            val_list.write(images[i]+'\n')
        else:
            train_list.write(images[i]+'\n')
    train_list.close()
    val_list.close()

def make_dataset(split='train', data_root=None, domain=None):
    assert split in ['train', 'val', 'test']
    sets = {'train':'train', 'val':'train', 'test':'test'}
    set = sets[split]
    data_list = os.path.join(data_root, set, domain, 'file_list.txt')
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    if domain == 'night':
        day_path = os.path.join(data_root, set, 'day', 'file_list.txt')
        day_list = open(day_path).readlines()
        samples = random.sample(list_read, len(day_list) - len(list_read))
        list_read = list_read + samples
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        if split == 'test':
            rgb_image_name = os.path.join(data_root, set, domain, 'RGB', line+'.png').replace('\\', '/')
            thermal_image_name = os.path.join(data_root, set, domain, 'thermal', line[:-5]+'_ir.png').replace('\\', '/')
            label_name = os.path.join(data_root, set, domain, 'label', line+'.png').replace('\\', '/')
        else:
            rgb_image_name = os.path.join(data_root, set, domain, 'RGB', 'fl_rgb'+line+'.png').replace('\\', '/')
            thermal_image_name = os.path.join(data_root, set, domain, 'thermal', 'fl_ir_aligned'+line+'.png').replace('\\', '/')
            if domain == 'day':
                label_name = os.path.join(data_root, set, domain, 'label', 'gray', 'fl_rgb'+line+'.png').replace('\\', '/')
            else:
                label_name = thermal_image_name       
        item = (rgb_image_name, thermal_image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list

class Freiburg_Dataset(data.Dataset):
    def __init__(self, split='train', data_root=None, domain=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, domain)
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        rgb_image_path, thermal_image_path, label_path = self.data_list[index]
        
        rgb = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        grayscale = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        thermal = cv2.imread(thermal_image_path, 0)
        img = np.concatenate((grayscale[...,None], thermal[...,None]),axis=2)
        img = np.float32(img) / 255.0
        img = cv2.resize(img, (640,320), interpolation=cv2.INTER_AREA)
        if self.split == 'train':
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = label[5:645,360:1640]
            label = cv2.resize(label,(640,320),cv2.INTER_NEAREST)
        else:
            label = Image.open(label_path)
            label = np.array(label)
            label = label[5:645,360:1640]
            label = np.array(label,dtype=np.float32)
            label = cv2.resize(label,(640,320),cv2.INTER_NEAREST)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label
    
def make_test_dataset(split='test', data_root=None, domain=None):
    assert split in ['train', 'val', 'test']
    sets = {'train':'train', 'val':'train', 'test':'test'}
    set = sets[split]
    data_list = os.path.join(data_root, set, domain, 'file_list.txt')
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        rgb_image_name = os.path.join(data_root, set, domain, 'RGB', line+'.png').replace('\\', '/')
        thermal_image_name = os.path.join(data_root, set, domain, 'thermal', line[:-5]+'_ir.png').replace('\\', '/')
        label_name = os.path.join(data_root, set, domain, 'label', line+'.png').replace('\\', '/')      
        item = (rgb_image_name, thermal_image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list

class Freiburg_test_Dataset(data.Dataset):
    def __init__(self, split='train', data_root=None, domain=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, domain)
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        rgb_image_path, thermal_image_path, label_path = self.data_list[index]
        
        rgb = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        grayscale = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        thermal = cv2.imread(thermal_image_path, 0)
        img = np.concatenate((grayscale[...,None], thermal[...,None]),axis=2)
        img = np.float32(img) / 255.0
        img = cv2.resize(img, (640,320), interpolation=cv2.INTER_AREA)

        label = Image.open(label_path)
        label = np.array(label)
        label = label[5:645,360:1640]
        label = np.array(label,dtype=np.float32)
        label = cv2.resize(label,(640,320),cv2.INTER_NEAREST)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label

