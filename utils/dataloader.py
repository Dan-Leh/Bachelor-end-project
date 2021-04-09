# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 23:23:44 2021

@author: sonne
"""
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    image = image.resize((nw, nh), Image.BICUBIC)
    new_im = Image.new('RGB', size, (128,128,128))
    new_im.paste(image, ((w-nw)//2,(h-nh)//2))
    
    label = label.resize((nw, nh), Image.BICUBIC)
    new_la = Image.new('L', size, (0))
    new_la.paste(label, ((w-nw)//2, (h-nh)//2))
    
    return new_im, new_la