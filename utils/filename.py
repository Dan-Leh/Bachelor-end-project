# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:06:32 2021

@author: sonne
"""
import os

path = r'E:\graduation project\mapillary_vistas_v2_part'

train_file = os.listdir(os.path.join(path,'training/images'))
with open(os.path.join(path,'train.txt'),'w') as f:
    for tr in train_file:
        f.write(tr[:-4]+'\n')
    f.close()
val_file = os.listdir(os.path.join(path,'validation/images'))
with open(os.path.join(path,'val.txt'),'w') as f:
    for v in val_file:
        f.write(v[:-4]+'\n')
    f.close()
