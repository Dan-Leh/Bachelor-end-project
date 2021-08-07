import os
import os.path
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    sets = {'train':'training', 'val':'validation'}
    set = sets[split]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, set, 'images', line_split[0]).replace('\\', '/')
            label_name = os.path.join(data_root, set, 'v1.2', 'labels', line_split[1]).replace('\\', '/')
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


# data_root = 'E:/graduation project/mapillary_vistas_v2_part/'
# data_list = 'E:/graduation project/mapillary_vistas_v2_part/trainset.txt'
# data = make_dataset('train',data_root,data_list)

# import json
# file1 = open(r'E:\graduation project\mapillary_vistas_v2_part\config_v1.2.json','r')
# content1 = file1.read()
# f1 = json.loads(content1)
# name_v1 = []
# for ann in f1['labels']:
#     name_v1.append(ann['readable'])

# t1 = open(r'E:\graduation project\mapillary_vistas_v2_part\classes_v1.0.txt','w')
# t2 = open(r'E:\graduation project\mapillary_vistas_v2_part\colors_v1.0.txt','w')
# for ann in f1['labels']:
#     name = ann['readable']
#     color = ann['color']
#     t1.write(name+'\n')
#     t2.write(str(color[0])+' '+str(color[1])+' '+str(color[2])+'\n')
# t1.close()
# t2.close()
# #%%
# train = open(r'E:\graduation project\mapillary_vistas_v2_part\train.txt','r')
# tra = train.readlines()
# val = open(r'E:\graduation project\mapillary_vistas_v2_part\val.txt','r')
# va = val.readlines()
# trainset = open(r'E:\graduation project\mapillary_vistas_v2_part\trainset.txt','w')
# valset = open(r'E:\graduation project\mapillary_vistas_v2_part\valset.txt','w')
# for tr in tra:
#     trainset.write(tr.strip()+'.jpg'+' '+tr.strip()+'.png'+'\n')
# trainset.close()
# for v in va:
#     valset.write(v.strip()+'.jpg'+' '+v.strip()+'.png'+'\n')
# valset.close()
# #%%
# file2 = open(r'E:\graduation project\mapillary_vistas_v2_part\config_v2.0.json','r')
# content2 = file2.read()
# f2 = json.loads(content2)
# name_v2 = []
# for ann in f2['labels']:
#     name_v2.append(ann['readable'])

class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        print('##############################################################################################',image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = cv2.resize(image, (473,473))
        image = np.float32(image)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label = Image.open(label_path)
        # label = label.resize((473,473), Image.NEAREST)
        label = np.array(label)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
# #%%
# import torch
# import torchvision.transforms as transforms

# import sys
# sys.path.append(r'E:\graduation project\semseg-master')
# from util import dataset, transform, config

# train_transform = transform.Compose([
#         transform.RandScale([0.5,2.0]),
#         transform.RandRotate([-10, 10], padding=[0.485, 0.456, 0.406], ignore_label=255),
#         transform.RandomGaussianBlur(),
#         transform.RandomHorizontalFlip(),
#         transform.Crop([473, 473], crop_type='rand', padding=[0.485, 0.456, 0.406], ignore_label=255),
#         transform.ToTensor()])
# data_root = 'E:/graduation project/mapillary_vistas_v2_part/'
# train_list = 'E:/graduation project/mapillary_vistas_v2_part/trainset.txt'
# train_data = SemData(split='train', data_root=data_root, data_list=train_list, transform=train_transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=8)
# for i, batch in enumerate(train_loader):
#     print(batch[0].shape, batch[1].shape)