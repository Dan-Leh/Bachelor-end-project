import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def plot_img(output, target):
    colors = [(165, 42, 42), (0, 192, 0), (250, 170, 31), (250, 170, 32), (196, 196, 196),
              (190, 153, 153), (180, 165, 180), (90, 120, 150), (250, 170, 33), (250, 170, 34),
              (128, 128, 128), (250, 170, 35), (102, 102, 156), (128, 64, 255), (140, 140, 200),
              (170, 170, 170), (250, 170, 36), (250, 170, 160), (250, 170, 37), (96, 96, 96),
              (230, 150, 140), (128, 64, 128), (110, 110, 110), (110, 110, 110), (244, 35, 232),
              (128, 196, 128), (150, 100, 100), (70, 70, 70), (150, 150, 150), (150, 120, 90),
              (220, 20, 60), (220, 20, 60), (255, 0, 0), (255, 0, 100), (255, 0, 200),
              (255, 255, 255), (255, 255, 255), (250, 170, 29), (250, 170, 28), (250, 170, 26),
              (250, 170, 25), (250, 170, 24), (250, 170, 22), (250, 170, 21), (250, 170, 20),
              (255, 255, 255), (250, 170, 19), (250, 170, 18), (250, 170, 12), (250, 170, 11),
              (255, 255, 255), (255, 255, 255), (250, 170, 16), (250, 170, 15), (250, 170, 15),
              (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (64, 170, 64),
              (230, 160, 50), (70, 130, 180), (190, 255, 255), (152, 251, 152), (107, 142, 35),
              (0, 170, 30), (255, 255, 128), (250, 0, 30), (100, 140, 180), (220, 128, 128),
              (222, 40, 40), (100, 170, 30), (40, 40, 40), (33, 33, 33), (100, 128, 160),
              (20, 20, 255), (142, 0, 0), (70, 100, 150), (250, 171, 30), (250, 172, 30),
              (250, 173, 30), (250, 174, 30), (250, 175, 30), (250, 176, 30), (210, 170, 100),
              (153, 153, 153), (153, 153, 153), (128, 128, 128), (0, 0, 80), (210, 60, 60),
              (250, 170, 30), (250, 170, 30), (250, 170, 30), (250, 170, 30), (250, 170, 30),
              (250, 170, 30), (192, 192, 192), (192, 192, 192), (192, 192, 192), (220, 220, 0),
              (220, 220, 0), (0, 0, 196), (192, 192, 192), (220, 220, 0), (140, 140, 20),
              (119, 11, 32), (150, 0, 255), (0, 60, 100), (0, 0, 142), (0, 0, 90), (0, 0, 230),
              (0, 80, 100), (128, 64, 64), (0, 0, 110), (0, 0, 70), (0, 0, 142), (0, 0, 192),
              (170, 170, 170), (32, 32, 32), (111, 74, 0), (120, 10, 10), (81, 0, 81),
              (111, 111, 0), (0, 0, 0)]

    h = output.shape[0]
    w = output.shape[1]
    pre_img = np.zeros((h,w,3))
    tar_img = np.zeros((h,w,3))
    for c in range(len(colors)):
        pre_img[:,:,0] += ((output[:,: ] == c )*( colors[c][0] )).astype('uint8')
        pre_img[:,:,1] += ((output[:,: ] == c )*( colors[c][1] )).astype('uint8')
        pre_img[:,:,2] += ((output[:,: ] == c )*( colors[c][2] )).astype('uint8')
        tar_img[:,:,0] += ((target[:,: ] == c )*( colors[c][0] )).astype('uint8')
        tar_img[:,:,1] += ((target[:,: ] == c )*( colors[c][1] )).astype('uint8')
        tar_img[:,:,2] += ((target[:,: ] == c )*( colors[c][2] )).astype('uint8')
    return pre_img, tar_img

# im1 = np.array([[1,1,2,2,0],
#                 [0,1,3,3,1],
#                 [2,1,1,1,3],
#                 [1,3,3,0,0],
#                 [0,0,1,0,2]])
# im2 = np.array([[1,1,2,2,0],
#                 [0,1,3,3,1],
#                 [2,1,1,1,3],
#                 [1,3,3,0,0],
#                 [0,0,1,0,2]])
# img1, img2 = plot_img(im1, im2)

# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()