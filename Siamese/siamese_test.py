# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:58:36 2021

@author: sonne
"""

import argparse
import logging
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import time
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
from PIL import Image
import cv2

import sys
sys.path.append('/home/jsun/Project/Siamese/')
from model.siamese_net import Siamese_net
from dataset.dataset import Freiburg_test_Dataset
from util import transform
from model.Loss import CrossEntropy2d, MMD, MaxSquareloss, IW_MaxSquareloss
from util.util import AverageMeter, intersectionAndUnion

IMG_MEAN = np.array((0.369,0.5), dtype=np.float32)
IMG_STD = np.array((0.321,0.5), dtype=np.float32)

BATCH_SIZE = 1
NUM_WORKERS = 0
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
NUM_CLASSES = 14
MODEL_PATH = '/home/jsun/Project/Siamese/snapshots/train_epoch_100.pth'
DATA_ROOT = '/home/jsun/Project/Freiburg/'
SAVE_PATH = '/home/jsun/Project/Siamese/results/'
DOMAIN = 'day'

COLORS = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]])

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Where restore model parameters from.")
    parser.add_argument("--domain", type=str, default=DOMAIN,
                        help="which domain to test.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="where to save results.")
    parser.add_argument("--gpu", type=int, default=2,
                        help="choose gpu device.")
    return parser.parse_args()

args = get_arguments()
   
def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    print(input_size)
    cudnn.enabled = True
    gpu = args.gpu
   
    model = Siamese_net(args.num_classes, 50) 
    model.cuda(args.gpu)
    cudnn.benchmark == True
    
    global logger
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    #logger.info(model)
    
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path,map_location=lambda storage,loc: storage.cuda(args.gpu))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    
    mean = IMG_MEAN
    std = IMG_STD
    
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True) 
    
    gray_folder = os.path.join(args.save_path, args.domain, 'gray')
    color_folder = os.path.join(args.save_path, args.domain, 'color')

    test_transform = transform.Compose([transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)])
    test_data = Freiburg_test_Dataset('test',  args.data_root, args.domain, transform=test_transform)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, \
                        num_workers=args.num_workers, pin_memory=True)
    
    colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]]).astype('uint8')
    names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', \
             'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Ignore']

    test(test_loader, test_data.data_list, model, args.num_classes, interp, \
                         gray_folder, color_folder, colors)

    cal_acc(test_data.data_list, gray_folder, args.num_classes, names)
    
def test(test_loader, data_list, model, classes, interp, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    
    for i, (img, label) in enumerate(test_loader):
        grayscale = img[:,0,:,:]
        thermal = img[:,1,:,:]
        grayscale = grayscale.unsqueeze(dim=1)
        thermal = thermal.unsqueeze(dim=1)
        grayscale = grayscale.cuda(args.gpu)
        thermal = thermal.cuda(args.gpu)

        prediction = model(grayscale)
        prediction = interp(prediction)
        prediction = F.softmax(prediction, dim=1)
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)[0]

        gray = np.uint8(prediction)
        gray = cv2.resize(gray, (1280,640), cv2.INTER_NEAREST)
        color = colorize(gray, colors)
        rgb_image_path, _, _ = data_list[i]
        image_name = rgb_image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, _, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = Image.open(target_path)
        target = np.asarray(target)
        target = target[5:645,360:1640]
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

if __name__ == '__main__':
    main()