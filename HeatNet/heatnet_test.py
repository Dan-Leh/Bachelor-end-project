"""
HeatNet implemented according to the code of the original paper.
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
from torchvision import transforms
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
from PIL import Image
import sys
sys.path.append('/home/jsun/Project/HeatNet-master/')

from model.discriminator import FCDiscriminator
from model.Loss import MSE_loss, CE_Loss, CrossEntropy2d
from model.conv_net import HeatNet
from util.dataset import Freiburg_Dataset
from util import transform
from util.util import AverageMeter, intersectionAndUnion

import time
import cv2
    
cv2.ocl.setUseOpenCL(False)

IMG_MEAN = [0.5, 0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5, 0.5]

NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
INPUT_SIZE_TARGET = '640,320'
NUM_CLASSES = 14
MODEL_PATH = '/home/jsun/Project/HeatNet-master/new_training/snapshots/train_epoch_80.pth'

DATA_ROOT = '/home/jsun/Project/Freiburg/'
DOMAIN = 'night'
SAVE_PATH = '/home/jsun/Project/Freiburg/train/night/label_uni/'

GPU = 0

COLORS = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]])

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def get_parser():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Where restore model parameters from.")
    parser.add_argument("--domain", type=str, default=DOMAIN,
                        help="which domain to test.")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH,
                        help="where to save results.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    # parser.add_argument("--colors", type=int, default=COLORS,
    #                     help="color palette.")
    return parser.parse_args()


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    mean = IMG_MEAN
    std = IMG_STD

    gray_folder = os.path.join(args.save_path, 'gray')
    color_folder = os.path.join(args.save_path, 'color')

    test_transform = transform.Compose([transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)])
    
    #test_transform = transform.Compose([transform.ToTensor()])
    
    test_data = Freiburg_Dataset('train',  args.data_root, args.domain, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, \
                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]]).astype('uint8')
    #names = [line.rstrip('\n') for line in open(args.names_path)]
    names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', \
             'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Ignore']

    model = HeatNet(num_classes=args.num_classes)
    #pre_trained_weights = torch.load(args.model_path)
    #model.load_state_dict({k: v for k, v in pre_trained_weights.items() if k in model.state_dict()},strict=False)
    # logger.info(model)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True
    
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path,map_location=lambda storage,loc: storage.cuda(args.gpu))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        
    interp = nn.Upsample(size=(input_size[1],input_size[0]), mode='bilinear', align_corners=True)
    test1(test_loader, test_data.data_list, model, args.num_classes, interp, \
                          gray_folder, color_folder, colors)
        
    # test(test_loader, test_data.data_list, model, args.num_classes, mean, std, \
    #      base_size=2048, crop_h=473, crop_w=473, scales=[1.0], gray_folder=gray_folder, \
    #      color_folder=color_folder, colors=colors)

    #cal_acc(test_data.data_list, gray_folder, args.num_classes, names)

def test1(test_loader, data_list, model, classes, interp, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    
    for i, (image, _, label) in enumerate(test_loader):
        rgb, ir = image[:,:3,:,:], image[:,3,:,:]
        ir = ir.unsqueeze(1)
        rgb, ir = rgb.cuda(args.gpu), ir.cuda(args.gpu)
        import time
        s = time.time()
        prediction = model([rgb,ir])
        e = time.time()
        print((e-s)*1000)
        prediction = prediction['pred_label_a']
        #print(prediction.shape)
        prediction = interp(prediction)
        #print(prediction.shape)
        prediction = F.softmax(prediction, dim=1)
        #print(prediction.shape)
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)[0]
        #print(prediction.shape)

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

def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))

        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
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
        target = cv2.resize(target,(640,320),cv2.INTER_NEAREST)
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