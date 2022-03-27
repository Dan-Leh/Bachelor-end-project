# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:59:21 2021

@author: sonne
"""

"""
The code is for the training of the single-head network using maximum squares loss on RGB-to-thermal adaptation.
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
sys.path.append('/home/jsun/Project/Siamese')

from model.deeplab_multi import DeeplabMulti
from dataset.dataset import Freiburg_Dataset, Freiburg_test_Dataset
from util import transform
from model.Loss import CrossEntropy2d, MaxSquareloss, IW_MaxSquareloss
from util.util import AverageMeter, intersectionAndUnion

IMG_MEAN = np.array((0.369,0.5), dtype=np.float32)
IMG_STD = np.array((0.321,0.5), dtype=np.float32)

BATCH_SIZE = 8
NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
INPUT_SIZE_TARGET = '640,320'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
ALPHA = 0.9
NUM_CLASSES = 14
START_EPOCH = 0
FIRST_EPOCHS = 50
EPOCHS = 150
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = ''
ROOT = '/home/jsun/Project/Freiburg'
LOG_SAVE_PATH = '/home/jsun/Project/Siamese/logs'
SNAPSHOT_DIR = '/home/jsun/Project/Siamese/snapshots'
VAL_FOLDER = '/home/jsun/Project/Siamese/val'
TARGET_MODE = 'maxsquare'
WEIGHT_DECAY = 0.0005
LAMBDA_SEG = 0.1
LAMBDA_TARGET = 0.1
THRESHOLD = 0.95
GPU = 2

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
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-target", type=float, default=LAMBDA_TARGET,
                        help="lambda_target")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="threshold")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Alpha of RMSprop optimizer")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="The start epoch.")
    parser.add_argument("--first-epochs", type=int, default=FIRST_EPOCHS,
                        help="The first epochs.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="The total epoch.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help="The target mode.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--val-folder", type=str, default=VAL_FOLDER,
                        help="The folder for validation.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--logs", type=str, default=LOG_SAVE_PATH,
                        help="where to save the logs")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def target_loss(pred, label, mode, gpu):
    if mode == 'maxsquare':
        Loss = MaxSquareloss(ignore_index=-1, num_class=14).cuda(gpu)
    else:
        Loss = IW_MaxSquareloss(ignore_index=-1, num_class=14, ratio=0.2).cuda(gpu)
    return Loss(pred, label)

def target_hard_loss(pred, label, gpu):
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(gpu)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_1(optimizer, i_iter, total_iter, lr):
    lr = lr_poly(lr, i_iter, total_iter, power=0.9)
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate_2(optimizer):
    optimizer.param_groups[0]['lr'] /= 2
    
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color
    
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
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    print(input_size)
    cudnn.enabled = True
    gpu = args.gpu
   
    model = DeeplabMulti(args.num_classes)
    
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=args.momentum, \
                            weight_decay=args.weight_decay)
    
    optimizer.zero_grad()
    
    model.cuda(args.gpu)
    cudnn.benchmark == True
    
    global writer, logger
    writer = SummaryWriter(args.logs)
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    #logger.info(model)
    
    if args.restore_from:
        checkpoint = torch.load(args.restore_from, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.restore_from, checkpoint))
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    mean = IMG_MEAN
    std = IMG_STD
    
    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True) 
    
    trainloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'day', train_transform), \
                        batch_size=args.batch_size, shuffle=True, \
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    test_transform = transform.Compose([transform.ToTensor(),
                                        transform.Normalize(mean=mean, std=std)])
    test_data = Freiburg_test_Dataset('test',  ROOT, 'day', transform=test_transform)
    testloader = data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    
    colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]]).astype('uint8')
    names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', \
             'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Ignore']
    
    for epoch in range(args.start_epoch, args.epochs):
        
        epoch_log = epoch + 1
        trainloader_iter = iter(trainloader)
        
        max_iter = len(trainloader)
        
        if 0 <= epoch < args.first_epochs:
            
            loss_seg_value = 0
            
            for i_iter in range(max_iter):
                current_iter = epoch * max_iter + i_iter + 1
                adjust_learning_rate_1(optimizer, current_iter, max_iter*args.first_epochs, args.learning_rate)
                
                img, label = trainloader_iter.next()
                # print(img.shape)
                grayscale = img[:,0,:,:]
                grayscale = grayscale.unsqueeze(dim=1)
                grayscale = grayscale.repeat(1,3,1,1)
                
                loss = train1(model, grayscale, label, epoch_log, i_iter, max_iter, \
                              current_iter, optimizer, interp)
                
                loss_seg_value += loss[-1]
            
            writer.add_scalar('loss_seg_value', loss_seg_value / max_iter, epoch_log)
        else:   
        
            loss_seg = 0
            loss_target = 0
            
            for i_iter in range(max_iter):
                current_iter = epoch * max_iter + i_iter + 1 
                adjust_learning_rate_1(optimizer, current_iter-max_iter*args.first_epochs, \
                                       max_iter*(args.epochs-args.first_epochs), args.learning_rate)
            
                img, label = trainloader_iter.next()           
                grayscale = img[:,0,:,:]
                thermal = img[:,1,:,:]

                grayscale = grayscale.unsqueeze(dim=1)
                thermal = thermal.unsqueeze(dim=1)
                grayscale = grayscale.repeat(1,3,1,1)
                thermal = thermal.repeat(1,3,1,1)
            
                loss = train2(model, grayscale, thermal, label, epoch_log, i_iter, \
                             max_iter, current_iter, optimizer, interp)
            
                loss_seg += loss[2]
                loss_target += loss[5]
            
            writer.add_scalar('loss_seg', loss_seg/max_iter, epoch_log)
            writer.add_scalar('loss_target', loss_target/max_iter, epoch_log)
        
        if epoch_log >= 1:
            filename = args.snapshot_dir + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'model_state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, filename)
        if epoch_log / 1 > 2:
            if (epoch_log-2) % 10 == 0:
                continue
            else:
                deletename = args.snapshot_dir + '/train_epoch_' + str(epoch_log-2) + '.pth'
                os.remove(deletename)
        
        for modality in ['grayscale', 'thermal']:
            gray_folder = os.path.join(args.val_folder, 'day', modality, 'gray')
            color_folder = os.path.join(args.val_folder, 'day', modality, 'color')
            evaluate(testloader, test_data.data_list, model, args.num_classes, modality, \
                     interp, gray_folder, color_folder, colors)
            cal_acc(test_data.data_list, gray_folder, args.num_classes, names, modality, epoch_log)

def train1(model, gray, label, epoch_log, i_iter, max_iter, current_iter, optimizer, interp):
    model.train()
    optimizer.zero_grad()
    
    gray = gray.cuda(args.gpu)
    pred1, pred2 = model(gray)
    pred1 = interp(pred1)
    pred2 = interp(pred2)
    loss_seg1 = loss_calc(pred1, label, args.gpu)
    loss_seg2 = loss_calc(pred2, label, args.gpu)
    loss_seg_batch = loss_seg1 + args.lambda_seg * loss_seg2
    loss_seg_batch.backward()
    
    loss_seg_value1 = loss_seg1.data.cpu().numpy()
    loss_seg_value2 = loss_seg2.data.cpu().numpy()
    loss_seg_value_batch = loss_seg_batch.data.cpu().numpy()
    optimizer.step()
    
    if i_iter > 0 and i_iter % 10 == 0:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Loss_seg1 {loss_seg_value1:.4f} '
                    'Loss_seg2 {loss_seg_value2:.4f} '
                    'Loss_seg {loss_seg_value_batch:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                                  loss_seg_value1=loss_seg_value1,
                                                                  loss_seg_value2=loss_seg_value2,
                                                                  loss_seg_value_batch=loss_seg_value_batch))
    
    writer.add_scalar('loss_seg_value1', loss_seg_value1, current_iter)
    writer.add_scalar('loss_seg_value2', loss_seg_value2, current_iter)
    writer.add_scalar('loss_seg_value_batch', loss_seg_value_batch, current_iter)
    writer.add_scalar('learning_rate_stage1_0', optimizer.param_groups[0]['lr'], current_iter)
    writer.add_scalar('learning_rate_stage1_1', optimizer.param_groups[1]['lr'], current_iter)
    
    return loss_seg_value1, loss_seg_value2, loss_seg_value_batch
    
def train2(model, gray, thermal, label, epoch_log, i_iter, max_iter, current_iter, \
          optimizer, interp):  
    model.train()
    optimizer.zero_grad()
    
    gray = gray.cuda(args.gpu)
    thermal = thermal.cuda(args.gpu)
    
    # source loss
    pred1, pred2 = model(gray)
    pred1 = interp(pred1)
    pred2 = interp(pred2)
    loss_seg1 = loss_calc(pred1, label, args.gpu)
    loss_seg2 = loss_calc(pred2, label, args.gpu)
    loss_seg_batch = loss_seg1 + args.lambda_seg * loss_seg2
    loss_seg_batch.backward()
    
    loss_seg_value1 = loss_seg1.data.cpu().numpy()
    loss_seg_value2 = loss_seg2.data.cpu().numpy()
    loss_seg_value_batch = loss_seg_batch.data.cpu().numpy()
    
    # target loss
    pred_target1, pred_target2 = model(thermal)
    pred_target1 = interp(pred_target1)
    pred_target2 = interp(pred_target2)
    pred_target_P1 = F.softmax(pred_target1, dim=1)
    pred_target_P2 = F.softmax(pred_target2, dim=1)
    
    label_target1 = pred_target_P1
    label_target2 = pred_target_P2
    
    maxpred_1, argpred_1 = torch.max(pred_target_P1.detach(), dim=1)
    maxpred_2, argpred_2 = torch.max(pred_target_P2.detach(), dim=1)
    
    loss_target_1 = args.lambda_target * target_loss(pred_target1, label_target1, \
                                                     args.target_mode, args.gpu)
    # print(loss_target_1)
    loss_target_1_value = loss_target_1.data.cpu().numpy()
    
    pred_c = (pred_target_P1 + pred_target_P2) / 2
    maxpred_c, argpred_c = torch.max(pred_c, dim=1)
    mask = (maxpred_1 > args.threshold) | (maxpred_2 > args.threshold)
    
    label_target2 = torch.where(mask, argpred_c, torch.ones(1).to(args.gpu, dtype=torch.long)*(-1))
    loss_target_2 = args.lambda_seg * args.lambda_target * target_hard_loss(pred_target2, label_target2, args.gpu)
    # print(loss_target_2)
    loss_target_2_value = loss_target_2.data.cpu().numpy()
    
    loss_target = loss_target_1 + loss_target_2
    loss_target_value_batch = loss_target.data.cpu().numpy()
    loss_target.backward()
    
    optimizer.step()
    
    if i_iter > 0 and  i_iter % 10 == 0:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Loss_seg {loss_seg_value_batch:.4f} '
                    'Loss_target_value1 {loss_target_value1:.4f} '
                    'Loss_target_value2 {loss_target_value2:.4f} '
                    'Loss_target {loss_target_value_batch:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                                  loss_seg_value_batch=loss_seg_value_batch,
                                                                  loss_target_value1=loss_target_1_value,
                                                                  loss_target_value2=loss_target_2_value,
                                                                  loss_target_value_batch=loss_target_value_batch))
    
    writer.add_scalar('loss_seg_value1', loss_seg_value1, current_iter)
    writer.add_scalar('loss_seg_value2', loss_seg_value2, current_iter)
    writer.add_scalar('loss_seg_value_batch', loss_seg_value_batch, current_iter)
    writer.add_scalar('loss_target_value1', loss_target_1_value, current_iter)
    writer.add_scalar('loss_target_value2', loss_target_2_value, current_iter)
    writer.add_scalar('loss_target_value_batch', loss_target_value_batch, current_iter)
    writer.add_scalar('learning_rate_stage2_0', optimizer.param_groups[0]['lr'], current_iter)
    writer.add_scalar('learning_rate_stage2_1', optimizer.param_groups[1]['lr'], current_iter)
    
    return loss_seg_value1, loss_seg_value2, loss_seg_value_batch, \
           loss_target_1_value, loss_target_2_value, loss_target_value_batch

def evaluate(test_loader, data_list, model, classes, modality, interp, gray_folder, \
             color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()
    
    for i, (img, label) in enumerate(test_loader):
       
        if modality == 'grayscale':
            inputs = img[:,0,:,:]
        else:
            inputs = img[:,1,:,:]
        inputs = inputs.unsqueeze(dim=1)
        inputs = inputs.repeat(1,3,1,1)
        inputs = inputs.cuda(args.gpu)

        prediction = model(inputs)
        pred = prediction[0]
        pred = interp(pred)
        pred_P = F.softmax(pred, dim=1)
        def flip(x, dim):
            dim = x.dim() + dim if dim < 0 else dim
            inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
            return x[inds]
        inputs_flip = flip(inputs, -1)
        prediction_flip = model(inputs_flip)
        pred_flip = prediction_flip[0]
        pred_flip = interp(pred_flip)
        pred_P_flip = F.softmax(pred_flip, dim=1)
        pred_P_2 = flip(pred_P_flip, -1)
        pred_c = (pred_P + pred_P_2) / 2
        pred = pred_c.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)[0]

        gray = np.uint8(pred)
        gray = cv2.resize(gray, (1280,640), cv2.INTER_NEAREST)
        color = colorize(gray, colors)
        rgb_image_path, _, _ = data_list[i]
        image_name = rgb_image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

def cal_acc(data_list, pred_folder, classes, names, modality, epoch):
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

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    mIoU_12 = sum(iou_class[:12]) / 12

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Eval result on 12 classes: mIoU {:.4f}.'.format(mIoU_12))
    
    if modality == 'grayscale':
        writer.add_scalar('mIoU-grayscale', mIoU_12, epoch)
    else:
        writer.add_scalar('mIoU-thermal', mIoU_12, epoch)
# def train(model, gray, thermal, label, epoch_log, i_iter, max_iter, current_iter, \
#           optimizer, interp, criterion):
    
#     # batch_time = AverageMeter('Time', ':6.3f')
#     # data_time = AverageMeter('Data', ':6.3f')
#     # losses = AverageMeter('Loss', ':.4f')
    
#     model.train()
    
#     end = time.time()
    
#     gray = gray.cuda(args.gpu)
#     thermal = thermal.cuda(args.gpu)
#     # print(gray.shape,thermal.shape)
    
#     # p1, p2, z1, z2, pred, pred_aux = model(gray, thermal)
    
#     # loss_similarity = -(criterion(p1,z2).mean() + criterion(p2,z1).mean()) * 0.5
#     # #print(loss_similarity)
#     # loss_similarity_value = loss_similarity.data.cpu().numpy()
    
#     # pred = interp(pred)
#     # pred_aux = interp(pred_aux)
#     # loss_seg1 = loss_calc(pred, label, args.gpu)
#     # loss_seg2 = loss_calc(pred_aux, label, args.gpu)
#     # loss_seg = loss_seg1 + args.lambda_seg * loss_seg2
#     # #print(loss_seg1, loss_seg2, loss_seg)
#     # loss_seg1_value = loss_seg1.data.cpu().numpy()
#     # loss_seg2_value = loss_seg2.data.cpu().numpy()
#     # loss_seg_value = loss_seg.data.cpu().numpy()
    
#     # loss_batch = loss_similarity + loss_seg
#     # #print(loss_batch)
#     # loss_batch_value = loss_batch.data.cpu().numpy()
    
#     x1, x2, pred, pred_aux = model(gray, thermal)
    
#     loss_gap = MMD(x1, x2)
#     #print(loss_gap)
#     loss_gap_value = loss_gap.data.cpu().numpy()
    
#     pred = interp(pred)
#     pred_aux = interp(pred_aux)
#     loss_seg1 = loss_calc(pred, label, args.gpu)
#     loss_seg2 = loss_calc(pred_aux, label, args.gpu)
#     loss_seg = loss_seg1 + args.lambda_seg * loss_seg2
#     #print(loss_seg1, loss_seg2, loss_seg)
#     loss_seg1_value = loss_seg1.data.cpu().numpy()
#     loss_seg2_value = loss_seg2.data.cpu().numpy()
#     loss_seg_value = loss_seg.data.cpu().numpy()
    
#     loss_batch = loss_gap * 0.25 + loss_seg
#     #print(loss_batch)
#     loss_batch_value = loss_batch.data.cpu().numpy()
    
#     optimizer.zero_grad()
#     loss_batch.backward()
#     optimizer.step()
    
#     if i_iter % 10 == 0:
#         logger.info('Epoch: [{}/{}][{}/{}] '
#                     'Seg_Loss {loss_seg:.4f} '
#                     'Gap_Loss {loss_gap:.4f} '
#                     'Loss {loss_batch:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
#                                                                    loss_seg = loss_seg_value,
#                                                                    loss_gap = loss_gap_value,
#                                                                    loss_batch = loss_batch_value))
    
#     writer.add_scalar('loss_seg1_batch', loss_seg1_value, current_iter)
#     writer.add_scalar('loss_seg2_batch', loss_seg2_value, current_iter)
#     writer.add_scalar('loss_seg_batch', loss_seg_value, current_iter)
#     writer.add_scalar('loss_similarity_batch', loss_gap_value, current_iter)
#     writer.add_scalar('loss_batch', loss_batch_value, current_iter)
    
#     return loss_seg_value, loss_gap_value, loss_batch_value
    
if __name__ == '__main__':
    main()
