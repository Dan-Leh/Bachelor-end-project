# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:19:48 2021

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
from model.HeatNet_PSP import HeatNet_PSP

from util.dataset import Freiburg_Dataset
from util import transform

IMG_MEAN = [0.485, 0.456, 0.406, 0.0]
IMG_STD = [0.229, 0.224, 0.225, 1.0]

BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
INPUT_SIZE_TARGET = '640,320'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
ALPHA = 0.9
NUM_CLASSES = 14
START_EPOCH = 150
EPOCHS_ONE_STAGE = 50
EPOCHS = 200
POWER = 0.9
RESTORE_FROM = ''
WEIGHT = ''
ROOT = '/home/jsun/Project/Freiburg/'
LOG_SAVE_PATH = '/home/jsun/Project/HeatNet-master/two_stage_training/logs/'
SNAPSHOT_DIR = '/home/jsun/Project/HeatNet-master/two_stage_training/snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 5e-5
LAMBDA_SEG = 0.4
LAMBDA_ADV = 0.01
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

GPU = 0

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="HeatNet-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
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
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                        help="lambda_adv.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Alpha of RMSprop optimizer")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="The start epoch.")
    parser.add_argument("--epochs-one-stage", type=int, default=EPOCHS_ONE_STAGE,
                        help="Epochs for stage one.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="The total epoch.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--weight", type=str, default=None,
                        help="Weight to initialize the model.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
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

def adjust_learning_rate(optimizer):
    optimizer.param_groups[0]['lr'] /= 2
    
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
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)
    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)
    # print(input_size)
    cudnn.enabled = True

    # create network
    model = HeatNet_PSP(classes=args.num_classes)    
    model.train()
    model.cuda(args.gpu)
    # model = torch.nn.DataParallel(model.cuda())

    cudnn.benchmark = True
   
    optimizer= optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=args.alpha, \
                             weight_decay = args.weight_decay)
    optimizer.zero_grad()
    
    # initializae D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)
    
    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    optimizer_D1 = optim.Adam(model_D1.parameters(),lr=args.learning_rate_D,betas=(0.9,0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(),lr=args.learning_rate_D,betas=(0.9,0.99))
    optimizer_D2.zero_grad()
    
    global writer, logger
    writer = SummaryWriter(args.logs)
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    logger.info(model)
    
    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
    
    if args.restore_from:
        logger.info("=> loading checkpoint '{}'".format(args.restore_from))
        
        # checkpoint = torch.load(args.restore_from, map_location=torch.device('cpu'))        
        checkpoint = torch.load(args.restore_from, map_location=lambda storage, loc: storage.cuda(args.gpu))
        
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_D1.load_state_dict(checkpoint['model_D1_state_dict'])
        optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
        model_D2.load_state_dict(checkpoint['model_D2_state_dict'])
        optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
        
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.restore_from, checkpoint['epoch']))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir) 
    
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial learning
    source_label = 0
    target_label = 1

    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    mean = IMG_MEAN
    std = IMG_STD
       
    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    target_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    # create dataloader   
    trainloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'day', train_transform), \
                        batch_size=args.batch_size, shuffle=False, \
                        num_workers=args.num_workers, pin_memory=True)
    
    targetloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'night', target_transform), \
                        batch_size=args.batch_size, shuffle=False, \
                        num_workers=args.num_workers, pin_memory=True)  
           
    for epoch in range(args.start_epoch, args.epochs):
        
        epoch_log = epoch + 1
        
        trainloader_iter = iter(trainloader)     
        targetloader_iter = iter(targetloader)

        max_iter = len(trainloader)  # how many steps in one epoch
        
        if epoch > 0 and epoch % 30 == 0: 
            adjust_learning_rate(optimizer)
        
        if epoch > args.epochs_one_stage and (epoch - args.epochs_one_stage) % 30 == 0:
            adjust_learning_rate(optimizer_D1)
            adjust_learning_rate(optimizer_D2)
        
        if 0 <= epoch < args.epochs_one_stage:
            loss_rgb_seg = 0
            loss_thermal_seg = 0
            loss_seg_stage1 = 0
        
            for i_iter in range(max_iter):
                current_iter = epoch * max_iter + i_iter + 1
            
                images_source, labels_source = trainloader_iter.next()
                images_target, labels_target = targetloader_iter.next()

                loss = train_stage_one(epoch_log, model, images_source, images_target, 
                                       labels_source, labels_target, \
                                       optimizer, i_iter+1, current_iter, max_iter, interp, interp_target)
            
                loss_rgb_seg += loss[0]
                loss_thermal_seg += loss[1]
                loss_seg_stage1 += loss[2]
        
            writer.add_scalar('loss_rgb_seg', loss_rgb_seg/max_iter, epoch_log)
            writer.add_scalar('loss_thermal_seg', loss_thermal_seg/max_iter, epoch_log)
            writer.add_scalar('loss_seg_stage1', loss_seg_stage1/max_iter, epoch_log)
            
            if (epoch_log % 1 == 0):
                filename = args.snapshot_dir + 'train_epoch_' + str(epoch_log) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch_log, 'model_state_dict': model.state_dict(), \
                            'optimizer': optimizer.state_dict()}, filename)
                if epoch_log / 1 > 2:
                    if (epoch_log-2) % 10 == 0:
                         continue
                    else:
                        deletename = args.snapshot_dir + '/train_epoch_' + str(epoch_log - 1 * 2) + '.pth'
                        os.remove(deletename)
        
        else:
            loss_seg_stage2 = 0
            loss_adv = 0
            loss_D1 = 0
            loss_D2 = 0
            
            for i_iter in range(max_iter):
                current_iter = epoch * max_iter + i_iter + 1
                
                images_source, labels_source = trainloader_iter.next()
                images_target, labels_target = targetloader_iter.next()

                loss = train_stage_two(epoch_log, model, model_D1, model_D2, images_source, labels_source, \
                                       images_target, source_label, target_label, optimizer, optimizer_D1, \
                                       optimizer_D2, i_iter, current_iter, max_iter, interp, interp_target) 
                
                loss_seg_stage2 += loss[0]
                loss_adv += loss[1]
                loss_D1 += loss[2]
                loss_D2 += loss[3]
            
            writer.add_scalar('loss_seg_stage2', loss_seg_stage2/max_iter, epoch_log)
            writer.add_scalar('loss_adv', loss_adv/max_iter, epoch_log)
            writer.add_scalar('loss_D1', loss_D1/max_iter, epoch_log)
            writer.add_scalar('loss_D2', loss_D2/max_iter, epoch_log)
        
            if (epoch_log % 1 == 0):
                filename = args.snapshot_dir + '/train_epoch_' + str(epoch_log) + '.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch_log, 'model_state_dict': model.state_dict(), \
                            'optimizer': optimizer.state_dict(), \
                            'model_D1_state_dict': model_D1.state_dict(), \
                            'model_D2_state_dict': model_D2.state_dict(), \
                            'optimizer_D1': optimizer_D1.state_dict(), \
                            'optimizer_D2': optimizer_D2.state_dict()}, filename)
                if epoch_log / 1 > 2:
                    if (epoch_log-2) % 10 == 0:
                         continue
                    else:
                        deletename = args.snapshot_dir + '/train_epoch_' + str(epoch_log - 1 * 2) + '.pth'
                        os.remove(deletename)

def train_stage_one(epoch_log, model, images_source, images_target, labels_source, labels_target, optimizer, \
                    i_iter, current_iter, max_iter, interp, interp_target):
    
    optimizer.zero_grad()
    
    images_source = images_source.cuda(args.gpu)
    pred1, pred2 = model(images_source)
    pred1, pred2 = interp(pred1), interp(pred2)
    loss_rgb_seg1 = loss_calc(pred1, labels_source, args.gpu)
    loss_rgb_seg2 = loss_calc(pred2, labels_source, args.gpu)
    loss1 = loss_rgb_seg2 + args.lambda_seg * loss_rgb_seg1
    loss1.backward()
    loss_rgb_seg = loss1.data.cpu().numpy()
    
    images_target = images_target.cuda(args.gpu)
    pred_target1, pred_target2 = model(images_target)
    pred_target1, pred_target2 = interp_target(pred_target1), interp_target(pred_target2)
    loss_thermal_seg1 = loss_calc(pred_target1, labels_target, args.gpu)
    loss_thermal_seg2 = loss_calc(pred_target2, labels_target, args.gpu)
    loss2 = loss_thermal_seg2 + args.lambda_seg * loss_thermal_seg1
    loss2.backward()
    loss_thermal_seg = loss2.data.cpu().numpy()
    
    loss_seg_stage1 = (loss_rgb_seg + loss_thermal_seg) / 2
    optimizer.step()
    
    if i_iter % 10 == 0:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Loss_rgb_seg {loss_rgb_seg:.4f} '
                    'Loss_thermal_seg {loss_thermal_seg:.4f} '
                    'Loss_seg_stage1 {loss_seg_stage1:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                          loss_rgb_seg=loss_rgb_seg,
                                                          loss_thermal_seg=loss_thermal_seg,
                                                          loss_seg_stage1=loss_seg_stage1))
    
    writer.add_scalar('loss_rgb_seg_batch', loss_rgb_seg, current_iter)
    writer.add_scalar('loss_thermal_seg_batch', loss_thermal_seg, current_iter)
    writer.add_scalar('loss_seg_stage1_batch', loss_seg_stage1, current_iter)
    
    return loss_rgb_seg, loss_thermal_seg, loss_seg_stage1

def train_stage_two(epoch_log, model, model_D1, model_D2, images_source, labels_source, images_target, \
                    source_label, target_label, optimizer, optimizer_D1, optimizer_D2, i_iter, \
                    current_iter, max_iter, interp, interp_target):

    optimizer.zero_grad()   
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()


    # train Seg, don't accumulate gradients in D
    for param in model_D1.parameters():
        param.requires_grad = False
    for param in model_D2.parameters():
        param.requires_grad = False
           
    # train with source data
    images_source = images_source.cuda(args.gpu)
    
    pred1, pred2 = model(images_source)   
    # print(pred1.shape,pred2.shape)                              
    pred1 = interp(pred1)                                    # b, 66, 650, 1920
    pred2 = interp(pred2)                                    # b, 66, 650, 1920
    # print(pred1.shape,pred2.shape)

    loss_seg1 = loss_calc(pred1, labels_source, args.gpu)
    loss_seg2 = loss_calc(pred2, labels_source, args.gpu)
    loss_seg_batch = loss_seg1 + args.lambda_seg * loss_seg2
    # print(loss_seg)
    loss_seg_batch.backward()
    
    loss_seg_value1 = loss_seg1.data.cpu().numpy()
    loss_seg_value2 = loss_seg2.data.cpu().numpy()

    # train with target data
    images_target = images_target.cuda(args.gpu)
    
    pred_target1, pred_target2 = model(images_target)
    pred_target1 = interp_target(pred_target1)
    pred_target2 = interp_target(pred_target2)

    D_out1 = model_D1(F.softmax(pred_target1, dim=1))
    D_out2 = model_D2(F.softmax(pred_target2, dim=1))
    
    loss_adv_target1 = MSE_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(args.gpu))
    loss_adv_target2 = MSE_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda(args.gpu))
    loss_adv_batch = args.lambda_adv * (loss_adv_target1 + 0.2 * loss_adv_target2)
    # print(loss_adv)

    loss_adv_batch.backward()
    loss_adv_target_value1 = loss_adv_target1.detach().cpu().numpy()
    loss_adv_target_value2 = loss_adv_target2.detach().cpu().numpy()
    # print(loss_adv_target_value1,loss_adv_target_value2)
            
    # train D

    for param in model_D1.parameters():
        param.requires_grad = True
    for param in model_D2.parameters():
        param.requires_grad = True
        
    # train with source
    pred1 = pred1.detach()
    pred2 = pred2.detach()
    
    D_out1 = model_D1(F.softmax(pred1, dim=1))
    D_out2 = model_D2(F.softmax(pred2, dim=1))
        
    loss_D1 = MSE_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda(args.gpu))
    loss_D2 = MSE_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda(args.gpu))
        
    loss_D1 = loss_D1 / 2
    loss_D2 = loss_D2 / 2

    loss_D1.backward()
    loss_D2.backward()
        
    loss_D_value1 = loss_D1.detach().cpu().numpy()
    loss_D_value2 = loss_D2.detach().cpu().numpy()
        
    # train with target
    pred_target1 = pred_target1.detach()
    pred_target2 = pred_target2.detach()
        
    D_out1 = model_D1(F.softmax(pred_target1, dim=1))
    D_out2 = model_D2(F.softmax(pred_target2, dim=1))
        
    loss_D1 = MSE_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda(args.gpu))
    loss_D2 = MSE_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).cuda(args.gpu))
        
    loss_D1 = loss_D1 / 2
    loss_D2 = loss_D2 / 2
        
    loss_D1.backward()
    loss_D2.backward()
        
    loss_D_value1 += loss_D1.detach().cpu().numpy()
    loss_D_value2 += loss_D2.detach().cpu().numpy()
    
    optimizer.step()
    optimizer_D1.step()
    optimizer_D2.step()
    
    if i_iter % 10 == 0:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Seg_Loss {seg_loss:.4f} '
                    'Adv_Loss {adv_loss:.4f} '
                    'Dis_Loss {dis_loss:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                          seg_loss=loss_seg_batch,
                                                          adv_loss=loss_adv_batch,
                                                          dis_loss=loss_D_value1+loss_D_value2))

    
    writer.add_scalar('loss_seg_value1_batch', loss_seg_value1, current_iter)
    writer.add_scalar('loss_seg_value2_batch', loss_seg_value2, current_iter)
    writer.add_scalar('loss_seg_batch', loss_seg_batch, current_iter)
    writer.add_scalar('loss_adv_target_value1_batch', loss_adv_target_value1, current_iter)
    writer.add_scalar('loss_adv_target_value2_batch', loss_adv_target_value2, current_iter)
    writer.add_scalar('loss_adv_target_batch', loss_adv_batch, current_iter)
    writer.add_scalar('loss_D_value1_batch', loss_D_value1, current_iter)
    writer.add_scalar('loss_D_value2_batch', loss_D_value2, current_iter)
    writer.add_scalar('loss_dis_batch',loss_D_value1+loss_D_value2, current_iter)
    
    return loss_seg_batch, loss_adv_batch, loss_D_value1, loss_D_value2


if __name__ == '__main__':
    main()