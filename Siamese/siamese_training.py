# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:59:21 2021

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

import sys
sys.path.append('/home/jsun/Project/Siamese')
from model.siamese_net import Siamese_net
from dataset.dataset import Freiburg_Dataset
from util import transform
from model.Loss import CrossEntropy2d, MMD, MaxSquareloss, IW_MaxSquareloss

IMG_MEAN = np.array((0.459,0.0), dtype=np.float32)
IMG_STD = np.array((0.226,1.0), dtype=np.float32)

BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
INPUT_SIZE_TARGET = '640,320'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
ALPHA = 0.9
NUM_CLASSES = 14
START_EPOCH = 0
EPOCHS = 100
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = ''
ROOT = 'E:/graduation project/Freiburg'
LOG_SAVE_PATH = 'C:/Users/sonne/Downloads/graduation-project-2020-sonnefred-main/Siamese/logs'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = 'C:/Users/sonne/Downloads/graduation-project-2020-sonnefred-main/Siamese/snapshots'
WEIGHT_DECAY = 0.0001
LAMBDA_SEG = 0.4

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese-ResNet Network")
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
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
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
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="The total epoch.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
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

def target_loss(pred, label, gpu):
    Loss = IW_MaxSquareloss().cuda(gpu)
    return Loss(pred, label)

def target_hard_loss(pred, label, gpu):
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(gpu)
    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, total_iter):
    lr = lr_poly(args.learning_rate, i_iter, total_iter, power=0.9)
    optimizer.param_groups[0]['lr'] = lr
    
# def adjust_learning_rate(optimizer):
#     optimizer.param_groups[0]['lr'] /= 2
    
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
   
    model = Siamese_net(args.num_classes, 50)
    
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, \
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
    
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)
    
    for epoch in range(args.start_epoch, args.epochs):       
        epoch_log = epoch + 1
        
        trainloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'day', train_transform), \
                        batch_size=args.batch_size, shuffle=True, \
                        num_workers=args.num_workers, pin_memory=True)          
        trainloader_iter = iter(trainloader)        
        
        max_iter = len(trainloader)
        
        loss_seg = 0
        loss_gap = 0
        total_loss = 0
            
        for i_iter in range(max_iter):
            current_iter = epoch * max_iter + i_iter + 1
            adjust_learning_rate(optimizer, current_iter, max_iter*args.epochs)
            
            img, label = trainloader_iter.next()           
            grayscale = img[:,0,:,:]
            thermal = img[:,1,:,:]
            grayscale = grayscale.unsqueeze(dim=1)
            thermal = thermal.unsqueeze(dim=1)
            
            loss = train(model, grayscale, thermal, label, epoch_log, i_iter, \
                         max_iter, current_iter, optimizer, interp)
            
            loss_seg += loss[0]
            loss_gap += loss[1]
            total_loss += loss[2]
            
        writer.add_scalar('loss_seg', loss_seg/max_iter, epoch_log)
        writer.add_scalar('loss_gap', loss_gap/max_iter, epoch_log)
        writer.add_scalar('total_loss', total_loss/max_iter, epoch_log)
        
        if (epoch_log % 1 == 0):
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

def train(model, gray, thermal, label, epoch_log, i_iter, max_iter, current_iter, \
          optimizer, interp):  
    model.train()
    gray = gray.cuda(args.gpu)
    thermal = thermal.cuda(args.gpu)
    
    # source loss
    pred1, pred2 = model(gray)
    pred1 = interp(pred1)
    pred2 = interp(pred2)
    loss_seg1 = loss_calc(pred1, label, args.gpu)
    loss_seg2 = loss_calc(pred2, label, args.gpu)
    loss_seg_batch = loss_seg1 + args.lambda_seg * loss_seg2
    print(loss_seg_batch)
    loss_seg_batch.backward()
    
    loss_seg_value1 = loss_seg1.data.cpu().numpy()
    loss_Seg_value2 = loss_seg2.data.cpu().numpy()
    
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
    
    # loss_target_1 = args.lambda_target * target_loss(pred_target1, label_target1)
    loss_target_1 = 0.25 * target_loss(pred_target1, label_target1, args.gpu)
    print(loss_target_1)
    pred_c = (pred_target_P1 + pred_target_P2) / 2
    maxpred_c, argpred_c = torch.max(pred_c, dim=1)
    # mask = (maxpred_1 > args.threshold) | (maxpred_2 > args.threshold)
    mask = (maxpred_1 > 0.5) | (maxpred_2 > 0.5)
    
    label_target2 = torch.where(mask, argpred_c, torch.ones(1).to(args.gpu, dtype=torch.long)*(-1))
    # loss_target_2 = args.lambda_seg * args.lambda_target * target_hard_loss(pred_target2, label_target2)
    loss_target_2 = 0.25 * 0.5 * target_hard_loss(pred_target2, label_target2, args.gpu)
    print(loss_target_2)
    loss_target = loss_target_1 + loss_target_2
    loss_target.backward()
    
    optimizer.step()
    optimizer.zero_grad()

               
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
