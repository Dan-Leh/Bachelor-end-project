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
import random
from tensorboardX import SummaryWriter
from PIL import Image
import cv2
import sys
sys.path.append('/home/jsun/Project/HeatNet-master/')

from model.discriminator import FCDiscriminator
from model.Loss import MSE_loss, CE_Loss, CrossEntropy2d
# from model.HeatNet_PSP import HeatNet_PSP
from model.conv_net import HeatNet
from util.dataset import Freiburg_Dataset
from util import transform
from util.util import AverageMeter, intersectionAndUnion


"""
HeatNet implemented according to the code of the original paper.
"""

# IMG_MEAN = [0.485, 0.456, 0.406, 0.0]
# IMG_STD = [0.229, 0.224, 0.225, 1.0]

IMG_MEAN = [0.5, 0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5, 0.5]

BATCH_SIZE = 4
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
INPUT_SIZE_TARGET = '640,320'
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
ALPHA = 0.9
NUM_CLASSES = 14
START_EPOCH = 0
EPOCHS_ONE_STAGE = 50
EPOCHS = 200
POWER = 0.9
RESTORE_FROM = ''
WEIGHT = ''
ROOT = '/home/jsun/Project/Freiburg/'
LOG_SAVE_PATH = '/home/jsun/Project/HeatNet-master/two_stage_train/logs/'
SNAPSHOT_DIR = '/home/jsun/Project/HeatNet-master/two_stage_train/snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
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

def criterion_semseg(pred, label, gpu):
    label = label.long().cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    
    return criterion(pred, label)

def criterion_semseg_weighted(pred, label, gpu):
    label = label.long().cuda(gpu)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    return criterion(pred, label)

def adjust_learning_rate(optimizer):
    optimizer.param_groups[0]['lr'] /= 2

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

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
    model = HeatNet(num_classes=args.num_classes,  num_dis=1)
    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True
   
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
    
    optimizer.zero_grad()
    
    # critics_params = []
    # for p in model.module.critics:
    #     critics_params.append({'params': p.parameters()})
    
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
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.restore_from, checkpoint['epoch']))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir) 
    
    criterion_conf = torch.nn.MSELoss()
    # labels for adversarial learning
    source_label = 0
    target_label = 1
    
    state = 'train_critic'
    counter = 1
    model.setPhase(state)
    
    mean = IMG_MEAN
    std = IMG_STD
       
    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    target_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    
    # create dataloader   
    trainloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'day', train_transform), \
                        batch_size=args.batch_size, shuffle=True, \
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    targetloader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'night', target_transform), \
                        batch_size=args.batch_size, shuffle=True, \
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
    test_data_day = Freiburg_Dataset('test', ROOT, 'day', test_transform)
    test_loader_day = data.DataLoader(test_data_day, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
     
    test_data_night = Freiburg_Dataset('test', ROOT, 'night', test_transform)
    test_loader_night = data.DataLoader(test_data_night, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    path = '/home/jsun/Project/HeatNet-master/val/'
    gray_folder_day = path + 'day/gray'
    color_folder_day = path + 'day/color'
    gray_folder_night = path + 'night/gray'
    color_folder_night = path + 'night/color'
    pred_folder_day = gray_folder_day
    pred_folder_night = gray_folder_night
    
    colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]]).astype('uint8')

    names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', \
             'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Ignore']
    
    best_iou = [0, 0, 0, 0]
        
    for epoch in range(args.start_epoch, args.epochs):
        
        epoch_log = epoch + 1
        trainloader_iter = iter(trainloader)
        targetloader_iter = iter(targetloader)
        
        max_iter = len(trainloader)
        
        for i_iter in range(1, max_iter+1):
            
            current_iter = i_iter + max_iter * epoch
            
            images_source, labels_source = trainloader_iter.next()
            images_target, labels_target, cert = targetloader_iter.next()
            
            rgb_day = images_source[:,:3,:,:].cuda(args.gpu)
            ir_day = images_source[:,3,:,:].cuda(args.gpu)
            ir_day = ir_day.unsqueeze(1)
            rgb_night = images_target[:,:3,:,:].cuda(args.gpu)
            ir_night = images_target[:,3,:,:].cuda(args.gpu)
            ir_night = ir_night.unsqueeze(1)
            
            state = train(model, optimizer, rgb_day, ir_day, rgb_night, ir_night, labels_source,\
                  labels_target, source_label, target_label, cert, state, counter, \
                  epoch_log, i_iter, current_iter, max_iter, criterion_conf)
        
        lr_scheduler.step()
        
        evaluate(test_loader_day, test_data_day.data_list, model, args.num_classes,\
                  gray_folder_day, color_folder_day, colors, 'day')
        evaluate(test_loader_night, test_data_night.data_list, model, args.num_classes,\
                  gray_folder_night, color_folder_night, colors, 'night')
        
        mIoU_day = cal_acc(test_data_day.data_list, pred_folder_day, args.num_classes, names)
        mIoU_night = cal_acc(test_data_night.data_list, pred_folder_night, args.num_classes, names)       
        mean_iou = (mIoU_day + mIoU_night) / 2
        
        writer.add_scalar('mIoU_day', mIoU_day, epoch_log)
        writer.add_scalar('mIoU_night', mIoU_night, epoch_log)
        writer.add_scalar('mean_IoU', mean_iou, epoch_log)
        
        is_best = False
        if mean_iou > best_iou[0]:
            is_best = True
            best_iou[0] = mean_iou
            best_iou[1] = mIoU_day
            best_iou[2] = mIoU_night
            best_iou[3] = epoch_log
            
        filename = args.snapshot_dir + '/train_epoch_' + str(epoch_log) + '.pth'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({'epoch': epoch_log, 
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_iou': best_iou[:3]}, filename)
        if is_best:
            exist = os.listdir('/home/jsun/Project/HeatNet-master/best')
            if exist != []:
                os.remove('/home/jsun/Project/HeatNet-master/best/' + exist[0])
            best_name = '/home/jsun/Project/HeatNet-master/best' + '/train_epoch_' + str(epoch_log) + '.pth'
            torch.save({'epoch': epoch_log, 
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()}, best_name)
        
        if epoch_log / 1 > 2:
            if (epoch_log-2) % 10 == 0:
                continue
            else:
                deletename = args.snapshot_dir + '/train_epoch_' + str(epoch_log - 1 * 2) + '.pth'
                os.remove(deletename)    
        
def train(model, optimizer, rgb_day, ir_day, rgb_night, ir_night, label_day, label_night, \
          source_label, target_label, cert, state, counter, epoch_log, i_iter, current_iter,\
          max_iter, criterion):
    model.train()
    optimizer.zero_grad()
    
    output = model([rgb_day, ir_day], [rgb_night, ir_night])
    
    critics_day = torch.zeros(1).cuda(args.gpu)
    critics_night = torch.zeros(1).cuda(args.gpu)
    for c_a in output['critics_a']:
        critics_day += torch.sum(criterion(c_a, torch.full_like(c_a, 1)))
    for c_b in output['critics_b']:
        critics_night += torch.sum(criterion(c_b, torch.full_like(c_b, 0)))
    total_critics = critics_day + critics_night
    
    if state == 'train_seg':
        # segmentation loss
        seg_loss_day = criterion_semseg(output['pred_label_a'], label_day, args.gpu)
        seg_loss_ir = criterion_semseg_weighted(output['pred_label_b'], label_night, args.gpu)
        cert = cert.cuda(args.gpu)
        seg_loss_ir = torch.mean(cert * seg_loss_ir)
        seg_loss = seg_loss_day + seg_loss_ir
        
        seg_loss_day_batch = seg_loss_day.data.cpu().numpy()
        seg_loss_ir_batch = seg_loss_ir.data.cpu().numpy()
        seg_loss_batch = seg_loss.data.cpu().numpy()
        # critic loss
        weights = [1.] * 7
        adv_loss = torch.zeros(1).cuda(args.gpu)
        adv_weighting = (torch.full_like(cert, 1.0) - cert).unsqueeze(0)
        for m, c_a in enumerate(output['critics_a']):
            adv_loss += torch.mean(F.interpolate(adv_weighting, size=(c_a.size(2),c_a.size(3)),
                                                 mode='bilinear')*criterion(c_a,torch.full_like(c_a,0)))
        for m, c_b in enumerate(output['critics_b']):
            adv_loss += torch.mean(F.interpolate(adv_weighting, size=(c_b.size(2),c_b.size(3)),
                                                 mode='bilinear')*criterion(c_b,torch.full_like(c_b,1)))*weights[m]
        
        adv_loss_batch = adv_loss.data.cpu().numpy()
        total_loss = seg_loss + 0.01 * adv_loss
        
        total_loss.backward()
        optimizer.step()
        
        writer.add_scalar('seg_loss', seg_loss_batch, current_iter)
        writer.add_scalar('seg_loss_day', seg_loss_day_batch, current_iter)
        writer.add_scalar('seg_loss_ir', seg_loss_ir_batch, current_iter)
        writer.add_scalar('adv_loss', adv_loss_batch, current_iter)
        # print(seg_loss_batch, seg_loss_day_batch,seg_loss_ir_batch, adv_loss_batch)
        if i_iter % 10 == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Seg_loss {seg_loss:.4f} '
                        'Seg_loss_day {seg_loss_day:.4f} '
                        'Seg_loss_ir {seg_loss_ir:.4f} '
                        'Adv_loss {adv_loss:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                          seg_loss = seg_loss_batch,
                                                          seg_loss_day = seg_loss_day_batch,
                                                          seg_loss_ir = seg_loss_ir_batch,
                                                          adv_loss = adv_loss_batch[0]))
    else:
        critic_loss_batch = total_critics.data.cpu().numpy()
        total_loss = total_critics
        total_loss.backward()
        optimizer.step()
        
        writer.add_scalar('critic_loss', critic_loss_batch, current_iter)
        
        if (i_iter-1) % 10 == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Critic_loss {critic_loss:.4f} '.
                        format(epoch_log, args.epochs, i_iter, max_iter,
                                                          critic_loss = critic_loss_batch[0]))
    
    # switch learning phase
    counter -= 1
    if counter == 0:
        if state == 'train_seg':
            state = 'train_critic'
            counter = 1
        else:
            state = 'train_seg'
            counter = 1
 
        model.setPhase(state)
        
    return state

def evaluate(test_loader, data_list, model, classes, gray_folder, color_folder, colors, domain):
    logger.info('>>>>>>>>>> Start Evaluation >>>>>>>>>>')
    model.eval()
    
    for i, (img, label) in enumerate(test_loader):
        rgb = img[:,:3,:,:].cuda(args.gpu)
        ir = img[:,3,:,:].cuda(args.gpu)
        ir = ir.unsqueeze(1)
        prediction = model([rgb,ir])
        prediction = prediction['pred_label_a']
        prediction = F.softmax(prediction, dim=1)
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)[0]
        
        gray =  np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('>>>>>>>>>> End Evaluation >>>>>>>>>>')

def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    for i, (image_path, _, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = Image.open(target_path)
        target = np.asarray(target)
        target = target[5:645, 360:1640]
        target = cv2.resize(target, (640,320), cv2.INTER_NEAREST)
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / sum(target_meter.val + 1e-10)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class[:12])
    mAcc = np.mean(accuracy_class[:12])
    
    logger.info('Eval result: mIoU/mAcc {:.4f}/{:.4f}.'.format(mIoU, mAcc))
    
    return mIoU

       
if __name__ == '__main__':
    main()
