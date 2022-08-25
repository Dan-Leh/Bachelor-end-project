"""
This is the code to train and evaluate SH-TSDA, the Single-Head network with Teacher-Student Domain Adaptation.
It is equivalent to the Multi-head network (also with the same pretrained weights), but with only the 'nighttime' head being used for learning and inference.

It was adapted from: https://github.com/tue-mps-edu/graduation-project-2020-sonnefred/blob/main/multi-head/train_dn_v3.py
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
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter
from PIL import Image
import cv2
import sys
sys.path.append('/home/danleh/graduation-project-2020-sonnefred/multi-head/')

from util.Loss import MSE_loss, CE_Loss, CrossEntropy2d
from model.res_deeplab_dn import ResNet_Deeplab
from util.dataset_dn import Freiburg_Dataset
from util import transform
from util.util import AverageMeter, intersectionAndUnion


BATCH_SIZE = 8
NUM_WORKERS = 4
IGNORE_LABEL = 255
INPUT_SIZE = '640,320'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-5
MOMENTUM = 0.9
ALPHA = 0.9
NUM_CLASSES = 14
START_EPOCH = 0
FIRST_EPOCH = 50
EPOCHS = 150
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/data/models/multi-head/multitask_v2/trained_weights.pth'
ROOT = '/data/freiburg/'
LOG_SAVE_PATH = '/home/danleh/BEP-Dan/SH-TSDA/logs/'
SNAPSHOT_DIR = '/home/danleh/BEP-Dan/SH-TSDA/snapshots/'
BEST_DIR = '/home/danleh/BEP-Dan/SH-TSDA/best/'
WEIGHT_DECAY = 0.0005
GPU = 0


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Alpha of RMSprop optimizer")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="The start epoch.")
    parser.add_argument("--first-epoch", type=int, default=FIRST_EPOCH,
                        help="The epoch for first stage.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="The total epoch.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--best-dir", type=str, default=BEST_DIR,
                        help="Where to save the best model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--logs", type=str, default=LOG_SAVE_PATH,
                        help="where to save the logs")
    return parser.parse_args()


args = get_arguments()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def criterion_semseg(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, total_iter):
    lr = lr_poly(args.learning_rate, i_iter, total_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        

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

    # print(input_size)
    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = ResNet_Deeplab(num_classes=args.num_classes, layers=101)
    
    global writer, logger
    writer = SummaryWriter(args.logs)
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    logger.info(model)
    
    model.train()
    model.cuda(args.gpu)
    
    optimizer_both = optim.SGD(model.optim_night_parameters(args), lr=args.learning_rate, \
                             momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_both.zero_grad()
    
    
    if args.restore_from:
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                # print i_parts
                if not args.num_classes == 14 or (i_parts[1] != 'layer5' and i_parts[1] != 'layer6'):
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                    # print i_parts
            model.load_state_dict(new_params)
        else:
            # saved_state_dict = torch.load(args.restore_from)
            saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage.cuda(args.gpu))
            # args.start_epoch = saved_state_dict['epoch']
            model.load_state_dict(saved_state_dict['state_dict'])
            # optimizer_day.load_state_dict(saved_state_dict['optimizer_day'])
            # optimizer_night.load_state_dict(saved_state_dict['optimizer_night'])
            # model_D.load_state_dict(saved_state_dict['D_state_dict'])
            # optimizer_D.load_state_dict(saved_state_dict['optimizer_D'])
        

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir) 
    
    train_transform = transform.Compose([
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    
    # create dataloader   
    day_loader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'day', train_transform), \
                    batch_size=args.batch_size, shuffle=True, \
                    num_workers=args.num_workers, pin_memory=True)
    night_loader = data.DataLoader(Freiburg_Dataset('train', ROOT, 'night', train_transform), \
                    batch_size=args.batch_size, shuffle=True, \
                    num_workers=args.num_workers, pin_memory=True)
    
    test_data_day = Freiburg_Dataset('test', ROOT, 'day', test_transform)
    test_loader_day = data.DataLoader(test_data_day, batch_size=1, shuffle=False,\
                                      num_workers=0, pin_memory=True)
    test_data_night = Freiburg_Dataset('test', ROOT, 'night', test_transform)
    test_loader_night = data.DataLoader(test_data_night, batch_size=1, shuffle=False,\
                                      num_workers=0, pin_memory=True)
    
    path = '/home/danleh/BEP-Dan/SH-TSDA/test/'
    gray_folder_day = path + 'day/gray'
    color_folder_day = path + 'day/color'
    gray_folder_night = path + 'night/gray'
    color_folder_night = path + 'night/color'
    
    colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],
                [190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],
                [70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],
                [0, 0, 0]]).astype('uint8')

    names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', \
             'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Ignore']
    
    best_iou = [0, 0, 0, 0]
           
    for epoch in range(100, args.epochs):
        epoch_log = epoch + 1
        
        day_loader_iter = iter(day_loader)        
        night_loader_iter = iter(night_loader)
        max_iter = len(day_loader_iter)
        
        loss_seg_day = 0
        loss_seg_night = 0
        
        for i_iter in range(max_iter):
            current_iter = epoch * max_iter + i_iter + 1
            #if epoch_log <= args.first_epoch:
            adjust_learning_rate(optimizer_both, current_iter, max_iter * args.epochs)
            #else:
             #   adjust_learning_rate(optimizer_both, current_iter, max_iter * (args.epochs - args.first_epoch))
            
            day, label_day = day_loader_iter.next()
            night, label_night = night_loader_iter.next()
            
            loss = train(model, optimizer_both, day, night, label_day, label_night, epoch_log, current_iter, i_iter, max_iter)
            
            loss_seg_day += loss[0]
            loss_seg_night += loss[1]
        
        writer.add_scalar('loss_seg_day', loss_seg_day / max_iter, epoch + 1)
        writer.add_scalar('loss_seg_night', loss_seg_night / max_iter, epoch + 1)

        evaluate(test_loader_day, test_data_day.data_list, model, args.num_classes, gray_folder_day, \
                 color_folder_day, colors)
        evaluate(test_loader_night, test_data_night.data_list, model, args.num_classes, gray_folder_night, \
                 color_folder_night, colors)
        
        mIoU_day = cal_acc(test_data_day.data_list, gray_folder_day, args.num_classes, names)
        mIoU_night = cal_acc(test_data_night.data_list, gray_folder_night, args.num_classes, names)
        
        writer.add_scalar('mIoU-day', mIoU_day, epoch_log)
        writer.add_scalar('mIoU-night', mIoU_night, epoch_log)
        
        mean_iou = (mIoU_day + mIoU_night) / 2
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
                    'state_dict': model.state_dict(),
                    'optimizer_both': optimizer_both.state_dict()}, filename)
        if is_best:
            exist = os.listdir(args.best_dir)
            if exist != []:
                os.remove(args.best_dir + exist[0])
            best_name = args.best_dir + '/train_epoch_' + str(epoch_log) + '.pth'
            torch.save({'epoch': epoch_log, 
                        'state_dict': model.state_dict(),
                        'optimizer_both': optimizer_both.state_dict()}, best_name)
            
        if epoch_log / 1 > 2:
            if (epoch_log-2) % 10 == 0:
                continue
            else:
                deletename = args.snapshot_dir + '/train_epoch_' + str(epoch_log - 1 * 2) + '.pth'
                os.remove(deletename) 

def train(model, optimizer_both, day, night, label_day, label_night, epoch_log, current_iter, i_iter, max_iter):
    model.train()
    optimizer_both.zero_grad()
    
    day = day.cuda(args.gpu)
    night = night.cuda(args.gpu)

    pred_day = model(day, False)
    loss_seg_day = criterion_semseg(pred_day, label_day, args.gpu)
    loss_seg_day_batch = loss_seg_day.data.cpu().numpy()
    loss_seg_day.backward()
    optimizer_both.step()
    
    pred_night = model(night, False)   
    loss_seg_night = criterion_semseg(pred_night, label_night, args.gpu)
    loss_seg_night_batch = loss_seg_night.data.cpu().numpy()
    loss_seg_night.backward()
    optimizer_both.step()
    
    writer.add_scalar('loss_seg_day_batch', loss_seg_day_batch, current_iter)
    writer.add_scalar('loss_seg_night_batch', loss_seg_night_batch, current_iter)
    
    if i_iter % 10 == 0:
        logger.info('Epoch: [{}/{}][{}/{}] '
                    'Seg_Loss_Day {seg_loss_day:.4f} '
                    'Seg_Loss_Night {seg_loss_night:.4f} '.format(epoch_log, args.epochs, i_iter, max_iter,
                                                                    seg_loss_day=loss_seg_day_batch,
                                                                    seg_loss_night=loss_seg_night_batch))
    
    return loss_seg_day_batch, loss_seg_night_batch


def evaluate(test_loader, data_list, model, classes, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>> Start Evaluation >>>>>>>>>>')
    model.eval()
    
    for i, (img, label) in enumerate(test_loader):
        img = img.cuda(args.gpu)
        
        prediction = model(img, False)
        prediction = F.interpolate(prediction, (640,1280), mode='bilinear', align_corners=True)
        prediction = F.softmax(prediction, dim=1)
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)[0]
        
        gray =  np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
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
    
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = Image.open(target_path)
        target = np.asarray(target)
        target = target[5:645, 360:1640]
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
