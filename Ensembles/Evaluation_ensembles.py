import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import sys
import os
import cv2
from PIL import Image
import pandas as pd
sys.path.append('/home/danleh/graduation-project-2020-sonnefred/multi-head/')
from model.res_deeplab_dn import ResNet_Deeplab
from util.dataset_dn import Freiburg_Dataset
from util import transform
from util.util import AverageMeter, intersectionAndUnion

NUM_CLASSES = 14
ROOT = '/data/freiburg/'
GPU = 0
PATH_WEIGHTS = '/home/danleh/BEP-Dan/d_specific/best/train_epoch_122.pth' # => Weights of MH-TSDA

model = ResNet_Deeplab(num_classes=NUM_CLASSES, layers=101)
saved_state_dict = torch.load(PATH_WEIGHTS, map_location=lambda storage, loc: storage.cuda(GPU))
model.load_state_dict(saved_state_dict['state_dict'])
model.cuda()

test_transform = transform.Compose([
   transform.ToTensor(),
   transform.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])  
   
test_data_day = Freiburg_Dataset('test', ROOT, 'day', test_transform)
test_loader_day = data.DataLoader(test_data_day, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)
test_data_night = Freiburg_Dataset('test', ROOT, 'night', test_transform)
test_loader_night = data.DataLoader(test_data_night, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)

#classifier model:
class dnClassifier(nn.Module):
    def __init__(self):
        super(dnClassifier, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=10, stride = 10)
        self.maxpool1=nn.MaxPool2d(kernel_size=15)
        self.fc1 = nn.Linear(40, 1)
        
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
device = torch.device("cuda")
state_dict = torch.load("/home/danleh/bep-project-2021-Dan-Leh/Ensembles/Classifier_weights.pth")
Classifier_model = dnClassifier()
Classifier_model.load_state_dict(state_dict)
Classifier_model.to(device)

def evaluate(domain, test_loader, data_list, model, gray_folder, color_folder, colors):
    model.eval()
    for i, (img, label) in enumerate(test_loader):
        img = img.cuda(GPU)

        # determining domain:
        pred_domain = Classifier_model(img)
        pred_domain = pred_domain.squeeze(dim=1)
        pred_domain = torch.sigmoid(pred_domain) # Sigmoid to map predictions between 0 and 1
        pred_domain = torch.round(pred_domain) # round predictions to 0 and 1
        predicted_domain='day' if pred_domain==0 else 'night'
        # day/night classifier segmentation:
        if predicted_domain == 'day':
            prediction_classifier = model(img, True) 
        else:
            prediction_classifier = model(img, False)
        prediction_classifier = F.interpolate(prediction_classifier, (640,1280), mode='bilinear', align_corners=True)
        prediction_classifier = F.softmax(prediction_classifier, dim=1)
        prediction_classifier = prediction_classifier.detach().cpu().numpy()
        prediction_classifier = np.argmax(prediction_classifier, axis=1)[0]
        save_eval(i,prediction_classifier, data_list, gray_folder+'/classifier', color_folder +'/classifier')
        
        # day & night head predictions:
        prediction_day_head = model(img, True) 
        prediction_day_head = F.interpolate(prediction_day_head, (640,1280), mode='bilinear', align_corners=True)
        prediction_day_head = F.softmax(prediction_day_head, dim=1)
        prediction_day_head = prediction_day_head.detach().cpu().numpy()

        prediction_night_head = model(img, False)
        prediction_night_head = F.interpolate(prediction_night_head, (640,1280), mode='bilinear', align_corners=True)
        prediction_night_head = F.softmax(prediction_night_head, dim=1)
        prediction_night_head = prediction_night_head.detach().cpu().numpy()
        
        #soft voting: argmax of average/sum of daytime & nighttime head
        prediction_softvote = np.argmax(prediction_day_head+prediction_night_head, axis=1)[0]
        save_eval(i,prediction_softvote, data_list, gray_folder+'/softvote', color_folder +'/softvote')

        #maximum likelihood: take whichever head's softmax activation is highest for each pixel
        prediction_maxlikelihood=np.argmax(np.concatenate((prediction_day_head[0],prediction_night_head[0]),axis=0),axis=0)
        prediction_maxlikelihood[prediction_maxlikelihood>=14]=prediction_maxlikelihood[prediction_maxlikelihood>=14]-14
        save_eval(i,prediction_maxlikelihood, data_list, gray_folder+'/maxlikelihood', color_folder +'/maxlikelihood')

def save_eval(i,prediction, data_list, gray_folder, color_folder):
    gray =  np.uint8(prediction)
    color = colorize(gray, colors)
    image_path, _ = data_list[i]
    image_name = image_path.split('/')[-1].split('.')[0]
    gray_path = os.path.join(gray_folder, image_name + '.png')
    color_path = os.path.join(color_folder, image_name + '.png')
    cv2.imwrite(gray_path, gray)
    color.save(color_path)

colors = np.array([[128, 64, 128],[244, 35, 232],[70, 70, 70],[196, 196, 196],[190, 153, 153],[153, 153, 153],[107, 142, 35],[152, 251, 152],[70, 130, 180],[220, 20, 60],[0, 0, 142],[119, 11, 32],[138,43,226],[0, 0, 0]]).astype('uint8')

path = '/home/danleh/BEP-Dan/Evaluation_ensembles/'
gray_folder_day = path + 'day/gray'
color_folder_day = path + 'day/color'
gray_folder_night = path + 'night/gray'
color_folder_night = path + 'night/color'

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

evaluate('day', test_loader_day, test_data_day.data_list, model, gray_folder_day,color_folder_day, colors)
evaluate('night', test_loader_night, test_data_night.data_list, model, gray_folder_night,color_folder_night, colors)


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = Image.open(target_path)
        target = np.asarray(target)
        target = target[5:645, 360:1640]
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class[:12])
    
    iou_class[13]=mIoU
    
    return iou_class

names = ['Road', 'Sidewalk', 'Building', 'Curb', 'Fence', 'Pole', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Car', 'Bicycle', 'Background', 'Mean']

IoU_day_classifier = cal_acc(test_data_day.data_list, gray_folder_day+'/classifier', NUM_CLASSES, names)
IoU_night_classifier = cal_acc(test_data_night.data_list, gray_folder_night+'/classifier', NUM_CLASSES, names)

IoU_day_softvote = cal_acc(test_data_day.data_list, gray_folder_day+'/softvote', NUM_CLASSES, names)
IoU_night_softvote = cal_acc(test_data_night.data_list, gray_folder_night+'/softvote', NUM_CLASSES, names)

IoU_day_maxlikelihood = cal_acc(test_data_day.data_list, gray_folder_day+'/maxlikelihood', NUM_CLASSES, names)
IoU_night_maxlikelihood = cal_acc(test_data_night.data_list, gray_folder_night+'/maxlikelihood', NUM_CLASSES, names)

IoU_avg_classifier=(IoU_day_classifier+IoU_night_classifier)/2
results_classifier=np.array([IoU_day_classifier, IoU_night_classifier, IoU_avg_classifier])*100
IoU_avg_softvote=(IoU_day_softvote+IoU_night_softvote)/2
results_softvote=np.array([IoU_day_softvote, IoU_night_softvote, IoU_avg_softvote])*100
IoU_avg_maxlikelihood=(IoU_day_maxlikelihood+IoU_night_maxlikelihood)/2
results_maxlikelihood=np.array([IoU_day_maxlikelihood, IoU_night_maxlikelihood, IoU_avg_maxlikelihood])*100

df_classifier = pd.DataFrame(results_classifier, columns=names, index=['Daytime IoU', 'Nighttime IoU', 'Average IoU']).round(decimals=2)
df_softvote = pd.DataFrame(results_softvote, columns=names, index=['Daytime IoU', 'Nighttime IoU', 'Average IoU']).round(decimals=2)
df_maxlikelihood = pd.DataFrame(results_maxlikelihood, columns=names, index=['Daytime IoU', 'Nighttime IoU', 'Average IoU']).round(decimals=2)

print(df_classifier)
print(df_softvote)
print(df_maxlikelihood)