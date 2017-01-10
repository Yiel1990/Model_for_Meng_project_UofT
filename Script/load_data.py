import os
from PIL import Image
import numpy as np
import pandas as pd
import processing
from pylab import *

img_width = 48
img_height = 48
classes = ['Angry', 'Disgust', 'Fear', 'Happy','Sad','Surprise','Neutral'];
channel = 1

def load_training_data():
    data = np.empty((28709,channel,img_width,img_height),dtype="float32")
    tmp_pre = np.empty((1,channel,img_width,img_height),dtype="float32")
    label = np.empty((28709,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/NEWFER2013/Train/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/NEWFER2013/Train/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

def load_validation_data():
    data = np.empty((3589,channel,img_width,img_height),dtype="float32")
    label = np.empty((3589,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/NEWFER2013/Val/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/NEWFER2013/Val/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

def load_testing_data():
    data = np.empty((3589,channel,img_width,img_height),dtype="float32")
    label = np.empty((3589,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/NEWFER2013/Test/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/NEWFER2013/Test/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

def load_LBP_training_data():
    data = np.empty((28709,channel,img_width,img_height),dtype="float32")
    tmp_pre = np.empty((1,channel,img_width,img_height),dtype="float32")
    label = np.empty((28709,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/FER2013/Train/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/FER2013/Train/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

def load_LBP_validation_data():
    data = np.empty((3589,channel,img_width,img_height),dtype="float32")
    label = np.empty((3589,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/FER2013/Val/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/FER2013/Val/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

def load_LBP_testing_data():
    data = np.empty((3589,channel,img_width,img_height),dtype="float32")
    label = np.empty((3589,),dtype="uint8")
    total = 0
    for index in range(0,7):
        imgs = os.listdir(os.path.join("../Data/FER2013/Test/",classes[index]))
        num = len(imgs)
        for i in range(num):
            img = array((Image.open("../Data/FER2013/Test/" + classes[index] + "/" + imgs[i]).convert('L')).resize((img_width, img_height)),'f')
            tmpdata = processing.Process(img)
            data[total,0,:,:] = tmpdata
            label[total] = index
            total += 1
    return data,label

