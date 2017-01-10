from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import cPickle 
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import processing
from PIL import Image
import sys
from pylab import *
import load_data
import model

weights_path = '/home/yiou/Downloads/keras-master/project/Script/ICML_model/Model.1199-0.6701.hdf5'
img_rows, img_cols = 48, 48
channel = 1
model = model.model_ICML()
model.load_weights(weights_path)
print("Model loaded successfully")
model.layers.pop()

filename = "img-1.png"
img = array((Image.open(filename).convert('L')).resize((img_rows, img_cols)),'f')
tmpdata = processing.Process(img)
inputdata = np.empty((1,channel,img_rows,img_cols),dtype="float32")
inputdata[0,0,:,:] = tmpdata
out = model.predict_proba(inputdata,batch_size=128, verbose=1)



print(out)







