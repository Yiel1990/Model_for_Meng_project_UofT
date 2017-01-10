from __future__ import print_function
from __future__ import division
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

img_rows, img_cols = 48, 48
channel = 1
classes = ['Angry', 'Disgust', 'Fear', 'Happy','Sad','Surprise','Neutral'];

weights_path_ICML = './ICML_model/Model.1156-0.6729.hdf5'
model_ICML = model.model_ICML()
model_ICML.load_weights(weights_path_ICML)
print("Model loaded successfully")

weights_path_VGG = './VGG_model/Model.356-0.6796.hdf5'
model_VGG = model.model_VGG_face()
model_VGG.load_weights(weights_path_VGG)
print("Model loaded successfully")

VGG_19_weigth_path = "./VGG_19_model/VGG_19_Model.310-0.6668.hdf5"
model_VGG_19 = model.model_VGG_19_face()
model_VGG_19.load_weights(VGG_19_weigth_path)
print("Model loaded successfully") 

test_data,test_label = load_data.load_testing_data()

out_ICML = model_ICML.predict_proba(test_data,batch_size=128, verbose=1)
out_VGG = model_VGG.predict_proba(test_data,batch_size=128, verbose=1)
out_VGG_19 =model_VGG_19.predict_proba(test_data,batch_size=128, verbose=1)


final_out = out_ICML + out_VGG + out_VGG_19

#print (final_out.shape)
#print("\n")
#print (final_out[0][0])
result_label = np.empty((3589,),dtype="uint8")

for i in range(0,3589):
    tmp_index = 0
    tmp_value = 0
    for j in range(0,7):
        if final_out[i][j] > tmp_value:
            tmp_index = j
            tmp_value = final_out[i][j]
    result_label[i] = tmp_index

print("\n")
counter = 0
for i in range(0,3589):
    if result_label[i] == test_label[i]:
        counter += 1
accuracy = counter / len(out_ICML)
print("The accuracy for the multiple-CNN model on Test set of Fer2013 is " + str(accuracy))



