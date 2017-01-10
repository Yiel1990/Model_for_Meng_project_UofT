from __future__ import print_function
from __future__ import division
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
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
import load_data
import model


img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
img_channels = 1


test_data,test_label = load_data.load_testing_data()

ICML_weight_path = "./ICML_model/Model.1156-0.6729.hdf5"
VGG_weight_path = "./VGG_model/Model.356-0.6796.hdf5"
VGG_19_weigth_path = "./VGG_19_model/VGG_19_Model.310-0.6668.hdf5"

model_ICML = model.model_ICML()
model_VGG_face = model.model_VGG_face()
model_VGG_19 = model.model_VGG_19_face()

model_ICML.load_weights(ICML_weight_path)
model_VGG_face.load_weights(VGG_weight_path)
model_VGG_19.load_weights(VGG_19_weigth_path)

out_ICML = model_ICML.predict_classes(test_data,batch_size=128,verbose=1)
out_VGG = model_VGG_face.predict_classes(test_data,batch_size = batch_size,verbose=1)
out_VGG_19 = model_VGG_19.predict_classes(test_data,batch_size = batch_size,verbose=1)

print("\n")
counter = 0
for i in range(0,3589):
    if out_ICML[i] == test_label[i]:
        counter += 1
accuracy = counter / len(out_ICML)
print("The accuracy for the ICML model on Test set of Fer2013 is " + str(accuracy))


print("\n")
counter = 0
for i in range(0,3589):
    if out_VGG[i] == test_label[i]:
        counter += 1
accuracy = counter / len(out_ICML)
print("The accuracy for the VGG_face model on Test set of Fer2013 is " + str(accuracy))

print("\n")
counter = 0
for i in range(0,3589):
    if out_VGG_19[i] == test_label[i]:
        counter += 1
accuracy = counter / len(out_ICML)
print("The accuracy for the VGG_19 model on Test set of Fer2013 is " + str(accuracy))

