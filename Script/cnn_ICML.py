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
import load_data
import model

img_rows, img_cols = 48, 48
batch_size = 128
nb_classes = 7
nb_epoch = 1500
img_channels = 1

Train_x, Train_y = load_data.load_training_data();

Val_x, Val_y = load_data.load_validation_data();

Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)


model = model.model_ICML()

filepath='./tmp/ICML_Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,samplewise_std_normalization=False,zca_whitening=False,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=False)

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,batch_size=batch_size),samples_per_epoch=Train_x.shape[0],nb_epoch=nb_epoch,validation_data=(Val_x, Val_y),callbacks=[checkpointer])

