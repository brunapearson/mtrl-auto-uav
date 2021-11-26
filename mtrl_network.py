import numpy as np
import time
import os
import glob
import cv2
import math
from math import *
from PIL import Image, ImageDraw
# from scipy.misc import imsave
import matplotlib.pyplot as plt
plt.ion()


from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
import tensorflow as tf

# from pyquaternion import Quaternion
# from tensorflow.keras.models import Model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import concatenate
from keras.optimizers import Adam, SGD, Adamax, Nadam

def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lx)

def euc_loss1y(y_true, y_pred):
    ly = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * ly)
    #return (50 * ly)

def euc_loss1z(y_true, y_pred):
    lz = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.1 * lz)

def euc_loss1rw(y_true, y_pred):
    lw = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    #return (50 * lw)
    return (0.1 * lw)

def euc_loss1rx(y_true, y_pred):
    lp = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.05 * lp)

def euc_loss1ry(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    #return (50 * lq)
    return (0.1 * lq)

def euc_loss1rz(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    #return (50 * lq)
    return (0.1 * lq)



def create_model():
    #Create the convolutional stacks
    # cv2.imshow('input', Input)
    #input_img = Input(shape=(224,224,3))
    input_img = Input(shape=(224, 224, 3))

    '''model = Sequential()
    model.add(Conv2D(16, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))'''
    
    print("**********************entered model***********************")
    print(input_img.shape)
    x = Conv2D(16, kernel_size=3, activation='relu')(input_img)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(x.shape)
    x = MaxPooling2D(pool_size=(2,2))(x)
    print("maxpooling outpt??????????????????????????????")
    print(x.shape)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    print("conv2d}}}}}}}}}}}}}}}}}}}}}")
    print(x.shape)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.20)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(20, activation='relu')(x)

    n = Conv2D(16, kernel_size=3, activation='relu')(input_img)
    n = MaxPooling2D(pool_size=(2,2))(n)
    n = Conv2D(32, kernel_size=3, activation='relu')(n)
    n = MaxPooling2D(pool_size=(2,2))(n)
    n = Conv2D(64, kernel_size=3, activation='relu')(n)
    n = MaxPooling2D(pool_size=(2,2))(n)
    n = Flatten()(n)
    n = Dense(500, activation='relu')(n)
    #n = Dropout(0.50)(n)
    n = Dense(100, activation='relu')(n)
    n = Dense(20, activation='relu')(n)

    #output
    output_x = Dense(1, activation='linear', name='input_x')(n)
    output_y = Dense(1, activation='linear', name='input_y')(n)
    output_z = Dense(1, activation='linear', name='input_z')(n)

    output_qw = Dense(1, activation='linear', name='input_qw')(x)
    output_qx = Dense(1, activation='linear', name='input_qx')(x)
    output_qy = Dense(1, activation='linear', name='input_qy')(x)
    output_qz = Dense(1, activation='linear', name='input_qz')(x)


    model = Model(inputs=input_img, outputs=[output_x,output_y,output_z,output_qw,output_qx,output_qy,output_qz])
    return model