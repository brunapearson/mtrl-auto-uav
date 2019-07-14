import setup_path
import airsim

import numpy as np
import time
import os
import glob
import cv2
import math
from math import *
from PIL import Image, ImageDraw
from scipy.misc import imsave
import matplotlib.pyplot as plt
plt.ion()

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
from pyquaternion import Quaternion
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
n_intervention = 0

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
    input_img = Input(shape=(224,224,3))

    x = Conv2D(16, kernel_size=3, activation='relu')(input_img)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
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

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    model.compile(optimizer=Adam(lr=1e-04), loss=[euc_loss1x,euc_loss1y,euc_loss1z,euc_loss1rw,euc_loss1rx,euc_loss1ry,euc_loss1rz])
    return model

#convert horizonal fov to vertical fov
def hfov2vfov(hfov, image_sz):
    aspect = image_sz[0]/image_sz[1]
    vfov = 2*math.atan( tan(hfov/2) * aspect)
    return vfov

#compute bounding box size
def compute_bb(image_sz, obj_sz, hfov, distance):
    vfov = hfov2vfov(hfov,image_sz)
    box_h = ceil(obj_sz[0] * image_sz[0] / (math.tan(hfov/2)*distance*2))
    box_w = ceil(obj_sz[1] * image_sz[1] / (math.tan(vfov/2)*distance*2))
    return box_h, box_w

def generate_depth_viz(img,thres=0):
    if thres > 0:
        img[img > thres] = thres
    else:
        img = np.reciprocal(img)
    return img

def moveUAV(client,pred_pos,yaw):
    client.moveToPositionAsync(pred_pos[0], pred_pos[1], pred_pos[2],5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = yaw))
    time.sleep(1)

def interventions_counter(client,depth_img,uav_size,pred_pos,yaw):
    global n_intervention

    current_pos = client.simGetGroundTruthKinematics().position
    p_x = ((current_pos.x_val - pred_pos[0])*(0.75))+current_pos.x_val
    p_y = ((current_pos.y_val - pred_pos[1])*(-0.75))+current_pos.y_val
    p_z = ((current_pos.z_val-pred_pos[2])*0.15)+current_pos.z_val #snowy

    #control max_height and min_height
    if p_z < -150:
        p_z=(-30)
    elif p_z==0 or p_z > (-2):
        p_z=(-4)

    hfov=radians(120)#90
    coll_thres=3 #3 forest
    intervention_thres = 50

    [h,w] = np.shape(depth_img)
    [roi_h,roi_w] = compute_bb((h,w), uav_size, hfov, coll_thres)

    img2d_box = depth_img[int((h-roi_h)/2):int((h+roi_h)/2),int((w-roi_w)/2):int((w+roi_w)/2)]

    if (int(np.mean(img2d_box)) < intervention_thres):
        n_intervention += 1
        client.moveToPositionAsync(-30, p_y, p_z,5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = yaw))
        time.sleep(1)
    else:
        client.moveToPositionAsync(p_x, p_y, p_z,5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = yaw))
        time.sleep(1)

def recover_collision(client):

    collision_info = client.simGetCollisionInfo()

    if (collision_info.has_collided == True):
        landed = client.getMultirotorState().landed_state
        if landed == airsim.LandedState.Landed:
            client.takeoffAsync().join()
            client.moveToPositionAsync(-60, 50, -16,5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = 0))
            time.sleep(2)
        else:
            current_pos = client.simGetGroundTruthKinematics().position
            p_z = current_pos.z_val
            if p_z==0 or p_z > (-2):
                p_z = - 4
            else:
                p_z = current_pos.z_val * (-1)

            client.moveToPositionAsync(-30, 50, p_z,5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = 90))
            time.sleep(2)


def get_image(client):
    image_buf = np.zeros((1, 432 , 768, 4))
    image_response = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    image_rgba = cv2.cvtColor(image_rgba,cv2.COLOR_RGBA2BGR)
    image_buf = image_rgba.copy()
    image_buf = cv2.resize(image_buf,(224,224))

    return image_buf


#confirm connection to simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
print("arming the drone...")
client.armDisarm(True)
client.takeoffAsync().join()

#get uav current state
state = client.getMultirotorState()

#define start position
pos = [-1,-5,-6] #start position x,y,z
#pos = [-10,55,-34] #start position x,y,z for snowy mountain
uav_size = [0.29*3,0.98*2] #height:0.29 x width:0.98 - allow some tolerance

#move uav to initial position
moveUAV(client,pos,0)

#load trained model
model = load_trained_model('model\model_0.5004407.h5')



for i in range(150):

    #get depth image
    depth = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)])
    # get numpy array
    img1d = depth[0].image_data_float

    # reshape array to 2D array H X W
    img2d = np.reshape(img1d,(depth[0].height, depth[0].width))

    #feed predicted position
    input_image = get_image(client)
    input_image = np.array(input_image, dtype=np.float32)
    m = 27.824116
    input_image -= m
    #input_image = np.expand_dims(input_image, axis=0) #expand from 3D to 4D
    input_image = input_image.reshape(1,224,224,3)
    pos_x,pos_y,pos_z,rot_w,rot_x,rot_y,rot_z = model.predict(input_image)
    action = [pos_x[0][0],pos_y[0][0],pos_z[0][0]]

    q = client.simGetVehiclePose().orientation
    q.w_val = rot_w[0][0]
    q.x_val = rot_x[0][0]
    q.y_val = rot_y[0][0]
    q.z_val = rot_z[0][0]
    pitch, roll, yaw  = airsim.to_eularian_angles(q)

    #update position
    pos[0] = float(pos_x[0][0])
    pos[1] = float(pos_y[0][0])
    pos[2] = float(pos_z[0][0])

    #move uav to correct position
    interventions_counter(client,img2d,uav_size,pos,yaw)
    recover_collision(client)

print('Total number of intervention: ', n_intervention)
