################################################################################

# End-to-End Multi-Task Regression-based Learning approach capable of deﬁning
# ﬂight commands for navigation and exploration under the forest canopy

# Copyright (c) 2019 - Maciel-Pearson, B.G., Akçay, S., Atapour-Abarghouei, A.,
# Holder, C. and Breckon, T.P., Durham University, UK

# License : https://github.com/brunapearson/mtrl-auto-uav/blob/master/LICENSE

################################################################################

import setup_path
import airsim

################################################################################

import numpy as np
import time
import os
import sys
import glob
import cv2
import math
from math import *
from PIL import Image, ImageDraw
from scipy.misc import imsave
import matplotlib.pyplot as plt
plt.ion()

################################################################################

from mtrl_network import*
import mtrl_network

################################################################################

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

################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
n_intervention = 0
n_collision = 0

################################################################################

# Credits: the functions in this block are heavily based on the work of:
# https://github.com/microsoft/AirSim/blob/master/PythonClient/computer_vision/cv_navigate.py

# convert horizonal fov to vertical fov
def hfov2vfov(hfov, image_sz):
    aspect = image_sz[0]/image_sz[1]
    vfov = 2*math.atan( tan(hfov/2) * aspect)
    return vfov

# compute bounding box size
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

################################################################################

# load pre-trained model
def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    model.compile(optimizer=Adam(lr=1e-04), loss=[euc_loss1x,euc_loss1y,euc_loss1z,euc_loss1rw,euc_loss1rx,euc_loss1ry,euc_loss1rz])
    return model

def moveUAV(client,pred_pos,yaw):
    client.moveToPositionAsync(pred_pos[0], pred_pos[1], pred_pos[2],5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = yaw))
    time.sleep(1)

def interventions_counter(client,depth_img,uav_size,pred_pos,yaw,behaviour,smoothness_x,smoothness_y,smoothness_z):
    global n_intervention

    current_pos = client.simGetGroundTruthKinematics().position

    if behaviour=="search":
        # check change in position to maximize exploration in x direction
        if int(pred_pos[0])<0:
            p_x = int(abs(pred_pos[0])-abs(current_pos.x_val)*(-2))
        else:
            p_x = int(abs(pred_pos[0])+abs(current_pos.x_val))
        # check change in position to maximize exploration in y direction
        if int(pred_pos[1])<0:
            p_y = int(abs(pred_pos[1])-abs(current_pos.y_val)*(-2))
        else:
            p_y = int(abs(pred_pos[1])+abs(current_pos.y_val))

    if behaviour=="flight":
        p_x = ((current_pos.x_val - pred_pos[0])*(smoothness_x))+current_pos.x_val
        p_y = ((current_pos.y_val - pred_pos[1])*(smoothness_y))+current_pos.y_val

    p_z = ((current_pos.z_val-pred_pos[2])*smoothness_z)+current_pos.z_val #snowy

    # control max_height and min_height
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
    global n_collision

    # verify if a collision has happen
    collision_info = client.simGetCollisionInfo()

    # if a collision happened verify if has forced the drone to land
    if (collision_info.has_collided == True):
        n_collision += 1
        landed = client.getMultirotorState().landed_state
        # if the drone has landed take off again and change flight position
        if landed == airsim.LandedState.Landed:
            client.takeoffAsync().join()
            client.moveToPositionAsync(-60, 50, -16,5,drivetrain = airsim.DrivetrainType.ForwardOnly,lookahead=-1,adaptive_lookahead=1, yaw_mode = airsim.YawMode(is_rate = False, yaw_or_rate = 0))
            time.sleep(2)
        else:
            # if the drone has not landed verify the height and adjust it to acceptable flight conditions
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

################################################################################

# confirm connection to simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
print("arming the drone...")
client.armDisarm(True)
client.takeoffAsync().join()

# get uav current state
state = client.getMultirotorState()
# define start position
pos = [-1,-5,-6] #start position x,y,z
# define uav size
uav_size = [0.29*3,0.98*2] #height:0.29 x width:0.98 - allow some tolerance

################################################################################

# read user's input
n_predictions = int(sys.argv[1])
behaviour = str(sys.argv[2])
pos[0] = int(sys.argv[3])
pos[1] = int(sys.argv[4])
pos[2] = int(sys.argv[5])
smoothness_x = float(sys.argv[6])
smoothness_y = float(sys.argv[7])
smoothness_z = float(sys.argv[8])

################################################################################

#move uav to initial position
moveUAV(client,pos,0)

#load pre-trained model
model = load_trained_model('models\model_0.5004407.h5')

for i in range(n_predictions):

    # get depth image
    depth = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)])
    # get numpy array
    img1d = depth[0].image_data_float

    # reshape array to 2D array H X W
    img2d = np.reshape(img1d,(depth[0].height, depth[0].width))

    # feed predicted position
    input_image = get_image(client)
    input_image = np.array(input_image, dtype=np.float32)
    m = 27.824116
    input_image -= m
    input_image = input_image.reshape(1,224,224,3)
    pos_x,pos_y,pos_z,rot_w,rot_x,rot_y,rot_z = model.predict(input_image)
    action = [pos_x[0][0],pos_y[0][0],pos_z[0][0]]

    q = client.simGetVehiclePose().orientation
    q.w_val = rot_w[0][0]
    q.x_val = rot_x[0][0]
    q.y_val = rot_y[0][0]
    q.z_val = rot_z[0][0]
    pitch, roll, yaw  = airsim.to_eularian_angles(q)

    # update position
    pos[0] = float(pos_x[0][0])
    pos[1] = float(pos_y[0][0])
    pos[2] = float(pos_z[0][0])

    # move uav to correct position and monitor the number of interventions
    interventions_counter(client,img2d,uav_size,pos,yaw, behaviour,smoothness_x,smoothness_y,smoothness_z)
    # in case a collision happens, this function will attempt to regain flight conditions
    recover_collision(client)

################################################################################
print('Total number of intervention: ', n_intervention)
print('Total number of collisions:', n_collision)
################################################################################
