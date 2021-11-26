#!/usr/bin/env python3



# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py
import bpy
import os, sys
sys.path.append(os.getcwd())
sys.path.append('/home/naitri/.local/lib/python3.7/site-packages')
sys.path.append('/usr/lib/python3.7/lib-dynload')
sys.path.append('/usr/local/lib/python3.7/dist-packages')

from mtrl_network import*

import math
import os
import random
import Imath
import array
import numpy as np
import cv2
import OpenEXR

from tensorflow import keras
from keras.optimizers import Adam, SGD, Adamax, Nadam
xplots = []
yplots = []
zplots = []
def SetRenderSettings():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 64 # Lowering this makes rendering fast but noisy
    bpy.context.scene.render.resolution_x = 320 # Reduce to speed up
    bpy.context.scene.render.resolution_y = 240 # Reduce to speed up

# load pre-trained model
def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    #model.compile(optimizer=Adam(lr=1e-04), loss=[euc_loss1x,euc_loss1y,euc_loss1z,euc_loss1rw,euc_loss1rx,euc_loss1ry,euc_loss1rz])
    model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

    return model

def Exr2Depth(exrfile):
    file = OpenEXR.InputFile(exrfile)

    # Compute the size
    dw = file.header()['dataWindow']
    ImgSize = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    [Width, Height] = ImgSize

    # R and G channel stores the flow in (u,v) (Read Blender Notes)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    # start = timeit.default_timer()
    (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    # stop = timeit.default_timer()
    # print('Time: ', stop - start) 
    
    D = np.array(R).reshape(Height, Width, -1)
    D = (D <= 20. ) * D
 
    return D
    
def Render():    
        # DO NOT CHANGE THIS FUNCTION BELOW THIS LINE!        
    

    # Render in .exr
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    # Render Second Camera for Third Person view of the Drone
    cam = bpy.data.objects['Camera2']    
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = os.path.join(path_dir, 'EXR', 'ThirdView', 'Frame%04d'%(bpy.data.scenes[0].frame_current))
    bpy.ops.render.render(write_still=True)

    # Render Drone Camera
    cam = bpy.data.objects['Camera']    
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = os.path.join(path_dir, 'EXR','Frames', 'Frame%04d'%(bpy.data.scenes[0].frame_current))
    bpy.ops.render.render(write_still=True)

    # Render in .png
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    # Render Second Camera for Third Person view of the Drone
    cam = bpy.data.objects['Camera2']    
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = os.path.join(path_dir, 'PNG','ThirdView', 'Frame%04d'%(bpy.data.scenes[0].frame_current))
    bpy.ops.render.render(write_still=True)

    # Render Drone Camera
    cam = bpy.data.objects['Camera']    
    bpy.context.scene.camera = cam
    bpy.context.scene.render.filepath = os.path.join(path_dir, 'PNG','Frames', 'Frame%04d'%(bpy.data.scenes[0].frame_current))
    bpy.ops.render.render(write_still=True)
    
def VisionAndPlanner(GoalLocation):
    # USE cv2.imread to the latest frame from 'Frames' Folder
    # HINT: You can get the latest frame using: bpy.data.scenes[0].frame_current
    # USE ReadEXR() function provided below to the latest depth image saved from 'Depth' Folder
    # Compute Commands to go left and right (VelX) using any method you like
    path_dir = "/home/naitri/Documents/mtrl-auto-uav/blender/render"
    
            # print(os.path.join(path_dir, 'Frames', 'Frame%04d'%(bpy.data.scenes[0].frame_current)))

    ""
    #D = Exr2Depth(os.path.join(path_dir, 'Depth', 'Depth%04d.exr'%(bpy.data.scenes[0].frame_current)))
    #D = Exr2Depth(os.path.join(path_dir, 'Image%04d.exr'%(bpy.data.scenes[0].frame_current)))
    PATH_TO_EXR_FILE = os.path.join(path_dir, 'PNG','Frames', 'Frame%04d.png'%(bpy.data.scenes[0].frame_current))
    print("******************************",PATH_TO_EXR_FILE)
    D = cv2.imread(PATH_TO_EXR_FILE) 

    cv2.imshow('depth', D)
   
    depth = cv2.resize(D, (224,224))
    depth =depth.reshape(1,224,224,3)
    #x = np.expand_dims(depth, axis=0)
    #cv2.imshow('resized', depth)
    #print(depth.shape)
    pos_x,pos_y,pos_z,rot_w,rot_x,rot_y,rot_z = model.predict(depth)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(pos_x)
    # xplots.append(pos_x[0][0])
    # yplots.append(pos_y[0][0])
    # zplots.append(pos_z[0][0])
    return pos_x/8, pos_y, pos_z

    # You can visualize depth and image as follows
    # cv2.imshow('Depth', D)
    # cv2.imshow('Image', I)
    #cv2.waitKey(0)
    
    

def Controller(TrafficCylinder, Camera):
    GoalReached = False # We are far from the goal when we start
    MaxFrames = 10 # Run it for a maximum of 100 frames
    OutOfBounds = False
   
    GoalLocation = [TrafficCylinder.location[0], TrafficCylinder.location[1], TrafficCylinder.location[2]]
    
    while(not (GoalReached or OutOfBounds or bpy.data.scenes['Scene'].frame_current>=MaxFrames)): 
        Render()   
        VelX, y ,z = VisionAndPlanner(GoalLocation)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(VelX, y ,z)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # World Axis Convention: +Y is front, +Z is up and +X is right
        # Change Location of the active object (in our case the Camera or our Drone)
        Camera.location[0] += 1# Your controller changes this with feedback from vision
        
        Camera.location[1] += 1 # Forward velocity if fixed, m/frame, DO NOT CHANGE!
        # Camera.location[2] += z # Perfect altittude hold, this does not change
        
        # Increment Frame counter so you know it's the next step
        bpy.data.scenes['Scene'].frame_set(bpy.data.scenes['Scene'].frame_current + 1)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&")
        #print(bpy.context.object.location)
        print(Camera.location)
        
        # DO NOT MODIFY BELOW THIS!
        DistToGoal = math.sqrt((GoalLocation[0]-Camera.location[0])**2 + (GoalLocation[1]-Camera.location[1])**2 + (GoalLocation[2]-bpy.context.object.location[2])**2)
        if(DistToGoal <= 2.):
            GoalReached = True
        
            
path_dir = bpy.data.scenes["Scene"].node_tree.nodes["File Output"].base_path
def main():
    # plot_ = plt.fig(figsize=plt.figaspect(0.5))
    # chart_ = plot_.add_subplot(2,2,2,projection='3d')
    # chart_.set_xlabel('X-axis')
    # chart_.set_ylabel('Y-axis')
    # chart_.set_zlabel('Z-axis')
    SetRenderSettings()
    # Reset Frame to 0
    bpy.data.scenes['Scene'].frame_set(0)
    # Deselect all objects
    for obj in bpy.data.objects:
        obj.select_set(False)

    # Get Variables for objects we want to read/modify
    TrafficCylinder = bpy.data.objects['Cylinder.070']
    Camera = bpy.data.objects['Camera'] # This is the drone

    # Set camera to start point (DO NOT CHANGE THE START POINT OF THE CAMERA!)
    Camera.location[0] = 20.0
    Camera.location[1] = 0.0
    Camera.location[2] = 5.0
    Controller(TrafficCylinder, Camera)
    chart_.scatter3D(xplots,yplots,zplots)
    plt.show()
model = load_trained_model('/home/naitri/Documents/mtrl-auto-uav/models/model_0.5004407.h5')

if __name__=="__main__":
    main()