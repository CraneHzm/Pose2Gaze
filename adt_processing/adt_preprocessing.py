import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import math
from math import tan
import random
from scipy.linalg import pinv
import projectaria_tools.core.mps as mps
import shutil
import json
from PIL import Image
from utils import remake_dir
import pandas as pd
import pylab as p
from IPython.display import display
import time


from projectaria_tools import utils
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinSkeletonProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
   bbox2d_to_image_coordinates,
   utils as adt_utils,
    Aria3dPose
)


dataset_path = '/datasets/public/zhiming_datasets/adt/'
dataset_processed_path = '/scratch/hu/pose_forecast/adt_pose2gaze/'
remake_dir(dataset_processed_path)
remake_dir(dataset_processed_path + "train/")
remake_dir(dataset_processed_path + "test/")
dataset_info = pd.read_csv('adt.csv')
save_images = False


for i, seq in enumerate(dataset_info['sequence_name']):        
    action = dataset_info['action'][i]
    print("\nprocessing {}th seq: {}, action: {}...".format(i, seq, action))
    seq_path = dataset_path + seq + '/'
    if dataset_info['training'][i] == 1:
        save_path = dataset_processed_path + 'train/' + seq + '_'        
        if save_images:
            img_path = dataset_processed_path + 'train/' + seq + '_images/'
            remake_dir(img_path)
    if dataset_info['training'][i] == 0:
        save_path = dataset_processed_path + 'test/' + seq + '_'        
        if save_images:
            img_path = dataset_processed_path + 'test/' + seq + '_images/'
            remake_dir(img_path)
        
    paths_provider = AriaDigitalTwinDataPathsProvider(seq_path)
    all_device_serials = paths_provider.get_device_serial_numbers()
    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    print("loading ground truth data done")
    
    stream_id = StreamId("214-1")
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)
    frame_num = len(img_timestamps_ns)
    print("There are {} frames".format(frame_num))

    # get all available skeletons in a sequence
    skeleton_ids = gt_provider.get_skeleton_ids()
    skeleton_info = gt_provider.get_instance_info_by_id(skeleton_ids[0])
    print("skeleton ", skeleton_info.name, " wears ", skeleton_info.associated_device_serial)

    useful_frame = []
    gaze_data = np.zeros((frame_num, 6))
    head_data = np.zeros((frame_num, 3))
    joint_number = 21
    pose_data = np.zeros((frame_num, joint_number*3))
    
    local_time = time.asctime(time.localtime(time.time()))
    print('\nProcessing starts at ' + local_time)    
    for j in range(frame_num):
        timestamps_ns = img_timestamps_ns[j]
        
        skeleton_with_dt = gt_provider.get_skeleton_by_timestamp_ns(timestamps_ns, skeleton_ids[0])
        assert skeleton_with_dt.is_valid(), "skeleton is not valid"
        
        skeleton = skeleton_with_dt.data()
        # use the 21 body joints
        body_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]
        joints = np.array(skeleton.joints)[body_joints, :].reshape(joint_number*3)
        pose_data[j] = joints
        
        # convert image to numpy array
        if save_images:
            image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamps_ns, stream_id)
            image = image_with_dt.data().to_numpy_array()
            # pad SLAM camera gray-scale image to 3 channel for color visualization
            image = np.repeat(image[..., np.newaxis], 3, axis=2) if len(image.shape) < 3 else image
            
        # get the Aria pose
        aria3dpose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamps_ns)
        if not aria3dpose_with_dt.is_valid():
            print("aria 3d pose is not available")
        aria3dpose = aria3dpose_with_dt.data()        
        transform_scene_device = aria3dpose.transform_scene_device.matrix()

        # get projection function
        cam_calibration = gt_provider.get_aria_camera_calibration(stream_id)
        assert cam_calibration is not None, "no camera calibration"

        eye_gaze_with_dt = gt_provider.get_eyegaze_by_timestamp_ns(timestamps_ns)
        assert eye_gaze_with_dt.is_valid(), "Eye gaze not available"
        
        # Project the gaze center in CPF frame into camera sensor plane, with multiplication performed in homogenous coordinates
        eye_gaze = eye_gaze_with_dt.data()
        gaze_center_in_cpf = np.array([tan(eye_gaze.yaw), tan(eye_gaze.pitch), 1.0], dtype=np.float64) * eye_gaze.depth
        transform_cpf_sensor = gt_provider.raw_data_provider_ptr().get_device_calibration().get_transform_cpf_sensor(cam_calibration.get_label())
        gaze_center_in_camera = transform_cpf_sensor.inverse().matrix() @ np.hstack((gaze_center_in_cpf, 1)).T
        gaze_center_in_camera = gaze_center_in_camera[:3] / gaze_center_in_camera[3:]
        gaze_center_in_pixels = cam_calibration.project(gaze_center_in_camera)
        
        extrinsic_matrix = cam_calibration.get_transform_device_camera().matrix()
        gaze_center_in_device = (extrinsic_matrix @ np.hstack((gaze_center_in_camera, 1)))[0:3]
        gaze_center_in_scene = (transform_scene_device @ np.hstack((gaze_center_in_device, 1)))[0:3]
        head_position = joints[4*3:5*3]
        gaze_direction = gaze_center_in_scene - head_position
        gaze_direction = [x / np.linalg.norm(gaze_direction) for x in gaze_direction]
        
        # calculate head direction
        head_center_in_cpf = np.array([0, 0, 1.0], dtype=np.float64)
        head_center_in_camera = transform_cpf_sensor.inverse().matrix() @ np.hstack((head_center_in_cpf, 0)).T
        head_center_in_camera = head_center_in_camera[:3]
        
        head_center_in_device = (extrinsic_matrix @ np.hstack((head_center_in_camera, 0)))[0:3]
        head_center_in_scene = (transform_scene_device @ np.hstack((head_center_in_device, 0)))[0:3]        
        head_direction = head_center_in_scene        
        head_direction = [x / np.linalg.norm(head_direction) for x in head_direction]
        head_data[j, 0:3] = head_direction        
        
        if gaze_center_in_pixels is not None:
            x_pixel = gaze_center_in_pixels[1]
            y_pixel = gaze_center_in_pixels[0]
            gaze_center_in_pixels[0] = x_pixel
            gaze_center_in_pixels[1] = y_pixel
            
            if save_images:
                rotated_image = np.rot90(image, k=-1)
                flipped_image = np.fliplr(rotated_image)
                img = Image.fromarray(flipped_image)
                path = img_path + str(j) + '.png'
                img.save(path)
                
            useful_frame.append(j)
            gaze_2d = np.divide(gaze_center_in_pixels, cam_calibration.get_image_size())
            gaze_data[j, 0:3] = gaze_direction
            gaze_data[j, 3:5] = gaze_2d
            gaze_data[j, 5] = j

            
    gaze_data = gaze_data[useful_frame, :]
    head_data = head_data[useful_frame, :]
    pose_data = pose_data[useful_frame, :]    
    gaze_path = save_path + 'gaze.npy'
    head_path = save_path + 'head.npy'
    pose_path = save_path + 'pose_xyz.npy'
    np.save(gaze_path, gaze_data)
    np.save(head_path, head_data)
    np.save(pose_path, pose_data)
    local_time = time.asctime(time.localtime(time.time()))
    print('\nProcessing ends at ' + local_time)