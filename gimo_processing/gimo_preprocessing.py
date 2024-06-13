# extract data from the GIMO dataset

import numpy as np
import os
import json
import pickle
import pandas as pd
import trimesh
from utils import remake_dir
from scipy.spatial.transform import Rotation
from human_body_prior.tools.model_loader import load_vposer
import smplx
import torch

dataset_path = "/datasets/public/zhiming_datasets/gimo/"
dataset_processed_path = "/scratch/hu/pose_forecast/gimo_pose2gaze/"
remake_dir(dataset_processed_path)
remake_dir(dataset_processed_path + "train/")
remake_dir(dataset_processed_path + "test/")

dataset_info = pd.read_csv('dataset_actions.csv')
training_item_num = 0
training_data_num = 0
test_item_num = 0
test_data_num = 0
check_gaze_head_distance = True # discard data when gaze head distance is too big/too unrealistic
gaze_head_distance_thres = 35 # degree
data_num_all = 0
data_used_num_all = 0


for i, seq in enumerate(dataset_info['sequence_path']):
    start_frame = dataset_info['start_frame'][i]
    end_frame = dataset_info['end_frame'][i]
    data_num = end_frame - start_frame + 1
    data_num_all = data_num_all + data_num
    #print(start_frame, end_frame, seq)
    scene = dataset_info['scene'][i]    
    action = dataset_info['action'][i]
    if pd.isna(action):
        action = 'None'
    if dataset_info['training'][i] != 1 and dataset_info['training'][i] != 0:
        continue
    print("\nprocessing scene: {}, seq: {}, action: {}...".format(scene, seq, action))
    
    transform_path = dataset_info['transformation'][i]
    transform_info = json.load(open(os.path.join(dataset_path, scene, seq, transform_path), 'r'))    
    scale = transform_info['scale']    
    trans_pose2scene = np.array(transform_info['transformation'])
    trans_pose2scene[:3, 3] /= scale
    transform_norm = np.loadtxt(os.path.join(dataset_path, scene,'scene_obj', 'transform_norm.txt')).reshape((4, 4))
    transform_norm[:3, 3] /= scale
    transform_pose = transform_norm @ trans_pose2scene

    # Process eye gaze data
    gaze_data = np.zeros((data_num, 3))    
    gaze_mask = np.zeros((data_num, 1))
    for j in range(data_num):
        frame = start_frame + j        
        gaze_ply_path = os.path.join(dataset_path, scene, seq,'eye_pc','{}_center.ply'.format(frame))        
        if os.path.exists(gaze_ply_path):
            gaze = trimesh.load_mesh(gaze_ply_path)
            gaze.apply_scale(1 / scale)
            gaze.apply_transform(transform_norm)   
            points = gaze.vertices
            if np.sum(abs(points)) > 1e-8:
                gaze_mask[j] = 1                         
            gaze_data[j] = gaze.vertices[0:1]            
            #print(gaze_data[j])

    gaze_valid_id = np.where(gaze_mask==1)[0]
    gaze_invalid_id = np.where(gaze_mask==0)[0]
    #print(gaze_invalid_id.shape)
    #print(gaze_data.shape)
    gaze_data_valid = gaze_data[gaze_valid_id, :]    
    gaze_data[gaze_invalid_id, :] *= 0        
    gaze_data[gaze_invalid_id, :] += np.mean(gaze_data_valid, axis=0, keepdims=True)
    #print(gaze_data)
    
    pose_data = np.zeros((data_num, 23*3))
    # body orientations: head, pelvis, spine1, spine2, spine3, neck
    head_data = np.zeros((data_num, 3))
    pelvis_data = np.zeros((data_num, 3))
    spine1_data = np.zeros((data_num, 3))
    spine2_data = np.zeros((data_num, 3))
    spine3_data = np.zeros((data_num, 3))
    neck_data = np.zeros((data_num, 3))
        
    # Process human pose data
    vposer, _ = load_vposer('vposer_v1_0', vp_model='snapshot')
    body_mesh_model = smplx.create('smplx_models',
                                   model_type='smplx',
                                   gender='neutral', ext='npz',
                                   num_pca_comps=12,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1,
                                   num_betas=10,
                                   num_expression_coeffs=10)
    joint_names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw"]
    for j in range(data_num):
        frame = start_frame + j   
        pose_path = os.path.join(dataset_path, scene, seq,'smplx_local','{}.pkl'.format(frame))
        if os.path.exists(pose_path):
            # Load human pose
            pose = pickle.load(open(pose_path, 'rb'))
            #print(pose)
            pose_trans = pose['trans'].detach().cpu().numpy().reshape((3, 1))
            pose_ori = pose['orient'].detach().cpu().numpy()
            pose_latent = pose['latent']
            body_pose = vposer.decode(pose_latent, output_type='aa').cpu().unsqueeze(0)
            pose_trans_global = (transform_pose[:3, :3] @ pose_trans + transform_pose[:3, 3:]).reshape(3)
            pose_trans_global = torch.from_numpy(pose_trans_global.copy()).float()
            #print(pose_trans_global)
            pose_trans_local = torch.zeros(3)
            #print(pose_trans_local)
            
            R = Rotation.from_rotvec(pose_ori).as_matrix()
            R_s = transform_pose[:3, :3] @ R
            pose_ori_global_mat = Rotation.from_matrix(R_s).as_matrix()
            pose_ori_global = Rotation.from_matrix(R_s).as_rotvec()
            pose_ori_global = torch.from_numpy(pose_ori_global.copy()).float()
            
            # Decode global human pose
            pose_global = {}            
            pose_global['body_pose'] = body_pose
            pose_global['pose_embedding'] = pose_latent.cpu().unsqueeze(0)
            pose_global['global_orient'] = pose_ori_global.cpu().unsqueeze(0)
            pose_global['transl'] = pose_trans_global.cpu().unsqueeze(0)
            smplx_global = body_mesh_model(return_verts=True, **pose_global)
            pose_global = smplx_global.joints[0][:23, :].detach().cpu().numpy()
            pose_data[j] = pose_global.reshape((1, 23*3))
            
            head_position = pose_data[j, 15*3:16*3]
            gaze_data[j] = gaze_data[j] - head_position
            gaze_data[j] = [x / np.linalg.norm(gaze_data[j]) for x in gaze_data[j]]                                                           
            # Calculate head orientation
            body_pose = body_pose.detach().numpy().reshape((21, 3))            
            head_direction = [0, 0, 1]
            head_rot = Rotation.identity().as_matrix()
            head_rotation_list = [14, 11, 8, 5, 2]
            for k in head_rotation_list:                
                rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
                head_rot = np.matmul(rot, head_rot)
            
            head_direction = np.matmul(head_rot, head_direction)
            head_data[j] = np.matmul(pose_ori_global_mat, head_direction)
                                                    
            # calculate neck orientation
            neck_direction = [0, 0, 1]
            neck_rot = Rotation.identity().as_matrix()
            neck_rotation_list = [11, 8, 5, 2]
            for k in neck_rotation_list:                
                rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
                neck_rot = np.matmul(rot, neck_rot)
            
            neck_direction = np.matmul(neck_rot, neck_direction)
            neck_data[j] = np.matmul(pose_ori_global_mat, neck_direction)
            
            # calculate spine3 direction
            spine3_direction = [0, 0, 1]
            spine3_rot = Rotation.identity().as_matrix()
            spine3_rotation_list = [8, 5, 2]
            for k in spine3_rotation_list:                
                rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
                spine3_rot = np.matmul(rot, spine3_rot)
            
            spine3_direction = np.matmul(spine3_rot, spine3_direction)
            spine3_data[j] = np.matmul(pose_ori_global_mat, spine3_direction)
            
            # calculate spine2 direction
            spine2_direction = [0, 0, 1]
            spine2_rot = Rotation.identity().as_matrix()
            spine2_rotation_list = [5, 2]
            for k in spine2_rotation_list:                
                rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
                spine2_rot = np.matmul(rot, spine2_rot)
            
            spine2_direction = np.matmul(spine2_rot, spine2_direction)
            spine2_data[j] = np.matmul(pose_ori_global_mat, spine2_direction)
            
            # calculate spine1 direction
            spine1_direction = [0, 0, 1]
            spine1_rot = Rotation.identity().as_matrix()
            spine1_rotation_list = [2]
            for k in spine1_rotation_list:                
                rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
                spine1_rot = np.matmul(rot, spine1_rot)
            
            spine1_direction = np.matmul(spine1_rot, spine1_direction)
            spine1_data[j] = np.matmul(pose_ori_global_mat, spine1_direction)

            # calculate pelvis direction
            pelvis_direction = [0, 0, 1]
            pelvis_data[j] = np.matmul(pose_ori_global_mat, pelvis_direction)
            

    if check_gaze_head_distance:    
        gaze_head_distance = np.mean(np.arccos(np.sum(gaze_data*head_data, 1)))/np.pi * 180.0
        print("gaze head distance:{}".format(gaze_head_distance))
        if gaze_head_distance > gaze_head_distance_thres:
            continue
        else:
            data_used_num_all = data_used_num_all + data_num
            
    # save the data
    if dataset_info['training'][i] == 1:
        gaze_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"gaze.npy"
        pose_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_xyz.npy"
        head_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"head.npy"
        
        pelvis_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pelvis.npy"
        spine1_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine1.npy"
        spine2_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine2.npy"
        spine3_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine3.npy"
        neck_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"neck.npy"
                
        training_item_num +=1
        training_data_num += data_num
        print("Training item number: {}, training data size: {}".format(training_item_num, training_data_num))
        
    if dataset_info['training'][i] == 0:
        gaze_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" + "gaze.npy"
        pose_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_xyz.npy"
        head_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"head.npy"
        
        pelvis_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pelvis.npy"
        spine1_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine1.npy"
        spine2_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine2.npy"
        spine3_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine3.npy"
        neck_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"neck.npy"
                
        test_item_num +=1
        test_data_num += data_num
        print("Test item number: {}, test data size: {}".format(test_item_num, test_data_num))
    
    np.save(gaze_file, gaze_data)
    np.save(pose_file, pose_data)
    np.save(head_file, head_data)
    np.save(pelvis_file, pelvis_data)
    np.save(spine1_file, spine1_data)
    np.save(spine2_file, spine2_data)
    np.save(spine3_file, spine3_data)
    np.save(neck_file, neck_data)
    
if check_gaze_head_distance:
    data_used_ratio = data_used_num_all/data_num_all*100
    print("gaze head distance thres: {}, data used ratio: {:.1f}%".format(gaze_head_distance_thres, data_used_ratio))
    