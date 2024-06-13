# extract data from the EgoBody dataset

import numpy as np
import os
import json
import pickle
import pandas as pd
from utils import remake_dir
from scipy.spatial.transform import Rotation
import smplx
import torch
import glob


dataset_path = "/datasets/public/zhiming_datasets/egobody/"
dataset_processed_path = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
timestamp_frames_path = './timestamps_frames/'
smplx_model_path = "./"
remake_dir(dataset_processed_path)
remake_dir(dataset_processed_path + "train/")
remake_dir(dataset_processed_path + "test/")
data_split_info = pd.read_csv('./data_splits.csv')
train_split_list = list(data_split_info['train'])
val_split_list = list(data_split_info['val'])
test_split_list = list(data_split_info['test'])

################################################ read body idx info
dataset_info = pd.read_csv('./data_info_actions.csv')
recording_name_list = list(dataset_info['recording_name'])
start_frame_list = list(dataset_info['start_frame'])
end_frame_list = list(dataset_info['end_frame'])
body_idx_fpv_list = list(dataset_info['body_idx_fpv'])
gender_0_list = list(dataset_info['body_idx_0'])
gender_1_list = list(dataset_info['body_idx_1'])

body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
gender_0_dict = dict(zip(recording_name_list, gender_0_list))
gender_1_dict = dict(zip(recording_name_list, gender_1_list))
start_frame_dict = dict(zip(recording_name_list, start_frame_list))
end_frame_dict = dict(zip(recording_name_list, end_frame_list))

def row(A):
    return A.reshape((1, -1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    # print(xyz_source_coord.shape)
    # print(xyz_source_coord)
    # print(trans_mtx[:3, :3])
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])

def load_head_hand_eye_data(csv_path):
    joint_count = 26

    # load head and eye tracking of hololens data
    data = np.loadtxt(csv_path, delimiter=',')

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    left_hand_transs = np.zeros((n_frames, joint_count, 3))
    left_hand_transs_available = np.ones(n_frames, dtype=bool)
    right_hand_transs = np.zeros((n_frames, joint_count, 3))
    right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        head_transs[i_frame, :] = frame[1:17].reshape((4, 4))[:3, 3]

        # left hand
        left_hand_transs_available[i_frame] = (frame[17] == 1)
        left_start_id = 18
        for i_j in range(joint_count):
            j_start_id = left_start_id + 16 * i_j
            j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
            left_hand_transs[i_frame, i_j, :] = j_trans
        # right hand
        right_hand_transs_available[i_frame] = (frame[left_start_id + joint_count * 4 * 4] == 1)
        right_start_id = left_start_id + joint_count * 4 * 4 + 1
        for i_j in range(joint_count):
            j_start_id = right_start_id + 16 * i_j
            j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
            right_hand_transs[i_frame, i_j, :] = j_trans

        # assert(j_start_id + 16 == 851)
        gaze_available[i_frame] = (frame[851] == 1)
        # if gaze_available[i_frame] != True:
        #     print(i_frame)
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    return (timestamps, head_transs, left_hand_transs, left_hand_transs_available,
            right_hand_transs, right_hand_transs_available, gaze_data, gaze_available)
    # return (timestamps, head_transs, gaze_data, gaze_available)


def get_eye_gaze_point(gaze_data):
    origin_homog = gaze_data[:4]
    direction_homog = gaze_data[4:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    # if no distance was recorded, set 1m by default
    dist = gaze_data[8] if gaze_data[8] > 0.0 else 1.0
    point = origin_homog + direction_homog * dist
    return point[:3], origin_homog, direction_homog, dist


for i, seq in enumerate(dataset_info['recording_name']):
    calib_trans_dir = os.path.join(dataset_path, 'calibrations', seq)  # extrinsics
    # fpv_recording_dir = glob.glob(os.path.join(timestamps_path, 'egocentric_color', seq, '202*'))[0]

    start_frame = dataset_info['start_frame'][i]
    end_frame = dataset_info['end_frame'][i]
    data_num = end_frame - start_frame + 1
    # data_num_all = data_num_all + data_num
    
    scene = dataset_info['scene_name'][i]    
    action = dataset_info['action'][i]
    if pd.isna(action):
        action = 'None'
    
    if seq in train_split_list:
        split = 'train'
    elif seq in val_split_list:
        split = 'val'
    elif seq in test_split_list:
        split = 'test'
    else:
        print('Error: {} not in all splits.'.format(dataset_info['recording_name'][i]))
        exit()
    print("processing {}th file: scene: {}, seq: {}, action: {}...\n".format(i+1, scene, seq, action))
    
    ################################## read hololens world <-> kinect master RGB cam extrinsics
    holo2kinect_dir = os.path.join(calib_trans_dir, 'cal_trans', 'holo_to_kinect12.json')
    with open(holo2kinect_dir, 'r') as f1:
        trans_holo2kinect = np.array(json.load(f1)['trans'])
    # trans_kinect2holo = np.linalg.inv(trans_holo2kinect)
    
    kinect2global_dir = os.path.join(calib_trans_dir, 'cal_trans', 'kinect12_to_world', scene+'.json')
    with open(kinect2global_dir, 'r') as f2:
        trans_kinect2global = np.array(json.load(f2)['trans'])


    ######## get body idx for camera wearer/second person
    interactee_idx = int(body_idx_fpv_dict[seq].split(' ')[0])
    camera_wearer_idx = 1 - interactee_idx

    ######### get gender for camera weearer/second person
    interactee_gender = body_idx_fpv_dict[seq].split(' ')[1]
    if camera_wearer_idx == 0:
        camera_wearer_gender = gender_0_dict[seq].split(' ')[1]
    elif camera_wearer_idx == 1:
        camera_wearer_gender = gender_1_dict[seq].split(' ')[1]


    fitting_root_camera_wearer = os.path.join(dataset_path, 'smplx_camera_wearer_{}'.format(split), seq)
    fitting_root_interactee = os.path.join(dataset_path, 'smplx_interactee_{}'.format(split), seq)


    file_name =  timestamp_frames_path + seq + "_" + "timestamps.npy"
    timestamp_dict = np.load(file_name, allow_pickle=True).item()

    # Process eye gaze data
    gaze_dir = glob.glob(os.path.join(dataset_path, 'egocentric_gaze', seq, '202*'))[0]
    holo_gaze_file_path = glob.glob(os.path.join(gaze_dir, '*_head_hand_eye.csv'))[0]
    gaze_point3d_dict = {}
    gaze_data = []
    # gaze_data = np.zeros((data_num, 3)) 
    (timestamps, _, _, _, _, _, gaze_data_all, gaze_available) = load_head_hand_eye_data(holo_gaze_file_path)
    for pv_timestamp in timestamp_dict.keys():
        gaze_ts = match_timestamp(int(pv_timestamp), timestamps)
        # print(gaze_ts)
        if gaze_available[gaze_ts]:
            point, origin_homog, direction_homog, dist = get_eye_gaze_point(gaze_data_all[gaze_ts]) 
            # print(point)
            point = points_coord_trans(point, trans_holo2kinect)  # trans 3d points from hololens world coord to kinect world
            point = points_coord_trans(point, trans_kinect2global)  # trans 3d points from kinect world coord to global world
            cur_frame_id = timestamp_dict[pv_timestamp]
            gaze_point3d_dict[cur_frame_id] = point

    for key, value in gaze_point3d_dict.items():
        gaze_data.append(value)
    
    if len(gaze_data) == 0:
        print("\n\n\n<<<<<<<<no gaze available at: {}>>>>>>>>\n\n\n".format(dataset_info['recording_name'][i]))
        continue
    gaze_data = np.array(gaze_data)
    gaze_data = np.squeeze(gaze_data)


    pose_data = np.zeros((gaze_data.shape[0], 23*3))
    pose_interactee_data = np.zeros((gaze_data.shape[0], 23*3))
    head_data = np.zeros((gaze_data.shape[0], 3))
    neck_data = np.zeros((gaze_data.shape[0], 3))
    pelvis_data = np.zeros((gaze_data.shape[0], 3))
    spine3_data = np.zeros((gaze_data.shape[0], 3))
    spine2_data = np.zeros((gaze_data.shape[0], 3))
    spine1_data = np.zeros((gaze_data.shape[0], 3))


    # Process human pose data
    body_model = smplx.create(os.path.join(smplx_model_path, 'smplx_model'), 
                                       model_type='smplx',
                                       gender='neutral', 
                                       ext='npz', 
                                       num_pca_comps=12,
                                       create_global_orient=True, 
                                       create_transl=True, 
                                       create_body_pose=True,
                                       create_betas=True,
                                       create_left_hand_pose=True, 
                                       create_right_hand_pose=True,
                                       create_expression=True, 
                                       create_jaw_pose=True, 
                                       create_leye_pose=True,
                                       create_reye_pose=True)

    for i_frame in range(start_frame_dict[seq], end_frame_dict[seq]+1):
        frame_id = 'frame_{}'.format("%05d"%i_frame)
        frame = i_frame - start_frame_dict[seq]  

        # Load camera_wearer pose
        with open(os.path.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl'), 'rb') as f1:
            pose = pickle.load(f1)
        # Load interactee pose
        with open(os.path.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl'), 'rb') as f2:
            pose_interactee = pickle.load(f2)

        # Decode global camera_wearer pose  
        torch_param = {}
        for key in pose.keys():
            if key in ['body_pose', 'pose_embedding', 'global_orient', 'transl']:
                torch_param[key] = torch.tensor(pose[key])
            else:
                continue

        pose_trans = pose['transl']
        pose_ori = pose['global_orient'].squeeze()
        body_pose = pose['body_pose'].reshape((21, 3))  

        pose_ori_R = Rotation.from_rotvec(pose_ori).as_matrix()
        # pose_ori_global = points_coord_trans(pose_ori_R, trans_kinect2global)
        pose_ori_global = trans_kinect2global[:3, :3] @ pose_ori_R
        pose_ori_global_mat = Rotation.from_matrix(pose_ori_global).as_matrix()

        smplx_global = body_model(return_verts=True, **torch_param)
        pose_global = smplx_global.joints[0][:23, :].detach().cpu().numpy()
        pose_global = points_coord_trans(pose_global, trans_kinect2global)
        pose_data[frame] = pose_global.reshape((1, 23*3))


        # Decode global interactee pose    
        torch_param_interactee = {}
        for key in pose_interactee.keys():
            if key in ['body_pose', 'pose_embedding', 'global_orient', 'transl']:
                torch_param_interactee[key] = torch.tensor(pose_interactee[key])
            else:
                continue

        pose_trans_interactee = pose_interactee['transl']
        pose_ori_interactee = pose_interactee['global_orient'].squeeze()
        body_pose_interactee = pose_interactee['body_pose'].reshape((21, 3))  

        pose_ori_R_interactee = Rotation.from_rotvec(pose_ori_interactee).as_matrix()
        # pose_ori_global = points_coord_trans(pose_ori_R, trans_kinect2global)
        pose_ori_global_interactee = trans_kinect2global[:3, :3] @ pose_ori_R_interactee
        pose_ori_global_mat_interactee = Rotation.from_matrix(pose_ori_global_interactee).as_matrix()

        smplx_global_interactee = body_model(return_verts=True, **torch_param_interactee)
        pose_global_interactee = smplx_global_interactee.joints[0][:23, :].detach().cpu().numpy()
        pose_global_interactee = points_coord_trans(pose_global_interactee, trans_kinect2global)
        pose_interactee_data[frame] = pose_global_interactee.reshape((1, 23*3))

        head_position = pose_data[frame, 15*3:16*3]
        gaze_data[frame] = gaze_data[frame] - head_position
        gaze_data[frame] = [x / np.linalg.norm(gaze_data[frame]) for x in gaze_data[frame]]            

        # Calculate head direction
        head_direction = [0, 0, 1]
        head_rot = Rotation.identity().as_matrix()
        head_rotation_list = [14, 11, 8, 5, 2]
        for k in head_rotation_list:                
            rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
            head_rot = np.matmul(rot, head_rot)
        
        head_direction = np.matmul(head_rot, head_direction)
        head_data[frame] = np.matmul(pose_ori_global_mat, head_direction)

        # calculate neck direction
        neck_direction = [0, 0, 1]
        neck_rot = Rotation.identity().as_matrix()
        neck_rotation_list = [11, 8, 5, 2]
        for k in neck_rotation_list:                
            rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
            neck_rot = np.matmul(rot, neck_rot)
        neck_direction = np.matmul(neck_rot, neck_direction)
        neck_data[frame] = np.matmul(pose_ori_global_mat, neck_direction)

        # calculate spine3 direction
        spine3_direction = [0, 0, 1]
        spine3_rot = Rotation.identity().as_matrix()
        spine3_rotation_list = [8, 5, 2]
        for k in spine3_rotation_list:                
            rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
            spine3_rot = np.matmul(rot, spine3_rot)
        spine3_direction = np.matmul(spine3_rot, spine3_direction)
        spine3_data[frame] = np.matmul(pose_ori_global_mat, spine3_direction)
        
        # calculate spine2 direction
        spine2_direction = [0, 0, 1]
        spine2_rot = Rotation.identity().as_matrix()
        spine2_rotation_list = [5, 2]
        for k in spine2_rotation_list:                
            rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
            spine2_rot = np.matmul(rot, spine2_rot)
        spine2_direction = np.matmul(spine2_rot, spine2_direction)
        spine2_data[frame] = np.matmul(pose_ori_global_mat, spine2_direction)
        
        # calculate spine1 direction
        spine1_direction = [0, 0, 1]
        spine1_rot = Rotation.identity().as_matrix()
        spine1_rotation_list = [2]
        for k in spine1_rotation_list:                
            rot = Rotation.from_rotvec(body_pose[k, :]).as_matrix()
            spine1_rot = np.matmul(rot, spine1_rot)
        spine1_direction = np.matmul(spine1_rot, spine1_direction)
        spine1_data[frame] = np.matmul(pose_ori_global_mat, spine1_direction)

        # calculate pelvis direction
        pelvis_direction = [0, 0, 1]
        pelvis_direction = np.array(pelvis_direction)
        pelvis_data[frame] = np.matmul(pose_ori_global_mat, pelvis_direction)
        
    if split == 'train' or split == 'val':
        pose_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_xyz.npy"
        pose_interactee_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_interactee.npy"
        gaze_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"gaze.npy"
        head_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"head.npy"

        pelvis_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pelvis.npy"
        spine1_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine1.npy"
        spine2_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine2.npy"
        spine3_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine3.npy"
        neck_file = dataset_processed_path + "train/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"neck.npy"

    if split == 'test':
        pose_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_xyz.npy"
        pose_interactee_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pose_interactee.npy"
        gaze_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" + "gaze.npy"
        head_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"head.npy"

        pelvis_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"pelvis.npy"
        spine1_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine1.npy"
        spine2_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine2.npy"
        spine3_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"spine3.npy"
        neck_file = dataset_processed_path + "test/" + action + "_" + scene + "_" + seq + "_" + str(start_frame) + "_" +"neck.npy"

    np.save(pose_file, pose_data)
    np.save(pose_interactee_file, pose_interactee_data)
    np.save(gaze_file, gaze_data)
    np.save(head_file, head_data)
    np.save(pelvis_file, pelvis_data)
    np.save(spine1_file, spine1_data)
    np.save(spine2_file, spine2_data)
    np.save(spine3_file, spine3_data)
    np.save(neck_file, neck_data)