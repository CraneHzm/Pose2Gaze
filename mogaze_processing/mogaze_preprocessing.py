# Extract data from the MoGaze dataset
import h5py
import numpy as np
from utils import quaternion_matrix, remake_dir, euler2xyz, euler2xyz_head, euler2xyz_neck, euler2xyz_torso, euler2xyz_pelvis, euler2xyz_base
from scipy.spatial.transform import Rotation as R


# the original mogaze dataset downloaded from https://humans-to-robots-motion.github.io/mogaze/
dataset_path = "/datasets/public/zhiming_datasets/mogaze/"
dataset_processed_path = "/scratch/hu/pose_forecast/mogaze_pose2gaze/"
# ["7_2"] is not used due to the low quality of eye gaze data
data_idx = ["1_1", "1_2", "2_1", "4_1", "5_1", "6_1", "6_2", "7_1", "7_3"]
original_fps = 120.0
# downsample the original data to 30.0 fps
downsample_rate = 4
# check the quality of eye gaze data
check_eye_gaze = True
# gaze confidence level >= 0.6 is considered as high-quality
confidence_level_threshold = 0.6
# a recording is used only if it contains more than 80% of high-quality eye gaze data
confidence_ratio_threshold = 0.8
# drop the beginning: each action starts with a waiting time, thus we drop sometime in the beginning
drop_beginning = True
# for each recording, we use at most the last three seconds to avoid the waiting time (no action) at the beginning
time_use = 3 #seconds
data_all = 0
data_used = 0


for data in data_idx:
    print('processing data p{}'.format(data))
    data_processed_path = dataset_processed_path + "/p" + data + "/"
    remake_dir(data_processed_path)
    
    # load human pose data
    f_pose = h5py.File(dataset_path + 'p' + data + "_human_data.hdf5", 'r')
    pose = f_pose['data'][:]
    print("Human pose shape: {}".format(pose.shape))
    
    # load objects    
    f_object = h5py.File(dataset_path + 'p' + data + "_object_data.hdf5", 'r')
    objects = {}
    object_names = ['table', 'cup_red', 'shelf_laiva', 'shelf_vesken', 'plate_blue', 'jug', 'goggles', 'plate_green', 'plate_red', 'cup_green', 'cup_blue', 'red_chair', 'cup_pink', 'plate_pink', 'bowl', 'blue_chair']                       
    object_idx = 0
    for key in f_object['bodies'].keys():
        # Object Data: ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "rot_w"]
        objects[object_names[object_idx]] = f_object['bodies/' + key][:]
        object_idx = object_idx + 1        
    print("Object shape : {}".format(objects['table'].shape))
    # goggles object is used to calibrate the eye gaze data
    goggles_object = objects["goggles"]
    
    # load eye gaze
    gaze = np.zeros((pose.shape[0], 3))
    gaze_confidence = np.zeros((pose.shape[0], 1))
    with h5py.File(dataset_path + 'p' + data + "_gaze_data.hdf5", 'r') as f_gaze:
        gaze_data = f_gaze['gaze'][:, 2:5]
        confidence = f_gaze['gaze'][:, -1]
        calib = f_gaze['gaze'].attrs['calibration']
        for i in range(gaze.shape[0]):
                gaze_confidence[i] = confidence[i]
                rotmat = quaternion_matrix(calib)
                rotmat = np.dot(quaternion_matrix(goggles_object[i, 3:7]), rotmat)
                endpos = gaze_data[i]
                # Borrowed from https://github.com/PhilippJKratzer/humoro/blob/master/humoro/player_pybullet.py
                if endpos[2] < 0:
                    endpos *= -1  # mirror gaze point if wrong direction
                endpos = np.dot(rotmat, endpos) # gaze calibration
                direction = endpos - goggles_object[i][0:3]
                direction = [x / np.linalg.norm(direction) for x in direction]
                gaze[i] = direction                       
            
    print("Eye gaze shape : {}".format(gaze.shape))
    
    # segment the data
    f_seg = h5py.File(dataset_path + 'p' + data + "_segmentations.hdf5", 'r')
    segments = f_seg['segments'][:]
    pick_num = 0
    place_num = 0
    for i in range(len(segments) - 1):
        if data == "6_1" and i == 248:  # error in the dataset, null twice in a row
            i += 1
        if data == "7_2" and i == 0:  # error in the dataset, null twice in a row
            i += 1
                
        current_segment = segments[i]
        next_segment = segments[i + 1]
        current_goal = next_segment[2].decode("utf-8")
        current_object = current_segment[2].decode("utf-8")
                               
        action_time = (current_segment[1] - current_segment[0] + 1)/original_fps # seconds                
        #print("Action: {}, time: {:.1f} s".format(action, action_time))
                        
        start_frame = current_segment[0]
        end_frame = current_segment[1]
        data_all = data_all + end_frame - start_frame + 1
        #print("start: {}, end: {}".format(start_frame, end_frame))    
                        
        if check_eye_gaze:
            # reset the start and end frame
            for frame in range(start_frame, end_frame+1):
                if gaze_confidence[frame] >= confidence_level_threshold:
                    start_frame = frame
                    break
            for frame in range(end_frame, start_frame, -1):
                if gaze_confidence[frame] >= confidence_level_threshold:
                    end_frame = frame
                    break
            #print("new start: {}, new end: {}".format(start_frame, end_frame))
            
            # ratio of the high-quality eye gaze data    
            ratio = np.sum(gaze_confidence[start_frame:end_frame+1]>=confidence_level_threshold)/(end_frame-start_frame+1)
            if ratio >= confidence_ratio_threshold:
                # replace low-quality eye gaze data with previous high-quality one
                for frame in range(start_frame+1, end_frame):
                    if gaze_confidence[frame] < confidence_level_threshold:
                        gaze[frame] = gaze[frame-1]                
            else:
                continue
        
        data_used = data_used + end_frame - start_frame + 1
        
        if current_goal == "null":
            action = "place"
            place_num = place_num + 1
        else:
            action = "pick"
            pick_num = pick_num + 1
        
        
        # segment data
        length = end_frame-start_frame+1
        length_use = int(time_use*original_fps)
        if drop_beginning and length > length_use:
            start_frame = end_frame - length_use
            pose_seg = pose[start_frame:end_frame+1:downsample_rate, :]
            #print("Human pose segment shape: {}".format(pose_seg.shape))
            # calculate the xyz positions of human joints
            pose_xyz = euler2xyz(pose_seg)
            # calculate the head direction from human pose
            head_direction = euler2xyz_head(pose_seg)
            # calculate the neck, torso, pelvis, base directions from human pose
            neck_direction = euler2xyz_neck(pose_seg)
            torso_direction = euler2xyz_torso(pose_seg)
            pelvis_direction = euler2xyz_pelvis(pose_seg)
            base_direction = euler2xyz_base(pose_seg)
            gaze_seg = gaze[start_frame:end_frame+1:downsample_rate, :]
            #print("Eye gaze segment shape: {}".format(gaze_seg.shape))
            objects_seg = {}
            for key in objects.keys():
                objects_seg[key] = objects[key][start_frame:end_frame+1:downsample_rate, :]            
            #print("Object segment shape : {}".format(objects_seg['cup_red'].shape))
        else:
            pose_seg = pose[start_frame:end_frame+1:downsample_rate, :]
            #print("Human pose segment shape: {}".format(pose_seg.shape))
            # calculate the xyz positions of human joints
            pose_xyz = euler2xyz(pose_seg)
            # calculate the head direction from human pose
            head_direction = euler2xyz_head(pose_seg)
            neck_direction = euler2xyz_neck(pose_seg)
            torso_direction = euler2xyz_torso(pose_seg)
            pelvis_direction = euler2xyz_pelvis(pose_seg)
            base_direction = euler2xyz_base(pose_seg)            
            gaze_seg = gaze[start_frame:end_frame+1:downsample_rate, :]
            #print("Eye gaze segment shape: {}".format(gaze_seg.shape))
            objects_seg = {}
            for key in objects.keys():
                objects_seg[key] = objects[key][start_frame:end_frame+1:downsample_rate, :]            
            #print("Object segment shape : {}".format(objects_seg['cup_red'].shape))        
                     
                     
        # save the data      
        if action == "pick":
            num = pick_num
        if action == "place":
            num = place_num
                
        pose_euler_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "pose_euler.npy"
        pose_xyz_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "pose_xyz.npy"
        head_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "head.npy"        
        neck_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "neck.npy"
        torso_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "torso.npy"
        pelvis_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "pelvis.npy"
        base_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "base.npy"
        gaze_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "gaze.npy"
        objects_file = data_processed_path + action + "_" + str(num).zfill(3) + "_" + "objects.npy"
                  
        np.save(pose_euler_file, pose_seg)
        np.save(pose_xyz_file, pose_xyz)            
        np.save(head_file, head_direction)
        np.save(neck_file, neck_direction)
        np.save(torso_file, torso_direction)
        np.save(pelvis_file, pelvis_direction)
        np.save(base_file, base_direction)
        np.save(gaze_file, gaze_seg)
        np.save(objects_file, objects_seg)
        
print("data used: {}, data all: {}, ratio: {:.1f}%".format(data_used, data_all, data_used/data_all*100))