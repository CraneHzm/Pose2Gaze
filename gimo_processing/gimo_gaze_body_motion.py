import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "/scratch/hu/pose_forecast/gimo_pose2gaze/"
actions = ['change']
#actions = ['change', 'interact', 'rest']
test_dirs = ['train', 'test']
action_number = len(actions)

joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw"]

joint_number = len(joint_names)
gaze_motion = {}
for i in range(joint_number):
    gaze_motion[joint_names[i]] = []
    
for test_dir in test_dirs:
    path = data_dir + "/" + test_dir + "/"
    file_names = sorted(os.listdir(path))
    gaze_file_names = {}
    pose_file_names = {}
    
    for action_idx in np.arange(action_number):              
        gaze_file_names[actions[ action_idx ]] = []
        pose_file_names[actions[ action_idx ]] = []
        
    for name in file_names:
        name_split = name.split('_')
        action = name_split[0]                
        if action in actions:                
            data_type = name_split[-1][:-4]
            if(data_type == 'gaze'):
                gaze_file_names[action].append(name)
            if(data_type == 'xyz'):
                pose_file_names[action].append(name)  
                
    for action_idx in np.arange(action_number):
        action = actions[ action_idx ]
        segments_number = len(gaze_file_names[action])
        print("Reading dir {}, action {}, segments number {}".format(test_dir, action, segments_number))
        
        for seg_num in range(segments_number):
            gaze_data_path = path + gaze_file_names[action][seg_num]
            gaze_data = np.load(gaze_data_path)[1:, :]
            if gaze_data.shape[0] <= 30:
                continue
            pose_data_path = path + pose_file_names[action][seg_num]
            pose_data = np.load(pose_data_path)
            motion_data = pose_data[1:, :] - pose_data[:-1, :]
            motion_data = motion_data.reshape(-1, joint_number, 3)
            for i in range(motion_data.shape[0]):
                for j in range(motion_data.shape[1]):
                    direction = motion_data[i, j, :]
                    direction = [x / np.linalg.norm(direction) for x in direction]
                    motion_data[i, j, :] = direction
            for i in range(joint_number):
                joint_data = motion_data[:, i, :]
                gaze_motion[joint_names[i]].append(np.mean(np.sum(gaze_data*joint_data, 1)))
                
cosine_similarity = np.mean(gaze_motion['pelvis'])
print("Base cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['spine1'])
print("Pelvis cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['spine3'])
print("Torso cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['neck'])
print("Neck cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['head'])
print("Head cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_collar'])
print("L_Collar cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_collar'])
print("R_Collar cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_shoulder'])
print("L_Shoulder cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_shoulder'])
print("R_Shoulder cosine similarity: {:.2f}".format(cosine_similarity)) 
cosine_similarity = np.mean(gaze_motion['left_elbow'])
print("L_Elbow cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_elbow'])
print("R_Elbow cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_wrist'])
print("L_Wrist cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_wrist'])
print("R_Wrist cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_hip'])
print("L_Hip cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_hip'])
print("R_Hip cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_knee'])
print("L_Knee cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_knee'])
print("R_Knee cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_ankle'])
print("L_Ankle cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_ankle'])
print("R_Ankle cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['left_foot'])
print("L_Foot cosine similarity: {:.2f}".format(cosine_similarity))
cosine_similarity = np.mean(gaze_motion['right_foot'])
print("R_Foot cosine similarity: {:.2f}".format(cosine_similarity))

cosine_similarity = 0
for i in range(joint_number):
    cosine_similarity += np.mean(gaze_motion[joint_names[i]])
cosine_similarity = cosine_similarity/joint_number
print("Average cosine similarity: {:.2f}".format(cosine_similarity))