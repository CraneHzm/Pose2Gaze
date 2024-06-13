import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "/scratch/hu/pose_forecast/mogaze_pose2gaze/"
actions = ['pick']
#actions = ['pick', 'place']
action_number = len(actions)
test_id = 0
# 0 means using all the data
if test_id == 0:
    subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']

gaze_head_cosine_similarity = []
gaze_neck_cosine_similarity = []
gaze_torso_cosine_similarity = []
gaze_pelvis_cosine_similarity = []
gaze_base_cosine_similarity = []

for subj in subjects:
    path = data_dir + "/" + subj + "/"
    file_names = sorted(os.listdir(path))  
    gaze_file_names = {}
    head_file_names = {}
    neck_file_names = {}
    torso_file_names = {}
    pelvis_file_names = {}
    base_file_names = {}
    
    for action_idx in np.arange(action_number):              
        gaze_file_names[actions[ action_idx ]] = []
        head_file_names[actions[ action_idx ]] = []
        neck_file_names[actions[ action_idx ]] = []
        torso_file_names[actions[ action_idx ]] = []
        pelvis_file_names[actions[ action_idx ]] = []
        base_file_names[actions[ action_idx ]] = []
        
    for name in file_names:
        name_split = name.split('_')
        action = name_split[0]                
        if action in actions:                
            data_type = name_split[-1][:-4]
            if(data_type == 'gaze'):
                gaze_file_names[action].append(name)
            if(data_type == 'head'):
                head_file_names[action].append(name)
            if(data_type == 'neck'):
                neck_file_names[action].append(name)
            if(data_type == 'torso'):
                torso_file_names[action].append(name)
            if(data_type == 'pelvis'):
                pelvis_file_names[action].append(name)
            if(data_type == 'base'):
                base_file_names[action].append(name)                        
                
    for action_idx in np.arange(action_number):
        action = actions[ action_idx ]
        segments_number = len(gaze_file_names[action])
        print("Reading subject {}, action {}, segments number {}".format(subj, action, segments_number))
        
        for seg_num in range(segments_number):
            gaze_data_path = path + gaze_file_names[action][seg_num]
            gaze_data = np.load(gaze_data_path)
            if gaze_data.shape[0] <= 30:
                continue
            head_data_path = path + head_file_names[action][seg_num]
            head_data = np.load(head_data_path)            
            neck_data_path = path + neck_file_names[action][seg_num]
            neck_data = np.load(neck_data_path)            
            torso_data_path = path + torso_file_names[action][seg_num]
            torso_data = np.load(torso_data_path)            
            pelvis_data_path = path + pelvis_file_names[action][seg_num]
            pelvis_data = np.load(pelvis_data_path)            
            base_data_path = path + base_file_names[action][seg_num]
            base_data = np.load(base_data_path)
            gaze_head_cosine_similarity.append(np.mean(np.sum(gaze_data*head_data, 1)))
            gaze_neck_cosine_similarity.append(np.mean(np.sum(gaze_data*neck_data, 1)))
            gaze_torso_cosine_similarity.append(np.mean(np.sum(gaze_data*torso_data, 1)))
            gaze_pelvis_cosine_similarity.append(np.mean(np.sum(gaze_data*pelvis_data, 1)))
            gaze_base_cosine_similarity.append(np.mean(np.sum(gaze_data*base_data, 1)))            
               
print("Gaze-Head cosine similarity: {:.2f}".format(np.mean(gaze_head_cosine_similarity)))
print("Gaze-Neck cosine similarity: {:.2f}".format(np.mean(gaze_neck_cosine_similarity)))
print("Gaze-Torso cosine similarity: {:.2f}".format(np.mean(gaze_torso_cosine_similarity)))
print("Gaze-Pelvis cosine similarity: {:.2f}".format(np.mean(gaze_pelvis_cosine_similarity)))
print("Gaze-Base cosine similarity: {:.2f}".format(np.mean(gaze_base_cosine_similarity)))