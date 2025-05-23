import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
actions = ['catch', 'chat', 'dance', 'discuss', 'learn', 'perform', 'teach']
action_number = len(actions)
test_dirs = ['train', 'test']

# 30 fps
time_interval_start = -45 # -1500 ms
time_interval_end = 45 # 1500 ms

joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw"]

joint_number = len(joint_names)
gaze_motion_all = {}
for action_idx in np.arange(action_number):
    gaze_motion_all[actions[ action_idx ]] = []
    
for interval in range(time_interval_start, time_interval_end+1, 15):
    print("Time interval: {} ms".format(interval//3*100))
    gaze_motion = {}
    for action_idx in np.arange(action_number):
        gaze_motion[actions[ action_idx ]] = []
        
    for test_dir in test_dirs:
        path = data_dir + "/" + test_dir + "/"
        file_names = sorted(os.listdir(path))  
        gaze_file_names = {}
        pose_file_names = {}
        pose_interactee_file_names = {}
        
        for action_idx in np.arange(action_number):              
            gaze_file_names[actions[ action_idx ]] = []
            pose_file_names[actions[ action_idx ]] = []
            pose_interactee_file_names[actions[ action_idx ]] = []
            
        for name in file_names:
            name_split = name.split('_')
            action = name_split[0]                
            if action in actions:                
                data_type = name_split[-1][:-4]
                if(data_type == 'gaze'):
                    gaze_file_names[action].append(name)
                if(data_type == 'xyz'):
                    pose_file_names[action].append(name)
                if(data_type == 'interactee'):
                    pose_interactee_file_names[action].append(name)               
                    
                    
        for action_idx in np.arange(action_number):
            action = actions[ action_idx ]
            segments_number = len(gaze_file_names[action])
            print("Reading dir {}, action {}, segments number {}".format(test_dir, action, segments_number))
            
            for seg_num in range(segments_number):
                gaze_data_path = path + gaze_file_names[action][seg_num]
                gaze_data = np.load(gaze_data_path)
                if gaze_data.shape[0] <= 30:
                    continue
                pose_data_path = path + pose_file_names[action][seg_num]
                pose_data = np.load(pose_data_path)
                pose_interactee_data_path = path + pose_interactee_file_names[action][seg_num]
                pose_interactee_data = np.load(pose_interactee_data_path)
                motion_data = pose_interactee_data - pose_data
                motion_data = motion_data.reshape(-1, joint_number, 3)
                
                for i in range(motion_data.shape[0]):
                    for j in range(motion_data.shape[1]):
                        direction = motion_data[i, j, :]
                        direction = [x / np.linalg.norm(direction) for x in direction]
                        motion_data[i, j, :] = direction
                        
                if interval < 0:
                    cosine_similarity = 0
                    for i in range(joint_number):
                        joint_data = motion_data[:, i, :]
                        cosine_similarity += np.mean(np.sum(gaze_data[-interval:, :]*joint_data[:interval, :], 1))
                    cosine_similarity = cosine_similarity/joint_number
                    gaze_motion[actions[ action_idx ]].append(cosine_similarity)
                elif interval == 0:
                    cosine_similarity = 0
                    for i in range(joint_number):
                        joint_data = motion_data[:, i, :]
                        cosine_similarity += np.mean(np.sum(gaze_data*joint_data, 1))
                    cosine_similarity = cosine_similarity/joint_number
                    gaze_motion[actions[ action_idx ]].append(cosine_similarity)
                else:
                    cosine_similarity = 0
                    for i in range(joint_number):
                        joint_data = motion_data[:, i, :]
                        cosine_similarity += np.mean(np.sum(gaze_data[:-interval, :]*joint_data[interval:, :], 1))
                    cosine_similarity = cosine_similarity/joint_number
                    gaze_motion[actions[ action_idx ]].append(cosine_similarity)

    for action_idx in np.arange(action_number):              
        gaze_motion_all[actions[ action_idx ]].append(np.mean(gaze_motion[actions[ action_idx ]]))
        
# Visualization
time = np.arange(-1500, 1501, 500)
plt.figure(figsize=(7, 5))
for action_idx in np.arange(action_number):
    plt.plot(time, gaze_motion_all[actions[ action_idx ]], label=actions[ action_idx ])
    max_motion = np.argmax(gaze_motion_all[actions[ action_idx ]])
    plt.scatter(time[max_motion], gaze_motion_all[actions[ action_idx ]][max_motion], s=15) 
    plt.plot([time[max_motion], time[max_motion]], [0, gaze_motion_all[actions[ action_idx ]][max_motion]], color='#1f77b4', linestyle='--', alpha = 0.7)

plt.xlabel("Time Interval (ms)", fontsize=22)
plt.ylabel("Cosine Similarity", fontsize=22)
plt.xticks(np.linspace(-1500,1500,7),['-1500','-1000','-500', '0', '500', '1000', '1500'],fontsize=20)
plt.yticks(np.arange(0.5, 1.1, 0.1), fontsize=20)
plt.axis([-1500, 1500, 0.5, 1])
plt.grid(linestyle = '--', linewidth = 0.8, alpha = 0.5)
plt.legend(fontsize=18, ncol=2)
plt.savefig('gaze_two_body_motion_egobody.pdf',bbox_inches='tight',dpi=600,pad_inches=0.1)
plt.savefig('gaze_two_body_motion_egobody.png',bbox_inches='tight',dpi=600,pad_inches=0.1)