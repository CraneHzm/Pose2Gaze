import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "/scratch/hu/pose_forecast/adt_pose2gaze/"
test_dirs = ['train', 'test']

actions = ['decoration', 'meal', 'work']
action_number = len(actions)

# 30 fps
time_interval_start = -12 # -400 ms
time_interval_end = 30 # 1000 ms

joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'right_hip', 'right_knee', 'right_ankle','right_foot']
joint_number = len(joint_names)
gaze_motion_all = {}
for action_idx in np.arange(action_number):
    gaze_motion_all[actions[ action_idx ]] = []
        
for interval in range(time_interval_start, time_interval_end+1, 3):
    print("Time interval: {} ms".format(interval//3*100))
    gaze_motion = {}
    for action_idx in np.arange(action_number):
        gaze_motion[actions[ action_idx ]] = []
        
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
            action = name_split[2]                
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
                gaze_data = np.load(gaze_data_path)[:, :3]
                gaze_data = gaze_data[1:, :]
                if gaze_data.shape[0] <= 30:
                    continue
                pose_data_path = path + pose_file_names[action][seg_num]
                pose_data = np.load(pose_data_path)
                motion_data = pose_data[1:, :] - pose_data[:-1, :]
                motion_data = motion_data.reshape(-1, joint_number, 3)
                for i in range(motion_data.shape[0]):
                    for j in range(motion_data.shape[1]):
                        direction = motion_data[i, j, :]
                        if np.linalg.norm(direction) == 0:
                            direction = [0, 0, 1]
                        else:
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
time = np.arange(-400, 1001, 100)
plt.figure(figsize=(7, 5))
for action_idx in np.arange(action_number):
    plt.plot(time, gaze_motion_all[actions[ action_idx ]], label=actions[ action_idx ])
    max_motion = np.argmax(gaze_motion_all[actions[ action_idx ]])
    plt.scatter(time[max_motion], gaze_motion_all[actions[ action_idx ]][max_motion], s=15) 
    plt.plot([time[max_motion], time[max_motion]], [0, gaze_motion_all[actions[ action_idx ]][max_motion]], color='#1f77b4', linestyle='--', alpha = 0.7)

plt.xlabel("Time Interval (ms)", fontsize=22)
plt.ylabel("Cosine Similarity", fontsize=22)
plt.xticks(np.linspace(-400,1000,8),['-400','-200','0','200','400','600','800', '1000'],fontsize=20)
plt.yticks(np.arange(0, 0.7, 0.1), fontsize=20)
plt.axis([-400, 1000, 0, 0.62])
plt.grid(linestyle = '--', linewidth = 0.8, alpha = 0.5)
plt.legend(fontsize=18, ncol=1) 
plt.savefig('gaze_body_motion_adt.pdf',bbox_inches='tight',dpi=600,pad_inches=0.1)
plt.savefig('gaze_body_motion_adt.png',bbox_inches='tight',dpi=600,pad_inches=0.1)