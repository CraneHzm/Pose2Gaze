import numpy as np
import os
import matplotlib.pyplot as plt


data_dir = "/scratch/hu/pose_forecast/adt_pose2gaze/"
test_dirs = ['train', 'test']
actions = ['decoration', 'meal', 'work']
action_number = len(actions)

# 30 fps
time_interval_start = -12 # -400 ms
time_interval_end = 24 # 800 ms

gaze_head_all = {}
for action_idx in np.arange(action_number):              
    gaze_head_all[actions[ action_idx ]] = []    
       
for interval in range(time_interval_start, time_interval_end+1, 3):
    gaze_head = {}
    for action_idx in np.arange(action_number):              
        gaze_head[actions[ action_idx ]] = []    
    
    for test_dir in test_dirs:
        path = data_dir + "/" + test_dir + "/"
        file_names = sorted(os.listdir(path))  
        gaze_file_names = {}
        head_file_names = {}
        
        for action_idx in np.arange(action_number):              
            gaze_file_names[actions[ action_idx ]] = []
            head_file_names[actions[ action_idx ]] = []
            
        for name in file_names:
            name_split = name.split('_')
            action = name_split[2]
            if action in actions:                
                data_type = name_split[-1][:-4]
                if(data_type == 'gaze'):
                    gaze_file_names[action].append(name)
                if(data_type == 'head'):
                    head_file_names[action].append(name)
                    
        for action_idx in np.arange(action_number):
            action = actions[ action_idx ]
            segments_number = len(gaze_file_names[action])
            print("Reading dir {}, action {}, segments number {}".format(test_dir, action, segments_number))
            
            for seg_num in range(segments_number):
                gaze_data_path = path + gaze_file_names[action][seg_num]
                gaze_data = np.load(gaze_data_path)[:, :3] # only use gaze direction
                if gaze_data.shape[0] <= 30:
                    continue
                head_data_path = path + head_file_names[action][seg_num]
                head_data = np.load(head_data_path)            
                
                if interval < 0:
                    gaze_head[actions[ action_idx ]].append(np.mean(np.sum(gaze_data[-interval:, :]*head_data[:interval, :], 1)))
                elif interval == 0:
                    gaze_head[actions[ action_idx ]].append(np.mean(np.sum(gaze_data*head_data, 1)))
                else:
                    gaze_head[actions[ action_idx ]].append(np.mean(np.sum(gaze_data[:-interval, :]*head_data[interval:, :], 1)))
    
    for action_idx in np.arange(action_number):              
        gaze_head_all[actions[ action_idx ]].append(np.mean(gaze_head[actions[ action_idx ]]))
        
# Visualisation
time = np.arange(-400, 801, 100)
plt.figure(figsize=(7, 5))
for action_idx in np.arange(action_number):
    plt.plot(time, gaze_head_all[actions[ action_idx ]], label=actions[ action_idx ])
    max_head = np.argmax(gaze_head_all[actions[ action_idx ]])
    plt.scatter(time[max_head], gaze_head_all[actions[ action_idx ]][max_head], s=15) 
    plt.plot([time[max_head], time[max_head]], [0, gaze_head_all[actions[ action_idx ]][max_head]], color='#1f77b4', linestyle='--', alpha = 0.7)

plt.xlabel("Time Interval (ms)", fontsize=22)
plt.ylabel("Cosine Similarity", fontsize=22)
plt.xticks(np.linspace(-400,800,7),['-400','-200','0','200','400','600','800'],fontsize=20)
plt.yticks(np.arange(0.5, 1.1, 0.1), fontsize=20)
plt.axis([-400, 800, 0.5, 1])
plt.grid(linestyle = '--', linewidth = 0.8, alpha = 0.5)
plt.legend(fontsize=18, ncol=1)
plt.savefig('gaze_head_adt.pdf',bbox_inches='tight',dpi=600,pad_inches=0.1)
plt.savefig('gaze_head_adt.png',bbox_inches='tight',dpi=600,pad_inches=0.1)
#plt.show()