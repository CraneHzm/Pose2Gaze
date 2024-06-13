# visualise data in the EgoBody dataset

import numpy as np
import matplotlib.pyplot as plt

# play human pose using a skeleton
class Player_Skeleton:
    def __init__(self, fps=30.0):
        """ init function
        
        Keyword arguments:
        fps -- frames per second of the data (default 30)
        """
        
        self._fps = fps
        # # names of all the 23 joints + gaze + head direction
        # self._joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
        #                      "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        #                      "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        #                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "gaze", "head_direction"]   
        # # names of all the 23 joints
        # self._joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
        #                      "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        #                      "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        #                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw"]  
        # names of all the 23 joints on two pose
        self._joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", 
                             "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
                             "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
                             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "gaze", "head_direction",

                             "pelvis1", "left_hip1", "right_hip1", "spine11", "left_knee1", "right_knee1", 
                             "spine21", "left_ankle1", "right_ankle1", "spine31", "left_foot1", "right_foot1",
                             "neck1", "left_collar1", "right_collar1", "head1", "left_shoulder1", "right_shoulder1",
                             "left_elbow1", "right_elbow1", "left_wrist1", "right_wrist1", "jaw1"]  
                                                                                         
        self._joint_ids = {name: idx for idx, name in enumerate(self._joint_names)}
                            
    #    # parent of every joint
    #     self._joint_parent_names = {
    #                                   # root
    #                                   'pelvis':         'pelvis',
    #                                   "left_hip":       'pelvis',
    #                                   "right_hip":      'pelvis',
    #                                   "spine1":         'pelvis',
    #                                   "left_knee":      "left_hip",
    #                                   "right_knee":     "right_hip",
    #                                   "spine2":         "spine1",
    #                                   "left_ankle":     "left_knee",
    #                                   "right_ankle":    "right_knee",
    #                                   "spine3":         "spine2",
    #                                   "left_foot":      "left_ankle",       
    #                                   "right_foot":     "right_ankle",
    #                                   "neck":           "spine3",
    #                                   "left_collar":    "spine3",
    #                                   "right_collar":   "spine3", 
    #                                   "head":           "neck",
    #                                   "left_shoulder":  "left_collar",
    #                                   "right_shoulder": "right_collar",
    #                                   "left_elbow":     "left_shoulder",    
    #                                   "right_elbow":    "right_shoulder",   
    #                                   "left_wrist":     "left_elbow",
    #                                   "right_wrist":    "right_elbow",
    #                                   "jaw":            'head',
    #                                   'gaze':           'head',
    #                                   'head_direction':      'head'}
        
        # parent of every joint
        self._joint_parent_names = {
                                      # root
                                      'pelvis':         'pelvis',
                                      "left_hip":       'pelvis',
                                      "right_hip":      'pelvis',
                                      "spine1":         'pelvis',
                                      "left_knee":      "left_hip",
                                      "right_knee":     "right_hip",
                                      "spine2":         "spine1",
                                      "left_ankle":     "left_knee",
                                      "right_ankle":    "right_knee",
                                      "spine3":         "spine2",
                                      "left_foot":      "left_ankle",       
                                      "right_foot":     "right_ankle",
                                      "neck":           "spine3",
                                      "left_collar":    "spine3",
                                      "right_collar":   "spine3", 
                                      "head":           "neck",
                                      "left_shoulder":  "left_collar",
                                      "right_shoulder": "right_collar",
                                      "left_elbow":     "left_shoulder",    
                                      "right_elbow":    "right_shoulder",   
                                      "left_wrist":     "left_elbow",
                                      "right_wrist":    "right_elbow",
                                      "jaw":            'head',
                                      'gaze':           'head',
                                      'head_direction':      'head',
                                      
                                      'pelvis1':         'pelvis1',
                                      "left_hip1":       'pelvis1',
                                      "right_hip1":      'pelvis1',
                                      "spine11":         'pelvis1',
                                      "left_knee1":      "left_hip1",
                                      "right_knee1":     "right_hip1",
                                      "spine21":         "spine11",
                                      "left_ankle1":     "left_knee1",
                                      "right_ankle1":    "right_knee1",
                                      "spine31":         "spine21",
                                      "left_foot1":      "left_ankle1",       
                                      "right_foot1":     "right_ankle1",
                                      "neck1":           "spine31",
                                      "left_collar1":    "spine31",
                                      "right_collar1":   "spine31", 
                                      "head1":           "neck1",
                                      "left_shoulder1":  "left_collar1",
                                      "right_shoulder1": "right_collar1",
                                      "left_elbow1":     "left_shoulder1",    
                                      "right_elbow1":    "right_shoulder1",   
                                      "left_wrist1":     "left_elbow1",
                                      "right_wrist1":    "right_elbow1",
                                      "jaw1":            'head1'}
        
        # id of joint parent
        self._joint_parent_ids = [self._joint_ids[self._joint_parent_names[child_name]] for child_name in self._joint_names]
        # print(self._joint_parent_ids)
        # the links that we want to show
        # a link = child -> parent
        # self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
        # colors: 0 for middle, 1 for left, 2 for right
        self._link_colors = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 3, 4, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0]
        # self._link_colors = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0]
        # self._link_colors = [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0]
       
        self._fig = plt.figure()
        self._ax = plt.gca(projection='3d')
        self._plots = []
        for i in range(len(self._joint_links)):
            if self._link_colors[i] == 0:
                color = "#f9cb9c"
            if self._link_colors[i] == 1:
                color = "#3498db"                
            if self._link_colors[i] == 2:
                color = "#e74c3c"
            if self._link_colors[i] == 3:
                color = "#6aa84f"
            if self._link_colors[i] == 4:
                color = "#a64d79"                                                    
            self._plots.append(self._ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color))
            
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")
              

    # play the sequence of human pose in xyz representations on both poses
    def play_xyz(self, pose_xyz, gaze, head, pose2_xyz):
    # def play_xyz(self, pose_xyz, pose2_xyz):
        gaze_direction = pose_xyz[:, 15*3:16*3] + gaze
        head_direction = pose_xyz[:, 15*3:16*3] + head
        pose_xyz = np.concatenate((pose_xyz, gaze_direction), axis = 1)
        pose_xyz = np.concatenate((pose_xyz, head_direction), axis = 1)

        pose_xyz = np.concatenate((pose_xyz, pose2_xyz), axis = 1)
        for i in range(pose_xyz.shape[0]):       
            joint_number = len(self._joint_names)        
            pose_xyz_tmp = pose_xyz[i].reshape(joint_number, 3)
            # swap y and z axis in the original data
            pose_xyz_new = pose_xyz_tmp.copy()
            pose_xyz_new[:, 1] = pose_xyz_tmp[:, 2]
            pose_xyz_new[:, 2] = pose_xyz_tmp[:, 1]
            pose_xyz_tmp = pose_xyz_new
            for j in range(len(self._joint_links)):
                idx = self._joint_links[j]
                start_point = pose_xyz_tmp[idx]
                end_point = pose_xyz_tmp[self._joint_parent_ids[idx]]                               
                x = np.array([start_point[0], end_point[0]])
                y = np.array([start_point[1], end_point[1]])
                z = np.array([start_point[2], end_point[2]])
                self._plots[j][0].set_xdata(x)
                self._plots[j][0].set_ydata(y)                       
                self._plots[j][0].set_3d_properties(z)
                                      
            r = 2
            x_root, y_root, z_root = pose_xyz_tmp[0,0], pose_xyz_tmp[0,1], pose_xyz_tmp[0,2]
            self._ax.set_xlim3d([-r + x_root, r + x_root])
            self._ax.set_ylim3d([-r + y_root, r + y_root])
            self._ax.set_zlim3d([-r + z_root+1, r + z_root-1])
            self._ax.set_aspect('auto')          
            plt.show(block=False)
            self._fig.canvas.draw()
            past_time = f"{i/self._fps:.1f}"
            plt.title(f"Time: {past_time} s", fontsize=15)
            plt.pause(0.01)

            
if __name__ == "__main__":
    data_path = '/scratch/hu/pose_forecast/egobody_pose2gaze/'
    data_path = data_path + "/train/catch_cnb_dlab_0215_recording_20220215_S22_S21_01_1541_"         
    pose_xyz_data_path = data_path + "pose_xyz.npy"
    gaze_data_path = data_path + "gaze.npy"
    head_data_path = data_path + "head.npy"  
    pose_interactee_xyz_data_path = data_path + "pose_interactee.npy"

    # pose_xyz data has 69 dimensions, corresponding to the xyz positions of 23 joints
    pose_xyz = np.load(pose_xyz_data_path)
    print("Human xyz pose shape: {}".format(pose_xyz.shape))

    pose_interactee_xyz = np.load(pose_interactee_xyz_data_path)
    print("Human xyz pose 2 shape: {}".format(pose_interactee_xyz.shape))
    
    # gaze data has 3 dimensions, corresponding to the gaze direction (3)
    gaze = np.load(gaze_data_path)
    print("Eye gaze shape: {}".format(gaze.shape))

    # head data has 3 dimensions, corresponding to the head direction (3)
    head = np.load(head_data_path)
    print("Head shape: {}".format(head.shape))
    
    player = Player_Skeleton()
    player.play_xyz(pose_xyz, gaze, head, pose_interactee_xyz)   
    # player.play_xyz(pose_xyz, pose2_xyz)