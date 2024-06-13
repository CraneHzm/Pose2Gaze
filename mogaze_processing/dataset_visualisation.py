# visualise data in the MoGaze dataset
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
        # names of all the 21 joints
        self._joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',  
                             'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder', 
                             'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle', 
                             'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe', 'gaze','head_direction']                                                                                   
        self._joint_ids = {name: idx for idx, name in enumerate(self._joint_names)}
                                                 
       # parent of every joint
        self._joint_parent_names = {
                                      # root
                                      'base':           'base',
                                      'pelvis':         'base',
                                      'torso':          'pelvis', 
                                      'neck':           'torso', 
                                      'head':           'neck', 
                                      'linnerShoulder': 'torso',
                                      'lShoulder':      'linnerShoulder', 
                                      'lElbow':         'lShoulder', 
                                      'lWrist':         'lElbow', 
                                      'rinnerShoulder': 'torso', 
                                      'rShoulder':      'rinnerShoulder', 
                                      'rElbow':         'rShoulder', 
                                      'rWrist':         'rElbow', 
                                      'lHip':           'base', 
                                      'lKnee':          'lHip', 
                                      'lAnkle':         'lKnee', 
                                      'lToe':           'lAnkle', 
                                      'rHip':           'base', 
                                      'rKnee':          'rHip', 
                                      'rAnkle':         'rKnee', 
                                      'rToe':           'rAnkle',
                                      'gaze':           'head',
                                      'head_direction':           'head'}                               
        # id of joint parent
        self._joint_parent_ids = [self._joint_ids[self._joint_parent_names[child_name]] for child_name in self._joint_names]
        # print(self._joint_parent_ids)
        # the links that we want to show
        # a link = child -> parent
        self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # colors: 0 for middle, 1 for left, 2 for right
        self._link_colors = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4]
       
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
                color = "#a64d79"        
            if self._link_colors[i] == 4:
                color = "#6aa84f"                                    
            self._plots.append(self._ax.plot([0, 0], [0, 0], [0, 0], lw=2, c=color))
            
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")
              
    # play the sequence of human pose in xyz representations
    def play_xyz(self, pose_xyz, gaze, head_direction):      
        # gaze position = head position + gaze direction
        gaze_pos = pose_xyz[:, 4*3:5*3] + gaze    
        head_pos = pose_xyz[:, 4*3:5*3] + head_direction
        pose_xyz = np.concatenate((pose_xyz, gaze_pos), axis = 1)
        pose_xyz = np.concatenate((pose_xyz, head_pos), axis = 1)
        for i in range(pose_xyz.shape[0]):       
            joint_number = len(self._joint_names)        
            pose_xyz_tmp = pose_xyz[i].reshape(joint_number, 3)                        
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
                                      
            r = 0.5
            x_root, y_root, z_root = pose_xyz_tmp[0,0], pose_xyz_tmp[0,1], pose_xyz_tmp[0,2]
            self._ax.set_xlim3d([-r + x_root, r + x_root])
            self._ax.set_ylim3d([-r + y_root, r + y_root])
            self._ax.set_zlim3d([-r + z_root, r + z_root])            
            self._ax.set_aspect('auto')          
            plt.show(block=False)
            self._fig.canvas.draw()
            past_time = f"{i/self._fps:.1f}"
            plt.title(f"Time: {past_time} s", fontsize=15)
            plt.pause(0.00001)

            
if __name__ == "__main__":
    data_path = "/scratch/hu/pose_forecast/mogaze_pose2gaze/p1_1/pick_001_"
    pose_euler_data_path = data_path + "pose_euler.npy"
    pose_xyz_data_path = data_path + "pose_xyz.npy"
    gaze_data_path = data_path + "gaze.npy"
    head_data_path = data_path + "head.npy"   
    objects_data_path = data_path + "objects.npy"    
        
    # pose_euler data has 66 dimensions, corresponding to base translation position (3) + base rotation (3) + rotation of 20 joints (20*3)
    pose_euler = np.load(pose_euler_data_path)    
    print("Human euler pose shape: {}".format(pose_euler.shape))                  
    
    # pose_xyz data has 63 dimensions, corresponding to the xyz positions of 21 joints
    pose_xyz = np.load(pose_xyz_data_path)
    print("Human xyz pose shape: {}".format(pose_xyz.shape))          
    
    # gaze data has 3 dimensions, corresponding to the gaze direction/vector (3) in the world coordinate system
    gaze = np.load(gaze_data_path)
    print("Eye gaze shape: {}".format(gaze.shape))

    # head data has 3 dimensions, corresponding to the head direction/vector (3) in the world coordinate system
    head = np.load(head_data_path)
    print("Head shape: {}".format(head.shape))
        
    # objects data contains 16 tracked objects in the scene
    # each object data has 7 dimensions: ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "rot_w"]
    objects = np.load(objects_data_path, allow_pickle = True).item()
    for key in objects.keys():
        print("Object name: {}, shape {}".format(key, objects[key].shape))
        
    player = Player_Skeleton()
    player.play_xyz(pose_xyz, gaze, head)