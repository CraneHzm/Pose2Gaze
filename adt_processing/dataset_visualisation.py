# visualise data in the ADT dataset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# play human pose using a skeleton
class Player_Skeleton:
    def __init__(self, fps=30.0, show_images = False):
        """ init function

        Keyword arguments:
        fps -- frames per second of the data (default 30)
        """
        
        self._fps = fps
        self.show_images = show_images            
        # names of all the 21 joints
        self._joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',
                             'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder',
                             'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle',
                             'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe', 'gaze', 'head_direction']
              
        self._joint_ids = {name: idx for idx, name in enumerate(self._joint_names)}

        # parent of every joint
        self._joint_parent_names = {
            # root
            'base': 'base',
            'pelvis': 'base',
            'torso': 'pelvis',
            'neck': 'torso',
            'head': 'neck',
            'linnerShoulder': 'torso',
            'lShoulder': 'linnerShoulder',
            'lElbow': 'lShoulder',
            'lWrist': 'lElbow',
            'rinnerShoulder': 'torso',
            'rShoulder': 'rinnerShoulder',
            'rElbow': 'rShoulder',
            'rWrist': 'rElbow',
            'lHip': 'base',
            'lKnee': 'lHip',
            'lAnkle': 'lKnee',
            'lToe': 'lAnkle',
            'rHip': 'base',
            'rKnee': 'rHip',
            'rAnkle': 'rKnee',
            'rToe': 'rAnkle',
            'gaze': 'head',
            'head_direction': 'head',}

        # id of joint parent
        self._joint_parent_ids = [self._joint_ids[self._joint_parent_names[child_name]] for child_name in self._joint_names]
        # print(self._joint_parent_ids)
        # the links that we want to show
        # a link = child -> parent        
        self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]        
        # colors: 0 for middle, 1 for left, 2 for right
        self._link_colors = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4]

        if show_images:
            self._fig = plt.figure(figsize=(10, 5))
            # self._ax = self._fig.add_axes(Axes3D(self._fig))
            self._ax = self._fig.add_subplot(1, 2, 1, projection='3d')
            self._ax1 = self._fig.add_subplot(1, 2, 2)
        else:
            self._fig = plt.figure()
            self._ax = plt.gca(projection='3d')
            
        self._plots = []
        for i in range(len(self._joint_links)):
            if self._link_colors[i] == 0:
                color = "#3498db"
            if self._link_colors[i] == 1:
                color = "#3498db"
            if self._link_colors[i] == 2:
                color = "#3498db"
            if self._link_colors[i] == 3:
                color = "#6aa84f"
            if self._link_colors[i] == 4:
                color = "#a64d79"                
            self._plots.append(self._ax.plot([0, 0], [0, 0], [0, 0], lw=2.0, c=color))

        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")

    # play the sequence of human pose in xyz representations
    def play_xyz(self, path, pose_xyz, gaze, head):
        gaze_direction = pose_xyz[:, 4*3:5*3] + gaze[:, :3]*0.5
        head_direction = pose_xyz[:, 4*3:5*3] + head[:, :3]*0.5
        pose_xyz = np.concatenate((pose_xyz, gaze_direction), axis = 1)        
        pose_xyz = np.concatenate((pose_xyz, head_direction), axis = 1)
        useful_frames = gaze[:, 5]
        
        for i in range(pose_xyz.shape[0]):        
            joint_number = len(self._joint_names)
            pose_xyz_tmp = pose_xyz[i].reshape(joint_number, 3)                    
            for j in range(len(self._joint_links)):
                idx = self._joint_links[j]
                start_point = pose_xyz_tmp[idx]
                end_point = pose_xyz_tmp[self._joint_parent_ids[idx]]
                x = np.array([start_point[0], end_point[0]])
                y = np.array([start_point[2], end_point[2]])
                z = np.array([start_point[1], end_point[1]])
                self._plots[j][0].set_xdata(x)
                self._plots[j][0].set_ydata(y)
                self._plots[j][0].set_3d_properties(z)
                        
            if self.show_images:
                frame = int(useful_frames[i])
                if i % 30 == 0:
                    image_path = path + 'images/' + str(frame) + '.png'
                    img = Image.open(image_path)
                    img_width = 640
                    img_height = 640
                    img = img.resize((img_width, img_height))
                    
                    self._ax1.clear()
                    self._ax1.imshow(img)
                    self._ax1.grid(False)
                    self._ax1.axis('off')
                    
                    gaze_x = gaze[i, 3] * img_width
                    gaze_y = gaze[i, 4] * img_height
                    circle = patches.Circle([gaze_x, gaze_y], radius=20.0, facecolor="red")
                    self._ax1.add_patch(circle)
                                              
            r = 1.0
            x_root, y_root, z_root = pose_xyz_tmp[0, 0], pose_xyz_tmp[0, 2], pose_xyz_tmp[0, 1]
            self._ax.set_xlim3d([-r + x_root, r + x_root])
            self._ax.set_ylim3d([-r + y_root, r + y_root])
            self._ax.set_zlim3d([-r + z_root, r + z_root])
            #self._ax.view_init(elev=30, azim=-110)

            self._ax.grid(False)
            #self._ax.axis('off')
            
            self._ax.set_aspect('auto')
            plt.show(block=False)
            self._fig.canvas.draw()
            past_time = f"{i / self._fps:.1f}"
            plt.title(f"Time: {past_time} s", fontsize=15)
            plt.pause(0.000000001)

            
if __name__ == "__main__":
    data_path = '/scratch/hu/pose_forecast/adt_pose2gaze/train/Apartment_release_work_skeleton_seq106_'
    
    gaze_path = data_path + 'gaze.npy'
    head_path = data_path + 'head.npy'
    pose_path = data_path + 'pose_xyz.npy'
    show_images = False # display the ego-centric images
    if show_images:
        from PIL import Image
        import matplotlib.patches as patches
    
    # gaze direction (3) + 2D gaze positions (2) + frame id (1)
    gaze = np.load(gaze_path)
    print("Gaze shape: {}".format(gaze.shape))
    
    # head direction (3)
    head = np.load(head_path)
    print("Head shape: {}".format(head.shape))
    
    # joint positions (21*3)
    pose = np.load(pose_path)
    print("Pose shape: {}".format(pose.shape))

    player = Player_Skeleton(show_images = show_images)
    player.play_xyz(data_path, pose, gaze, head)