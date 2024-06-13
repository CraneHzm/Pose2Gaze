from torch.utils.data import Dataset
import numpy as np
import os


class egobody_dataset(Dataset):

    def __init__(self, data_dir, seq_len, actions = 'all', joints_used = 'all', train_flag = 1, sample_rate=1):
        actions = self.define_actions(actions)
        self.sample_rate = sample_rate
        if train_flag == 1:
            data_dir = data_dir + 'train/'
        if train_flag == 0:
            data_dir = data_dir + 'test/'
            
        if joints_used == 'all':
            # names of all the 23 joints
            # ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw"]  
            self.joints_used = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        elif joints_used == 'torso':
            self.joints_used = np.array([0, 3, 6, 9, 12, 15, 22])
        elif joints_used == 'arm':
            self.joints_used = np.array([13, 14, 16, 17, 18, 19, 20, 21])
        elif joints_used == 'leg':
            self.joints_used = np.array([1, 2, 4, 5, 7, 8, 10, 11])
        elif joints_used == 'torso+arm':
            self.joints_used = np.array([0, 3, 6, 9, 12, 15, 22, 13, 14, 16, 17, 18, 19, 20, 21])
        elif joints_used == 'torso+leg':
            self.joints_used = np.array([0, 3, 6, 9, 12, 15, 22, 1, 2, 4, 5, 7, 8, 10, 11])
        elif joints_used == 'arm+leg':
            self.joints_used = np.array([13, 14, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 10, 11])
        else:
            self.joints_used = np.array(joints_used)  
            
        pose_head_gaze = self.load_data(data_dir, seq_len, actions)
        self.pose_head_gaze = pose_head_gaze
        
    def define_actions(self, action):
        """
        Define the list of actions we are using.

        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included.
        """
        
        # 'None' means no action label available
        actions = ['None', 'catch', 'chat', 'dance', 'discuss', 'learn', 'perform', 'teach']
        if action in actions:
            return [action]
            
        if action == "all":
            return actions
        raise( ValueError, "Unrecognised action: %d" % action )
        
    def load_data(self, data_dir, seq_len, actions):
        action_number = len(actions)
        pose_head_gaze = []  
        file_names = sorted(os.listdir(data_dir))
        pose_xyz_file_names = {}
        pose_interactee_file_names = {}
        head_file_names = {}
        gaze_file_names = {}
        for action_idx in np.arange(action_number):
            pose_xyz_file_names[actions[ action_idx ]] = []            
            pose_interactee_file_names[actions[ action_idx ]] = []            
            head_file_names[actions[ action_idx ]] = []
            gaze_file_names[actions[ action_idx ]] = []        
        for name in file_names:
            name_split = name.split('_')
            action = name_split[0]
            if action in actions:            
                data_type = name_split[-1][:-4]
                if(data_type == 'xyz'):
                    pose_xyz_file_names[action].append(name)
                if(data_type == 'interactee'):
                    pose_interactee_file_names[action].append(name)
                if(data_type == 'head'):
                    head_file_names[action].append(name)    
                if(data_type == 'gaze'):
                    gaze_file_names[action].append(name)
                    
        for action_idx in np.arange(action_number):
            action = actions[ action_idx ]
            segments_number = len(pose_xyz_file_names[action])
            print("Reading action {}, segments number {}".format(action, segments_number))
            for i in range(segments_number):
                pose_xyz_data_path = data_dir + pose_xyz_file_names[action][i]
                pose_xyz_data = np.load(pose_xyz_data_path)
                # use the given joints
                dim_used = np.sort(np.concatenate((self.joints_used * 3, self.joints_used * 3 + 1, self.joints_used * 3 + 2)))
                pose_xyz_data = pose_xyz_data[:, dim_used]

                pose_interactee_data_path = data_dir + pose_interactee_file_names[action][i]
                pose_interactee_data = np.load(pose_interactee_data_path)
                pose_interactee_data = pose_interactee_data[:, dim_used]
                
                head_data_path = data_dir + head_file_names[action][i]
                head_data = np.load(head_data_path)
                           
                gaze_data_path = data_dir + gaze_file_names[action][i]
                gaze_data = np.load(gaze_data_path)
                                            
                num_frames = pose_xyz_data.shape[0]
                if num_frames < seq_len:
                    continue
                    #raise( ValueError, "sequence length {} is larger than frame number {}".format(seq_len, num_frames))
                    
                pose_head_gaze_data = np.concatenate((pose_xyz_data, pose_interactee_data), axis=1)
                pose_head_gaze_data = np.concatenate((pose_head_gaze_data, head_data), axis=1)
                pose_head_gaze_data = np.concatenate((pose_head_gaze_data, gaze_data), axis=1)
                
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                #print(fs_sel)
                seq_sel = pose_head_gaze_data[fs_sel, :]
                seq_sel = seq_sel[0::self.sample_rate, :, :]
                #print(seq_sel.shape)
                if len(pose_head_gaze) == 0:
                    pose_head_gaze = seq_sel
                else:
                    pose_head_gaze = np.concatenate((pose_head_gaze, seq_sel), axis=0)
    
        return pose_head_gaze
        
  
    def __len__(self):
        return np.shape(self.pose_head_gaze)[0]

    def __getitem__(self, item):
        return self.pose_head_gaze[item]
        
        
if __name__ == "__main__":
    data_dir = "/scratch/hu/pose_forecast/egobody_pose2gaze/"
    seq_len = 30
    joints_used = 'all'
    
    actions = 'all'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))
    
    actions = 'None'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))

    actions = 'catch'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))
    
    actions = 'chat'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))
    
    actions = 'dance'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))
    
    actions = 'discuss'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))    
    actions = 'learn'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))    
    actions = 'perform'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))

    actions = 'teach'
    train_dataset = egobody_dataset(data_dir, seq_len, actions, joints_used, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_head_gaze.shape))