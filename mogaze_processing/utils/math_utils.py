import numpy as np
import math
from scipy.spatial.transform import Rotation as R


def quaternion_matrix(quaternion):
    """Return rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    # We assum the ROS convention (x, y, z, w)
    quaternion_tmp = np.array([0.0] * 4)
    quaternion_tmp[1] = quaternion[0]  # x
    quaternion_tmp[2] = quaternion[1]  # y
    quaternion_tmp[3] = quaternion[2]  # z
    quaternion_tmp[0] = quaternion[3]  # w
    q = np.array(quaternion_tmp, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])
        
        
def euler2xyz(pose_euler):
    """
    Convert the human pose in MoGaze dataset from euler representation to xyz representation.
    """     
    # names of all the 21 joints
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head', 'linnerShoulder',  
                         'lShoulder', 'lElbow', 'lWrist', 'rinnerShoulder', 'rShoulder', 
                         'rElbow', 'rWrist', 'lHip', 'lKnee', 'lAnkle', 
                         'lToe', 'rHip', 'rKnee', 'rAnkle', 'rToe']                                                                                   
    joint_ids = {name: idx for idx, name in enumerate(joint_names)}
                         
    # translation of the 20 joints (excluding base)                                    
    joint_trans = np.array([ [3.477253617581374e-11, 9.836376779497708e-11, 0.07405368236962184], 
                          [-1.0047164753054265e-05, -3.0239514916602615e-05, 0.20083891659362765],
                          [-1.2600904410529728e-10, -1.7764181211437644e-09, 0.23359122887702816], 
                          [3.1869674957629284e-09, -0.018057959551160224, 0.13994911705542687],
                          [0.036502066101529215, 0.00031381713305931404, 0.1832770110533476], 
                          [0.15280389623890986, -1.1382671917429665e-09, -8.843481859722201e-10],
                          [0.24337935578849854, 3.1846829220430116e-09, 1.4634072008156868e-08], 
                          [0.2587863246939117, -0.001083441187206969, -2.579650873773653e-05],
                          [-0.035725060316154335, 0.00031381736864692205, 0.18327701132224603], 
                          [-0.15280389611921505, 1.5047363672598406e-10, -3.5061608640836945e-10],
                          [-0.24337935345513284, 1.7367526549061563e-09, 8.129849550012404e-09], 
                          [-0.26714046782640866, -0.0018438929787374019, 0.0002976280123487421],
                          [0.09030938177511867, -3.9169896274006325e-10, 6.535337361293777e-11], 
                          [2.3363012617863798e-09, -6.968930551857006e-09, -0.3833567539280885],
                          [-0.00022773850984784621, -0.0010201879980305337, -0.35354742240551973], 
                          [-8.62756280059957e-10, -0.13546407147725723, -0.05870108800865852],
                          [-0.09030938169548784, 3.1646243068853757e-10, -6.760126625141776e-11], 
                          [-2.5236199070228006e-09, -1.1574579905441608e-09, -0.38335675152180304],
                          [0.0005504841925316687, 0.0004893600191017677, -0.3414929392712298], 
                          [-1.2153921360447088e-09, -0.13546407016650086, -0.058701085864675616]])
         

    # parent of every joint
    joint_parent_names = {
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
                                  'rToe':           'rAnkle'}                               
    # id of joint parent
    joint_parent_ids = [joint_ids[joint_parent_names[child_name]] for child_name in joint_names]
        
    # forward kinematics
    joint_number = len(joint_names)
    pose_xyz = np.zeros((pose_euler.shape[0], joint_number*3))
    for i in range(pose_euler.shape[0]):        
        # xyz position in the world coordinate system
        pose_xyz_tmp = np.zeros((joint_number, 3))
        pose_xyz_tmp[0] = [pose_euler[i][0], pose_euler[i][1], pose_euler[i][2]]                        
        pose_rot_mat = np.zeros((joint_number, 3, 3))
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            pose_rot_mat[j] = R.from_euler('XYZ', rot).as_matrix()
                          
        for j in range(1, joint_number):
            pose_rot_mat_parent = pose_rot_mat[joint_parent_ids[j]]
            pose_xyz_tmp[j] = np.matmul(pose_rot_mat_parent, joint_trans[j-1]) + pose_xyz_tmp[joint_parent_ids[j]]
            pose_rot_mat[j] = np.matmul(pose_rot_mat_parent, pose_rot_mat[j])
        
        pose_xyz[i] = pose_xyz_tmp.reshape(joint_number*3)
    return pose_xyz

    
def euler2xyz_head(pose_euler):
    """
    Calculate head direction from human pose
    """     
    # names of the joints
    joint_names = ['base', 'pelvis', 'torso', 'neck', 'head']
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    head_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        head_direction[i] = [0, -1, 0]
        head_direction[i] = np.matmul(pose_rot, head_direction[i])
        
    return head_direction
    
    
def euler2xyz_neck(pose_euler):
    """
    Calculate neck direction from human pose
    """     
    # names of the joints
    joint_names = ['base', 'pelvis', 'torso', 'neck']
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    neck_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        neck_direction[i] = [0, -1, 0]
        neck_direction[i] = np.matmul(pose_rot, neck_direction[i])
        
    return neck_direction


def euler2xyz_torso(pose_euler):
    """
    Calculate torso direction from human pose
    """     
    # names of the joints
    joint_names = ['base', 'pelvis', 'torso']
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    torso_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        torso_direction[i] = [0, -1, 0]
        torso_direction[i] = np.matmul(pose_rot, torso_direction[i])
        
    return torso_direction


def euler2xyz_pelvis(pose_euler):
    """
    Calculate pelvis direction from human pose
    """     
    # names of the joints
    joint_names = ['base', 'pelvis']
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    pelvis_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        pelvis_direction[i] = [0, -1, 0]
        pelvis_direction[i] = np.matmul(pose_rot, pelvis_direction[i])
        
    return pelvis_direction


def euler2xyz_base(pose_euler):
    """
    Calculate base direction from human pose
    """     
    # names of the joints
    joint_names = ['base']
    # forward kinematics
    joint_number = len(joint_names)        
    data_size = pose_euler.shape[0]
    base_direction = np.zeros((data_size, 3))
    
    for i in range(data_size):
        pose_rot = R.identity().as_matrix()    
        for j in range(joint_number):
            rot = np.array([pose_euler[i][(j+1)*3], pose_euler[i][(j+1)*3 + 1], pose_euler[i][(j+1)*3 + 2]])
            rot_mat = R.from_euler('XYZ', rot).as_matrix()
            pose_rot = np.matmul(pose_rot, rot_mat)          
            
        base_direction[i] = [0, -1, 0]
        base_direction[i] = np.matmul(pose_rot, base_direction[i])
        
    return base_direction