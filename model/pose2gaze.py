from torch import nn
import torch
from model import graph_convolution_network
import utils.util as util
import torch.nn.functional as F

        
class pose2gaze(nn.Module):
    def __init__(self, opt):
        super(pose2gaze, self).__init__()        
        self.opt = opt                    
        self.joint_number = opt.joint_number
        self.input_n = 15
        self.use_pose_dct = opt.use_pose_dct              
        latent_features = opt.gcn_latent_features
        residual_gcns_num = opt.residual_gcns_num
        gcn_dropout = opt.gcn_dropout
        head_cnn_channels = opt.head_cnn_channels
        gaze_cnn_channels = opt.gaze_cnn_channels

        # 1D CNN for extracting features from head directions/body orientations
        in_channels_head = 3
        cnn_kernel_size = 3
        cnn_padding = (cnn_kernel_size -1)//2
        out_channels_1_head = head_cnn_channels
        out_channels_2_head = head_cnn_channels
        out_channels_head = head_cnn_channels
        
        self.head_cnn = nn.Sequential(
        nn.Conv1d(in_channels = in_channels_head, out_channels=out_channels_1_head, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
        nn.LayerNorm([out_channels_1_head, self.input_n], elementwise_affine=True),
        nn.Tanh(),
        nn.Conv1d(in_channels=out_channels_1_head, out_channels=out_channels_2_head, kernel_size=cnn_kernel_size, padding = cnn_padding, padding_mode='replicate'),
        nn.LayerNorm([out_channels_2_head, self.input_n], elementwise_affine=True),
        nn.Tanh(),
        nn.Conv1d(in_channels=out_channels_2_head, out_channels=out_channels_head, kernel_size=cnn_kernel_size, padding = cnn_padding, padding_mode='replicate'),
        nn.Tanh()
        )        
        
        # GCN for extracting features from body poses
        self.pose_gcn = graph_convolution_network.graph_convolution_network(in_features=3,
                                               latent_features=latent_features,
                                               node_n=self.joint_number,
                                               seq_len=self.input_n,
                                               p_dropout=gcn_dropout,
                                               residual_gcns_num=residual_gcns_num)
      
        # 1D CNN for generating eye gaze from extracted features
        in_channels_gaze = latent_features*self.joint_number + out_channels_head
        cnn_kernel_size = 3
        cnn_padding = (cnn_kernel_size -1)//2
        out_channels_1_gaze = gaze_cnn_channels
        out_channels_gaze = 3
        
        self.gaze_cnn = nn.Sequential(
        nn.Conv1d(in_channels = in_channels_gaze, out_channels=out_channels_1_gaze, kernel_size=cnn_kernel_size, padding=cnn_padding, padding_mode='replicate'),
        nn.LayerNorm([out_channels_1_gaze, self.input_n], elementwise_affine=True),
        nn.Tanh(),
        nn.Conv1d(in_channels=out_channels_1_gaze, out_channels=out_channels_gaze, kernel_size=cnn_kernel_size, padding = cnn_padding, padding_mode='replicate'),        
        nn.Tanh()        
        )
        
        dct_m, idct_m = util.get_dct_matrix(self.input_n)
        self.dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        self.idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)

        self.act_f = nn.Tanh()
        
    def forward(self, src, input_n=15):
        bs, seq_len, features = src.shape
        pose = src.clone()[:, :, :self.joint_number*3]
        head = src.clone()[:, :, self.joint_number*3:self.joint_number*3+3]
                       
        # extract features from head directions/body orientations
        head_features = head.clone().permute(0,2,1)
        head_features = self.head_cnn(head_features)
        
        # extract features from body poses
        if self.use_pose_dct:
            pose = torch.matmul(self.dct_m, pose).permute(0, 2, 1)
        else:    
            pose = pose.permute(0, 2, 1)        
        pose = pose.reshape(bs, self.joint_number, 3, input_n).permute(0, 2, 1, 3)
        pose_features = self.pose_gcn(pose)
        pose_features = pose_features.permute(0, 2, 1, 3).reshape(bs, -1, input_n)
        if self.use_pose_dct:
            pose_features = pose_features.permute(0, 2, 1)
            pose_features = torch.matmul(self.idct_m, pose_features).permute(0, 2, 1)
            pose_features = self.act_f(pose_features)
            
        # fuse head and pose features
        features = torch.cat((pose_features, head_features), dim=1)
        # generate eye gaze from extracted features
        prediction = self.gaze_cnn(features).permute(0, 2, 1)            
        # normalize to unit vectors
        prediction = F.normalize(prediction, dim=2)        

        return prediction