import os
import argparse
from pprint import pprint


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None
    
    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str,
                                 default='./dataset/',
                                 help='path to dataset')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether to evaluate existing models or not')                                 
        self.parser.add_argument('--ckpt', type=str, default='./checkpoint/', help='path to save checkpoint')               
        self.parser.add_argument('--test_id', type=int, default=1, help='id of the test participant') 
        self.parser.add_argument('--actions', type=str, default='all', help='actions to use for training and test')
        self.parser.add_argument('--action_num', type=int, default=3, help='number of actions')        
        self.parser.add_argument('--train_sample_rate', type=int, default=1, help='sample the training data')
        self.parser.add_argument('--save_predictions', dest='save_predictions', action='store_true',
                                 help='whether to save the prediction results or not')
        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--joint_number', type=int, default=21, help='number of joints to use')        
        self.parser.add_argument('--joints_used', type=str,
                                 default='all',
                                 help='the length of "joints_used" should equal to "joint_number"')
        self.parser.add_argument('--head_cnn_channels', type=int, default=32, help='number of channels used in the head_CNN')        
        self.parser.add_argument('--gcn_latent_features', type=int, default=16, help='number of latent features used in the gcn')
        self.parser.add_argument('--residual_gcns_num', type=int, default=4, help='number of residual gcns to use')
        self.parser.add_argument('--gcn_dropout', type=float, default=0.3, help='drop out probability in the gcn')               
        self.parser.add_argument('--gaze_cnn_channels', type=int, default=64, help='number of channels used in the gaze_CNN')                
        self.parser.add_argument('--use_pose_dct', type=int, default=1, help='use dct to process pose data')        
        # ===============================================================
        #                     Running options
        # ===============================================================       
        self.parser.add_argument('--seq_len', type=int, default=45, help='the total length of the used sequence, e.g. 15 past frames, 15 present frames, 15 future frames.')
        self.parser.add_argument('--generation_setting', type=str,
                                 default='present',
                                 help="generate eye gaze from 'past', 'present', or 'future' poses")
        self.parser.add_argument('--learning_rate', type=float, default=0.005)
        self.parser.add_argument('--gamma', type=float, default=0.95, help='decay learning rate by gamma')
        self.parser.add_argument('--epoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--validation_epoch', type=int, default=10, help='interval of epoches to test')
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        
    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, make_dir=True):
        self._initial()
        self.opt = self.parser.parse_args()               
        ckpt = self.opt.ckpt
        if make_dir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
        self._print()
        return self.opt