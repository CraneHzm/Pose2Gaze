from utils import adt_dataset, seed_torch
from model import pose2gaze
from utils.opt import options
from utils import log
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import datetime
import torch.optim as optim
import os
import math


def main(opt):
    # set the random seed to ensure reproducibility
    seed_torch.seed_torch(seed=0)
    torch.set_num_threads(1) 

    data_dir = opt.data_dir
    seq_len = opt.seq_len
    joints_used = opt.joints_used
    if joints_used == 'all':
        opt.joint_number = 21
    if joints_used == 'torso':
        opt.joint_number = 5
    if joints_used == 'arm':
        opt.joint_number = 8
    if joints_used == 'leg':
        opt.joint_number = 8
    if joints_used == 'torso+arm':
        opt.joint_number = 13
    if joints_used == 'torso+leg':
        opt.joint_number = 13
    if joints_used == 'arm+leg':
        opt.joint_number = 16    
     
    learning_rate = opt.learning_rate
    print('>>> create model')
    net = pose2gaze.pose2gaze(opt=opt).to(opt.cuda_idx)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=opt.learning_rate)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    print('>>> loading datasets')
    
    
    train_dataset = adt_dataset.adt_dataset(data_dir, seq_len, opt.actions, joints_used = joints_used, train_flag = 1, sample_rate = opt.train_sample_rate)
    train_data_size = train_dataset.pose_head_gaze.shape
    print("Training data size: {}".format(train_data_size))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dataset = adt_dataset.adt_dataset(data_dir, seq_len, opt.actions, joints_used = joints_used, train_flag = 0)
    valid_data_size = valid_dataset.pose_head_gaze.shape
    print("Validation data size: {}".format(valid_data_size))                
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # training
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTraining starts at ' + local_time)
    start_time = datetime.datetime.now()
    start_epoch = 1

    err_best = 1000
    best_epoch = 0
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma, last_epoch=-1)
    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False            
        learning_rate = exp_lr.optimizer.param_groups[0]["lr"]
            
        train_start_time = datetime.datetime.now()
        result_train = run_model(net, optimizer, is_train=1, data_loader=train_loader, opt=opt)
        train_end_time = datetime.datetime.now()
        train_time = (train_end_time - train_start_time).seconds*1000
        train_batch_num = math.ceil(train_data_size[0]/opt.batch_size)
        train_time_per_batch = math.ceil(train_time/train_batch_num)
        #print('\nTraining time per batch: {} ms'.format(train_time_per_batch))
        
        exp_lr.step()
        rng_state = torch.get_rng_state()
        if epo % opt.validation_epoch == 0:
            print('>>> training epoch: {:d}, lr: {:.12f}'.format(epo, learning_rate))
            print('Training data size: {}'.format(train_data_size))
            print('Average baseline error: {:.1f} degree'.format(result_train['baseline_error_average']))            
            print('Average training error: {:.1f} degree'.format(result_train['prediction_error_average']))
            
            test_start_time = datetime.datetime.now()
            result_valid = run_model(net, is_train=0, data_loader=valid_loader, opt=opt)                        
            test_end_time = datetime.datetime.now()
            test_time = (test_end_time - test_start_time).seconds*1000
            test_batch_num = math.ceil(valid_data_size[0]/opt.test_batch_size)
            test_time_per_batch = math.ceil(test_time/test_batch_num)
            #print('\nTest time per batch: {} ms'.format(test_time_per_batch))
            print('Validation data size: {}'.format(valid_data_size))
            
            print('Average baseline error: {:.1f} degree'.format(result_valid['baseline_error_average']))
            print('Average validation error: {:.1f} degree'.format(result_valid['prediction_error_average']))
            
            if result_valid['prediction_error_average'] < err_best:
                err_best = result_valid['prediction_error_average']
                is_best = True
                best_epoch = epo
                
            print('Best validation error: {:.1f} degree, best epoch: {}'.format(err_best, best_epoch))                                                
            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds/60
            print('\nTotal training time: {:.1f} min'.format(total_training_time))
            local_time = time.asctime(time.localtime(time.time()))
            print('\nTraining ends at ' + local_time)
            
            result_log = np.array([epo, learning_rate])
            head = np.array(['epoch', 'lr'])
            for k in result_train.keys():
                result_log = np.append(result_log, [result_train[k]])
                head = np.append(head, [k])
            for k in result_valid.keys():
                result_log = np.append(result_log, [result_valid[k]])
                head = np.append(head, ['valid_' + k])
            log.save_csv_log(opt, head, result_log, is_create=(epo == 1))           
            log.save_ckpt({'epoch': epo,
                           'lr': learning_rate,
                           'err': result_valid['prediction_error_average'],
                           'state_dict': net.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            opt=opt)
                            
        torch.set_rng_state(rng_state)

        
def eval(opt):
    data_dir = opt.data_dir
    seq_len = opt.seq_len
    joints_used = opt.joints_used
    if joints_used == 'all':
        opt.joint_number = 21
    if joints_used == 'torso':
        opt.joint_number = 5
    if joints_used == 'arm':
        opt.joint_number = 8
    if joints_used == 'leg':
        opt.joint_number = 8
    if joints_used == 'torso+arm':
        opt.joint_number = 13
    if joints_used == 'torso+leg':
        opt.joint_number = 13
    if joints_used == 'arm+leg':
        opt.joint_number = 16
        
    print('>>> create model')
    net = pose2gaze.pose2gaze(opt=opt).to(opt.cuda_idx)    
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    #load model    
    model_name = 'model.pt'
    model_path = os.path.join(opt.ckpt, model_name)    
    print(">>> loading ckpt from '{}'".format(model_path))
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    
    print('>>> loading datasets')        
    test_dataset = adt_dataset.adt_dataset(data_dir, seq_len, opt.actions, joints_used = joints_used, train_flag = 0)
    test_data_size = test_dataset.pose_head_gaze.shape
    print("Test data size: {}".format(test_data_size))                
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # test
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest starts at ' + local_time)
    start_time = datetime.datetime.now()
    if opt.save_predictions:
        result_test, predictions = run_model(net, is_train=0, data_loader=test_loader, opt=opt)
    else:
        result_test = run_model(net, is_train=0, data_loader=test_loader, opt=opt)
    
    print('Average baseline error: {:.1f} degree'.format(result_test['baseline_error_average']))
    print('Average prediction error: {:.1f} degree'.format(result_test['prediction_error_average']))
                                               
    end_time = datetime.datetime.now()
    total_test_time = (end_time - start_time).seconds/60
    print('\nTotal test time: {:.1f} min'.format(total_test_time))  
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest ends at ' + local_time)


    if opt.save_predictions:    
        # body poses + head directions + predictions + ground truth
        prediction = predictions[:, :, opt.joint_number*3+3: opt.joint_number*3+6]
        ground_truth = predictions[:, :, opt.joint_number*3+6: opt.joint_number*3+9]
        prediction_errors = np.arccos(np.sum(prediction*ground_truth, 2))/np.pi * 180.0        
        print('Average prediction error: {:.1f} degree'.format(np.mean(prediction_errors)))
        
        predictions_path = os.path.join(opt.ckpt, "predictions.npy")
        np.save(predictions_path, predictions)        
        prediction_errors_path = os.path.join(opt.ckpt, "prediction_errors.npy")
        np.save(prediction_errors_path, prediction_errors)


def acos_safe(x, eps=1e-6):
    slope = np.arccos(1-eps) / eps
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

    
def run_model(net, optimizer=None, is_train=1, data_loader=None, opt=None):
    if is_train == 1:
        net.train()
    else:
        net.eval()
            
    if opt.is_eval and opt.save_predictions:
        predictions = []
                        
    prediction_error_average = 0
    baseline_error_average = 0                
    
    n = 0
    # E.g. if opt.seq_len == 45, it contains 15 past frames, 15 present frames, and 15 future frames.
    input_n = opt.seq_len//3
        
    for i, (data) in enumerate(data_loader):
        batch_size, seq_n, dim = data.shape
        joint_number = opt.joint_number
        # when only one sample in this batch
        if batch_size == 1 and is_train == 1:
            continue
        n += batch_size
        data = data.float().to(opt.cuda_idx)
        
        # generate eye gaze from past body poses
        if opt.generation_setting == 'past':
            # past body poses
            body_poses = data.clone()[:, :input_n, :joint_number*3]
            # past head directions
            head_directions = data.clone()[:, :input_n, joint_number*3:joint_number*3+3]
            input = torch.cat((body_poses, head_directions), dim=2)
            # future eye gaze
            ground_truth = data.clone()[:, input_n:input_n*2, joint_number*3+3:joint_number*3+6]
            prediction = net(input, input_n=input_n)            
        # generate eye gaze from present body poses
        elif opt.generation_setting == 'present':
            # present body poses
            body_poses = data.clone()[:, input_n:input_n*2, :joint_number*3]
            # present head directions
            head_directions = data.clone()[:, input_n:input_n*2, joint_number*3:joint_number*3+3]
            input = torch.cat((body_poses, head_directions), dim=2)
            # present eye gaze
            ground_truth = data.clone()[:, input_n:input_n*2, joint_number*3+3:joint_number*3+6]
            prediction = net(input, input_n=input_n)                          
        # generate eye gaze from future body poses
        elif opt.generation_setting == 'future':
            # future body poses
            body_poses = data.clone()[:, input_n*2:input_n*3, :joint_number*3]
            # present head directions
            head_directions = data.clone()[:, input_n:input_n*2, joint_number*3:joint_number*3+3]
            input = torch.cat((body_poses, head_directions), dim=2)
            # present eye gaze
            ground_truth = data.clone()[:, input_n:input_n*2, joint_number*3+3:joint_number*3+6]
            prediction = net(input, input_n=input_n)
                        
        if opt.is_eval and opt.save_predictions:
            # body poses + head directions + predictions + ground truth
            prediction_cpu = torch.cat((input, prediction), dim=2)
            prediction_cpu = torch.cat((prediction_cpu, ground_truth), dim=2)            
            prediction_cpu = prediction_cpu.cpu().data.numpy()
            if len(predictions) == 0:
                predictions = prediction_cpu                
            else:
                predictions = np.concatenate((predictions, prediction_cpu), axis=0)           
            
        # training process
        loss = torch.mean(acos_safe(torch.sum(ground_truth*prediction, 2)))/torch.tensor(math.pi) * 180.0
        
        if is_train == 1:            
            optimizer.zero_grad()
            loss.backward()                        
            optimizer.step()
            
        # Calculate prediction errors
        error = torch.mean(acos_safe(torch.sum(ground_truth*prediction, 2)))/torch.tensor(math.pi) * 180.0
        prediction_error_average += error.cpu().data.numpy() * batch_size
        
        # Use head directions as the baseline
        baseline_error = torch.mean(acos_safe(torch.sum(ground_truth*head_directions, 2)))/torch.tensor(math.pi) * 180.0
        baseline_error_average += baseline_error.cpu().data.numpy() * batch_size
            
    result = {}
    result["prediction_error_average"] = prediction_error_average / n
    result["baseline_error_average"] = baseline_error_average / n
    
    
    if opt.is_eval and opt.save_predictions:        
        return result, predictions
    else:
        return result
    
    
if __name__ == '__main__':    
    option = options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)