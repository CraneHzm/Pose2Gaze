python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --use_pose_dct 1;

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'catch';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'chat';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'dance';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'discuss';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'learn';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'perform';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'teach';

python pose2gaze_egobody.py --data_dir /scratch/hu/pose_forecast/egobody_pose2gaze/ --ckpt ./checkpoints/egobody_past/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'past' --gamma 0.8 --train_sample_rate 5 --learning_rate 0.001 --epoch 20 --is_eval --actions 'all';