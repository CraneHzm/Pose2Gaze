python pose2gaze_adt.py --data_dir /scratch/hu/pose_forecast/adt_pose2gaze/ --ckpt ./checkpoints/adt_future/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'future' --gamma 0.9 --train_sample_rate 2 --learning_rate 0.001 --epoch 30 --use_pose_dct 1;

python pose2gaze_adt.py --data_dir /scratch/hu/pose_forecast/adt_pose2gaze/ --ckpt ./checkpoints/adt_future/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'future' --gamma 0.9 --train_sample_rate 2 --learning_rate 0.001 --epoch 30 --is_eval --actions 'decoration';

python pose2gaze_adt.py --data_dir /scratch/hu/pose_forecast/adt_pose2gaze/ --ckpt ./checkpoints/adt_future/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'future' --gamma 0.9 --train_sample_rate 2 --learning_rate 0.001 --epoch 30 --is_eval --actions 'meal';

python pose2gaze_adt.py --data_dir /scratch/hu/pose_forecast/adt_pose2gaze/ --ckpt ./checkpoints/adt_future/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'future' --gamma 0.9 --train_sample_rate 2 --learning_rate 0.001 --epoch 30 --is_eval --actions 'work';

python pose2gaze_adt.py --data_dir /scratch/hu/pose_forecast/adt_pose2gaze/ --ckpt ./checkpoints/adt_future/ --cuda_idx cuda:4 --joints_used 'all' --seq_len 45 --generation_setting 'future' --gamma 0.9 --train_sample_rate 2 --learning_rate 0.001 --epoch 30 --is_eval --actions 'all';