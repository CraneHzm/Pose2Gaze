python pose2gaze_gimo.py --data_dir /scratch/hu/pose_forecast/gimo_pose2gaze/ --ckpt ./checkpoints/gimo_present/ --cuda_idx cuda:3 --joints_used 'all' --seq_len 45 --generation_setting 'present' --learning_rate 0.001 --gamma 0.9 --epoch 20 --use_pose_dct 1;

python pose2gaze_gimo.py --data_dir /scratch/hu/pose_forecast/gimo_pose2gaze/ --ckpt ./checkpoints/gimo_present/ --cuda_idx cuda:3 --joints_used 'all' --seq_len 45 --generation_setting 'present' --learning_rate 0.001 --gamma 0.9 --epoch 20 --is_eval --actions 'change';

python pose2gaze_gimo.py --data_dir /scratch/hu/pose_forecast/gimo_pose2gaze/ --ckpt ./checkpoints/gimo_present/ --cuda_idx cuda:3 --joints_used 'all' --seq_len 45 --generation_setting 'present' --learning_rate 0.001 --gamma 0.9 --epoch 20 --is_eval --actions 'interact';

python pose2gaze_gimo.py --data_dir /scratch/hu/pose_forecast/gimo_pose2gaze/ --ckpt ./checkpoints/gimo_present/ --cuda_idx cuda:3 --joints_used 'all' --seq_len 45 --generation_setting 'present' --learning_rate 0.001 --gamma 0.9 --epoch 20 --is_eval --actions 'rest';

python pose2gaze_gimo.py --data_dir /scratch/hu/pose_forecast/gimo_pose2gaze/ --ckpt ./checkpoints/gimo_present/ --cuda_idx cuda:3 --joints_used 'all' --seq_len 45 --generation_setting 'present' --learning_rate 0.001 --gamma 0.9 --epoch 20 --is_eval --actions 'all';