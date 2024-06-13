python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'past' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'pick' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'place' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'present' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'pick' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'place' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'future' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'all' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'pick' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p7/ --cuda_idx cuda:3 --joints_used 'all' --test_id 7 --actions 'place' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;