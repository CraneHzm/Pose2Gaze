python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'past' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'pick' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_past_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'place' --seq_len 45 --generation_setting 'past' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'present' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'pick' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_present_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'place' --seq_len 45 --generation_setting 'present' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'future' --train_sample_rate 2;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'all' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'pick' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;

python pose2gaze_mogaze.py --data_dir /scratch/hu/pose_forecast/mogaze_pose2gaze/ --ckpt ./checkpoints/mogaze_future_p1/ --cuda_idx cuda:5 --joints_used 'all' --test_id 1 --actions 'place' --seq_len 45 --generation_setting 'future' --train_sample_rate 2 --is_eval;