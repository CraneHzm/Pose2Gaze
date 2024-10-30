## Code to process and analyse the MoGaze dataset


## Usage:
Step 1: Download the dataset at https://humans-to-robots-motion.github.io/mogaze/.

Step 2: Set 'dataset_path' and 'dataset_processed_path' in 'mogaze_preprocessing.py' and run it to process the dataset.

Step 3: It is optional but highly recommended to set 'data_path' in 'dataset_visualisation.py' to visualise and get familiar with the dataset.

Step 4: Set 'data_dir' and 'actions' in 'mogaze_gaze_body_orientation.py' and run it to analyse the correlations between eye gaze and body orientations, corresponding to the results in Table 1 of the Pose2Gaze paper.

Step 5: Set 'data_dir' in 'mogaze_gaze_head_intervals.py' and run it to analyse the correlations between eye gaze and head orientation at different time intervals, corresponding to the results in Figure 1 of the Pose2Gaze paper.

Step 6: Set 'data_dir' and 'actions' in 'mogaze_gaze_body_motion.py' and run it to analyse the correlations between eye gaze and body motions, corresponding to the results in Table 2 of the Pose2Gaze paper.

Step 7: Set 'data_dir' in 'mogaze_gaze_body_motion_intervals.py' and run it to analyse the correlations between eye gaze and body motions at different time intervals, corresponding to the results in Figure 2 of the Pose2Gaze paper.


## Citations

```bibtex
@article{hu24pose2gaze,
	author={Hu, Zhiming and Xu, Jiahui and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={Pose2Gaze: Eye-body Coordination during Daily Activities for Gaze Prediction from Full-body Poses},
	year={2024}}
	
@article{kratzer2020mogaze,
	title={MoGaze: A dataset of full-body motions that includes workspace geometry and eye-gaze},
	author={Kratzer, Philipp and Bihlmaier, Simon and Midlagajni, Niteesh Balachandra and Prakash, Rohit and Toussaint, Marc and Mainprice, Jim},
	journal={IEEE Robotics and Automation Letters},
	volume={6},
	number={2},
	pages={367--373},
	year={2020}}
```