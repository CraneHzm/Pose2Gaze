## Code to process and analyse the EgoBody dataset


## Usage:
Step 1: Download the dataset at https://egobody.ethz.ch/.

Step 2: Set 'dataset_path' and 'dataset_processed_path' in 'egobody_preprocessing.py' and run it to process the dataset. If you meet the error "RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported", follow this link to solve it https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported.

Step 3: It is optional but highly recommended to set 'data_path' in 'dataset_visualisation.py' to visualise and get familiar with the dataset.

Step 4: Set 'data_dir' and 'actions' in 'egobody_gaze_body_orientation.py' and run it to analyse the correlations between eye gaze and body orientations, corresponding to the results in Table 1 of the Pose2Gaze paper.

Step 5: Set 'data_dir' in 'egobody_gaze_head_intervals.py' and run it to analyse the correlations between eye gaze and head orientation at different time intervals, corresponding to the results in Figure 1 of the Pose2Gaze paper.

Step 6: Set 'data_dir' and 'actions' in 'egobody_gaze_body_motion.py' and run it to analyse the correlations between eye gaze and body motions, corresponding to the results in Table 2 of the Pose2Gaze paper.

Step 7: Set 'data_dir' in 'egobody_gaze_body_motion_intervals.py' and run it to analyse the correlations between eye gaze and body motions at different time intervals, corresponding to the results in Figure 2 of the Pose2Gaze paper.

Step 8: Set 'data_dir' and 'actions' in 'egobody_gaze_two_body_motion.py' and run it to analyse the correlations between eye gaze and the directions pointing from a person's body to the interaction partner, corresponding to the results in Table 3 of the Pose2Gaze paper.

Step 9: Set 'data_dir' in 'egobody_gaze_two_body_motion_intervals.py' and run it to analyse the correlations between eye gaze and the directions pointing from a person's body to the interaction partner at different time intervals, corresponding to the results in Figure 3 of the Pose2Gaze paper.


## Citations

```bibtex
@article{hu24_pose2gaze,
	author={Hu, Zhiming and Xu, Jiahui and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={Pose2Gaze: Eye-body Coordination during Daily Activities for Gaze Prediction from Full-body Poses},
	year={2024}}

@inproceedings{zhang2022egobody,
	title={EgoBody: Human body shape, motion and social interactions from head-mounted devices},
	author={Zhang, Siwei and Ma, Qianli and Zhang, Yan and Qian, Zhiyin and Pollefeys, Marc and Bogo, Federica and Tang, Siyu},
	booktitle={Proceedings of the 2022 European Conference on Computer Vision},
	year={2022}}
```