## Code to process and analyse the ADT dataset

Note: processing the ADT dataset is much more complicated than other datasets because it relies on the Project Aria Tools. It would be easier to get started with other datasets first.


## Usage:
Step 1: Follow steps 1-3 on https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download to prepare the folder and download_links file for downloading the dataset.

Step 2: Follow the instructions on https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/getting_started to set up the environment for the Project Aria Tools.

Step 3: Follow the instructions on https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download to download main data for all sequences.

Step 4: Set 'dataset_path' and 'dataset_processed_path' in 'adt_preprocessing.py', put 'adt_preprocessing.py' and 'adt.csv' into the codebase of the Project Aria Tools (https://github.com/facebookresearch/projectaria_tools), and run it to process the dataset.

Step 5: It is optional but highly recommended to set 'data_path' in 'dataset_visualisation.py' to visualise and get familiar with the dataset.

Step 6: Set 'data_dir' in 'adt_gaze_head_intervals.py' and run it to analyse the correlations between eye gaze and head orientation at different time intervals, corresponding to the results in Figure 1 of the Pose2Gaze paper.

Step 7: Set 'data_dir' and 'actions' in 'adt_gaze_body_motion.py' and run it to analyse the correlations between eye gaze and body motions, corresponding to the results in Table 2 of the Pose2Gaze paper.

Step 8: Set 'data_dir' in 'adt_gaze_body_motion_intervals.py' and run it to analyse the correlations between eye gaze and body motions at different time intervals, corresponding to the results in Figure 2 of the Pose2Gaze paper.


## Citations

```bibtex
@article{hu24_pose2gaze,
	author={Hu, Zhiming and Xu, Jiahui and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={Pose2Gaze: Eye-body Coordination during Daily Activities for Gaze Prediction from Full-body Poses},
	year={2024}
}
			
@inproceedings{pan2023aria,
  title={Aria digital twin: A new benchmark dataset for egocentric 3d machine perception},
  author={Pan, Xiaqing and Charron, Nicholas and Yang, Yongqian and Peters, Scott and Whelan, Thomas and Kong, Chen and Parkhi, Omkar and Newcombe, Richard and Ren, Yuheng Carl},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20133--20143},
  year={2023}
}
```