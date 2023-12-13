# maxent_irl_maps

Code for running MaxEnt IRL for offroad navigation

## Dependencies

[ROS noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)

[`rosbag_to_dataset`](https://github.com/striest/rosbag_to_dataset/tree/feature/irl_postproc): Note that there is a separate IRL preproc branch for now. Additionally, this dependency is only necessary for generating new train data.

`torch_mpc`: Currently a private repo, ask @striest for access


## Key Scripts/Files

All scripts have relatively helpful descriptions of their run args (`python3 <script> -h`)

`scripts/run_experiment.py`: Main driver script. Need to provide a `--setup_fp` arg that points to a config file (a good example is: `configs/training/pointpillars_debug.yaml`).

`scripts/generate_metrics.py`: After running experiment, evaluate/visualize network with this script. 

`scripts/preprocess_dataset_no_pointpillars.py`: Take a `rosbag_to_dataset` dataset and process it for IRL.

`scripts/ros/gridmap_to_cvar_costmap.py`: ROS node that runs the trained CVaR IRL map in ROS.

`src/maxent_irl_costmaps/algos/mppi_irl_speedmaps.py`: Main IRL training code

`src/maxent_irl_costmaps/experiment_management/parse_configs.py`: Registry of strings->files to set up IRL experiments. New network definitions should be added here.

## Usage

Run via: 

```
cd scripts
python3 run_experiment.py --setup_fp <your config here>
```

This code is designed to train a network for inverse RL for a ROS-based autonomy stack similar to [TartanDrive](https://github.com/castacks/tartan_drive_2.0). At a high level, its input is rosbags with the following:

1. [GridMaps](https://github.com/ANYbotics/grid_map) of local terrain
2. [Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html) of robot state
3. [Images](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) FPV images (viz only for now)
4. Steering Angle (as stamped Float32 in deg, robot specific)

Note that we require both grid maps and odometry to be in the same frame.

Outputs will be a directory of trained networks that can be run on robot with the ROS script.