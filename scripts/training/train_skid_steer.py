import torch
import os
import argparse
import matplotlib.pyplot as plt

from torch_mpc.models.skid_steer import SkidSteer
from torch_mpc.algos.mppi import MPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.algos.mppi_irl import MPPIIRL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_fp', type=str, required=True, help='dir to save experiment results')
    parser.add_argument('--rosbag_dir', type=str, required=True, help='dir for rosbags to train from')
    parser.add_argument('--preprocess_dir', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/warty/odom', help='topic to extract odom from')
    parser.add_argument('--epochs', type=int, required=False, default=10, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size')
    parser.add_argument('--horizon', type=int, required=False, default=70, help='number of mppi steps to optimize over')
    args = parser.parse_args()

    dataset = MaxEntIRLDataset(bag_fp=args.rosbag_dir, preprocess_fp=args.preprocess_dir, map_features_topic=args.map_topic, odom_topic=args.odom_topic, horizon=int(args.horizon) * 1.5)

#    for i in range(10):
#        dataset.visualize()
#        plt.show()

    model = SkidSteer(v_lim=[0.5, 3.0], w_lim=[-1.5, 1.5])
    
    cfn = WaypointCostMapCostFunction(unknown_cost=0., goal_cost=0., map_params=dataset.metadata)
    mppi  = MPPI(model=model, cost_fn=cfn, num_samples=2048, num_timesteps=args.horizon, control_params={'sys_noise':torch.tensor([1.0, 0.5]), 'temperature':0.05})

    mppi_irl = MPPIIRL(dataset, mppi, args.batch_size)

    for ei in range(args.epochs):
        mppi_irl.update()


    torch.save(mppi_irl.network, 'baseline.pt')
#        if ((ei+1) % 5) == 0:
#            mppi_irl.visualize()
    mppi_irl.visualize()
