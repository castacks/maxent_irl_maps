"""
Script for converting experiment yamls into the actual objects to run experiments with
"""
import yaml
import torch
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.skid_steer import SkidSteer

from torch_mpc.algos.mppi import MPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.algos.mppi_irl import MPPIIRL

from maxent_irl_costmaps.networks.resnet import ResnetCostmapCNN
from maxent_irl_costmaps.networks.unet import UNet

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from maxent_irl_costmaps.experiment_management.experiment import Experiment

def setup_experiment(fp):
    """
    Expect the following top-level keys in the YAML:
        1. experiment: high-level params such as where to save to, epochs, etc.
        2. dataset
        3. network
        4. trajopt
        5. cost_function
        6. model
        7. metrics

    Design decision to use case statements instead of dicts of class types in case I want to
    handle params in specific ways for certain classes
    """
    experiment_dict = yaml.safe_load(open(fp, 'r'))
    experiment_keys = [
        'experiment',
        'dataset',
        'network',
        'netopt',
        'trajopt',
        'cost_function',
        'model',
        'metrics'
    ]
    res = {}
    #check validity of experiment YAML
    for k in experiment_keys:
        assert k in experiment_dict.keys(), "Expected key {} in yaml, found these keys: {}".format(k, experiment_dict.keys())

    #move to correct device
    device = experiment_dict['experiment']['device'] if 'device' in experiment_dict['experiment'].keys() else 'cpu'

    res['params'] = experiment_dict

    #setup dataset
    dataset_params = experiment_dict['dataset']
    if dataset_params['type'] == 'MaxEntIRLDataset':
        res['dataset'] = MaxEntIRLDataset(**dataset_params['params']).to(device)
    else:
        print('Unsupported dataset type {}'.format(dataset_params['type']))
        exit(1)

    #setup network
    network_params = experiment_dict['network']
    if network_params['type'] == 'ResnetCostMapCNN':
        res['network'] = ResnetCostmapCNN(
            in_channels = len(res['dataset'].feature_keys),
            out_channels = 1,
            **network_params['params']
        ).to(device)
    elif network_params['type'] == 'UNet':
        channels = len(res['dataset'].feature_keys)
        nx = int(res['dataset'].metadata['width'] / res['dataset'].metadata['resolution'])
        ny = int(res['dataset'].metadata['height'] / res['dataset'].metadata['resolution'])
        res['network'] = UNet(
            insize = [channels, nx, ny],
            outsize = [1, nx, ny],
            **network_params['params']
        )
    else:
        print('Unsupported network type {}'.format(network_params['type']))
        exit(1)

    #setup network opt
    netopt_params = experiment_dict['netopt']
    if netopt_params['type'] == 'Adam':
        res['netopt'] = torch.optim.Adam(res['network'].parameters(), **netopt_params['params'])
    else:
        print('Unsupported netopt type {}'.format(netopt_params['type']))
        exit(1)

    #setup model
    model_params = experiment_dict['model']
    if model_params['type'] == 'SteerSetpointKBM':
        res['model'] = SteerSetpointKBM(**model_params['params']).to(device)
    elif model_params['type'] == 'SkidSteer':
        res['model'] = SkidSteer(**model_params['params']).to(device)
    else:
        print('Unsupported model type {}'.format(model_params['type']))
        exit(1)

    #setup cost function
    cost_function_params = experiment_dict['cost_function']
    if cost_function_params['type'] == 'WaypointCostMapCostFunction':
        res['cost_function'] = WaypointCostMapCostFunction(
            map_params = res['dataset'].metadata,
            **cost_function_params['params']
        ).to(device)
    else:
        print('Unsupported cost function type {}'.format(cost_function_params['type']))
        exit(1)

    #setup trajopt
    trajopt_params = experiment_dict['trajopt']
    if trajopt_params['type'] == 'MPPI':
        res['trajopt'] = MPPI(
            model = res['model'],
            cost_fn = res['cost_function'],
            num_timesteps = res['dataset'].horizon,
            **trajopt_params['params']
        ).to(device)
    else:
        print('Unsupported trajopt function type {}'.format(trajopt_params['type']))
        exit(1)

    #setup algo
    algo_params = experiment_dict['algo']
    if algo_params['type'] == 'MPPIIRL':
        res['algo'] = MPPIIRL(
            network = res['network'],
            opt = res['netopt'],
            expert_dataset = res['dataset'],
            mppi = res['trajopt'],
            **algo_params['params']
    ).to(device)

    #setup experiment
    experiment_params = experiment_dict['experiment']
    res['experiment'] = Experiment(
        algo = res['algo'],
        params = res['params'],
        **experiment_params
    ).to(device)

    return res

#TEST
if __name__ == '__main__':
    fp = '../../../configs/training/yamaha_atv_20220628.yaml'
    res = setup_experiment(fp)

#    for i in range(10):
#        res['dataset'].visualize()
#        plt.show()

    res['experiment'].run()
