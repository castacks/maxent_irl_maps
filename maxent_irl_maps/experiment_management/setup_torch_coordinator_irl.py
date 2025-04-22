"""
Script for converting experiment yamls into the actual objects to run experiments with
"""
import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.skid_steer import SkidSteer

from torch_mpc.setup_mpc import setup_mpc

# TODO uncomment when torch_state_lattice_planner is back
# from torch_state_lattice_planner.setup_planner import setup_planner

from torch_coordinator.setup_torch_coordinator import setup_torch_coordinator

from maxent_irl_maps.algos.mppi_irl_speedmaps_torch_coordinator import MPPIIRLSpeedmapsTorchCoordinator

from maxent_irl_maps.geometry_utils import make_footprint

from maxent_irl_maps.networks.resnet import ResnetCategorical, ResnetExpCostCategoricalSpeed

from maxent_irl_maps.dataset.voxel_mapping_irl_dataset import VoxelMappingIRLDataset

from maxent_irl_maps.experiment_management.experiment import Experiment

def load_net_for_eval(model_fp, device='cuda', skip_mpc=True):
    """
    Set up IRL network for eval

    Args:
        model_fp: the path to find the model (i.e. aaa/itr_x.pt)
            assumes that it's still in the parent folder (i.e. we can find _params.yaml and dummy_dataset)
    """
    model_base_dir = os.path.split(model_fp)[0]
    param_fp = os.path.join(model_base_dir, "_params.yaml")
    dummy_dataset_fp = os.path.join(model_base_dir, 'dummy_dataset')
    config = yaml.safe_load(open(param_fp, 'r'))
    config['dataset']['params']['root_fp'] = dummy_dataset_fp
    res = setup_experiment(config, skip_mpc=skip_mpc)["algo"].to(device)

    res.network.load_state_dict(torch.load(model_fp, weights_only=True))
    res.network.eval()

    return res

def setup_torch_coordinator_irl(fp, skip_mpc=False):
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
    if isinstance(fp, str):
        experiment_dict = yaml.safe_load(open(fp, "r"))
    else:
        experiment_dict = fp

    experiment_keys = [
        "experiment",
        "dataset",
        "network",
        "netopt",
        "footprint",
        "solver",
        "metrics",
    ]
    res = {}
    # check validity of experiment YAML
    for k in experiment_keys:
        assert (
            k in experiment_dict.keys()
        ), "Expected key {} in yaml, found these keys: {}".format(
            k, experiment_dict.keys()
        )

    # move to correct device
    device = (
        experiment_dict["experiment"]["device"]
        if "device" in experiment_dict["experiment"].keys()
        else "cpu"
    )

    res["params"] = experiment_dict

    # setup dataset
    dataset_params = experiment_dict["dataset"]
    if dataset_params["type"] == "VoxelMappingMaxEntIRLDataset":
        res["dataset"] = VoxelMappingIRLDataset(**dataset_params["params"]).to(device)
    else:
        print("Unsupported dataset type {}".format(dataset_params["type"]))
        exit(1)

    config = yaml.safe_load(open(experiment_dict["torch_coordinator_config"], 'r'))
    res["torch_coordinator"] = setup_torch_coordinator(config)

    n_feats = len(res["torch_coordinator"].nodes["cvar_maxent_irl_costmap"].feature_keys)

    # setup network
    network_params = experiment_dict["network"]
    network_params["params"]["device"] = device
    if network_params["type"] == "ResnetCategorical":
        res["bev_network"] = ResnetCategorical(
            in_channels=n_feats, **network_params["params"],
        ).to(device)
    elif network_params["type"] == "ResnetExpCostCategoricalSpeed":
        res["bev_network"] = ResnetExpCostCategoricalSpeed(
            in_channels=n_feats, **network_params["params"],
        ).to(device)
    else:
        print("Unsupported network type {}".format(network_params["type"]))
        exit(1)

    #replace with new network (note that this is pretty hacky)
    res["torch_coordinator"].nodes["cvar_maxent_irl_costmap"].network = res["bev_network"]

    #also for now just grab the fpv net
    res["fpv_network"] = res["torch_coordinator"].nodes['image_featurizer'].image_pipeline.blocks[-1].net

    # setup network opt
    netopt_params = experiment_dict["netopt"]
    if netopt_params["type"] == "Adam":
        res["netopt"] = {}
        res["netopt"]["bev"] = torch.optim.Adam(
            res["bev_network"].parameters(), **netopt_params["params"]
        )
        res["netopt"]["fpv"] = torch.optim.Adam(
            res["fpv_network"].parameters(), **netopt_params["params"]
        )
    elif netopt_params["type"] == "AdamW":
        res["netopt"] = {}
        res["netopt"]["bev"] = torch.optim.AdamW(
            res["bev_network"].parameters(), **netopt_params["params"]
        )
        res["netopt"]["fpv"] = torch.optim.AdamW(
            res["fpv_network"].parameters(), **netopt_params["params"]
        )
    else:
        print("Unsupported netopt type {}".format(netopt_params["type"]))
        exit(1)

    # setup footprint
    footprint_config = experiment_dict["footprint"]
    res["footprint"] = make_footprint(**footprint_config["params"])

    # setup mpc
    if not skip_mpc:
        solver_params = experiment_dict["solver"]
        if solver_params["type"] == "mpc":
            mpc_config = yaml.safe_load(open(experiment_dict["solver"]["mpc_fp"], "r"))
            # have to make batching params match top-level config
            mpc_config["common"]["B"] = experiment_dict["algo"]["params"]["batch_size"]
            mpc_config["common"]["H"] = res["dataset"].H
            res["trajopt"] = setup_mpc(mpc_config)

        elif solver_params["type"] == "planner":
            planner_config = yaml.safe_load(
                open(experiment_dict["solver"]["planner_fp"], "r")
            )
            res["planner"] = setup_planner(planner_config, device)
    else:
        res["trajopt"] = res["planner"] = torch.zeros(0)

    # setup algo
    algo_params = experiment_dict["algo"]
    if algo_params["type"] == "MPPIIRLSpeedmapsTorchCoordinator":
        res["algo"] = MPPIIRLSpeedmapsTorchCoordinator(
            torch_coordinator=res["torch_coordinator"],
            bev_network=res["bev_network"],
            fpv_network=res["fpv_network"],
            bev_opt=res["netopt"]["bev"],
            fpv_opt=res["netopt"]["fpv"],
            expert_dataset=res["dataset"],
            mppi=res["trajopt"],
            footprint=res["footprint"],
            **algo_params["params"]
        ).to(device)

        assert algo_params["params"]["batch_size"] == 1, "Only support batch size 1 for coordinator train for now"

    # setup experiment
    experiment_params = experiment_dict["experiment"]
    res["experiment"] = Experiment(
        algo=res["algo"], params=res["params"], **experiment_params
    ).to(device)

    return res


# TEST
if __name__ == "__main__":
    fp = "../../../config/training/test.yaml"
    res = setup_experiment(fp)

    print(
        {
            k: v.shape if isinstance(v, torch.Tensor) else v
            for k, v in res["dataset"][1].items()
        }
    )

    #    for i in range(10):
    #        res['dataset'].visualize()
    #        plt.show()

    res["experiment"].run()
