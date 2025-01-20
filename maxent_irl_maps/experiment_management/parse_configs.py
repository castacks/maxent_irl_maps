"""
Script for converting experiment yamls into the actual objects to run experiments with
"""
import argparse
import yaml
import torch
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.skid_steer import SkidSteer

from torch_mpc.setup_mpc import setup_mpc

# TODO uncomment when torch_state_lattice_planner is back
# from torch_state_lattice_planner.setup_planner import setup_planner

from maxent_irl_maps.algos.mppi_irl_speedmaps import MPPIIRLSpeedmaps
from maxent_irl_maps.algos.planner_irl_speedmaps import PlannerIRLSpeedmaps

from maxent_irl_maps.geometry_utils import make_footprint

from maxent_irl_maps.networks.resnet import ResnetCategorical

from maxent_irl_maps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from maxent_irl_maps.experiment_management.experiment import Experiment


def setup_experiment(fp, skip_mpc=False):
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
    if dataset_params["type"] == "MaxEntIRLDataset":
        res["dataset"] = MaxEntIRLDataset(**dataset_params["params"]).to(device)
    else:
        print("Unsupported dataset type {}".format(dataset_params["type"]))
        exit(1)

    # setup network
    network_params = experiment_dict["network"]
    network_params["params"]["device"] = device
    if network_params["type"] == "ResnetCategorical":
        res["network"] = ResnetCategorical(
            in_channels=len(res["dataset"].feature_keys), **network_params["params"]
        ).to(device)

    else:
        print("Unsupported network type {}".format(network_params["type"]))
        exit(1)

    # setup network opt
    netopt_params = experiment_dict["netopt"]
    if netopt_params["type"] == "Adam":
        res["netopt"] = torch.optim.Adam(
            res["network"].parameters(), **netopt_params["params"]
        )
    elif netopt_params["type"] == "AdamW":
        res["netopt"] = torch.optim.AdamW(
            res["network"].parameters(), **netopt_params["params"]
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
            mpc_config["common"]["H"] = res["dataset"][0]["traj"].shape[0]
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
    if algo_params["type"] == "MPPIIRL":
        res["algo"] = MPPIIRL(
            network=res["network"],
            opt=res["netopt"],
            expert_dataset=res["dataset"],
            mppi=res["trajopt"],
            **algo_params["params"]
        ).to(device)

    elif algo_params["type"] == "MPPIIRLSpeedmaps":
        res["algo"] = MPPIIRLSpeedmaps(
            network=res["network"],
            opt=res["netopt"],
            expert_dataset=res["dataset"],
            mppi=res["trajopt"],
            footprint=res["footprint"],
            **algo_params["params"]
        ).to(device)

    elif algo_params["type"] == "PlannerIRLSpeedmaps":
        res["algo"] = PlannerIRLSpeedmaps(
            network=res["network"],
            opt=res["netopt"],
            expert_dataset=res["dataset"],
            planner=res["planner"],
            footprint=res["footprint"],
            **algo_params["params"]
        ).to(device)

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
