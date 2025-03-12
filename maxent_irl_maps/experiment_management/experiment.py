import os
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from maxent_irl_maps.os_utils import maybe_mkdir


class Experiment:
    """
    Wrapper around RL algorithms that drives the training and handles all the IO stuff. (i.e. making directories, saving networks, recording performance, etc.)
    """

    def __init__(
        self,
        algo,
        name,
        params,
        save_to="",
        epochs=10,
        steps_per_epoch=-1,
        save_every=10,
        device="cpu",
    ):
        self.algo = algo
        self.name = "{}_{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), name)
        self.base_fp = os.path.join(os.getcwd(), save_to, self.name)
        self.log_fp = os.path.join(self.base_fp, "_log")
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_every = save_every
        self.device = device
        self.params = params

    def build_experiment_dir(self):
        if os.path.exists(self.base_fp):
            i = input(
                "Directory {} already exists. input 'q' to stop the experiment (and anything else to keep going).".format(
                    self.base_fp
                )
            )
            if i.lower() == "q":
                exit(0)

        maybe_mkdir(self.base_fp, True)
        maybe_mkdir(os.path.join(self.base_fp, "dummy_dataset"), True)

    def run(self):
        for i in range(3):
            self.algo.visualize()
            plt.show()

        for e in range(self.epochs):
            self.algo.update(self.steps_per_epoch)

            if e == 0:
                self.build_experiment_dir()

                #create a sample dataset to load
                with open(os.path.join(self.algo.expert_dataset.root_fp, "normalizations.yaml"), "r") as fp:
                    normalizations = yaml.safe_load(fp)
                    target_fp = open(os.path.join(self.base_fp, "dummy_dataset","normalizations.yaml"), "w")
                    yaml.dump(normalizations, target_fp)

                src_fp = os.path.join(self.algo.expert_dataset.root_fp, self.algo.expert_dataset.dpt_fps[0])
                dst_fp = os.path.join(self.base_fp, "dummy_dataset", "traj_0.pt")
                torch.save(torch.load(src_fp, weights_only=False), dst_fp)
                
                #save params
                with open(os.path.join(self.base_fp, "_params.yaml"), "w") as fp:
                    yaml.dump(self.params, fp, default_flow_style=False)

            if e % self.save_every == 0:
                torch.save(
                    self.algo.network.state_dict(),
                    os.path.join(self.base_fp, "itr_{}.pt".format(e + 1)),
                )
            self.algo = self.algo.to("cuda")

        for i in range(10):
            self.algo.visualize()
            plt.show()

    def to(self, device):
        self.device = device
        return self
