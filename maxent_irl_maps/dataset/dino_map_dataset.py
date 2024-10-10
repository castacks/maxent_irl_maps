import os
import cv2
import yaml
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from maxent_irl_maps.os_utils import walk_bags


class DinoMapDataset(Dataset):
    """
    Dataset for geometric + dino maps
    """

    def __init__(self, fp, positional_n=8, dino_n=-1, device="cpu"):
        """
        Load fps and feature normalizations
        Args:
            fp: The location of the dataset
            positional_embedding: set to true to return location of each cell in map
        """
        super(DinoMapDataset, self).__init__()
        self.fp = fp
        self.dfps = self.get_data_fps()
        self.device = device

        # load metadata (have to hack origin to be in robot fram
        self.metadata = yaml.safe_load(
            open(os.path.join(self.fp, self.dfps[0], "map_metadata.yaml"), "r")
        )
        pose = np.load(os.path.join(self.fp, self.dfps[0], "traj.npy"))[0, :2]
        self.metadata["origin"] -= pose
        self.positional_n = positional_n
        self.positional_embedding = self.get_positional_embedding()

        self.dino_n = np.load(
            os.path.join(self.fp, self.dfps[0], "dino_map_data.npy")
        ).shape[0]
        self.geom_n = np.load(
            os.path.join(self.fp, self.dfps[0], "geometric_map_data.npy")
        ).shape[0]
        geom_fks = yaml.safe_load(
            open(os.path.join(self.fp, self.dfps[0], "map_metadata.yaml"))
        )["feature_keys"]

        if self.should_preprocess():
            print("recomputing feature mean/std...")
            self.feature_keys = (
                geom_fks
                + ["dino_{}".format(i) for i in range(self.dino_n)]
                + ["pe_{}".format(i) for i in range(2 * self.positional_n)]
            )
            self.preprocess()

        self.normalizations = torch.load(
            os.path.join(self.fp, "normalizations.pt"), map_location=self.device
        )

        self.dino_n = dino_n

        self.nfeats = (
            self.normalizations["geom_mean"].shape[0]
            + self.dino_n
            + self.positional_embedding.shape[0]
        )

        self.feature_keys = (
            geom_fks
            + ["dino_{}".format(i) for i in range(self.dino_n)]
            + ["pe_{}".format(i) for i in range(2 * self.positional_n)]
        )

        # import the camera info so we can project paths into the image frame
        self.config = yaml.safe_load(open(os.path.join(self.fp, "config.yaml"), "r"))

    def __getitem__(self, idx):
        return self.get_normalized_dpt(idx)

    def __len__(self):
        return len(self.dfps)

    def get_data_fps(self):
        """
        Get locations of all datapoints (assume dir of dirs)
        """
        fps = []
        for fp1 in os.listdir(self.fp):
            _fp = os.path.join(self.fp, fp1)
            if os.path.isdir(_fp):
                subfps = os.listdir(os.path.join(_fp, "data"))
                for sfp in subfps:
                    fps.append(os.path.join(_fp, "data", sfp))
        return sorted(fps)

    def should_preprocess(self):
        """
        Look to see if normalization stuff has been generated
        """
        return not os.path.exists(os.path.join(self.fp, "normalizations.pt"))

    def preprocess(self):
        """
        Compute normalizations for features
        """
        geom_feats = []
        dino_feats = []

        # just going to do a rough estimate bc im lazy
        for i in tqdm.tqdm(range(min(len(self), 500))):
            dpt = self.get_unnormalized_dpt(i)

            gmap = dpt["map_features"][: self.geom_n]
            dmap = dpt["map_features"][self.geom_n : self.geom_n + self.dino_n]

            gfeats = gmap.reshape(gmap.shape[0], -1)
            dfeats = dmap.reshape(dmap.shape[0], -1)

            geom_feats.append(gfeats)
            dino_feats.append(dfeats)

        geom_feats = torch.cat(geom_feats, axis=-1)
        dino_feats = torch.cat(dino_feats, axis=-1)

        self.normalizations = {
            "geom_mean": geom_feats.mean(axis=-1),
            "geom_std": geom_feats.std(axis=-1),
            "dino_mean": dino_feats.mean(axis=-1),
            "dino_std": dino_feats.std(axis=-1),
        }

        torch.save(
            {k: v.cpu() for k, v in self.normalizations.items()},
            os.path.join(self.fp, "normalizations.pt"),
        )

    def get_positional_embedding(self):
        """
        Precompute positional embedding of distance to center (can just do with metadata)
        """
        ox = self.metadata["origin"][0].item()
        oy = self.metadata["origin"][1].item()
        lx = self.metadata["length_x"]
        ly = self.metadata["length_y"]
        res = self.metadata["resolution"]
        nx = round(lx / res)
        ny = round(ly / res)

        xs = ox + torch.arange(nx, device=self.device) * res
        ys = oy + torch.arange(ny, device=self.device) * res

        xs, ys = torch.meshgrid(xs, ys, indexing="ij")
        ds = torch.hypot(xs, ys)
        ds = (ds / ds.max()).view(1, nx, ny)

        fft_as = torch.randn(self.positional_n, device=self.device).view(
            self.positional_n, 1, 1
        )
        fft_bs = torch.randn(self.positional_n, device=self.device).view(
            self.positional_n, 1, 1
        )

        fft_feats1 = (2 * np.pi * fft_as * ds + fft_bs).cos()
        fft_feats2 = (2 * np.pi * fft_as * ds + fft_bs).sin()

        fft_feats = torch.cat([fft_feats1, fft_feats2], dim=0)

        return fft_feats

    def get_unnormalized_dpt(self, idx=-1):
        """
        get datapoint with raw map features
        """
        if idx == -1:
            idx = np.random.randint(len(self))

        fp = os.path.join(self.fp, self.dfps[idx])

        image = cv2.imread(os.path.join(fp, "image.png")) / 255.0
        dino_map_data = torch.tensor(
            np.load(os.path.join(fp, "dino_map_data.npy"))[: self.dino_n],
            device=self.device,
            dtype=torch.float32,
        )
        geom_map_data = torch.tensor(
            np.load(os.path.join(fp, "geometric_map_data.npy")),
            device=self.device,
            dtype=torch.float32,
        )

        # eww TODO cleanup this and metadata later
        traj = torch.tensor(
            np.load(os.path.join(fp, "traj.npy")),
            device=self.device,
            dtype=torch.float32,
        )
        #        fake_vels = torch.zeros(traj.shape[0], 6, device=self.device, dtype=torch.float32)
        #        traj = torch.cat([traj, fake_vels], dim=-1)

        metadata = yaml.safe_load(open(os.path.join(fp, "map_metadata.yaml"), "r"))
        metadata["origin"] = torch.tensor(metadata["origin"], device=self.device)
        metadata["length_x"] = torch.tensor(
            metadata["length_x"], device=self.device, dtype=torch.float32
        )
        metadata["length_y"] = torch.tensor(
            metadata["length_y"], device=self.device, dtype=torch.float32
        )
        metadata["height"] = torch.tensor(
            metadata["length_x"], device=self.device, dtype=torch.float32
        )
        metadata["width"] = torch.tensor(
            metadata["length_y"], device=self.device, dtype=torch.float32
        )
        metadata["resolution"] = torch.tensor(
            metadata["resolution"], device=self.device, dtype=torch.float32
        )
        feature_keys = metadata.pop("feature_keys")

        positional_embedding = torch.clone(self.positional_embedding)

        # need to reorganize keys/values
        map_features = torch.cat(
            [geom_map_data, dino_map_data[: self.dino_n], positional_embedding], dim=0
        )

        steer = torch.zeros(traj.shape[0], 1, device=self.device)

        # irl indexing is transposed
        return {
            "map_features": map_features.permute(0, 2, 1),
            "metadata": metadata,
            "steer": steer,
            "traj": traj,
            "image": torch.tensor(
                image, dtype=torch.float32, device=self.device
            ).permute(2, 0, 1)[[2, 1, 0]],
            "feature_keys": self.feature_keys,
        }

    def get_normalized_dpt(self, idx=-1):
        """
        get datapoint with normalized map features
        """
        gmu = self.normalizations["geom_mean"].view(-1, 1, 1)
        gsig = self.normalizations["geom_std"].view(-1, 1, 1)
        dmu = self.normalizations["dino_mean"][: self.dino_n].view(-1, 1, 1)
        dsig = self.normalizations["dino_std"][: self.dino_n].view(-1, 1, 1)
        pmu = torch.zeros(2 * self.positional_n, device=self.device).view(-1, 1, 1)
        psig = torch.ones(2 * self.positional_n, device=self.device).view(-1, 1, 1)

        _mu = torch.cat([gmu, dmu, pmu], dim=0)
        _sig = torch.cat([gsig, dsig, psig], dim=0)

        udpt = self.get_unnormalized_dpt(idx)

        udpt["map_features"] = ((udpt["map_features"] - _mu) / _sig).clip(-10.0, 10.0)

        return udpt

    def visualize(self, dpt, fig=None, axs=None):
        if fig is None or axs is None:
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))

        extent = (
            dpt["metadata"]["origin"][0].item(),
            dpt["metadata"]["origin"][0].item() + dpt["metadata"]["length_x"],
            dpt["metadata"]["origin"][1].item(),
            dpt["metadata"]["origin"][1].item() + dpt["metadata"]["length_y"],
        )

        geom_map = dpt["map_features"][: self.geom_n]
        dino_map = dpt["map_features"][self.geom_n : self.geom_n + self.dino_n]
        pos_map = dpt["map_features"][-2 * self.positional_n :]

        if self.dino_n >= 3:
            dino_viz = dino_map[:3]
            dmin = dino_viz.min(dim=2)[0].min(dim=1)[0].view(3, 1, 1)
            dmax = dino_viz.max(dim=2)[0].max(dim=1)[0].view(3, 1, 1)
            dino_viz = (dino_viz - dmin) / (dmax - dmin)
            axs[2].imshow(
                dino_viz.permute(2, 1, 0).cpu(), origin="lower", extent=extent
            )

        sidx = dpt["feature_keys"].index("step")

        axs[0].imshow(dpt["image"].cpu())

        axs[1].imshow(geom_map[sidx].T.cpu(), origin="lower", extent=extent)

        axs[3].imshow(pos_map.mean(dim=0).T.cpu(), origin="lower", extent=extent)

        axs[1].plot(dpt["traj"][:, 0].cpu(), dpt["traj"][:, 1].cpu(), c="y")

        axs[2].plot(dpt["traj"][:, 0].cpu(), dpt["traj"][:, 1].cpu(), c="y")

        axs[3].plot(dpt["traj"][:, 0].cpu(), dpt["traj"][:, 1].cpu(), c="y")

        axs[0].set_title("FPV")
        axs[1].set_title("Geom")
        axs[2].set_title("Dino")
        axs[3].set_title("PE")

        return fig, axs

    def to(self, device):
        self.device = device
        self.normalizations = {k: v.to(device) for k, v in self.normalizations.items()}
        self.positional_embedding = self.positional_embedding.to(device)
        return self


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    fp = "/home/physics_atv/workspace/datasets/yamaha_preproc_dino/train"

    dataset = DinoMapDataset(fp)

    dl = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dl))

    for i in range(10):
        dpt = dataset.get_normalized_dpt()
        dataset.visualize(dpt)
        plt.show()
