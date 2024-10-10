"""
Make a costmap of the entirety of gascola

Note that there are some implementation limitations at the moment:
    1. We currently take in gridmap features and expect a cost function as a member of this class
    2. There may be smarter ways to handle costmaps overlapping, but for now I am just assigning a mean/stddev that is weighted by the distance to the cell

Also note that for use in downstream controls testing, we also need:
    1. All visited states (i.e. GPS pose)
    2. height low, height high
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from maxent_irl_maps.preprocess import load_data
from maxent_irl_maps.utils import quat_to_yaw
from maxent_irl_maps.os_utils import walk_bags


class GlobalCostmap:
    def __init__(
        self,
        config_fp,
        model_fp,
        bag_dir=None,
        odom_topic="/integrated_to_init",
        gps_topic="/odometry/filtered_odom",
        map_features_topic="/local_gridmap",
        dt=0.05,
        device="cpu",
    ):
        """
        Args:
            config_fp: yaml file containing metadata for the state visitation map
            bag_dir: The dir to parse for rosbags
            resolution: The discretization of the map
            dt: The dt for the trajectory
            speed_bins: Allow for binning of speeds in the dataset. Don't do speed bins if None, else expect a set if bin lower edges
        """
        self.config_fp = config_fp
        self.base_fp = bag_dir
        self.model_fp = model_fp
        self.odom_topic = odom_topic
        self.gps_topic = gps_topic
        self.map_features_topic = map_features_topic
        self.dt = dt

        config_data = yaml.safe_load(open(self.config_fp, "r"))
        self.metadata = {
            "origin": torch.tensor(
                [config_data["origin"]["x"], config_data["origin"]["y"]]
            ),
            "length_x": config_data["length_x"],
            "length_y": config_data["length_y"],
            "resolution": config_data["resolution"],
        }

        self.traj_fps = []

        self.model = torch.load(model_fp)

        self.initialize(bag_dir=bag_dir)
        self.device = device

    def to(self, device):
        self.device = device
        self.state_visitations = self.state_visitations.to(device)
        self.metadata["origin"] = self.metadata["origin"].to(device)
        return self

    def initialize(self, bag_dir=None):
        """
        Set map bounds and occgrid
        Args:
            bag_dir: optional directory to walk for bags.
        """
        nx = int(self.metadata["length_x"] / self.metadata["resolution"])
        ny = int(self.metadata["length_y"] / self.metadata["resolution"])

        self.state_visitations = torch.zeros(nx, ny).long()

        # for now, keep track of the closest we've been to a cell and use that as cost
        # TODO: think about a weighted Welford or something
        self.costmap_mean = torch.zeros(nx, ny)
        self.minimum_distances = torch.ones(nx, ny) * 1e8
        self.unknowns = torch.ones(nx, ny).bool()
        self.height_low = torch.ones(nx, ny) * 1e8 * 0.0
        self.height_high = torch.ones(nx, ny) * 1e8 * 0.0
        self.states = torch.zeros(0, 13)

        if bag_dir is not None:
            traj_fps = np.array(walk_bags(bag_dir))
            for i, tfp in enumerate(traj_fps):
                print(
                    "{}/{} ({})".format(i + 1, len(traj_fps), os.path.basename(tfp)),
                    end="\r",
                )
                res, traj = self.check_trajectory(tfp)
                if res:
                    self.insert_traj(traj)

    def check_trajectory(self, tfp):
        """
        Run a few sanity checks on a trajectory and retun whether it is valid
        NOTE: I'm maintaining a list of all the tfps that get passed through this.
            This isn't necessarily great, as checking =/= inserting in all cases
        Args:
            tfp: path to the trajectory rosbag to check
        Returns:
            tuple (check, traj) where check is True iff. the trajectory is valid and traj is the [T x 13] trajectory if valid, else None
        """
        xmin = self.metadata["origin"][0].item()
        xmax = xmin + self.metadata["length_x"]
        ymin = self.metadata["origin"][1].item()
        ymax = ymin + self.metadata["length_y"]

        try:
            traj, feature_keys = load_data(
                bag_fp=tfp,
                map_features_topic=self.map_features_topic,
                odom_topic=self.odom_topic,
                gps_topic=self.gps_topic,
                image_topic="/multisense/left/image_rect_color",
                horizon=10,
                dt=self.dt,
                fill_value=0.0,
            )

            # There's a bit of a data mis-match here. Traj is represented as a bunch of snippets. Need to condense into a single arr
            traj_new = {}
            for k, v in traj.items():
                if k == "metadata":
                    traj_new[k] = v
                elif k in ["traj", "gps_traj", "cmd", "steer"]:
                    traj_new[k] = torch.stack([vv[0] for vv in v], dim=0)
                else:
                    traj_new[k] = torch.stack(v, dim=0)

            traj = traj_new
            traj["feature_keys"] = feature_keys

        except:
            print("Couldn't load {}, skipping...".format(tfp))
            return False, None

        # check for gps jumps
        diffs = np.linalg.norm(
            traj["gps_traj"][1:, :3] - traj["gps_traj"][:-1, :3], axis=-1
        )
        if any(diffs > self.dt * 50.0):
            print("Jump in ({}) > 50m/s. skipping...".format(tfp))
            return False, None

        # check in map bounds
        elif traj["gps_traj"][:, 0].min() < xmin:
            print(
                "Traj {} x {:.2f} less than xmin {:.2f}, skipping...".format(
                    tfp, traj[:, 0].min(), xmin
                )
            )
            return False, None

        elif traj["gps_traj"][:, 0].max() > xmax:
            print(
                "Traj {} x {:.2f} more than xmax {:.2f}, skipping...".format(
                    tfp, traj[:, 0].max(), xmax
                )
            )
            return False, None

        elif traj["gps_traj"][:, 1].min() < ymin:
            print(
                "Traj {} y {:.2f} less than ymin {:.2f}, skipping...".format(
                    tfp, traj[:, 1].min(), ymin
                )
            )
            return False, None

        elif traj["gps_traj"][:, 1].max() > ymax:
            print(
                "Traj {} y {:.2f} more than ymax {:.2f}, skipping...".format(
                    tfp, traj[:, 1].max(), ymax
                )
            )
            return False, None

        else:
            self.traj_fps.append(tfp)
            return True, traj

    def add_dataset(self, dataset):
        """
        Args:
            dataset: the dataset to add. Expects the following keys:
                traj: odom from super odometry
                gps_traj: traj from gps
                map_features: tesor of map features in the same frame as traj
        """
        print("adding dataset..." + " " * 30)
        xmin = self.metadata["origin"][0].item()
        xmax = xmin + self.metadata["length_x"]
        ymin = self.metadata["origin"][1].item()
        ymax = ymin + self.metadata["length_y"]
        nx, ny = self.state_visitations.shape
        for i, batch in enumerate(dataset):
            if i + 1 == len(dataset):
                break

            print("{}/{}".format(i + 1, len(dataset)), end="\r")
            map_features = batch["map_features"]
            gps_traj = batch["gps_traj"]
            odom_traj = batch["traj"]
            metadata = batch["metadata"]
            img = batch["image"]

            # compute a mask of unknown cells
            unk_idx = batch["feature_keys"].index("unknown")

            # note that height values from the dataset are both relative and normalized
            height_low_idx = batch["feature_keys"].index("height_low")
            height_high_idx = batch["feature_keys"].index("height_high")

            # have to do this to get around the dataset normalization
            unk_mask = map_features[unk_idx] < map_features[unk_idx].mean()

            gps_pose = gps_traj[0]
            odom_pose = odom_traj[0]

            # filter out stationary
            if torch.linalg.norm(gps_pose[7:10]) < 1.0:
                continue

            # generate the occgrid
            ix = int((gps_pose[0] - xmin) / self.metadata["resolution"])
            iy = int((gps_pose[1] - ymin) / self.metadata["resolution"])

            # add states and state visitations
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                self.state_visitations[ix, iy] += 1
                self.states = torch.cat(
                    [self.states, gps_pose.cpu().unsqueeze(0)], dim=0
                )

            # update the costmap
            # first run the net to get costmap
            with torch.no_grad():
                self.model.network.eval()
                if hasattr(self.model.network, "ensemble_forward"):
                    res = self.model.network.ensemble_forward(map_features.unsqueeze(0))
                    costmap = res["costmap"].mean(dim=1).squeeze()
                else:
                    res = self.model.network.forward(map_features.unsqueeze(0))
                    costmap = res["costmap"].squeeze()

            # get heightmap
            height_low = map_features[height_low_idx]
            height_low_mean = self.model.expert_dataset.feature_mean[height_low_idx]
            height_low_std = self.model.expert_dataset.feature_std[height_low_idx]
            height_high = map_features[height_high_idx]
            height_high_mean = self.model.expert_dataset.feature_mean[height_high_idx]
            height_high_std = self.model.expert_dataset.feature_std[height_high_idx]
            ego_height = gps_pose[2]

            # need to de-relative/unnormalize
            height_low = height_low * height_low_std + height_low_mean + ego_height
            height_high = height_high * height_high_std + height_high_mean + ego_height

            # next, calculate cell locations in gps frame
            # note that costmap is in odom frame, and needs to move to gps frame.
            # do this by calculating the rotation offset and transforming in 2d
            pose_rot = quat_to_yaw(gps_pose[3:7]) - quat_to_yaw(odom_pose[3:7])

            # simplest to get vehicle offset, snap to gps frame, then rotate
            map_dxs = torch.arange(
                metadata["origin"][0] - odom_pose[0],
                metadata["origin"][0] - odom_pose[0] + metadata["height"],
                metadata["resolution"],
            )

            map_dys = torch.arange(
                metadata["origin"][1] - odom_pose[1],
                metadata["origin"][1] - odom_pose[1] + metadata["width"],
                metadata["resolution"],
            )

            map_poses = torch.stack(
                torch.meshgrid([map_dxs, map_dys], indexing="ij"), dim=-1
            ).to(
                gps_pose.device
            )  # W x H x 2
            R = torch.tensor(
                [[pose_rot.cos(), -pose_rot.sin()], [pose_rot.sin(), pose_rot.cos()]]
            ).to(gps_pose.device)

            gps_map_poses = (
                gps_pose[:2].view(1, 1, 2, 1)
                + torch.matmul(R.view(1, 1, 2, 2), map_poses.unsqueeze(-1))
            ).squeeze(
                -1
            )  # [W x H x 2]

            # at this point, we have a set of costs/positions. process into map idxs and update cells

            costmap_raster = costmap.T.flatten()
            unknown_raster = unk_mask.T.flatten()
            height_low_raster = height_low.T.flatten()
            height_high_raster = height_high.T.flatten()
            px_raster = gps_map_poses[..., 0].flatten()
            py_raster = gps_map_poses[..., 1].flatten()

            ixs = (
                (px_raster - self.metadata["origin"][0]) / self.metadata["resolution"]
            ).long()
            iys = (
                (py_raster - self.metadata["origin"][1]) / self.metadata["resolution"]
            ).long()

            in_bounds_mask = (ixs >= 0) & (ixs < nx) & (iys >= 0) & (iys < ny)

            costmap_raster = costmap_raster[in_bounds_mask]
            unknown_raster = unknown_raster[in_bounds_mask]
            height_low_raster = height_low_raster[in_bounds_mask]
            height_high_raster = height_high_raster[in_bounds_mask]
            ixs = ixs[in_bounds_mask]
            iys = iys[in_bounds_mask]
            px_raster = px_raster[in_bounds_mask]
            py_raster = py_raster[in_bounds_mask]

            dists = torch.linalg.norm(
                gps_pose[:2].view(1, 2) - torch.stack([px_raster, py_raster], dim=-1),
                dim=-1,
            )

            prev_unknowns = self.unknowns[ixs, iys].to(costmap_raster.device)
            prev_dists = self.minimum_distances[ixs, iys].to(costmap_raster.device)

            # we want to update cells in the following cases:
            # 1. the query is known, and is closer
            # 2. the query is unknown, but is closer for an unexplored cell
            # i.e. if we don't have a query yet, or we have a fill value, update if the new update is closer
            fill_mask = (~unknown_raster | prev_unknowns) & (dists < prev_dists)

            ixs = ixs[fill_mask]
            iys = iys[fill_mask]
            dists = dists[fill_mask]
            costmap_raster = costmap_raster[fill_mask]
            unknown_raster = unknown_raster[fill_mask]
            height_low_raster = height_low_raster[fill_mask]
            height_high_raster = height_high_raster[fill_mask]

            self.minimum_distances[ixs, iys] = dists.cpu()
            self.costmap_mean[ixs, iys] = costmap_raster.cpu()
            self.unknowns[ixs, iys] = unknown_raster.cpu()
            self.height_low[ixs, iys] = height_low_raster.cpu()
            self.height_high[ixs, iys] = height_high_raster.cpu()

            # TODO: there is a known issue where if multiple costmap cells query into a single global costmap cell, torch will take the last one. Correct behavior would be to take the closest

            # debug
            if i % 10000 == 0 or (i + 2 == len(dataset)):
                hvmin = torch.quantile(self.height_low[~self.unknowns], 0.01)
                hvmax = torch.quantile(self.height_high[~self.unknowns], 0.99)

                fig, axs = plt.subplots(2, 5, figsize=(30, 12))
                axs = axs.flatten()
                axs[0].imshow(self.state_visitations)
                axs[1].imshow(self.costmap_mean, cmap="plasma")
                axs[2].imshow(self.minimum_distances.clamp(0.0, 100.0), cmap="plasma")
                axs[3].imshow(costmap.cpu(), cmap="plasma")
                axs[4].imshow(self.unknowns)
                axs[5].imshow(img.permute(1, 2, 0)[..., [2, 1, 0]].cpu())
                axs[6].imshow(self.height_low, vmin=hvmin, vmax=hvmax)
                axs[7].imshow(self.height_high, vmin=hvmin, vmax=hvmax)
                axs[8].imshow(height_low.cpu(), vmin=hvmin, vmax=hvmax)
                axs[9].imshow(height_high.cpu(), vmin=hvmin, vmax=hvmax)

                axs[0].set_title("state visitations")
                axs[1].set_title("global costmap")
                axs[2].set_title("cell distances")
                axs[3].set_title("local costmap")
                axs[4].set_title("unknowns")
                axs[5].set_title("FPV")
                axs[6].set_title("height low")
                axs[7].set_title("height high")
                axs[8].set_title("local height low")
                axs[9].set_title("local height high")

                plt.show()

    def get_pixel_coordinates(self, poses, crop_params, local=True):
        """
        Args:
            poses: the (batched) pose to get the crop from [B x {x, y, th}]
            crop_params: the metadata of the crop to up/downsample to
            local: bool for whether to use the local or global rotation
                note that local  -> rotated to align with current pose x-forward
                          global -> rotated to align with pose base frame
        Returns:
            pxlist: list of pixel coordinates for the corresponding poses, crop_params
            invalid_mask: mask for which pixels went out of bounds
        """
        poses = poses.double()

        if not local:
            poses[:, 2] = 0

        crop_xs = (
            torch.arange(
                crop_params["origin"][0],
                crop_params["origin"][0] + crop_params["length_x"],
                crop_params["resolution"],
            )
            .double()
            .to(self.device)
        )
        crop_ys = (
            torch.arange(
                crop_params["origin"][1],
                crop_params["origin"][1] + crop_params["length_y"],
                crop_params["resolution"],
            )
            .double()
            .to(self.device)
        )
        crop_positions = torch.stack(
            torch.meshgrid(crop_xs, crop_ys, indexing="ij"), dim=-1
        )  # HxWx2 tensor
        crop_nx = int(crop_params["length_x"] / crop_params["resolution"])
        crop_ny = int(crop_params["length_y"] / crop_params["resolution"])

        translations = poses[
            :, :2
        ]  # Nx2 tensor, where each row corresponds to [x, y] position in metric space
        rotations = torch.stack(
            [
                poses[:, 2].cos(),
                -poses[:, 2].sin(),
                poses[:, 2].sin(),
                poses[:, 2].cos(),
            ],
            dim=-1,
        )  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]

        ## Reshape tensors to perform batch tensor multiplication.

        # The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]

        rotations = rotations.view(-1, 1, 1, 2, 2)  # [B x 1 x 1 x 2 x 2]
        crop_positions = crop_positions.view(
            1, crop_nx, crop_ny, 2, 1
        )  # [1 x H x W x 2 x 1]
        translations = translations.view(-1, 1, 1, 2, 1)  # [B x 1 x 1 x 2 x 1]

        # Apply each transform to all crop positions (res = [B x H x W x 2])
        crop_positions_transformed = (
            torch.matmul(rotations, crop_positions) + translations
        ).squeeze()

        # Obtain actual pixel coordinates
        map_origin = self.metadata["origin"].view(1, 1, 1, 2)
        pixel_coordinates = (
            (crop_positions_transformed - map_origin) / self.metadata["resolution"]
        ).long()

        #        pixel_coordinates_flipped = pixel_coordinates.swapaxes(-2,-3)

        # Obtain maximum and minimum values of map to later filter out of bounds pixels
        map_p_low = torch.tensor([0, 0]).to(self.device).view(1, 1, 1, 2)
        map_p_high = (
            torch.tensor(self.state_visitations.shape).to(self.device).view(1, 1, 1, 2)
        )
        invalid_mask = (pixel_coordinates < map_p_low).any(dim=-1) | (
            pixel_coordinates >= map_p_high
        ).any(dim=-1)

        # Indexing method: set all invalid idxs to a valid one (i.e. 0), index, then mask out the results

        # TODO: Per-channel fill value
        fill_value = 0
        pixel_coordinates[invalid_mask] = 0

        pxlist_flipped = pixel_coordinates.reshape(-1, 2)

        return pxlist_flipped, invalid_mask

    def get_state_visitations(self, poses, crop_params, local=True):
        # [B x C x W x H]
        crop_nx = int(crop_params["length_x"] / crop_params["resolution"])
        crop_ny = int(crop_params["length_y"] / crop_params["resolution"])
        fill_value = 0

        pxlist, invalid_mask = self.get_pixel_coordinates(poses, crop_params, local)
        values = self.state_visitations[pxlist[:, 0], pxlist[:, 1]]
        values = values.view(poses.shape[0], crop_nx, crop_ny)

        k1 = invalid_mask.float()
        values = (1.0 - k1) * values + k1 * fill_value

        value_sums = values.sum(dim=-1, keepdims=True).sum(dim=-2, keepdims=True) + 1e-4
        value_dist = values / value_sums

        # normalize to make a proper distribution
        return value_dist

    def get_costmap(self, poses, crop_params, local=True):
        # [B x C x W x H]
        crop_nx = int(crop_params["length_x"] / crop_params["resolution"])
        crop_ny = int(crop_params["length_y"] / crop_params["resolution"])
        fill_value = 0

        pxlist, invalid_mask = self.get_pixel_coordinates(poses, crop_params, local)
        values = self.costmap_mean[pxlist[:, 0], pxlist[:, 1]]
        values = values.view(poses.shape[0], crop_nx, crop_ny)

        k1 = invalid_mask.float()
        values = (1.0 - k1) * values + k1 * fill_value

        # normalize to make a proper distribution
        return values

    def create_anim(self, traj, save_to, local=True):
        """
        create an animation for viz purposes
        """
        print("making anim..." + " " * 30)

        traj = traj[np.linalg.norm(traj[:, 7:10], axis=-1) > 1.0]
        traj = torch.tensor(traj)
        traj = torch.stack([traj[:, 0], traj[:, 1], quat_to_yaw(traj[:, 3:7])], axis=-1)

        crop_params = {
            "origin": np.array([-40.0, -40.0]),
            "length_x": 80.0,
            "length_y": 80.0,
            "resolution": 0.5,
        }

        xmin = self.metadata["origin"][0]
        ymin = self.metadata["origin"][1]
        xmax = xmin + self.metadata["length_x"]
        ymax = ymin + self.metadata["length_y"]

        cxmin = crop_params["origin"][0]
        cymin = crop_params["origin"][1]
        cxmax = cxmin + crop_params["length_x"]
        cymax = cymin + crop_params["length_y"]

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        #        plt.show(block=False)

        def get_frame(t, fig, axs):
            print(t, end="\r")

            for ax in axs:
                ax.cla()

            svs = self.get_state_visitations(traj[t : t + 5], crop_params, local=local)
            costmap = self.get_costmap(traj[t : t + 5], crop_params, local=local)
            dx = traj[t + 5, 0] - traj[t, 0]
            dy = traj[t + 5, 1] - traj[t, 1]
            n = np.sqrt(dx * dx + dy * dy)
            l = max(self.metadata["length_x"], self.metadata["length_y"]) / 50.0

            axs[0].imshow(
                self.costmap_mean.T,
                origin="lower",
                extent=(xmin, xmax, ymin, ymax),
                cmap="plasma",
            )
            axs[0].arrow(
                traj[t, 0],
                traj[t, 1],
                l * dx / n,
                l * dy / n,
                color="r",
                head_width=l / 2.0,
            )
            axs[1].imshow(
                costmap[0].T,
                origin="lower",
                extent=(cxmin, cxmax, cymin, cymax),
                cmap="plasma",
            )
            axs[1].arrow(-2.0, 0.0, 3.0, 0.0, color="r", head_width=1.0)
            axs[2].imshow(
                svs[0].T,
                origin="lower",
                extent=(cxmin, cxmax, cymin, cymax),
                vmin=0.0,
                vmax=0.01,
            )
            axs[2].arrow(-2.0, 0.0, 3.0, 0.0, color="r", head_width=1.0)

            axs[0].set_title("Global costmap")
            axs[1].set_title("Local map of costmap")
            axs[2].set_title("Local map of global visitations")
            axs[0].set_xlabel("X (UTM)")
            axs[0].set_ylabel("Y (UTM)")
            axs[1].set_xlabel("X (local)")
            axs[1].set_ylabel("Y (local)")
            axs[2].set_xlabel("X (local)")
            axs[2].set_ylabel("Y (local)")

        anim = FuncAnimation(
            fig,
            lambda t: get_frame(t, fig, axs),
            frames=np.arange(traj.shape[0] - 10),
            interval=300 * self.dt,
        )

        #        plt.show()
        anim.save(save_to)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fp", type=str, required=True, help="path to global map config"
    )
    parser.add_argument(
        "--bag_dir",
        type=str,
        required=True,
        help="dir containing GPS for state visitations",
    )
    parser.add_argument(
        "--save_as", type=str, required=True, help="save buffer to this filepath"
    )
    args = parser.parse_args()

    buf = GlobalStateVisitationBuffer(args.config_fp, args.bag_dir)

    save_fp = args.save_as if args.save_as[-3:] == ".pt" else args.save_as + ".pt"

    torch.save(buf, save_fp)

    buf.create_anim(save_to="aaa", local=True)

#    if not os.path.exists('gsv_figs'):
#        os.mkdir('gsv_figs')
#    for i in range(10):
#        buf.create_anim(save_to = 'gsv_figs/gsv_{}.mp4'.format(i+1), local=True)
