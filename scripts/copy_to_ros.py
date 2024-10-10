import os
import yaml
import torch
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, required=True, help="path to expeiment to use for ros"
    )
    parser.add_argument(
        "--dst_dir", type=str, required=True, help="path to save model to"
    )
    args = parser.parse_args()

    ## TODO: copy all the network files, and make a new copy of the param file that points to placeholder
    exp_name = os.path.basename(str(args.src_dir).strip("/"))

    try:
        shutil.copytree(args.src_dir, os.path.join(args.dst_dir, exp_name))
    except:
        print("Model already exists. Delete if network files need updating")

    param_fp = os.path.join(os.path.split(args.src_dir)[0], "_params.yaml")

    params = yaml.safe_load(open(param_fp, "r"))

    # update dataset
    os.makedirs(os.path.join(args.dst_dir, exp_name, "dataset"), exist_ok=True)
    shutil.copy(
        os.path.join(params["dataset"]["params"]["root_fp"], "normalizations.yaml"),
        os.path.join(args.dst_dir, exp_name, "dataset", "normalizations.yaml"),
    )
    params["dataset"]["type"] = "PlaceHolderDataset"
    params["dataset"]["params"]["root_fp"] = os.path.join(
        args.dst_dir, exp_name, "dataset"
    )

    # copy planner params and normalizations
    shutil.copy(
        params["solver"]["planner_fp"],
        os.path.join(args.dst_dir, exp_name, "planner_config.yaml"),
    )
    params["solver"]["planner_fp"] = os.path.join(
        args.dst_dir, exp_name, "_planner_config.yaml"
    )

    yaml.dump(
        params, open(os.path.join(args.dst_dir, exp_name, "_run_params.yaml"), "w")
    )
