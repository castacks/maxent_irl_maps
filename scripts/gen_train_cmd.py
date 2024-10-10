import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", type=str, required=True, help="dir of experiments"
    )
    parser.add_argument(
        "--n", type=int, required=False, default=5, help="num of each experiment to run"
    )
    args = parser.parse_args()

    base_cmd = "python3 run_experiment.py --setup_fp {}"

    efps = os.listdir(args.config_dir)

    res = ""

    for efp in sorted(efps):
        ebfp = os.path.join(args.config_dir, efp)
        exp_cmd = base_cmd.format(ebfp)

        res = res + (exp_cmd + " && ")

    res = " && ".join([res[:-4] for _ in range(args.n)])

    print(res)
