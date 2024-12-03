import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="dir of experiments")
    args = parser.parse_args()

    base_cmd = "python3 generate_metrics.py --test_fp ~/workspace/datasets/irl_postpostproc_rr_2/test --mppi_eval_fp ../config/mpc/mppi_eval_config.yaml --viz --device cuda --save_fp {}/rr_metrics_cvar --model_fp {}/itr_5.pt"

    base_cmd2 = "python3 generate_metrics.py --test_fp ~/workspace/datasets/irl_postpostproc_multimap/gascola_loop_eval --mppi_eval_fp ../config/mpc/mppi_eval_config.yaml --viz --device cuda --save_fp {}/gascola_loop_metrics --model_fp {}/itr_5.pt"

    base_cmd3 = "python3 generate_metrics.py --test_fp ~/workspace/datasets/big_irl_dataset_postproc/test --mppi_eval_fp ../config/mpc/mppi_eval_config.yaml --viz --device cuda --save_fp {}/metrics_cvar --model_fp {}/itr_5.pt"

    efps = os.listdir(args.exp_dir)

    res = ""

    for efp in sorted(efps):
        ebfp = os.path.join(args.exp_dir, efp)
        exp_cmd = base_cmd.format(ebfp, ebfp)
        exp_cmd2 = base_cmd2.format(ebfp, ebfp)
        exp_cmd3 = base_cmd3.format(ebfp, ebfp)

        #        res = res + exp_cmd + " && " + exp_cmd2 + " && " + exp_cmd3 + " && "
        res = res + exp_cmd + " && " + exp_cmd3 + " && "

    print(res[:-4])
