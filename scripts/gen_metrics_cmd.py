import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--exp_dir', type=str, required=True, help='dir of experiments')
    args = parser.parse_args()

    base_cmd = "python3 generate_metrics.py --test_fp ~/workspace/datasets/irl_postpostproc_multimap_highres/test --viz --use_planner --device cuda --save_fp {}/metrics --model_fp {}/itr_5.pt"

    base_cmd2 = "python3 generate_metrics.py --test_fp ~/workspace/datasets/irl_postpostproc_multimap_highres/gascola_loop_eval --viz --use_planner --device cuda --save_fp {}/gascola_loop_metrics --model_fp {}/itr_5.pt"

    efps = os.listdir(args.exp_dir)

    res = ""

    for efp in sorted(efps):
        ebfp = os.path.join(args.exp_dir, efp)
        exp_cmd = base_cmd.format(ebfp, ebfp)
        exp_cmd2 = base_cmd2.format(ebfp, ebfp)

        res = res + exp_cmd + " && " + exp_cmd2 + " && "

    print(res[:-4])
