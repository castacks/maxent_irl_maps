"""
Really simple script for initializing a global state visitation buffer
"""

import os
import argparse
import torch

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True, help='path to config to initialize buffer')
    parser.add_argument('--save_as', type=str, required=True, help='path to location to save buffer')
    args = parser.parse_args()

    buf = GlobalStateVisitationBuffer(args.config_fp)

    save_fp = args.save_as if args.save_as[-3:] == '.pt' else args.save_as + '.pt'
    torch.save(buf, save_fp)
