"""
Look at the linear weights of a model
"""

import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fp", type=str, required=True, help="path to the model")
    args = parser.parse_args()

    model = torch.load(args.model_fp)

    weights = [w.weight.squeeze().detach() for w in model.network.cost_heads]
    weights = torch.stack(weights, dim=0)

    fks = model.expert_dataset.feature_keys

    wmean = weights.mean(dim=0)
    wstd = weights.std(dim=0)
    wzs = torch.abs(wmean / wstd)
    zidxs = torch.argsort(wzs, descending=True)

    for i in zidxs:
        fk = fks[i]
        ws = weights[:, i]
        print(
            "{:<20}:\t{:.2f} += {:.2f} (z+{:.2f})".format(
                fk, ws.mean(), ws.std(), wzs[i]
            )
        )
