import tqdm
import torch
import argparse
import matplotlib.pyplot as plt

from maxent_irl_costmaps.os_utils import walk_bags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='path to dataset dir to fix')
    args = parser.parse_args()

    fps = walk_bags(args.root_dir, extension='.pt')

    for fp in tqdm.tqdm(fps):
        res = torch.load(fp)

        gdata = res['gridmap_data']
        gdata2 = []

        fks = res['gridmap_feature_keys']

        for fi, fk in enumerate(fks):
            if 'VLAD' in fk or 'ganav' in fk:
                gdata2.append(gdata[fi].T)
            else:
                gdata2.append(gdata[fi])

        gdata2 = torch.stack(gdata2, dim=0)

        res['gridmap_data'] = gdata2
        torch.save(res, fp)

#        #debug
#        fig, axs = plt.subplots(1, 4)
#        axs[0].imshow(gdata2[fks.index('height_high')])
#        axs[1].imshow(gdata2[fks.index('dino_0')])
#        axs[2].imshow(gdata2[fks.index('VLAD_0')])
#        axs[3].imshow(gdata2[fks.index('ganav_2')])
#        plt.show()
