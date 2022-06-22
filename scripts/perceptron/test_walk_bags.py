import argparse

from maxent_irl_costmaps.os_utils import walk_bags

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, required=True, help='list of bags to walk')
    parser.add_argument('--save_to', type=str, required=False, default='bags.txt', help='file to dump output to')
    args = parser.parse_args()

    res = walk_bags(args.bag_dir)
    print(res)
    with open(args.save_to, 'w') as fp:
        fp.write(str(res))
