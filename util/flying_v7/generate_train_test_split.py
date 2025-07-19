import argparse
import glob
from pathlib import Path
import random
import json

def cumsum_grt(v, train_ratio=0.9):
    curr_sum = v[0]
    train_num = sum(v) * train_ratio
    i = 1
    for i in range(1, len(v)):
        if curr_sum > train_num:
            break
        curr_sum += v[i]
    if v[i-1] > sum(v)*0.05: i = i-1
    return i

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ade20k_dir', help='in this branch we use ADE20K_2021_17_01')
    parser.add_argument('--shapenet_dir', help='ShapeNetCore.v2')
    args = parser.parse_args()

    train_bg = list(Path(args.ade20k_dir).glob("images/ADE/training/*/*/*.jpg"))
    test_bg = list(Path(args.ade20k_dir).glob("images/ADE/validation/*/*/*.jpg"))

    obj_type = [x for x in Path(args.shapenet_dir).iterdir() if x.is_dir()]
    random.shuffle(obj_type)
    count_list = [len([d for d in t.iterdir() if d.is_dir()]) for t in obj_type]
    train_idx = cumsum_grt(count_list, 0.9)
    train_obj = [d.joinpath('models/model_normalized.obj') for t in obj_type[:train_idx] for d in t.iterdir() if d.is_dir()]
    test_obj = [d.joinpath('models/model_normalized.obj') for t in obj_type[train_idx:] for d in t.iterdir() if d.is_dir()]

    with open('configs/shapenet_train_test_split.json', 'wt') as fshape:
        data = {
            'train': [str(p.relative_to(args.shapenet_dir)) for p in train_obj],
            'test': [str(p.relative_to(args.shapenet_dir)) for p in test_obj]
        }
        json.dump(data, fshape, indent=4)
    print(f'shapenet train/test: {len(data["train"])}/{len(data["test"])}')

    with open('configs/ade20k_train_test_split.json', 'wt') as fade:
        data = {
            'train': [str(p.relative_to(args.ade20k_dir)) for p in train_bg],
            'test': [str(p.relative_to(args.ade20k_dir)) for p in test_bg]
        }
        json.dump(data, fade, indent=4)
    print(f'ade20k train/test: {len(data["train"])}/{len(data["test"])}')
    

if __name__ == '__main__':
    main()