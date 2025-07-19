import os
import yaml
import numpy as np
from omegaconf import OmegaConf

from config import CODE_ROOT
# from config import args
from loader.loader_eds import vis_subseq
from util.data import get_gt_corners
from loader.loader_ec import ECSubseq
from script.benchmark import benchmark, benchmark_pl
from src.evaluate_eds import CornerConfig, evaluate_seq


def evaluate_ec(model, evaluate_name):
    model.eval()

    os.system(f'mkdir -p output/evaluate_result/{evaluate_name}')

    EC_DATASETS = [
        "shapes_translation_8_88",
        "shapes_rotation_165_245",
        "shapes_6dof_485_565",
        "boxes_translation_330_410",
        "boxes_rotation_198_278",
    ]
    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    for seq_name in EC_DATASETS:
        dataset = ECSubseq(
            f'{args.eval_dataset_path}/ec_subseq',
            seq_name,
            -1,
            args.patch_size,
            args.event_representation,
            0.01,
            corner_config,
        )

        gt_features_path = f'{args.eval_dataset_path}/gt_tracks/{seq_name}.gt.txt'
        gt_start_corners = get_gt_corners(gt_features_path)

        dataset.override_keypoints(gt_start_corners)

        evaluate_seq(model, dataset, 0.2, seq_name, args.visualize, evaluate_name)

    metric = benchmark(evaluate_name, EC_DATASETS)
    return metric


def evaluate_ec_pl(event_model, rgb_model, evaluate_name, use_kalman=False, cfg=None, seqs=None, result_path=None):
    if event_model:
        event_model.eval()
    if rgb_model and rgb_model.name != 'klt':
        rgb_model.eval()

    if not cfg:
        with open(f'{CODE_ROOT}/configs/eval_real_defaults.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = OmegaConf.create(cfg)
        OmegaConf.set_struct(cfg, True)

    os.system(f'mkdir -p {CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}')
    if result_path:
        os.system(f'mkdir -p {CODE_ROOT}/{os.path.dirname(result_path)}')

    if seqs:
        EC_DATASETS = seqs
    else:
        EC_DATASETS = [
            "shapes_translation_8_88",
            "shapes_rotation_165_245",
            "shapes_6dof_485_565",
            "boxes_translation_330_410",
            "boxes_rotation_198_278",
        ]
    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    evt_runtime = []
    rgb_runtime = []

    for seq_name in EC_DATASETS:
        dataset = ECSubseq(
            f'{cfg.eval_dataset_path}/ec_subseq',
            seq_name,
            -1,
            cfg.patch_size,
            cfg.representation,
            0.01,
            corner_config,
            image_folder=cfg.image_folder,
            event_folder=cfg.event_folder,
        )

        gt_features_path = f'{cfg.eval_dataset_path}/gt_tracks/{seq_name}.gt.txt'
        gt_start_corners = get_gt_corners(gt_features_path)
        if cfg.one_track:
            gt_start_corners = gt_start_corners[:1]
        dataset.override_keypoints(gt_start_corners)

        # vis_subseq(dataset)

        time_metric = evaluate_seq(event_model, rgb_model, dataset, cfg.dt_track_vis, seq_name, cfg.visualize, evaluate_name, 
                                   use_kalman=use_kalman, result_path=result_path, rgb_fps=cfg.rgb_fps)
        if event_model:
            evt_runtime.append(time_metric['evt_runtime'])
        if rgb_model:
            rgb_runtime.append(time_metric['rgb_runtime'])

    metric = {}
    if not cfg.one_track:
        metric = benchmark_pl(evaluate_name, EC_DATASETS, result_path=result_path)

    if event_model:
        print(f'Evt Runtime: {np.array(evt_runtime).mean()}')
    if rgb_model:
        print(f'Rgb Runtime: {np.array(rgb_runtime).mean()}')

    return metric