import os
import yaml
import numpy as np
from omegaconf import OmegaConf

# from config import args
from config import CODE_ROOT
from util.data import get_gt_corners
from script.benchmark import benchmark, benchmark_pl
from src.evaluate_eds import CornerConfig, evaluate_seq
from loader.loader_eds import vis_subseq
from loader.loader_ec_vis import ECSubseq_vis
from loader.loader_eds_vis import EDSSubseq_vis
from loader.loader_dsec_vis import DSECSubseq_vis


def evaluate_for_vis_pl(event_model, rgb_model, evaluate_name, use_kalman=False, cfg=None, seqs=None, result_path=None):
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
        DATASETS = seqs
    else:
        if cfg.eval_dataset == 'ec-vis':
            DATASETS = [
                "dynamic_translation",
            ]
        elif cfg.eval_dataset == 'eds-vis':
            DATASETS = [
                # "05_rpg_building",
                # "10_office",
                # "11_all_characters",
                # "13_airplane",
                # "15_apartment_day",
                "peanuts_running_2360_2460",
            ]
        elif cfg.eval_dataset == 'dsec-vis':
            DATASETS = [
                # "interlaken_00_b",
                "zurich_city_12_a",
                # "zurich_city_13_a",
                # "zurich_city_13_b",
            ]

    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    for seq_name in DATASETS:
        if cfg.eval_dataset == 'ec-vis':
            dataset = ECSubseq_vis(
                f'{cfg.eval_dataset_path}/for_vis',
                seq_name,
                -1,
                cfg.patch_size,
                cfg.representation,
                0.01,
                corner_config,
                image_folder=cfg.image_folder,
                event_folder=cfg.event_folder,
            )
        elif cfg.eval_dataset == 'eds-vis':
            dataset = EDSSubseq_vis(
                f'{cfg.eval_dataset_path}/for_vis',
                seq_name,
                -1,
                cfg.patch_size,
                cfg.representation,
                0.005,
                corner_config,
                image_folder=cfg.image_folder,
                event_folder=cfg.event_folder,
            )
        elif cfg.eval_dataset == 'dsec-vis':
            dataset = DSECSubseq_vis(
                f'{cfg.eval_dataset_path}/for_vis',
                seq_name,
                -1,
                cfg.patch_size,
                cfg.representation,
                0.01,
                corner_config,
                image_folder=cfg.image_folder,
                event_folder=cfg.event_folder,
            )

        # gt_features_path = f'{cfg.eval_dataset_path}/gt_tracks/{seq_name}.gt.txt'
        # gt_start_corners = get_gt_corners(gt_features_path)

        if cfg.ref_frame_idx != 'none':
            dataset.override_refframe(cfg.ref_frame_idx)
        if cfg.frame_length != 'none':
            dataset.override_seqlength(cfg.frame_length)

        kp_path = dataset.first_image_path.split('.')[0] + '.npy'
        gt_start_corners = np.load(kp_path)
        dataset.override_keypoints(gt_start_corners)

        # vis_subseq(dataset)

        evaluate_seq(event_model, rgb_model, dataset, cfg.dt_track_vis, seq_name, cfg.visualize, evaluate_name, use_kalman=use_kalman, result_path=result_path)

    # metric = benchmark_pl(evaluate_name, EC_DATASETS, result_path=result_path)
    return