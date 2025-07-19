import os
import yaml
import numpy as np

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from prettytable import PrettyTable
from scipy.interpolate import interp1d

from config import *
from util.data import compute_tracking_errors, read_txt_results


L1_THRESHOLD = 31
EPE_THRESHOLD = 31
FRA_THRESHOLD = [1, 2, 4, 8, 16]
FRA_IMG_SIZE = 256


def benchmark(evaluate_name, EVAL_DATASETS=None, age=True, epe=False):

    error_threshold_range = np.arange(1, 32, 1)
    gt_dir = Path(f'{args.eval_dataset_path}/gt_tracks')
    results_dir = Path(f'output/evaluate_result/{evaluate_name}')

    if EVAL_DATASETS is None:
        EVAL_DATASETS = os.listdir(results_dir)
        EVAL_DATASETS = [seq.split('.')[0] for seq in EVAL_DATASETS]
        # EVAL_DATASETS = [
        # # eds
        #     "peanuts_light_160_386",
        #     "rocket_earth_light_338_438",
        #     "ziggy_in_the_arena_1350_1650",
        #     "peanuts_running_2360_2460",
        # # ec
        #     "shapes_translation_8_88",
        #     "shapes_rotation_165_245",
        #     "shapes_6dof_485_565",
        #     "boxes_translation_330_410",
        #     "boxes_rotation_198_278",
        # ]

    methods = ["network_pred"]
    metric = {}

    if age:

        table_keys = [
            "age_5_mu",
            "age_5_std",
            "te_5_mu",
            "te_5_std",
            "age_mu",
            "age_std",
            "inliers_mu",
            "inliers_std",
            "expected_age",
        ]
        tables = {}
        for k in table_keys:
            tables[k] = PrettyTable()
            tables[k].title = k
            tables[k].field_names = ["Sequence Name"] + methods

        for sequence_name in EVAL_DATASETS:
            track_data_gt = read_txt_results(
                str(gt_dir / f"{sequence_name}.gt.txt")
            )

            rows = {}
            for k in tables.keys():
                rows[k] = [sequence_name]

            for method in methods:
                inlier_ratio_arr, fa_rel_nz_arr = [], []

                track_data_pred = read_txt_results(
                    str(results_dir / f"{sequence_name}.txt")
                )

                if track_data_pred[0, 1] != track_data_gt[0, 1]:
                    track_data_pred[:, 1] += -track_data_pred[0, 1] + track_data_gt[0, 1]

                for thresh in error_threshold_range:
                    fa_rel, _ = compute_tracking_errors(
                        track_data_pred,
                        track_data_gt,
                        error_threshold=thresh,
                        asynchronous=False,
                    )

                    inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
                    if inlier_ratio > 0:
                        fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
                    else:
                        fa_rel_nz = [0]
                    inlier_ratio_arr.append(inlier_ratio)
                    fa_rel_nz_arr.append(np.mean(fa_rel_nz))

                mean_inlier_ratio, std_inlier_ratio = np.mean(inlier_ratio_arr), np.std(
                    inlier_ratio_arr
                )
                mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz_arr), np.std(fa_rel_nz_arr)
                expected_age = np.mean(np.array(inlier_ratio_arr) * np.array(fa_rel_nz_arr))

                rows["age_mu"].append(mean_fa_rel_nz)
                rows["age_std"].append(std_fa_rel_nz)
                rows["inliers_mu"].append(mean_inlier_ratio)
                rows["inliers_std"].append(std_inlier_ratio)
                rows["expected_age"].append(expected_age)

                fa_rel, te = compute_tracking_errors(
                    track_data_pred, track_data_gt, error_threshold=5, asynchronous=False
                )
                inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
                if inlier_ratio > 0:
                    fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
                else:
                    fa_rel_nz = [0]
                    te = [0]

                mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz), np.std(fa_rel_nz)
                mean_te, std_te = np.mean(te), np.std(te)
                rows["age_5_mu"].append(mean_fa_rel_nz)
                rows["age_5_std"].append(std_fa_rel_nz)
                rows["te_5_mu"].append(mean_te)
                rows["te_5_std"].append(std_te)

            # Load results
            for k in tables.keys():
                tables[k].add_row(rows[k])

        print('age_mu \t expected_age \t inliers_mu \t te_5_mu')
        for key in ['age_mu', 'expected_age', 'inliers_mu', 'te_5_mu']:
            seq_cnt = 0
            seq_sum = 0
            for seq in tables[key]._rows:
                if seq[1] == 0:
                    continue
                seq_cnt += 1
                seq_sum += seq[1]
            seq_avg = seq_sum / seq_cnt if seq_cnt != 0 else 0
            metric[key] = seq_avg
            print(f'{seq_avg:.8f}', end='\t') 
        print('')

    if epe:
        epe_all_seq = []
        epe_vis_seq = []
        epe_occ_seq = []
        for sequence_name in EVAL_DATASETS:
            if 'shapes' in sequence_name or 'boxes' in sequence_name:
                H, W = 180, 240
            else:
                H, W = 480, 640
            epe_all = []
            epe_vis = []
            epe_occ = []
            # id, time, x, y, vis
            track_data_gt = read_txt_results(
                str(gt_dir / f"{sequence_name}_occ.gt.txt")
            )

            track_data_pred = read_txt_results(
                str(results_dir / f"{sequence_name}.txt")
            )
            track_ids = np.unique(track_data_gt[...,0])
            for track_id in track_ids:
                track_gt = track_data_gt[track_data_gt[..., 0] == track_id]
                track_pred = track_data_pred[track_data_pred[..., 0] == track_id]

                x_interp = interp1d(
                    track_pred[:, 1],
                    track_pred[:, 2],
                    fill_value="extrapolate",
                )
                y_interp = interp1d(
                    track_pred[:, 1],
                    track_pred[:, 3],
                    fill_value="extrapolate",
                )

                pred_x = x_interp(track_gt[:, 1]).reshape((-1, 1))
                pred_y = y_interp(track_gt[:, 1]).reshape((-1, 1))
                pred_track_data_curr_interp = np.concatenate([pred_x, pred_y], axis=1)

                gt_outbound_mask = np.logical_or(
                    np.logical_or(track_gt[..., 2] < 0, W - 1 < track_gt[..., 2]),
                    np.logical_or(track_gt[..., 3] < 0, H - 1 < track_gt[..., 3])
                )
                pred_outbound_mask = np.logical_or(
                    np.logical_or(pred_track_data_curr_interp[..., 0] < 0, W - 1 < pred_track_data_curr_interp[..., 0]),
                    np.logical_or(pred_track_data_curr_interp[..., 1] < 0, H - 1 < pred_track_data_curr_interp[..., 1])
                )
                outbound_mask = np.logical_or(gt_outbound_mask, pred_outbound_mask)

                track_delta = track_gt[..., 2:4] - pred_track_data_curr_interp
                epe = np.linalg.norm(track_delta, axis=1)
                track_vis = track_gt[...,-1] == 1

                epe_all.append(epe[~outbound_mask])
                epe_vis.append(epe[np.logical_and(track_vis, ~outbound_mask)])
                epe_occ.append(epe[np.logical_and(~track_vis, ~outbound_mask)])

            epe_all = np.concatenate(epe_all)
            epe_vis = np.concatenate(epe_vis)
            epe_occ = np.concatenate(epe_occ)

            epe_all = epe_all[epe_all < EPE_THRESHOLD]
            epe_vis = epe_vis[epe_vis < EPE_THRESHOLD]
            epe_occ = epe_occ[epe_occ < EPE_THRESHOLD]

            epe_all_seq.append(epe_all.mean())
            epe_vis_seq.append(epe_vis.mean())
            epe_occ_seq.append(epe_occ.mean())

        epe_all_seq = np.array(epe_all_seq)
        epe_vis_seq = np.array(epe_vis_seq)
        epe_occ_seq = np.array(epe_occ_seq)
        
        print('epe_vis \t epe_occ \t epe_all')
        print(epe_vis_seq.mean(), epe_occ_seq.mean(), epe_all_seq.mean())

        metric['epe_vis'] = epe_vis_seq.mean()
        metric['epe_occ'] = epe_occ_seq.mean()
        metric['epe_all'] = epe_all_seq.mean()

    return metric


def benchmark_pl(evaluate_name=None, EVAL_DATASETS=None, eval_age=True, eval_epe=False, eval_fra=True, results_dir=None, result_path=None):

    with open(f'{CODE_ROOT}/configs/eval_real_defaults.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    error_threshold_range = np.arange(1, 32, 1)
    gt_dir = Path(f'{cfg.eval_dataset_path}/gt_tracks')
    if results_dir:
        results_dir = Path(results_dir)
    if evaluate_name:
        results_dir = Path(f'{CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}')

    if EVAL_DATASETS is None:
        EVAL_DATASETS = os.listdir(results_dir)
        EVAL_DATASETS = [seq for seq in EVAL_DATASETS if '.txt' in seq]
        EVAL_DATASETS = [seq.split('.')[0] for seq in EVAL_DATASETS]
        # EVAL_DATASETS = [
        # # eds
        #     "peanuts_light_160_386",
        #     "rocket_earth_light_338_438",
        #     "ziggy_in_the_arena_1350_1650",
        #     "peanuts_running_2360_2460",
        # # ec
        #     "shapes_translation_8_88",
        #     "shapes_rotation_165_245",
        #     "shapes_6dof_485_565",
        #     "boxes_translation_330_410",
        #     "boxes_rotation_198_278",
        # ]

    methods = ["network_pred"]
    metric = {}

    final_result = []

    if eval_age:

        table_keys = [
            "age_5_mu",
            "age_5_std",
            "te_5_mu",
            "te_5_std",
            "age_mu",
            "age_std",
            "inliers_mu",
            "inliers_std",
            "expected_age",
        ]
        tables = {}
        for k in table_keys:
            tables[k] = PrettyTable()
            tables[k].title = k
            tables[k].field_names = ["Sequence Name"] + methods

        for sequence_name in EVAL_DATASETS:
            track_data_gt = read_txt_results(
                str(gt_dir / f"{sequence_name}.gt.txt")
            )

            rows = {}
            for k in tables.keys():
                rows[k] = [sequence_name]

            for method in methods:
                inlier_ratio_arr, fa_rel_nz_arr = [], []

                track_data_pred = read_txt_results(
                    str(results_dir / f"{sequence_name}.txt")
                )

                if track_data_pred[0, 1] != track_data_gt[0, 1]:
                    track_data_pred[:, 1] += -track_data_pred[0, 1] + track_data_gt[0, 1]

                for thresh in error_threshold_range:
                    fa_rel, _ = compute_tracking_errors(
                        track_data_pred,
                        track_data_gt,
                        error_threshold=thresh,
                        asynchronous=False,
                    )

                    inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
                    if inlier_ratio > 0:
                        fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
                    else:
                        fa_rel_nz = [0]
                    inlier_ratio_arr.append(inlier_ratio)
                    fa_rel_nz_arr.append(np.mean(fa_rel_nz))

                mean_inlier_ratio, std_inlier_ratio = np.mean(inlier_ratio_arr), np.std(
                    inlier_ratio_arr
                )
                mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz_arr), np.std(fa_rel_nz_arr)
                expected_age = np.mean(np.array(inlier_ratio_arr) * np.array(fa_rel_nz_arr))

                rows["age_mu"].append(mean_fa_rel_nz)
                rows["age_std"].append(std_fa_rel_nz)
                rows["inliers_mu"].append(mean_inlier_ratio)
                rows["inliers_std"].append(std_inlier_ratio)
                rows["expected_age"].append(expected_age)

                fa_rel, te = compute_tracking_errors(
                    track_data_pred, track_data_gt, error_threshold=5, asynchronous=False
                )
                inlier_ratio = np.sum(fa_rel > 0) / len(fa_rel)
                if inlier_ratio > 0:
                    fa_rel_nz = fa_rel[np.nonzero(fa_rel)[0]]
                else:
                    fa_rel_nz = [0]
                    te = [0]

                mean_fa_rel_nz, std_fa_rel_nz = np.mean(fa_rel_nz), np.std(fa_rel_nz)
                mean_te, std_te = np.mean(te), np.std(te)
                rows["age_5_mu"].append(mean_fa_rel_nz)
                rows["age_5_std"].append(std_fa_rel_nz)
                rows["te_5_mu"].append(mean_te)
                rows["te_5_std"].append(std_te)

            # Load results
            for k in tables.keys():
                tables[k].add_row(rows[k])

        # for key in ['age_mu', 'expected_age', 'inliers_mu', 'te_5_mu']:
        for key in ['age_mu', 'expected_age']:
            print(tables[key].get_string())

        print('age_mu \t expected_age \t inliers_mu \t te_5_mu')
        for key in ['age_mu', 'expected_age', 'inliers_mu', 'te_5_mu']:
            seq_cnt = 0
            seq_sum = 0
            for seq in tables[key]._rows:
                if seq[1] == 0:
                    continue
                seq_cnt += 1
                seq_sum += seq[1]
            seq_avg = seq_sum / seq_cnt if seq_cnt != 0 else 0
            metric[key] = seq_avg
            print(f'{seq_avg:.8f}', end='\t')
            final_result.append(seq_avg)
        print('')

    if eval_epe or eval_fra:
        epe_all_seq, epe_vis_seq, epe_occ_seq = [], [], []
        fra_all_seq, fra_vis_seq, fra_occ_seq = [], [], []
        for sequence_name in EVAL_DATASETS:
            if 'shapes' in sequence_name or 'boxes' in sequence_name:
                H, W = 180, 240
            else:
                H, W = 480, 640
            H_factor, W_factor = FRA_IMG_SIZE / H, FRA_IMG_SIZE / W
            
            # id, time, x, y, vis
            track_data_gt = read_txt_results(
                str(gt_dir / f"{sequence_name}_occ.gt.txt")
            )
            track_data_pred = read_txt_results(
                str(results_dir / f"{sequence_name}.txt")
            )
            track_ids = np.unique(track_data_gt[...,0])

            epe_all, epe_vis, epe_occ = [], [], []
            fra_all, fra_vis, fra_occ = [0] * len(FRA_THRESHOLD), [0] * len(FRA_THRESHOLD), [0] * len(FRA_THRESHOLD)    # [d1, d2, d4, d8, d16] num_cnt
            fra_all_num, fra_vis_num, fra_occ_num = 0, 0, 0 # num_all
            for track_id in track_ids:
                track_gt = track_data_gt[track_data_gt[..., 0] == track_id]
                track_pred = track_data_pred[track_data_pred[..., 0] == track_id]

                x_interp = interp1d(
                    track_pred[:, 1],
                    track_pred[:, 2],
                    fill_value="extrapolate",
                )
                y_interp = interp1d(
                    track_pred[:, 1],
                    track_pred[:, 3],
                    fill_value="extrapolate",
                )

                pred_x = x_interp(track_gt[:, 1]).reshape((-1, 1))
                pred_y = y_interp(track_gt[:, 1]).reshape((-1, 1))
                pred_track_data_curr_interp = np.concatenate([pred_x, pred_y], axis=1)

                track_delta = track_gt[..., 2:4] - pred_track_data_curr_interp  # N, 2
                track_vis = track_gt[...,-1] == 1

                if eval_fra:
                    fra_all_num += track_vis.size
                    fra_vis_num += np.sum(track_vis)
                    fra_occ_num += np.sum(~track_vis)

                    track_delta_transform = np.zeros_like(track_delta)
                    track_delta_transform[..., 0] = track_delta[..., 0] * W_factor
                    track_delta_transform[..., 1] = track_delta[..., 1] * H_factor
                    epe_transform = np.linalg.norm(track_delta_transform, axis=1)

                    for fra_idx in range(len(FRA_THRESHOLD)):
                        fra_thre = FRA_THRESHOLD[fra_idx]
                        fra_all[fra_idx] += np.sum(epe_transform <= fra_thre)
                        fra_vis[fra_idx] += np.sum(epe_transform[track_vis] <= fra_thre)
                        fra_occ[fra_idx] += np.sum(epe_transform[~track_vis] <= fra_thre)

                if eval_epe:
                    epe = np.linalg.norm(track_delta, axis=1)
                    gt_outbound_mask = np.logical_or(
                        np.logical_or(track_gt[..., 2] < 0, W - 1 < track_gt[..., 2]),
                        np.logical_or(track_gt[..., 3] < 0, H - 1 < track_gt[..., 3])
                    )
                    pred_outbound_mask = np.logical_or(
                        np.logical_or(pred_track_data_curr_interp[..., 0] < 0, W - 1 < pred_track_data_curr_interp[..., 0]),
                        np.logical_or(pred_track_data_curr_interp[..., 1] < 0, H - 1 < pred_track_data_curr_interp[..., 1])
                    )
                    outbound_mask = np.logical_or(gt_outbound_mask, pred_outbound_mask) # True indicate outbound

                    lost_mask = np.abs(track_delta) > L1_THRESHOLD  # True indicate lost
                    lost_mask = np.any(lost_mask, axis=1)
                    outbound_mask = np.logical_or(lost_mask, outbound_mask)


                    epe_all.append(epe[~outbound_mask])
                    epe_vis.append(epe[np.logical_and(track_vis, ~outbound_mask)])
                    epe_occ.append(epe[np.logical_and(~track_vis, ~outbound_mask)])

            if eval_fra:
                fra_all_seq.append(np.mean(np.array(fra_all)) / fra_all_num)
                fra_vis_seq.append(np.mean(np.array(fra_vis)) / fra_vis_num)
                fra_occ_seq.append(np.mean(np.array(fra_occ)) / fra_occ_num)

            if eval_epe:
                epe_all = np.concatenate(epe_all)
                epe_vis = np.concatenate(epe_vis)
                epe_occ = np.concatenate(epe_occ)

                # epe_all = epe_all[epe_all < EPE_THRESHOLD]
                # epe_vis = epe_vis[epe_vis < EPE_THRESHOLD]
                # epe_occ = epe_occ[epe_occ < EPE_THRESHOLD]

                epe_all_seq.append(epe_all.mean())
                epe_vis_seq.append(epe_vis.mean())
                epe_occ_seq.append(epe_occ.mean())

        if eval_fra:
            fra_all_seq = np.array(fra_all_seq)
            fra_vis_seq = np.array(fra_vis_seq)
            fra_occ_seq = np.array(fra_occ_seq)

            for seq_idx in range(len(EVAL_DATASETS)):
                print(EVAL_DATASETS[seq_idx], '\t', fra_vis_seq[seq_idx], '\t', fra_occ_seq[seq_idx], '\t', fra_all_seq[seq_idx])

            print('fra_vis \t fra_occ \t fra_all')
            print(fra_vis_seq.mean(), '\t', fra_occ_seq.mean(), '\t', fra_all_seq.mean())

            final_result.append(fra_vis_seq.mean())
            final_result.append(fra_occ_seq.mean())
            final_result.append(fra_all_seq.mean())

            metric['fra_vis'] = fra_vis_seq.mean()
            metric['fra_occ'] = fra_occ_seq.mean()
            metric['fra_all'] = fra_all_seq.mean()

        if eval_epe:
            epe_all_seq = np.array(epe_all_seq)
            epe_vis_seq = np.array(epe_vis_seq)
            epe_occ_seq = np.array(epe_occ_seq)

            for seq_idx in range(len(EVAL_DATASETS)):
                print(EVAL_DATASETS[seq_idx], '\t', epe_vis_seq[seq_idx], '\t', epe_occ_seq[seq_idx], '\t', epe_all_seq[seq_idx])
            
            print('epe_vis \t epe_occ \t epe_all')
            print(epe_vis_seq.mean(), '\t', epe_occ_seq.mean(), '\t', epe_all_seq.mean())

            final_result.append(epe_vis_seq.mean())
            final_result.append(epe_occ_seq.mean())
            final_result.append(epe_all_seq.mean())

            metric['epe_vis'] = epe_vis_seq.mean()
            metric['epe_occ'] = epe_occ_seq.mean()
            metric['epe_all'] = epe_all_seq.mean()

        print(results_dir)

        if result_path:
            final_result = np.array(final_result)
            np.save(f'{CODE_ROOT}/{result_path}', final_result)
        print(final_result)

    return metric


if __name__ == '__main__':
    benchmark_pl(results_dir=RESULT_DIR)
