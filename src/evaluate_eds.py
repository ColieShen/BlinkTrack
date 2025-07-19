import os
import torch


# THREAD_NUM = 4
# torch.set_num_threads(THREAD_NUM)
# os.environ ['OMP_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['MKL_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_NUM}'

import cv2
import tqdm
import time
import yaml
import numpy as np
import imageio
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from omegaconf import OmegaConf, open_dict
from dataclasses import dataclass


from config import CODE_ROOT
# from config import args
# from util.vis import generate_track_colors, render_pred_tracks
from util.vis import generate_track_colors, generate_track_colors_y, render_pred_tracks
from util.data import TrackObserver, get_gt_corners
from util.timer import CudaTimer, cuda_timers
from script.benchmark import benchmark, benchmark_pl
from loader.loader_eds import EDSSubseq, vis_subseq
from model.kalman_util import CotrackerManualMapping, CotrackerParabolaFuncMapping
from model.kalman_filter import init_filter_state, kalman_filter_4


@dataclass
class CornerConfig:
    maxCorners: int
    qualityLevel: float
    minDistance: int
    k: float
    useHarrisDetector: bool
    blockSize: int


def evaluate_seq(event_model, rgb_model, sequence_dataset, dt_track_vis, sequence_name, visualize, evaluate_name, use_kalman=False, result_path=None, rgb_fps=None):
    print(f'Evaluating {sequence_name}')

    if event_model:
        event_model = event_model.cuda()

    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init, u_centers_init=sequence_dataset.u_centers
    )

    if 'boxes' in sequence_name or 'shapes' in sequence_name:
        dataset = 'ec'
    else:
        dataset = 'eds'

    if use_kalman:
        kalman_filter = kalman_filter_4()
        kalman_filter.prev_ts = sequence_dataset.t_init
        if rgb_model and rgb_model.name == 'blinktrack':
            rgb_model.kalman_filter = kalman_filter

    if event_model:
        if hasattr(event_model, 'module'):
            event_model = event_model.module
        event_model.reset(sequence_dataset.n_tracks)

    # if rgb_model:
    #     rgb_model.reset()
        
    evt_runtime = []
    rgb_runtime = []

    if event_model and event_model.pyramid:
        event_generator = sequence_dataset.events_2()
    else:
        event_generator = sequence_dataset.events()

    with torch.no_grad():

        n_tracks = sequence_dataset.n_tracks

        prev_state = None
        if use_kalman:
            # init kalman
            prev_state = {
                'predict': None,
                'update': None,
                'filter_state': init_filter_state(1, n_tracks)
            }
            for key in prev_state['filter_state']:
                prev_state['filter_state'][key] = prev_state['filter_state'][key].cuda()

            flow_pred = sequence_dataset.u_centers - sequence_dataset.u_centers_init
            flow_pred = flow_pred[None].cuda()

        if rgb_model:
            if rgb_model.name == 'blinktrack': # blinktrack
                ref_rgb = torch.from_numpy((sequence_dataset.frame_first).astype(np.float32))
                ref_rgb = ref_rgb[None, None].cuda() / 255  # 1, 1, H, W
                ref_xy = sequence_dataset.u_centers_init
                ref_xy = ref_xy[None, None].cuda()  # 1, 1, N, 2
                ref_ffeat = rgb_model('init_ref_view', ref_rgb, ref_xy)
            elif rgb_model.name == 'cotracker':
                ref_rgb = torch.from_numpy((sequence_dataset.frame_first).astype(np.float32))
                H, W = ref_rgb.shape
                ref_rgb = ref_rgb[None, None].expand(1, 3, H, W).cuda()  # 1, 3, H, W
                ref_rgb = 2 * (ref_rgb / 255.0) - 1.0

                queried_coords = sequence_dataset.u_centers_init / rgb_model.stride
                queried_coords = queried_coords[None].cuda()  # 1, N, 2
                queried_frames = torch.ones(queried_coords.shape[:2]).cuda()    # 1, N
                N = queried_frames.shape[1]

                S = rgb_model.window_len
                fmaps = rgb_model.fnet(ref_rgb).reshape(
                    1, -1, rgb_model.latent_dim, H // rgb_model.stride, W // rgb_model.stride
                ).repeat(1, S, 1, 1, 1)  # 1, S, D, H // stride, W // stride
                track_feat = rgb_model.get_track_feat(
                    fmaps,
                    queried_frames,
                    queried_coords,
                ).repeat(1, S, 1, 1) # 1, S, N, 128
                coords_init = queried_coords[None].repeat(1, S, 1, 1)  # 1, S, N, 2
                vis_init = torch.ones(1, S, N, 1).cuda() * 10  # 1, S, N, 1
                attention_mask = torch.ones(1, S, N).bool().cuda()  # 1, S, N
                track_mask = torch.ones(1, S, N, 1).bool().cuda()  # 1, S, N, 1
                iters = 6

                if use_kalman:
                    cotracker_uncert2R = CotrackerManualMapping()
                    # cotracker_uncert2R = CotrackerParabolaFuncMapping()

            else:
                rgb_model.init(sequence_dataset.frame_first, sequence_dataset.u_centers_init.numpy())

            next_rgb_idx = sequence_dataset.first_frame_idx + 1.0
            if rgb_fps is None or rgb_fps == 'None':
                next_rgb_idx = sequence_dataset.first_frame_idx + 1
            else:
                if dataset == 'ec':
                    next_rgb_idx = sequence_dataset.first_frame_idx + 24 / rgb_fps
                elif dataset == 'eds':
                    # next_rgb_idx = sequence_dataset.first_frame_idx + 75 / rgb_fps
                    next_rgb_idx = sequence_dataset.first_frame_idx + 8 # may have bug

        # Predict network tracks
        for event_data in tqdm.tqdm(
            event_generator,
            total=sequence_dataset.n_events - 1,
        ):

            if event_model and event_model.pyramid:
                t, x, xx = event_data
            else:
                t, x = event_data

            if event_model:
                if event_model.pyramid:
                    x = x.cuda()
                    xx = xx.cuda()
                    torch.cuda.synchronize()
                    evt_time1 = time.time()
                    y_hat = event_model(x, xx)
                else:
                    x = x.cuda()
                    torch.cuda.synchronize()
                    evt_time1 = time.time()
                    y_hat = event_model(x)

                uncert = y_hat[..., 2:]   # B, 2
                y_hat = y_hat[..., :2]
                # if R.shape[1] == 0:
                #     R = torch.zeros_like(y_hat)
                #     R[:] = -2

                if use_kalman:
                    flow = y_hat + flow_pred[0]
                    flow = flow[None]   # 1, B, 2

                    R = event_model.uncert2R(uncert)

                    # vis status to uncertainty
                    # R = F.softmax(uncert, dim=1)
                    # vis_pred = torch.argmax(R, dim=1).bool()
                    # R[vis_pred] = 0.001
                    # R[~vis_pred] = 4
                    # print(R)

                    # R = F.softplus(uncert)
                    # R = F.softplus(uncert, beta=0.1)
                    # R[R < 1] = 0.0001
                    # R[R > 1] = 2
                    # print(R)

                    R = torch.diag_embed(R, dim1=-2, dim2=-1)
                    R = R[None]  # 1, B, 2, 2
                    prev_state = kalman_filter.detach(prev_state)
                    flow, prev_state, _, _ = kalman_filter.update(flow, R, prev_state, t)
                    y_hat = (flow - flow_pred)[0]   # B, 2

                torch.cuda.synchronize()
                evt_time2 = time.time()
                evt_runtime.append(evt_time2 - evt_time1)

                sequence_dataset.accumulate_y_hat(y_hat)
                tracks_pred.add_observation(t, sequence_dataset.u_centers.cpu().numpy())

                if use_kalman:
                    # prev_state = kalman_filter.detach(prev_state)
                    # flow_pred, prev_state, Q = kalman_filter.predict(prev_state, sequence_dataset.t_now + sequence_dataset.dt)     # flow_pred 1, B, 2 from init
                    # sequence_dataset.u_centers = sequence_dataset.u_centers_init + flow_pred[0]
                    flow_pred = sequence_dataset.u_centers - sequence_dataset.u_centers_init
                    flow_pred = flow_pred[None]

            if rgb_model and sequence_dataset.check_next_rgb_time(int(next_rgb_idx)):
                t_, rgb = sequence_dataset.get_frame(int(next_rgb_idx), grayscale=True)
                if rgb_model.name == 'blinktrack':
                    rgb = torch.from_numpy(rgb.astype(np.float32))  # 1, 1, H, W
                    rgb = rgb[None, None].cuda() / 255

                    torch.cuda.synchronize()
                    rgb_time1 = time.time()
                    next_xy, prev_state, _ = rgb_model('predict_rgb', ref_xy, rgb, prev_state, ref_ffeat,
                                                       t_, sequence_dataset.u_centers[None, None].cuda())
                    torch.cuda.synchronize()
                    rgb_time2 = time.time()
                    rgb_runtime.append(rgb_time2 - rgb_time1)

                    next_xy = next_xy[-1][0, 0].to(sequence_dataset.u_centers.device)   # N, 2
                    # if event_model:
                    #     sequence_dataset.u_centers = next_xy[-1][0, 0]
                    # else:
                    #     sequence_dataset.u_centers = next_xy[-1][0, 0].cpu()
                elif rgb_model.name == 'cotracker':
                    rgb = torch.from_numpy(rgb.astype(np.float32))
                    rgb = rgb[None, None].expand(1, 3, H, W).cuda()  # 1, 3, H, W
                    rgb = 2 * (rgb / 255.0) - 1.0
                    coords_init_cur = sequence_dataset.u_centers[None, None].cuda() / rgb_model.stride  # 1, 1, N, 2

                    # torch.cuda.synchronize()
                    # rgb_time1 = time.time()

                    fmaps_cur = rgb_model.fnet(rgb).reshape(
                        1, 1, rgb_model.latent_dim, H // rgb_model.stride, W // rgb_model.stride
                    )  # 1, 1, D, H // stride, W // stride
                    fmaps = torch.concat([fmaps, fmaps_cur], dim=1)[:, 1:]

                    coords_init = torch.concat([coords_init, coords_init_cur], dim=1)[:, 1:]
                    vis_init = torch.concat([vis_init, vis_init[:,-1:]], dim=1)[:, 1:]

                    coords, vis = rgb_model.forward_window(
                        fmaps=fmaps,
                        coords=coords_init,
                        track_feat=attention_mask.unsqueeze(-1) * track_feat,
                        vis=vis_init,
                        track_mask=track_mask,
                        attention_mask=attention_mask,
                        iters=iters,
                    )

                    # torch.cuda.synchronize()
                    # rgb_time2 = time.time()

                    coords_init = coords[-1]
                    vis_init = vis[..., None]

                    coords_cur = coords_init[0, -1]
                    if use_kalman:
                        flow = coords_cur - queried_coords[0]
                        flow = flow[None]   # 1, N, 2
                        uncert = vis_init[0, -1]    # N, 1
                        R = cotracker_uncert2R(uncert)
                        R = torch.diag_embed(R, dim1=-2, dim2=-1)
                        R = R[None]  # 1, B, 2, 2
                        flow, prev_state, _, _ = kalman_filter.update(flow * rgb_model.stride, R, prev_state, t_)
                        flow /= rgb_model.stride
                        coords_cur = queried_coords[0] + flow[0]

                    next_xy = coords_cur.to(sequence_dataset.u_centers.device) * rgb_model.stride   # N, 2

                else:
                    # rgb_time1 = time.time()
                    next_xy = rgb_model.predict(rgb, sequence_dataset.u_centers.cpu().numpy())
                    # rgb_time2 = time.time()
                    next_xy = torch.from_numpy(next_xy).to(sequence_dataset.u_centers.device)
                
                # print(rgb_time2 - rgb_time1)

                sequence_dataset.u_centers = next_xy
                if rgb_fps is None or rgb_fps == 'None':
                    next_rgb_idx += 1
                else:
                    if dataset == 'ec':
                        next_rgb_idx += 24 / rgb_fps
                    elif dataset == 'eds':
                        next_rgb_idx += 75 / rgb_fps
                        

                tracks_pred.add_observation(t_, sequence_dataset.u_centers.cpu().numpy())
                flow_pred = sequence_dataset.u_centers - sequence_dataset.u_centers_init
                flow_pred = flow_pred[None]

            # sequence_dataset.u_centers[torch.abs(sequence_dataset.u_centers) > 1e10] = 1e10

        if visualize:
            # Visualize network tracks
            gif_img_arr = []
            tracks_pred_interp = tracks_pred.get_interpolators()
            # track_colors = generate_track_colors(sequence_dataset.n_tracks)
            track_colors = generate_track_colors_y(sequence_dataset.u_centers_init.cpu().numpy())
            for i, (t, img_now) in enumerate(
                tqdm.tqdm(
                    sequence_dataset.frames(),
                    total=sequence_dataset.n_frames - 1,
                    desc="Rendering predicted tracks... ",
                )
            ):
                fig_arr = render_pred_tracks(
                    tracks_pred_interp, t, img_now, track_colors, dt_track=dt_track_vis
                )
                gif_img_arr.append(fig_arr)
            make_video(gif_img_arr, f'{CODE_ROOT}/vis/{sequence_name}_{evaluate_name}.mp4', fps=5)
            # cv2.imwrite(f'{CODE_ROOT}/vis/{sequence_name}_{evaluate_name}_start.png', gif_img_arr[0])
            # cv2.imwrite(f'{CODE_ROOT}/vis/{sequence_name}_{evaluate_name}_end.png', gif_img_arr[-1])

            # imageio.mimsave(
            #     f'{CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}/{sequence_name}_tracks_pred.gif', gif_img_arr)
            # make_video(gif_img_arr, f'{CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}/{sequence_name}_{evaluate_name}.mp4', fps=5)

    # Save predicted tracks
    np.savetxt(
        f'{CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}/{sequence_name}.txt',
        tracks_pred.track_data,
        fmt=["%i", "%.9f", "%i", "%i"],
        delimiter=" ",
    )

    metrics = {}
    if event_model:
        metrics["evt_runtime"] = np.array(evt_runtime)[1:].mean()
    if rgb_model:
        metrics["rgb_runtime"] = np.array(rgb_runtime).mean()

    return metrics


def evaluate_eds(model, evaluate_name):
    model.eval()

    os.system(f'mkdir -p output/evaluate_result/{evaluate_name}')

    EDS_DATASETS = [
        "peanuts_light_160_386",
        "rocket_earth_light_338_438",
        "ziggy_in_the_arena_1350_1650",
        "peanuts_running_2360_2460",
    ]
    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    for seq_name in EDS_DATASETS:
        dataset = EDSSubseq(
            f'{args.eval_dataset_path}/eds_subseq',
            seq_name,
            -1,
            args.patch_size,
            args.event_representation,
            0.005,
            corner_config,
        )

        gt_features_path = f'{args.eval_dataset_path}/gt_tracks/{seq_name}.gt.txt'
        gt_start_corners = get_gt_corners(gt_features_path)

        dataset.override_keypoints(gt_start_corners)

        evaluate_seq(model, dataset, 0.2, seq_name, args.visualize, evaluate_name)

    metric = benchmark(evaluate_name, EDS_DATASETS)
    return metric


def evaluate_eds_pl(event_model, rgb_model, evaluate_name, use_kalman=False, cfg=None, seqs=None, result_path=None):
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
        EDS_DATASETS = seqs
    else:
        EDS_DATASETS = [
            "peanuts_light_160_386",
            "rocket_earth_light_338_438",
            "ziggy_in_the_arena_1350_1650",
            "peanuts_running_2360_2460",
        ]
    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    evt_runtime = []
    rgb_runtime = []

    for seq_name in EDS_DATASETS:
        dataset = EDSSubseq(
            f'{cfg.eval_dataset_path}/eds_subseq',
            seq_name,
            -1,
            cfg.patch_size,
            cfg.representation,
            0.005,
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
        metric = benchmark_pl(evaluate_name, EDS_DATASETS, result_path=result_path)

    if event_model:
        print(f'Evt Runtime: {np.array(evt_runtime).mean()}')
    if rgb_model:
        print(f'Rgb Runtime: {np.array(rgb_runtime).mean()}')

    return metric