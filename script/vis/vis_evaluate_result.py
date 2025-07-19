import os
import cv2
import time
import numpy as np
# import matplotlib
from datetime import datetime
# from  matplotlib import pyplot as plt
from scipy.interpolate import interp1d


TRACK_RESULT_PATH = 'xxx'
TRACK_GT_PATH = 'data/deep_ev_tracker_data/gt_tracks'
OUTPUT_PATH = 'vis'
DATA_PATH = 'data/deep_ev_tracker_data'

LENGTH = 5
THICKNESS = 2


def apply_colormap(data, colormap=cv2.COLORMAP_VIRIDIS):
    data = data.copy().astype(float)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.clip(data, 0, 1)
    data = (data * 255).astype(np.uint8)
    data = cv2.applyColorMap(data, colormap)
    return data


def make_video(images, outvid=None, fps=10, size=None, is_color=True, format="mp4v"):
# def make_video(images, outvid=None, fps=10, size=None, is_color=True, format="H264"):
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()


def draw_traj(img_list, traj, vis=None, length=5, thickness=2, color=(0, 255, 0), point=True):
    # traj L, 2
    if vis is None:
        vis = np.ones_like(traj[..., 0])
    color_opp = (255 - color[0], 255 - color[1], 255 - color[2])
    for idx in range(traj.shape[0]):
        for line_idx in range(length):
            if idx - line_idx - 1 < 0:
                break
            if vis[idx - line_idx]:
                cur_color = color
            else:
                cur_color = color_opp
            cv2.line(img_list[idx],
                     tuple(traj[idx - line_idx - 1].astype(int)),
                     tuple(traj[idx - line_idx].astype(int)),
                     cur_color,
                     thickness=thickness)
        if point:
            if vis[idx]:
                cur_color = color
            else:
                cur_color = color_opp
            cv2.circle(img_list[idx],
                       tuple(traj[idx].astype(int)), 1,
                       cur_color, thickness=thickness)
    return img_list


def draw_trajs(img_list, trajs, viss=None, length=5, thickness=2, color=(0, 255, 0), point=True):
    # trajs N, L, 2
    if viss is None:
        viss = np.ones_like(trajs[..., 0])
    for traj_id in range(trajs.shape[0]):
        img_list = draw_traj(img_list, trajs[traj_id], viss[traj_id], length=5, thickness=2, color=(0, 255, 0), point=True)
    return img_list



def vis_seq(seq):
    print(f'Visualizing {seq}')
    if 'boxes' in seq or 'shapes' in seq:
        dataset = 'ec'
    else:
        dataset = 'eds'
    seq_path = f'{DATA_PATH}/{dataset}_subseq/{seq}'

    img_list = []
    img_path = f'{seq_path}/images_corrected'
    for img in sorted(os.listdir(img_path)):
        if not img.endswith('.png'):
            continue
        img = cv2.imread(f'{img_path}/{img}')
        img_list.append(img)

    traj_gts = np.genfromtxt(f'{TRACK_GT_PATH}/{seq}_occ.gt.txt')
    traj_ids = np.unique(traj_gts[..., 0])
    traj_times = np.unique(traj_gts[..., 1])
    traj_times = np.sort(traj_times)

    for traj_id in traj_ids:
        traj_gt = traj_gts[traj_gts[..., 0] == traj_id]
        traj = traj_gt[...,2:4]
        img_list = draw_traj(img_list, traj, length=LENGTH, thickness=THICKNESS, color=(255, 0, 0))

    traj_preds = np.genfromtxt(f'{TRACK_RESULT_PATH}/{seq}.txt')        
    for traj_id in traj_ids:
        traj_gt = traj_gts[traj_gts[..., 0] == traj_id]
        traj_pred = traj_preds[traj_preds[..., 0] == traj_id]

        x_interp = interp1d(
            traj_pred[..., 1],
            traj_pred[..., 2],
            fill_value="extrapolate",
        )
        y_interp = interp1d(
            traj_pred[..., 1],
            traj_pred[..., 3],
            fill_value="extrapolate",
        )

        # pred_x = x_interp(traj_gt[:, 1]).reshape((-1, 1))
        # pred_y = y_interp(traj_gt[:, 1]).reshape((-1, 1))
        pred_x = x_interp(traj_times).reshape((-1, 1))
        pred_y = y_interp(traj_times).reshape((-1, 1))
        traj = np.concatenate([pred_x, pred_y], axis=1)
        img_list = draw_traj(img_list, traj, length=LENGTH, thickness=THICKNESS, color=(0, 0, 255))

    make_video(
        img_list, f'vis/result_{seq}_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{time.time()}.mp4', fps=5)
        

if __name__ == '__main__':

    seq_list = os.listdir(TRACK_RESULT_PATH)
    seq_list = [seq.split('.')[0] for seq in seq_list]
    for seq in seq_list:
        vis_seq(seq)