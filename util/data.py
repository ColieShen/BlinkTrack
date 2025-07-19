import os
import cv2
import math
import h5py
import torch
import numpy as np
import random
import pickle
import hdf5plugin
import torch.nn.functional as F


from math import pi
from tqdm import tqdm
from glob import glob
from typing import Iterator, Sequence
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from torchvision.transforms import InterpolationMode
from torch.utils.data.sampler import Sampler
from torchvision.transforms.functional import resize, rotate




def get_patch_voxel(voxel_grid, u_center, patch_size):

    if patch_size % 2 == 1:
        center = np.rint(u_center).astype(int).reshape((2,))
        if len(voxel_grid.shape) == 2:
            c = 1
            h, w = voxel_grid.shape
        else:
            h, w, c = voxel_grid.shape

        # Check out-of-bounds
        if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
            return torch.zeros((c, patch_size, patch_size), dtype=torch.float32)

        r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

        if len(voxel_grid.shape) == 2:
            x = np.array(voxel_grid[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]).astype(
                np.float32
            )
            x = np.expand_dims(x, axis=2)
        else:
            x = np.array(
                voxel_grid[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
            ).astype(np.float32)
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(x)
        x = torch.nn.functional.pad(x, pad_lrud)
    else:
        x = get_patch_voxel(voxel_grid, u_center, patch_size + 1)
        x = (x[...,1:,1:] + x[...,1:,:-1] + x[...,:-1,1:] + x[...,:-1,:-1]) / 4
        # voxel_grid = torch.from_numpy(np.array(voxel_grid).astype(np.float32))
        # if len(voxel_grid.shape) == 3:
        #     voxel_grid = voxel_grid.permute((2, 0, 1))
        # while len(voxel_grid.shape) < 4:
        #     voxel_grid = voxel_grid[None]
        # u_center = torch.from_numpy(u_center.astype(np.float32))
        # while len(u_center.shape) < 2:
        #     u_center = u_center[None]
        # x = extract_glimpse(
        #     voxel_grid,
        #     (patch_size, patch_size),
        #     u_center + 0.5,
        #     mode='bilinear'
        # )[0]

    return x



def compute_padding(center, patch_size, resolution):
    """
    Return patch crop area and required padding
    :param center: Integer center coordinates of desired patch crop
    :param resolution: Image res (w, h)
    :return:
    """
    w, h = resolution

    # Crop around the patch
    r_min = int(max(0, center[1] - patch_size // 2))
    r_max = int(min(h - 1, center[1] + patch_size // 2 + 1))
    c_min = int(max(0, center[0] - patch_size // 2))
    c_max = int(min(w - 1, center[0] + patch_size // 2 + 1))

    # Determine padding
    pad_l, pad_r, pad_u, pad_d = 0, 0, 0, 0
    if center[1] - patch_size // 2 < 0:
        pad_u = abs(center[1] - patch_size // 2)
    if center[1] + patch_size // 2 + 1 > h - 1:
        pad_d = center[1] + patch_size // 2 + 1 - (h - 1)
    if center[0] - patch_size // 2 < 0:
        pad_l = abs(center[0] - patch_size // 2)
    if center[0] + patch_size // 2 + 1 > w - 1:
        pad_r = center[0] + patch_size // 2 + 1 - (w - 1)

    return (
        (r_min, r_max),
        (c_min, c_max),
        (int(pad_l), int(pad_r), int(pad_u), int(pad_d)),
    )


def read_input(input_path, representation):
    input_path = str(input_path)

    assert os.path.exists(input_path), f"Path to input file {input_path} doesn't exist."

    if "time_surface" in representation:
        return h5py.File(input_path, "r")["time_surface"]

    elif "voxel" in representation:
        return h5py.File(input_path, "r")["voxel_grid"]

    elif "event_stack" in representation:
        return h5py.File(input_path, "r")["event_stack"]

    elif "grayscale" in representation:
        return cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    else:
        print("Unsupported representation")
        exit()


def extract_glimpse(
    input,
    size,
    offsets,
    centered=False,
    normalized=False,
    mode="nearest",
    padding_mode="zeros",
):
    """Returns a set of windows called glimpses extracted at location offsets
    from the input tensor. If the windows only partially overlaps the inputs,
    the non-overlapping areas are handled as defined by :attr:`padding_mode`.
    Options of :attr:`padding_mode` refers to `torch.grid_sample`'s document.
    The result is a 4-D tensor of shape [N, C, h, w].  The channels and batch
    dimensions are the same as that of the input tensor.  The height and width
    of the output windows are specified in the size parameter.
    The argument normalized and centered controls how the windows are built:
        * If the coordinates are normalized but not centered, 0.0 and 1.0 correspond
          to the minimum and maximum of each height and width dimension.
        * If the coordinates are both normalized and centered, they range from
          -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper left
          corner, the lower right corner is located at (1.0, 1.0) and the center
          is at (0, 0).
        * If the coordinates are not normalized they are interpreted as numbers
          of pixels.
    Args:
        input (Tensor): A Tensor of type float32. A 4-D float tensor of shape
            [N, C, H, W].
        size (tuple): 2-element integer tuple specified the
            output glimpses' size. The glimpse height must be specified first,
            following by the glimpse width.
        offsets (Tensor): A Tensor of type float32. A 2-D integer tensor of
            shape [N, 2]  containing the x, y locations of the center
            of each window.
        centered (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are centered relative to the image, in
            which case the (0, 0) offset is relative to the center of the input
            images. If false, the (0,0) offset corresponds to the upper left
            corner of the input images.
        normalized (bool, optional): An optional bool. Defaults to True. indicates
            if the offset coordinates are normalized.
        mode (str, optional): Interpolation mode to calculate output values.
            Defaults to 'bilinear'.
        padding_mode (str, optional): padding mode for values outside the input.
    Raises:
        ValueError: When normalized set False but centered set True
    Returns:
        output (Tensor): A Tensor of same type with input.
    """
    W, H = input.size(-1), input.size(-2)

    if normalized and centered:
        offsets = (offsets + 1) * offsets.new_tensor([W / 2, H / 2])
    elif normalized:
        offsets = offsets * offsets.new_tensor([W, H])
    elif centered:
        raise ValueError("Invalid parameter that offsets centered but not normlized")

    h, w = size
    xs = torch.arange(0, w, dtype=input.dtype, device=input.device) - (w - 1) / 2.0
    ys = torch.arange(0, h, dtype=input.dtype, device=input.device) - (h - 1) / 2.0

    # vy, vx = torch.meshgrid(ys, xs)
    vy, vx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2

    offsets_grid = offsets[:, None, None, :] + grid[None, ...]

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])
    ) / offsets_grid.new_tensor([W / 2, H / 2])

    return torch.nn.functional.grid_sample(
        input, offsets_grid, mode=mode, align_corners=True, padding_mode=padding_mode
    )


def get_patch_voxel_pairs(voxel_grid_0, voxel_grid_1, u_center, patch_size):
    center = np.rint(u_center).astype(int)
    if len(voxel_grid_0.shape) == 2:
        c = 1
        h, w = voxel_grid_0.shape
    else:
        h, w, c = voxel_grid_0.shape

    # Check out-of-bounds
    if not ((0 <= center[0] < w) and (0 <= center[1] < h)):
        return torch.zeros((c * 2, patch_size, patch_size), dtype=torch.float32)

    r_lims, c_lims, pad_lrud = compute_padding(center, patch_size, (w, h))

    if len(voxel_grid_0.shape) == 2:
        x0 = np.array(
            voxel_grid_0[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]
        ).astype(np.float32)
        x1 = np.array(
            voxel_grid_1[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1]]
        ).astype(np.float32)
        x0 = np.expand_dims(x0, axis=2)
        x1 = np.expand_dims(x1, axis=2)
    else:
        x0 = np.array(
            voxel_grid_0[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
        ).astype(np.float32)
        x1 = np.array(
            voxel_grid_1[r_lims[0] : r_lims[1], c_lims[0] : c_lims[1], :]
        ).astype(np.float32)
    x = np.concatenate([x0, x1], axis=2)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    x = torch.nn.functional.pad(x, pad_lrud)
    return x


def array_to_tensor(array):
    # Get patch inputs
    array = np.array(array)
    if len(array.shape) == 2:
        array = np.expand_dims(array, 0)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def get_gt_corners(results_txt_path):
    """
    Get initial corners from EKLT results
    :param results_txt_path:
    :return:
    """
    track_data = np.genfromtxt(results_txt_path)
    t_start = np.min(track_data[:, 1])
    return track_data[track_data[:, 1] == t_start, 2:]


class TrackInterpolator:
    def __init__(self, track_data, terminate_out_of_frame=False, img_size=None):
        self.n_corners = len(np.unique(track_data[:, 0]))
        self.track_interpolators = {}
        self.track_data = {}

        if terminate_out_of_frame:
            track_data = self.terminate_track(track_data, img_size)

        for track_idx in range(self.n_corners):
            track_data_curr = track_data[track_data[:, 0] == track_idx, 1:]
            if track_data_curr.shape[0] > 1:
                t, t_idx = np.unique(track_data_curr[:, 0], return_index=True)
                x = track_data_curr[t_idx, 1]
                y = track_data_curr[t_idx, 2]
                self.track_interpolators[track_idx] = {
                    "x": interp1d(t, x, kind="linear"),
                    "y": interp1d(t, y, kind="linear"),
                    "t_range": [
                        np.min(track_data_curr[:, 0]),
                        np.max(track_data_curr[:, 0]),
                    ],
                }
            else:
                self.track_interpolators[track_idx] = None
            self.track_data[track_idx] = track_data_curr

    def interpolate(self, track_idx, t_query):
        track_interpolator = self.track_interpolators[track_idx]

        if isinstance(track_interpolator, type(None)):
            return None
        elif (
            track_interpolator["t_range"][0]
            <= t_query
            <= track_interpolator["t_range"][1]
        ):
            return np.array(
                [track_interpolator["x"](t_query), track_interpolator["y"](t_query)]
            )
        else:
            return None

    def interpolate_list(self, track_idx, t_query_list):
        track_interpolator = self.track_interpolators[track_idx]

        if (
            track_interpolator["t_range"][0]
            <= np.min(t_query_list)
            <= track_interpolator["t_range"][1]
            and track_interpolator["t_range"][0]
            <= np.max(t_query_list)
            <= track_interpolator["t_range"][1]
        ):
            x_interp = track_interpolator["x"](t_query_list).reshape((-1, 1))
            y_interp = track_interpolator["y"](t_query_list).reshape((-1, 1))
            return np.concatenate([x_interp, y_interp], axis=1)
        else:
            print(
                f"Time range for interpolator is [{track_interpolator['t_range'][0]}, {track_interpolator['t_range'][1]}]"
                f"but queried time range is [{np.min(t_query_list)}, {np.max(t_query_list)}]"
            )
            return None

    def history(self, track_idx, t_query, dt_history):
        track_interpolator = self.track_interpolators[track_idx]
        track_data = self.track_data[track_idx]

        if (
            track_interpolator["t_range"][0]
            <= t_query
            <= track_interpolator["t_range"][1]
        ):
            time_mask = np.logical_and(
                track_data[:, 0] >= t_query - dt_history, track_data[:, 0] <= t_query
            )
            # time_mask = track_data[:, 0] <= t_query
            return track_data[time_mask, 1:]
        else:
            return None

    def terminate_track(self, track_data, img_size):
        new_track_data = []

        for track_idx in np.unique(track_data[:, 0]):
            track_data_curr = track_data[np.isclose(track_data[:, 0], track_idx), :]

            mask_oobx = np.logical_or(
                track_data_curr[:, 2] < 0, track_data_curr[:, 2] > img_size[0] - 1
            )
            mask_ooby = np.logical_or(
                track_data_curr[:, 3] < 0, track_data_curr[:, 3] > img_size[1] - 1
            )
            mask_oob = np.logical_or(mask_oobx, mask_ooby)

            if mask_oob.any():
                idx_oob = int(np.min(np.argwhere(mask_oob)))
                # pdb.set_trace()
                track_data_curr = track_data_curr[:idx_oob, :]

            new_track_data.append(track_data_curr)

        return np.concatenate(new_track_data, axis=0)


class TrackObserver:
    def __init__(self, t_init, u_centers_init):
        self.n_corners = u_centers_init.shape[0]
        idx_col = np.array(range(self.n_corners)).reshape((-1, 1))

        if isinstance(t_init, np.ndarray):
            time_col = t_init
        else:
            time_col = np.ones(idx_col.shape) * t_init
        self.track_data = np.concatenate([idx_col, time_col, u_centers_init], axis=1)

    def add_observation(self, t, u_centers, mask=None):
        idx_col = np.array(range(u_centers.shape[0])).reshape((-1, 1))
        time_col = np.ones(idx_col.shape) * t
        new_track_data = np.concatenate([idx_col, time_col, u_centers], axis=1)

        if not isinstance(mask, type(None)):
            new_track_data = new_track_data[mask, :]

        self.track_data = np.concatenate([self.track_data, new_track_data], axis=0)

    def get_interpolators(self):
        return TrackInterpolator(self.track_data)

    def terminate_oob(self, img_size, padding):
        """
        :param img_size: (H, W)
        :param padding: Padding that must be exceeded for termination to occur
        :return: None, modified internal track data
        """
        new_track_data = []

        for track_idx in np.unique(self.track_data[:, 0]):
            track_data_curr = self.track_data[
                np.isclose(self.track_data[:, 0], track_idx), :
            ]

            mask_oobx = np.logical_or(
                track_data_curr[:, 2] < -padding,
                track_data_curr[:, 2] > img_size[1] - 1 + padding,
            )
            mask_ooby = np.logical_or(
                track_data_curr[:, 3] < -padding,
                track_data_curr[:, 3] > img_size[0] - 1 + padding,
            )
            mask_oob = np.logical_or(mask_oobx, mask_ooby)

            if mask_oob.any():
                idx_oob = int(np.min(np.argwhere(mask_oob)))
                # pdb.set_trace()
                track_data_curr = track_data_curr[:idx_oob, :]

            new_track_data.append(track_data_curr)

        self.track_data = np.concatenate(new_track_data, axis=0)


def read_txt_results(results_txt_path):
    """
    Parse an output txt file from E-KLT or Ours of data rows formatted [id, t, x, y]
    :param results_txt_path:
    :return: TrackInterpolator
    """
    return np.genfromtxt(results_txt_path)


def compute_tracking_errors(
    pred_track_data, gt_track_data, asynchronous=True, error_threshold=5
):
    """
    Compute errors for async methods
    :param track_data: array of predicted tracks
    :param klt_track_data: array of gt tracks
    :param error_threshold: threshold for a live track (5 px used in HASTE paper)
    :return: None, prints the mean relative feature age and mean track-normed error
    """

    fa_rel_arr, te_arr = [], []

    for track_idx in np.unique(pred_track_data[:, 0]):
        gt_track_data_curr = gt_track_data[gt_track_data[:, 0] == track_idx, 1:]
        pred_track_data_curr = pred_track_data[pred_track_data[:, 0] == track_idx, 1:]

        if asynchronous:
            # Extend predicted tracks for asynchronous methods (no prediction -> no motion)
            if gt_track_data_curr[-1, 0] > pred_track_data_curr[-1, 0]:
                pred_track_data_curr = np.concatenate(
                    [
                        pred_track_data_curr,
                        np.array(
                            [
                                gt_track_data_curr[-1, 0],
                                pred_track_data_curr[-1, 1],
                                pred_track_data_curr[-1, 2],
                            ]
                        ).reshape((1, 3)),
                    ],
                    axis=0,
                )
        else:
            # Crop gt track for synchronous method (assumes synchronous method is approx. the same length)
            if gt_track_data_curr[-1, 0] > pred_track_data_curr[-1, 0]:
                gt_time_mask = np.logical_and(
                    gt_track_data_curr[:, 0] >= np.min(pred_track_data_curr[:, 0]),
                    gt_track_data_curr[:, 0] <= np.max(pred_track_data_curr[:, 0]),
                )
                gt_track_data_curr = gt_track_data_curr[gt_time_mask, :]

        gt_track_data_curr_cropped = gt_track_data_curr
        # If KLT could not track the feature, skip it
        if gt_track_data_curr_cropped.shape[0] < 2:
            continue

        # Else, compare against the KLT track
        else:
            # Create predicted track interpolators
            x_interp = interp1d(
                pred_track_data_curr[:, 0],
                pred_track_data_curr[:, 1],
                fill_value="extrapolate",
            )
            y_interp = interp1d(
                pred_track_data_curr[:, 0],
                pred_track_data_curr[:, 2],
                fill_value="extrapolate",
            )

            # Interpolate predicted track at GT timestamps
            pred_x = x_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_y = y_interp(gt_track_data_curr_cropped[:, 0]).reshape((-1, 1))
            pred_track_data_curr_interp = np.concatenate([pred_x, pred_y], axis=1)

            # Compute errors
            tracking_error = (
                gt_track_data_curr_cropped[:, 1:] - pred_track_data_curr_interp
            )
            tracking_error = tracking_error[
                1:, :
            ]  # discard initial location which has no error
            tracking_error = np.linalg.norm(tracking_error, axis=1).reshape((-1,))

            # Compute relative feature age (idx_end is inclusive)
            if (tracking_error > error_threshold).any():
                idx_end = int(np.min(np.argwhere(tracking_error > error_threshold)))
                if idx_end > 1:
                    idx_end = idx_end - 1
                else:
                    idx_end = 0
            else:
                idx_end = -1

            if idx_end == 0:
                fa_rel_arr.append(0)
            else:
                t_end_pred = gt_track_data_curr_cropped[idx_end, 0]
                fa = t_end_pred - gt_track_data_curr_cropped[0, 0]
                dt_track = gt_track_data_curr[-1, 0] - gt_track_data_curr[0, 0]
                # Ignore tracks that are short-lived as in HASTE and E-KLT?
                fa_rel = fa / dt_track

                if idx_end != -1:
                    te = np.mean(tracking_error[1 : idx_end + 1])
                else:
                    te = np.mean(tracking_error[1:])

                fa_rel_arr.append(fa_rel)
                te_arr.append(te)

    return np.array(fa_rel_arr).reshape((-1,)), np.array(te_arr).reshape((-1,))