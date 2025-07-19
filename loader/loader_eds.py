from datetime import datetime
import os
import cv2
import math
import yaml
import torch
import numpy as np
import pickle
from abc import ABC, abstractmethod
from tqdm import tqdm
from enum import Enum
from glob import glob
from typing import Iterator, Sequence
from pathlib import Path
from functools import partial
from omegaconf import OmegaConf
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from config import CODE_ROOT
from util.vis import event_voxel_to_rgb, time_surface_to_rgb
from util.data import array_to_tensor, extract_glimpse, get_patch_voxel, get_patch_voxel_pairs, read_input, skew


def vis_subseq(dataset):
    img_list = []
    next_rgb_idx = dataset.first_frame_idx + 1
    for t, evt in tqdm(dataset.globals(), total=dataset.n_events):
        # print(t)
        evt_img = event_voxel_to_rgb(np.array(evt[0]))
        H, W, _ = evt_img.shape
        img_list.append(evt_img)
        if dataset.check_next_rgb_time(next_rgb_idx):
            t_, rgb = dataset.get_frame(next_rgb_idx)
            # rgb = cv2.resize(rgb, (W, H))
            # rgb = rgb[..., None]
            # rgb = np.tile(rgb, (1, 1, 3))
            img_list.append(rgb)
            next_rgb_idx += 1

    make_video(img_list, f'{CODE_ROOT}/vis/subseq_vis_{datetime.now().strftime("%Y%m%d-%H%M%S")}.mp4', fps=2)


class PoseInterpolator:
    def __init__(self, pose_data, mode="linear"):
        """
        :param pose_data: Nx7 numpy array with [t, x, y, z, qx, qy, qz, qw] as the row format
        """
        self.pose_data = pose_data
        self.x_interp = interp1d(
            pose_data[:, 0], pose_data[:, 1], kind=mode, bounds_error=True
        )
        self.y_interp = interp1d(
            pose_data[:, 0], pose_data[:, 2], kind=mode, bounds_error=True
        )
        self.z_interp = interp1d(
            pose_data[:, 0], pose_data[:, 3], kind=mode, bounds_error=True
        )
        self.rot_interp = Slerp(pose_data[:, 0], Rotation.from_quat(pose_data[:, 4:]))
        #
        # self.qx_interp = interp1d(pose_data[:, 0], pose_data[:, 4], kind='linear')
        # self.qy_interp = interp1d(pose_data[:, 0], pose_data[:, 5], kind='linear')
        # self.qz_interp = interp1d(pose_data[:, 0], pose_data[:, 6], kind='linear')
        # self.qw_interp = interp1d(pose_data[:, 0], pose_data[:, 7], kind='linear')

    def interpolate(self, t):
        """
        Interpolate 6-dof pose from the initial pose data
        :param t: Query time at which to interpolate the pose
        :return: 4x4 Transformation matrix T_j_W
        """
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        return np.linalg.inv(T_W_j)

    def interpolate_colmap(self, t):
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        T_j_W = np.linalg.inv(T_W_j)
        quat = Rotation.from_matrix(T_j_W[:3, :3]).as_quat()
        return np.asarray(
            [T_j_W[0, 3], T_j_W[1, 3], T_j_W[2, 3], quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float32,
        )



class SequenceDataset(ABC):
    """
    Data class without ground-truth labels
    """

    def __init__(self):
        self.u_centers, self.u_centers_init = None, None
        self.n_tracks = None
        self.event_first, self.frame_first = None, None
        self.t_now, self.t_init = None, None
        self.n_events, self.n_frames = None, None
        self.patch_size = None
        self.has_poses = False
        self.device = "cpu"
        self.first_frame_idx, self.first_event_idx = 0, 0
        self.x_ref = torch.zeros(1)
        self.x_ref_2 = torch.zeros(1)

    def initialize(self, max_keypoints=30):
        self.frame_first = cv2.imread(
            self.first_image_path,
            cv2.IMREAD_GRAYSCALE,
        )
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])
        self.initialize_keypoints(max_keypoints)
        self.initialize_reference_patches()

    def override_keypoints(self, keypoints):
        self.u_centers = keypoints
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

        self.initialize_reference_patches()

    def override_refframe(self, img_frame_idx):
        pass

    def override_eventframe(self, img_frame_idx):
        pass

    def override_seqlength(self, seq_length):
        pass

    def initialize_keypoints(self, max_keypoints):
        self.u_centers = cv2.goodFeaturesToTrack(
            self.frame_first,
            max_keypoints,
            qualityLevel=self.corner_config.qualityLevel,
            minDistance=self.corner_config.minDistance,
            k=self.corner_config.k,
            useHarrisDetector=self.corner_config.useHarrisDetector,
            blockSize=self.corner_config.blockSize,
        ).reshape((-1, 2))
        self.u_centers = torch.from_numpy(self.u_centers.astype(np.float32))
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        if self.n_tracks == 0:
            raise ValueError("There are no corners in the initial frame")

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)
        self.x_ref_2 = self.x_ref_2.to(self.device)

    def accumulate_y_hat(self, y_hat):
        if y_hat.device != self.device:
            self.device = y_hat.device
            self.move_centers()

        self.u_centers += y_hat.detach()

    def frames(self):
        """
        :return: generator over frames
        """
        pass

    def events(self):
        """
        :return: generator over event representations
        """
        pass

    def events_2(self):
        """
        :return: generator over event representations
        """
        pass

    #
    # def get_track_data(self):
    #     track_data = []
    #     for i in range(self.u_centers.shape[0]):
    #         track_data.append([i, self.t_now, self.u_centers[i, 0], self.u_centers[i, 1]])
    #     return track_data

    def get_patches(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        # 0.5 offset is needed due to coordinate system of grid_sample
        x = extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )
        xx = extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size * 2, self.patch_size * 2),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )

        return x, xx

    def get_patches_new(self, arr_h5, padding=4):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param arr_h5: h5 file for the input event representation
        :return: (n_tracks, c, p, p) tensor
        """
        # Extract expanded patches from the h5 files
        u_centers_np = self.u_centers.detach().cpu().numpy()
        x_patches = []
        for i in range(self.n_tracks):
            u_center = u_centers_np[i, :]
            u_center_rounded = np.rint(u_center)
            u_center_offset = (
                u_center - u_center_rounded + ((self.patch_size + padding) // 2.0)
            )
            x_patch_expanded = get_patch_voxel(
                arr_h5, u_center_rounded.reshape((-1,)), self.patch_size + padding
            ).unsqueeze(0)
            x_patch = extract_glimpse(
                x_patch_expanded,
                (self.patch_size, self.patch_size),
                torch.from_numpy(u_center_offset).view((1, 2)) + 0.5,
                mode="nearest",
            )
            x_patches.append(x_patch)
        return torch.cat(x_patches, dim=0)

    @abstractmethod
    def initialize_reference_patches(self):
        pass

    def get_next(self):
        """
        Abstract method for getting input patches and epipolar lines
        :return: input patches (n_corners, C, patch_size, patch_size) and epipolar lines (n_corners, 3)
        """
        pass

    def get_frame(self, image_idx, grayscale=False):
        pass

    def get_frame_path(self, image_idx):
        pass



class EDSSubseq(SequenceDataset):
    # ToDo: Add to config file
    pose_r = 3
    pose_mode = False

    def __init__(
        self,
        root_dir,
        sequence_name,
        n_frames,
        patch_size,
        representation,
        dt,
        corner_config,
        include_prev=False,
        fused=False,
        grayscale_ref=True,
        use_colmap_poses=True,
        global_mode=False,
        image_folder="images_corrected",
        event_folder="events",
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.include_prev = include_prev
        self.dt, self.dt_us = dt, dt * 1e6
        self.grayscale_ref = grayscale_ref
        self.use_colmap_poses = use_colmap_poses
        self.global_mode = global_mode
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config
        self.image_folder = image_folder
        self.event_folder = event_folder

        # Determine number of frames
        self.frame_dir = self.root_dir / sequence_name / self.image_folder
        max_frames = len(list(self.frame_dir.iterdir())) - 1
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        if not self.pose_mode:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / self.event_folder
                / f"{dt:.4f}"
                / f"{self.representation}"
            )
        else:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / self.event_folder
                / f"pose_{self.pose_r:.0f}"
                / f"{self.representation}"
            )
        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated for a dt of {self.dt}"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(
            str(self.sequence_dir / "images_timestamps.txt")
        )

        # Read poses and camera matrix
        if self.use_colmap_poses:
            pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
        else:
            pose_data_path = self.sequence_dir / "stamped_groundtruth.txt"
        # self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        with open(str(self.root_dir / "calib.yaml"), "r") as fh:
            intrinsics = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]["intrinsics"]
            self.camera_matrix = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ]
            )
            self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2

        if self.include_prev:
            self.cropping_fn = get_patch_voxel_pairs
        else:
            self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.current_idx = 0
        self.t_init = self.frame_ts_arr[0] * 1e-6
        self.t_end = self.frame_ts_arr[-1] * 1e-6
        self.t_now = self.t_init

        # Pose interpolator for epipolar geometry
        # self.pose_interpolator = PoseInterpolator(self.pose_data)
        # self.T_last_W = self.pose_interpolator.interpolate(self.t_now)

        # Get counts
        # self.n_events = min(int(np.ceil((self.t_end - self.t_init) / self.dt)), len(glob(str(
        #     self.root_dir
        #     / sequence_name
        #     / self.event_folder
        #     / f"{dt:.4f}"
        #     / f"{self.representation}"
        #     / "*.h5"))))
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first imgs
        self.first_image_path = self.get_frame_path(self.first_frame_idx)
        
        # self.event_first = array_to_tensor(read_input(str(self.dir_representation / '0000000.h5'), self.representation))

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return
    
    def reset(self):
        self.t_now = self.t_init
        self.current_idx = 0
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        if "grayscale" in self.representation or self.grayscale_ref:
            ref_input = (
                torch.from_numpy(self.frame_first.astype(np.float32) / 255)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        else:
            ref_input = self.event_first.unsqueeze(0)
        self.x_ref, self.x_ref_2 = self.get_patches(ref_input)

        # for i in range(self.n_tracks):
        #     x = get_patch_voxel(ref_input, self.u_centers[i, :], self.patch_size)
        #     self.x_ref.append(x[:self.channels_in_per_patch, :, :].unsqueeze(0))
        # self.x_ref = torch.cat(self.x_ref, dim=0)

    def globals(self):
        for i in range(self.first_event_idx + 1, self.first_event_idx + self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )

            yield self.t_now, x.unsqueeze(0)

    def get_current_event(self):
        # Get patch inputs and set current time
        if not self.pose_mode:
            self.t_now += self.dt
            input_1 = read_input(
                self.dir_representation
                / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                self.representation,
            )
        else:
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[self.current_idx])[
                        1
                    ].replace(".h5", "")
                )
                * 1e-6
            )
            input_1 = read_input(
                self.event_representation_paths[self.current_idx], self.representation
            )

            if self.current_idx > 0 and self.current_idx % self.pose_r == 0:
                ref_input = cv2.imread(
                    str(
                        self.frame_dir
                        / (
                            "frame_"
                            + f"{self.current_idx // self.pose_r}".zfill(10)
                            + ".png"
                        )
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
                ref_input = (
                    torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.x_ref.device)
                )
                self.x_ref = extract_glimpse(
                    ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                    (self.patch_size, self.patch_size),
                    self.u_centers.detach() + 0.5,
                    mode="bilinear",
                )

        input_1 = np.array(input_1)
        input_1 = np.transpose(input_1, (2, 0, 1))
        input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

        x = extract_glimpse(
            input_1.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
        )
        x = torch.cat([x, self.x_ref], dim=1)

        return self.t_now, x

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        current_idx = 0

        for i in range(current_idx, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[current_idx])[
                        1
                    ].replace(".h5", "")
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[current_idx], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            current_idx += 1

            yield self.t_now, events_repr

    def events(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            self.current_idx = 0
        else:
            self.current_idx = self.first_event_idx + 1

        for self.current_idx in range(self.current_idx, self.first_event_idx + self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                if not os.path.exists(str(self.dir_representation / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5")):
                    print('===========================================')
                    print(f'{str(self.dir_representation / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5")} not exist')
                    print('not enough event! quit!')
                    print('===========================================')
                    break
                input_1 = read_input(
                    self.dir_representation
                    / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(
                            self.event_representation_paths[self.current_idx]
                        )[1].replace(".h5", "")
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[self.current_idx],
                    self.representation,
                )

                if self.current_idx > 0 and self.current_idx % self.pose_r == 0:
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / (
                                "frame_"
                                + f"{self.current_idx // self.pose_r}".zfill(10)
                                + ".png"
                            )
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x


    def events_2(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            self.current_idx = 0
        else:
            self.current_idx = self.first_event_idx + 1

        for self.current_idx in range(self.current_idx, self.first_event_idx + self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                if not os.path.exists(str(self.dir_representation / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5")):
                    print('===========================================')
                    print(f'{str(self.dir_representation / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5")} not exist')
                    print('not enough event! quit!')
                    print('===========================================')
                    break
                input_1 = read_input(
                    self.dir_representation
                    / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(
                            self.event_representation_paths[self.current_idx]
                        )[1].replace(".h5", "")
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[self.current_idx],
                    self.representation,
                )

                if self.current_idx > 0 and self.current_idx % self.pose_r == 0:
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / (
                                "frame_"
                                + f"{self.current_idx // self.pose_r}".zfill(10)
                                + ".png"
                            )
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            xx = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size * 2, self.patch_size * 2),
                self.u_centers.detach() + 0.5,
            )
            xx = torch.cat([xx, self.x_ref_2], dim=1)

            yield self.t_now, x, xx

    def frames(self):
        for i in range(self.first_frame_idx, self.first_frame_idx + self.n_frames):
            # Update time info
            self.t_now = self.frame_ts_arr[i] * 1e-6

            frame = cv2.imread(
                str(
                    self.sequence_dir
                    / self.image_folder
                    / ("frame_" + f"{i}".zfill(10) + ".png")
                ),
                cv2.IMREAD_UNCHANGED,
            )
            yield self.t_now, frame

    def get_next(self):
        """Strictly for pose supervision"""

        # Update time info
        self.t_now += self.dt

        self.current_idx += 1
        # DEBUG: Use grayscale frame timestamps
        # self.t_now = self.frame_ts_arr[self.current_idx]*1e-6

        # Get patch inputs
        input_1 = read_input(
            self.dir_representation
            / f"{str(int(self.current_idx * self.dt_us)).zfill(7)}.h5",
            self.representation,
        )
        x = array_to_tensor(input_1)
        x_patches = self.get_patches(x)

        # Get epipolar lines
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_last = T_now_W @ np.linalg.inv(self.T_last_W)
        T_last_now = np.linalg.inv(T_now_last)
        self.T_last_W = T_now_W
        F = (
            self.camera_matrix_inv.T
            @ skew(T_last_now[:3, 3])
            @ T_last_now[:3, :3]
            @ self.camera_matrix_inv
        )
        u_centers = self.u_centers.detach().cpu().numpy()
        u_centers_homo = np.concatenate(
            [u_centers, np.ones((u_centers.shape[0], 1))], axis=1
        )
        l_epi = torch.from_numpy(u_centers_homo @ F)

        return x_patches, l_epi
    
    def check_next_rgb_time(self, next_rgb_idx):
        if next_rgb_idx >= self.first_frame_idx + self.n_frames:
            return False

        next_rgb_t = self.frame_ts_arr[next_rgb_idx] * 1e-6
        return self.t_now + self.dt > next_rgb_t

    def get_frame(self, image_idx, grayscale=False):
        # Update time info
        # self.t_now = self.frame_ts_arr[idx]

        if grayscale:
            frame = cv2.imread(
                self.get_frame_path(image_idx),
                cv2.IMREAD_GRAYSCALE,
            )
        else:
            frame = cv2.imread(
                self.get_frame_path(image_idx),
                cv2.IMREAD_UNCHANGED,
            )
        return self.frame_ts_arr[image_idx] * 1e-6, frame

    def get_frame_path(self, image_idx):
        return str(
                    self.sequence_dir
                    / self.image_folder
                    / ("frame_" + f"{image_idx}".zfill(10) + ".png")
                )
    

def vis_seq(seq):
    print(f'Visualizing {seq}')

    with open(f'{CODE_ROOT}/configs/eval_real_defaults.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    if 'boxes' in seq or 'shapes' in seq:
        dataset = 'ec'
    else:
        dataset = 'eds'
    seq_path = f'{cfg.eval_dataset_path}/{dataset}_subseq/{seq}'

    if dataset == 'eds':
        timestamp = np.loadtxt(f'{seq_path}/images_timestamps.txt') # 1e-6s
    elif dataset == 'ec':
        timestamp = np.round(np.loadtxt(f'{seq_path}/images.txt') * 1e6) # 1e-6s

    vis_type = 'image'
    # vis_type = 'event'


    traj_gts = np.genfromtxt(f'{cfg.eval_dataset_path}/gt_tracks/{seq}_occ.gt.txt')
    traj_ids = np.unique(traj_gts[..., 0])
    traj_times = list(np.round(np.unique(traj_gts[..., 1]) * 1e6))

    img_list = []
    img_path = f'{seq_path}/{cfg.image_folder}'
    # for img in sorted(os.listdir(img_path)):
    #     if not img.endswith('.png'):
    #         continue
    #     img = cv2.imread(f'{img_path}/{img}', cv2.IMREAD_GRAYSCALE)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     img_list.append(img)
    final_timestamp = []

    for img_idx, img_timestamp in enumerate(timestamp):
        if img_timestamp not in traj_times:
            continue
        if vis_type == 'image':
            if dataset == 'eds':
                img = cv2.imread(f'{img_path}/frame_{img_idx:010d}.png', cv2.IMREAD_GRAYSCALE)
            elif dataset == 'ec':
                img = cv2.imread(f'{img_path}/frame_{img_idx:08d}.png', cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_list.append(img)
        elif vis_type == 'event':
            # TODO: ec
            if dataset == 'eds':
                evt_time = int((img_timestamp - timestamp[0]) // 5000 * 5000)
                event_repr = read_input(f'{seq_path}/{cfg.event_folder}/0.0050/time_surfaces_v2_5/{evt_time:07d}.h5', 'time_surfaces_v2_5')
            elif dataset == 'ec':
                evt_time = int((img_timestamp - timestamp[0]) // 10000 * 10000)
                event_repr = read_input(f'{seq_path}/{cfg.event_folder}/0.0100/time_surfaces_v2_5/{evt_time:07d}.h5', 'time_surfaces_v2_5')
            event_repr = np.transpose(event_repr, (2, 0, 1))
            img_list.append(time_surface_to_rgb(event_repr))

        final_timestamp.append(img_timestamp)
        
    final_timestamp = np.array(final_timestamp) / 1e6
    for traj_id in traj_ids:
        traj_gt = traj_gts[traj_gts[..., 0] == traj_id]
        x_interp = interp1d(
            traj_gt[..., 1],
            traj_gt[..., 2],
            fill_value="extrapolate",
        )
        y_interp = interp1d(
            traj_gt[..., 1],
            traj_gt[..., 3],
            fill_value="extrapolate",
        )
        vis_interp = interp1d(
            traj_gt[..., 1],
            traj_gt[..., 4],
            fill_value="extrapolate",
        )
        pred_x = x_interp(final_timestamp).reshape((-1, 1))
        pred_y = y_interp(final_timestamp).reshape((-1, 1))
        traj = np.concatenate([pred_x, pred_y], axis=1)
        viss = vis_interp(final_timestamp).reshape((-1, 1))
        viss[viss < 0.1] = 0    # TODO: not precise
        img_list = draw_traj(img_list, traj, vis=viss, length=5, thickness=2, color=(0, 255, 0))

    # traj_gts = np.genfromtxt(f'{cfg.eval_dataset_path}/gt_tracks/{seq}_occ.gt.txt')
    # for traj_id in traj_ids:
    #     traj_gt = traj_gts[traj_gts[..., 0] == traj_id]

    #     x_interp = interp1d(
    #         traj_gt[..., 1],
    #         traj_gt[..., 2],
    #         fill_value="extrapolate",
    #     )
    #     y_interp = interp1d(
    #         traj_gt[..., 1],
    #         traj_gt[..., 3],
    #         fill_value="extrapolate",
    #     )

    #     # pred_x = x_interp(traj_gt[:, 1]).reshape((-1, 1))
    #     # pred_y = y_interp(traj_gt[:, 1]).reshape((-1, 1))
    #     pred_x = x_interp(traj_times).reshape((-1, 1))
    #     pred_y = y_interp(traj_times).reshape((-1, 1))
    #     traj = np.concatenate([pred_x, pred_y], axis=1)
    #     img_list = draw_traj(img_list, traj, length=5, thickness=2, color=(255, 0, 0))

    make_video(
        img_list, f'vis/vis_{seq}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.mp4', fps=5)


if __name__ == '__main__':
    vis_seq('peanuts_light_160_386_1')
    vis_seq('shapes_6dof_485_565')
