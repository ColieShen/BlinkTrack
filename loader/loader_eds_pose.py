import os
import cv2
import yaml
import h5py
import math
import torch
import numpy as np
import pickle
import random
from abc import ABC, abstractmethod
from tqdm import tqdm
from enum import Enum
from glob import glob
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


from util.data import extract_glimpse, get_patch_voxel, read_input
from src.evaluate_eds import CornerConfig
from loader.loader_eds import PoseInterpolator
from loader.loader_multiflow import recurrent_collate


class EvalDatasetType(Enum):
    EC = 0
    EDS = 1


class PoseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_type,
        root_dir,
        n_event_representations_per_frame,
        n_train,
        n_val,
        batch_size,
        num_workers,
        patch_size,
        representation,
        n_frames_skip,
        **kwargs,
    ):
        super(PoseDataModule, self).__init__()
        assert dataset_type in [
            "EDS",
            "EC",
            "FPV",
        ], "Dataset type must be one of EDS, EC, or FPV"
        if dataset_type == "EDS":
            self.dataset_type = EvalDatasetType.EDS
        elif dataset_type == "EC":
            self.dataset_type = EvalDatasetType.EC
        else:
            raise NotImplementedError("Dataset type not supported for pose training")

        self.root_dir = root_dir
        self.n_event_representations_per_frame = n_event_representations_per_frame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.representation = representation
        self.n_frames_skip = n_frames_skip
        self.n_train = n_train
        self.n_val = n_val
        self.dataset_train, self.dataset_val = None, None

    def setup(self, stage=None):
        self.dataset_train = PoseDataset(
            self.dataset_type,
            self.root_dir,
            self.n_event_representations_per_frame,
            self.n_train,
            self.representation,
            self.patch_size,
            self.n_frames_skip,
        )
        # Change this later
        self.dataset_val = PoseDataset(
            self.dataset_type,
            self.root_dir,
            self.n_event_representations_per_frame,
            self.n_val,
            self.representation,
            self.patch_size,
            self.n_frames_skip,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
        )


class PoseDataset(Dataset):
    """
    Dataset encapsulating multiple sequences/subsequences.
    The events for each sequence/subsequence are loaded into memory and used to instantiate segments.
    Segments start at an image index, so that keypoints can be extracted.
    """

    def __init__(
        self,
        dataset_type,
        root_dir,
        n_event_representations_per_frame,
        max_segments,
        representation,
        patch_size,
        n_frames_skip,
    ):
        if dataset_type == EvalDatasetType.EDS:
            self.segment_dataset_class = EDSPoseSegmentDataset
        elif dataset_type == EvalDatasetType.EC:
            self.segment_dataset_class = ECPoseSegmentDataset
        else:
            raise NotImplementedError

        self.root_dir = Path(root_dir)
        self.idx2sequence = []
        self.n_event_representations_per_frame = n_event_representations_per_frame
        self.n_frames_skip = n_frames_skip
        self.max_segments = max_segments
        self.patch_size = patch_size
        self.representation = representation

        for sequence_dir in self.root_dir.iterdir():
            # ToDo: Change back for EDS Pose training
            if "." in str(sequence_dir) or str(sequence_dir.stem) in [
                "rocket_earth_dark",
                "peanuts_dark",
                "ziggy_and_fuzz",
            ]:
                # if '.' in str(sequence_dir):
                continue
            sequence_name = sequence_dir.stem

            # Don't consider frames after the last pose timestamp
            pose_interpolator = self.segment_dataset_class.get_pose_interpolator(
                sequence_dir
            )
            pose_ts_min = np.min(pose_interpolator.pose_data[:, 0])
            pose_ts_max = np.max(pose_interpolator.pose_data[:, 0]) - 4 * 0.010

            frame_ts_arr = self.segment_dataset_class.get_frame_timestamps(sequence_dir)
            inrange_mask = np.logical_and(
                frame_ts_arr > pose_ts_min, frame_ts_arr < pose_ts_max
            )
            frame_indices = np.nonzero(inrange_mask)[0]
            frame_paths = self.segment_dataset_class.get_frame_paths(sequence_dir)

            cached_mappings_path = sequence_dir / "valid_mappings.pkl"
            if cached_mappings_path.exists():
                with open(cached_mappings_path, "rb") as cached_mappings_f:
                    new_mappings = pickle.load(cached_mappings_f)
            else:
                frame_indices_skipped = list(
                    range(
                        np.min(frame_indices), np.max(frame_indices) - 4, n_frames_skip
                    )
                )
                new_mappings = []
                for i in tqdm(
                    frame_indices_skipped, desc="Checking corners in starting frames..."
                ):
                    img = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
                    kp = cv2.goodFeaturesToTrack(
                        img, 2, 0.3, 15, blockSize=11, useHarrisDetector=False, k=0.15
                    )
                    if not isinstance(kp, type(None)):
                        new_mappings.append((sequence_name, i))

                with open(cached_mappings_path, "wb") as cached_mappings_f:
                    pickle.dump(new_mappings, cached_mappings_f)

            random.shuffle(new_mappings)

            self.idx2sequence += new_mappings

        random.shuffle(self.idx2sequence)
        if len(self.idx2sequence) > self.max_segments:
            self.idx2sequence = self.idx2sequence[: self.max_segments]

    def __len__(self):
        return len(list(self.idx2sequence))

    def __getitem__(self, idx_segment):
        sequence, idx_start = self.idx2sequence[idx_segment]
        return self.segment_dataset_class(
            self.root_dir / sequence,
            idx_start,
            self.patch_size,
            self.representation,
            self.n_event_representations_per_frame,
        )


class SequenceDatasetV2(ABC):
    """Abstract class for real data.
    Defines loaders for timestamps, pose data, and input paths.
    Defines generators for frames, events, and even_events"""

    def __init__(self):
        self.idx_start = 0
        pass

    # TODO: fix these
    def initialize_pose_and_calib(self):
        # Loading
        (
            self.camera_matrix,
            self.camera_matrix_inv,
            self.distortion_coeffs,
        ) = self.get_calibration(self.sequence_dir)
        self.frame_ts_arr = self.get_frame_timestamps(self.sequence_dir)
        self.pose_interpolator = self.get_pose_interpolator(self.sequence_dir)

    def initialize_time_and_input_paths(self, pose_mode, dt_or_r):
        if pose_mode:
            self.event_representation_paths = self.get_even_event_paths(
                self.sequence_dir, self.representation, dt_or_r
            )
        else:
            self.event_representation_paths = self.get_event_paths(
                self.sequence_dir, self.representation, dt_or_r
            )
        self.frame_paths = self.get_frame_paths(self.sequence_dir)

        # Initial time and pose
        self.t_init = self.frame_ts_arr[self.idx_start]
        self.t_now = self.t_init
        self.T_first_W = self.pose_interpolator.interpolate(self.t_now)

    def initialize_centers_and_ref_patches(self, corner_config):
        # Initial keypoints and patches
        self.frame_first = cv2.imread(
            self.frame_paths[self.idx_start], cv2.IMREAD_GRAYSCALE
        )
        self.u_centers = self.get_keypoints(self.frame_first, corner_config)
        self.u_centers_init = self.u_centers.clone()
        self.n_tracks = self.u_centers.shape[0]

        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref = self.get_patches(ref_input)
        self.x_ref_2 = self.get_patches_2(ref_input)

    def accumulate_y_hat(self, y_hat):
        if y_hat.device != self.device:
            self.device = y_hat.device
            self.move_centers()

        self.u_centers += y_hat.detach()

    def configure_patches_iterator(self, corner_config, pose_mode, dt_or_r):
        # Initialize indices and timestamps
        self.initialize_time_and_input_paths(pose_mode, dt_or_r)
        self.initialize_centers_and_ref_patches(corner_config)
        self.n_events = len(self.event_representation_paths)
        self.n_frames = len(self.frame_paths)
        self.pose_mode = pose_mode
        self.dt_or_r = dt_or_r

    def get_patches_iterator(self):
        # Initialize reference patches
        for i in range(1, self.n_events):
            # Update time info
            if not self.pose_mode:
                self.t_now += self.dt_or_r
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )

            # Get patch inputs
            input_1 = read_input(
                self.event_representation_paths[i], self.representation
            )

            x = self.get_patches_new(input_1)
            if x.device != self.x_ref.device:
                self.x_ref = self.x_ref.to(x.device)

            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x

    @staticmethod
    @abstractmethod
    def get_frame_timestamps(sequence_dir):
        """
        :return: (-1,) array of timestamps in seconds
        """
        pass

    @staticmethod
    @abstractmethod
    def get_pose_interpolator(sequence_dir):
        """
        :return: PoseInterpolator object instantiated from sequence's pose data
        """
        pass

    @staticmethod
    @abstractmethod
    def get_calibration(sequence_dir):
        """
        :return: dict with keys 'camera_matrix', 'camera_matrix_inv', and 'distortion_coeffs'
        """
        pass

    @staticmethod
    @abstractmethod
    def get_frame_paths(sequence_dir):
        """
        :return: sorted list of frame paths
        """
        pass

    @staticmethod
    @abstractmethod
    def get_frames_iterator(sequence_dir):
        pass

    @staticmethod
    @abstractmethod
    def get_events_iterator(sequence_dir, dt):
        pass

    @staticmethod
    @abstractmethod
    def get_events(sequence_dir):
        pass

    @staticmethod
    def get_event_paths(sequence_dir, representation, dt):
        """
        :return: sorted list of event paths for a given representation and time-delta
        """
        return sorted(
            glob(
                str(
                    sequence_dir / "events" / f"{dt:.4f}" / f"{representation}" / "*.h5"
                )
            )
        )

    @staticmethod
    def get_even_event_paths(sequence_dir, representation, r):
        """
        :return: sorted list of event paths for a given representation and time-delta
        """
        # return sorted(glob(str(sequence_dir / "events" / f"pose_{r:.0f}" / representation / "*.h5")))
        return sorted(glob(str(sequence_dir / "events" / f"pose_{r:.0f}" / "*.h5")))

    @staticmethod
    def get_keypoints(frame_start, corner_config: CornerConfig):
        """
        :param frame_start:
        :param max_keypoints:
        :return: (N, 2) torch float32 tensor of initial keypoint locations
        """
        keypoints = cv2.goodFeaturesToTrack(
            frame_start,
            maxCorners=15,
            qualityLevel=corner_config.qualityLevel,
            minDistance=corner_config.minDistance,
            k=corner_config.k,
            useHarrisDetector=corner_config.useHarrisDetector,
            blockSize=corner_config.blockSize,
        ).reshape((-1, 2))

        if keypoints.shape[0] == 0:
            print("No corners in frame")
            exit()

        elif keypoints.shape[0] > corner_config.maxCorners:
            indices = list(range(keypoints.shape[0]))
            sampled_indices = random.sample(indices, corner_config.maxCorners)
            keypoints = keypoints[sampled_indices, :]

        return torch.from_numpy(keypoints.astype(np.float32))

    def move_centers(self):
        self.u_centers = self.u_centers.to(self.device)
        self.u_centers_init = self.u_centers_init.to(self.device)
        self.x_ref = self.x_ref.to(self.device)
        self.x_ref_2 = self.x_ref_2.to(self.device)

    def get_patches(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        # # OLD
        # # Round feature locations
        # self.u_centers = np.rint(self.u_centers)
        #
        # # Get patch crops
        # x_patches = []
        # for i_track in range(self.n_tracks):
        #     x_patches.append(get_patch_tensor(f, self.u_centers[i_track, :], self.patch_size))
        # x_patches = torch.cat(x_patches, dim=0)
        #
        # return x_patches

        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        return extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size, self.patch_size),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )
    
    def get_patches_2(self, f):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param f:
        :return:
        """
        # # OLD
        # # Round feature locations
        # self.u_centers = np.rint(self.u_centers)
        #
        # # Get patch crops
        # x_patches = []
        # for i_track in range(self.n_tracks):
        #     x_patches.append(get_patch_tensor(f, self.u_centers[i_track, :], self.patch_size))
        # x_patches = torch.cat(x_patches, dim=0)
        #
        # return x_patches

        if f.device != self.device:
            self.device = f.device
            self.move_centers()

        return extract_glimpse(
            f.repeat(self.u_centers.size(0), 1, 1, 1),
            (self.patch_size * 2, self.patch_size * 2),
            self.u_centers.detach() + 0.5,
            mode="nearest",
        )

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
    
    def get_patches_new_2(self, arr_h5, padding=4):
        """
        Return a tensor of patches for each feature centrally cropped around it's location
        :param arr_h5: h5 file for the input event representation
        :return: (n_tracks, c, p, p) tensor
        """

        # Extract expanded patches from the h5 files
        u_centers_np = self.u_centers.detach().cpu().numpy()
        x_patches = []
        xx_patches = []
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

            u_center_offset = (
                u_center - u_center_rounded + ((self.patch_size * 2 + padding) // 2.0)
            )
            xx_patch_expanded = get_patch_voxel(
                arr_h5, u_center_rounded.reshape((-1,)), self.patch_size * 2 + padding
            ).unsqueeze(0)
            xx_patch = extract_glimpse(
                xx_patch_expanded,
                (self.patch_size * 2, self.patch_size * 2),
                torch.from_numpy(u_center_offset).view((1, 2)) + 0.5,
                mode="nearest",
            )
            xx_patches.append(xx_patch)

        return torch.cat(x_patches, dim=0), torch.cat(xx_patches, dim=0)


class EDSSubseqDatasetV2(SequenceDatasetV2):
    resolution = (640, 480)

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_events(sequence_dir):
        """
        :param sequence_dir:
        :return: event dict with keys t, x, y, p. Values are numpy arrays.
        """
        with h5py.File(str(sequence_dir / "events_corrected.h5")) as h5f:
            return {
                "x": np.array(h5f["x"]),
                "y": np.array(h5f["y"]),
                "t": np.array(h5f["t"]),
                "p": np.array(h5f["p"]),
            }

    @staticmethod
    def get_frames_iterator(sequence_dir):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        frame_paths = EDSSubseqDatasetV2.get_frame_paths(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        assert len(frame_paths) == len(frame_ts_arr)

        for frame_idx in range(len(frame_paths)):
            yield frame_ts_arr[frame_idx], cv2.imread(
                frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE
            )

    @staticmethod
    def get_events_iterator(sequence_dir, dt):
        """
        :param sequence_dir: Path object
        :param dt: floating, seconds
        :return: Iterator over events between the frame timestamps
        """
        events = EDSSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)
        dt_elapsed = 0

        for t1 in np.arange(frame_ts_arr[0], frame_ts_arr[-1], dt):
            t1 = t1 * 1e6
            t0 = t1 - dt * 1e6
            idx0 = np.searchsorted(events["t"], t0, side="left")
            idx1 = np.searchsorted(events["t"], t1, side="right")

            yield dt_elapsed, {
                "x": events["x"][idx0:idx1],
                "y": events["y"][idx0:idx1],
                "p": events["p"][idx0:idx1],
                "t": events["t"][idx0:idx1],
            }

            dt_elapsed += dt

    @staticmethod
    def get_even_events_iterator(sequence_dir, r):
        """
        Return an iterator that (roughly) evenly splits events between frames into temporal bins.
        :param sequence_dir:
        :param r: number of temporal bins between frames
        :return:
        """
        events = EDSSubseqDatasetV2.get_events(sequence_dir)
        frame_ts_arr = EDSSubseqDatasetV2.get_frame_timestamps(sequence_dir)

        for i in range(len(frame_ts_arr) - 1):
            dt_us = (frame_ts_arr[i + 1] - frame_ts_arr[i]) * 1e6 // r

            t0 = frame_ts_arr[i] * 1e6
            for j in range(r):
                if j == r - 1:
                    t1 = frame_ts_arr[i + 1] * 1e6
                else:
                    t1 = t0 + dt_us

                idx0 = np.searchsorted(events["t"], t0, side="left")
                idx1 = np.searchsorted(events["t"], t1, side="right")
                yield t1 * 1e-6, {
                    "x": events["x"][idx0:idx1],
                    "y": events["y"][idx0:idx1],
                    "p": events["p"][idx0:idx1],
                    "t": events["t"][idx0:idx1],
                }
                t0 = t1

    @staticmethod
    def get_frame_paths(sequence_dir):
        return sorted(glob(str(sequence_dir / "images_corrected" / "*.png")))

    @staticmethod
    def get_frame_timestamps(sequence_dir):
        return np.genfromtxt(str(sequence_dir / "images_timestamps.txt")) * 1e-6

    @staticmethod
    def get_pose_interpolator(sequence_dir):
        colmap_pose_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
        if colmap_pose_path.exists():
            pose_data_path = sequence_dir / "colmap" / "stamped_groundtruth.txt"
            pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        else:
            pose_data = np.genfromtxt(
                str(sequence_dir / "stamped_groundtruth.txt"), skip_header=1
            )
        return PoseInterpolator(pose_data)

    @staticmethod
    def get_calibration(sequence_dir):
        with open(str(sequence_dir / ".." / "calib.yaml"), "r") as fh:
            data = yaml.load(fh, Loader=yaml.SafeLoader)["cam0"]
            camera_matrix = np.array(
                [
                    [data["intrinsics"][0], 0, data["intrinsics"][2]],
                    [0, data["intrinsics"][1], data["intrinsics"][3]],
                    [0, 0, 1],
                ]
            )
            camera_matrix_inv = np.linalg.inv(camera_matrix)
            distortion_coeffs = np.array(data["distortion_coeffs"]).reshape((-1,))
        return camera_matrix, camera_matrix_inv, distortion_coeffs


class EDSPoseSegmentDataset(EDSSubseqDatasetV2):
    def __init__(
        self, sequence_dir, idx_start, patch_size, representation, r=3, max_keypoints=2
    ):
        super().__init__()
        self.sequence_dir = sequence_dir
        self.idx_start = idx_start
        self.n_event_representations_per_frame = r
        self.patch_size = patch_size
        self.representation = representation
        self.device = torch.device("cpu")

        # Initial indices
        self.idx = self.idx_start
        self.event_representation_idx = (
            self.idx * self.n_event_representations_per_frame
        )
        self.initialize_pose_and_calib()
        self.initialize_time_and_input_paths(pose_mode=True, dt_or_r=r)

        # ToDO: Change back Pose Training
        max_keypoints = 4
        self.initialize_centers_and_ref_patches(
            CornerConfig(max_keypoints, 0.3, 15, 0.15, False, 11)
        )

    def get_next(self):
        self.event_representation_idx += 1
        self.t_now = (
            float(
                os.path.split(
                    self.event_representation_paths[self.event_representation_idx]
                )[1].replace(".h5", "")
            )
            * 1e-6
        )

        x_h5 = read_input(
            self.event_representation_paths[self.event_representation_idx],
            "time_surfaces_v2_5",
        )
        x_patches = self.get_patches_new(x_h5)
        x_patches = torch.cat([x_patches.to(self.x_ref.device), self.x_ref], dim=1)

        # # EPIPOLAR POSE SUPERVISION
        # # Get epipolar lines
        # T_now_W = self.pose_interpolator.interpolate(self.t_now)
        # T_first_now = np.linalg.inv(T_now_W @ np.linalg.inv(self.T_first_W))
        # F = self.camera_matrix_inv.T @ skew(T_first_now[:3, 3]) @ T_first_now[:3, :3] @ self.camera_matrix_inv
        # u_centers = self.u_centers.detach().cpu().numpy()
        # u_centers_homo = np.concatenate([u_centers, np.ones((u_centers.shape[0], 1))], axis=1)
        # l_epi = torch.from_numpy(u_centers_homo @ F)
        #
        # return x_patches, l_epi

        # REPROJECTION SUPERVISION
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_first = T_now_W @ np.linalg.inv(self.T_first_W)
        projection_matrix = (self.camera_matrix @ T_now_first[:3, :]).astype(np.float32)
        projection_matrices = (
            torch.from_numpy(projection_matrix)
            .unsqueeze(0)
            .repeat(x_patches.size(0), 1, 1)
        )

        return x_patches, projection_matrices
    

    def get_next_2(self):
        self.event_representation_idx += 1
        self.t_now = (
            float(
                os.path.split(
                    self.event_representation_paths[self.event_representation_idx]
                )[1].replace(".h5", "")
            )
            * 1e-6
        )

        x_h5 = read_input(
            self.event_representation_paths[self.event_representation_idx],
            "time_surfaces_v2_5",
        )
        x_patches, xx_patches = self.get_patches_new_2(x_h5)
        x_patches = torch.cat([x_patches.to(self.x_ref.device), self.x_ref], dim=1)
        xx_patches = torch.cat([xx_patches.to(self.x_ref_2.device), self.x_ref_2], dim=1)

        # # EPIPOLAR POSE SUPERVISION
        # # Get epipolar lines
        # T_now_W = self.pose_interpolator.interpolate(self.t_now)
        # T_first_now = np.linalg.inv(T_now_W @ np.linalg.inv(self.T_first_W))
        # F = self.camera_matrix_inv.T @ skew(T_first_now[:3, 3]) @ T_first_now[:3, :3] @ self.camera_matrix_inv
        # u_centers = self.u_centers.detach().cpu().numpy()
        # u_centers_homo = np.concatenate([u_centers, np.ones((u_centers.shape[0], 1))], axis=1)
        # l_epi = torch.from_numpy(u_centers_homo @ F)
        #
        # return x_patches, l_epi

        # REPROJECTION SUPERVISION
        T_now_W = self.pose_interpolator.interpolate(self.t_now)
        T_now_first = T_now_W @ np.linalg.inv(self.T_first_W)
        projection_matrix = (self.camera_matrix @ T_now_first[:3, :]).astype(np.float32)
        projection_matrices = (
            torch.from_numpy(projection_matrix)
            .unsqueeze(0)
            .repeat(x_patches.size(0), 1, 1)
        )

        return x_patches, xx_patches, projection_matrices
