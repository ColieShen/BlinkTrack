import os
import torch


# THREAD_NUM = 4
# torch.set_num_threads(THREAD_NUM)
# os.environ ['OMP_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['MKL_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_NUM}'

import os
import math
import time
import hydra
import torch
import numpy as np
import random
import pickle
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from typing import Iterator, Sequence
from pathlib import Path
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.sampler import Sampler

from util.vis import event_voxel_to_rgb
from util.cfg import propagate_keys
from util.data import get_patch_voxel, read_input
from util.augment import augment_input, augment_rotation, augment_rotation_2, augment_scale, augment_scale_2, augment_track, unaugment_rotation, unaugment_scale
from script.vis.vis_evaluate_result import draw_traj, make_video



MAX_ROTATION_ANGLE = 15
MAX_SCALE_CHANGE_PERCENTAGE = 20
MAX_PERSPECTIVE_THETA = 0.01
MAX_TRANSLATION = 3
GT_TRANSLATION_MULTIPLE = 0
TIME_TRANSLATION = 0
LOST_FRAME = [-1, -1]
LOST_TRANSLATION = [0, 0]


def retrieve_track_tuples(extra_dir, track_name):
    track_tuples = []
    for extra_seq_dir in tqdm(extra_dir.iterdir(), desc="Fetching track paths..."):
        # Ignore hidden dirs
        if str(extra_seq_dir.stem).startswith("."):
            continue

        # Check if has valid tracks
        track_path = os.path.join(str(extra_seq_dir), "tracks", f"{track_name}.gt.txt")
        if not os.path.exists(track_path):
            continue

        # Store paths
        track_data = np.genfromtxt(track_path)
        if track_data.size == 0:
            continue
        n_tracks = len(np.unique(track_data[:, 0]))
        for track_idx in range(n_tracks):
            track_tuples.append((track_path, track_idx))

    return track_tuples


def recurrent_collate(batch_dataloaders):
    return batch_dataloaders


class TrackData:
    """
    Dataloader for a single feature track. Returns input patches and displacement labels relative to
    the current feature location. Current feature location is either updated manually via accumulate_y_hat()
    or automatically via the ground-truth displacement.
    """

    def __init__(self, track_tuple, config):
        """
        Dataset for a single feature track
        :param track_tuple: (Path to track.gt.txt, track_id)
        :param config:
        """
        self.config = config

        # Track augmentation (disabled atm)
        if False:
            # if config.augment:
            self.flipped_lr = random.choice([True, False])
            self.flipped_ud = random.choice([True, False])
            # self.rotation_angle = round(random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE))
            self.rotation_angle = 0
        else:
            self.flipped_lr, self.flipped_ud, self.rotation_angle = False, False, 0
        self.last_aug_angle, self.last_aug_scale = 0.0, 1.0

        # Get input paths
        self.frame_paths = config.frame_paths
        self.event_paths = config.event_paths

        # TODO: Do this in a non-hacky way
        if "0.0100" in self.event_paths[0]:
            self.index_multiplier = 1
        elif "0.0200" in self.event_paths[0]:
            self.index_multiplier = 2
        elif "0.0300" in self.event_paths[0]:
            self.index_multiplier = 3
        elif "0.0400" in self.event_paths[0]:
            self.index_multiplier = 4
        else:
            print("Unsupported dt for feature track")
            raise NotImplementedError

        # Input and Labels
        ref_input = read_input(self.frame_paths[0], "grayscale")
        # ref_input = augment_input(
        #     ref_input, self.flipped_lr, self.flipped_ud, self.rotation_angle
        # )

        self.track_path, self.track_idx = track_tuple
        track_data = np.genfromtxt(self.track_path)
        self.track_data = track_data[track_data[:, 0] == self.track_idx, 2:4]
        self.track_vis = (track_data[track_data[:, 0] == self.track_idx, 4:] == 1).astype(bool)
        if self.track_vis.size == 0:
            self.track_vis = np.ones_like((track_data[track_data[:, 0] == self.track_idx, 2:3])).astype(bool)
        # self.track_data = augment_track(
        #     self.track_data,
        #     self.flipped_lr,
        #     self.flipped_ud,
        #     self.rotation_angle,
        #     (ref_input.shape[1], ref_input.shape[0]),
        # )

        self.u_center = self.track_data[0, :]
        self.u_center_gt = self.track_data[0, :]
        self.u_center_init = self.track_data[0, :]

        self.x_ref = get_patch_voxel(ref_input, self.u_center, config.patch_size)
        self.x_ref_2 = get_patch_voxel(ref_input, self.u_center, config.patch_size * 2)

        # Pathing for input data
        self.seq_name = Path(self.track_path).parents[1].stem

        # Operational
        self.time_idx = 0
        self.auto_update_center = False

        # Representation-specific Settings
        if "grayscale" in config.representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(config.representation[-1])
            # in v2, we have separate temporal bins for each event polarity
            if "v2" in config.representation:
                self.channels_in_per_patch *= 2

        self.lost_frame = np.random.randint(LOST_FRAME[0], LOST_FRAME[1] + 1)


    def reset(self):
        self.time_idx = 0
        self.u_center = self.u_center_init

    def set_current(self, idx):
        self.time_idx = idx
        self.u_center = self.track_data[self.time_idx * self.index_multiplier, :]
        self.u_center_gt = self.track_data[self.time_idx * self.index_multiplier, :]

    def accumulate_y_hat(self, y_hat):
        """
        Accumulate predicted flows if using predictions instead of gt patches
        :param y_hat: 2-element Tensor
        """
        # Disregard confidence
        y_hat = y_hat[:2]

        # Unaugment the predicted label
        if self.config.augment:
            # y_hat = unaugment_perspective(y_hat.detach().cpu(), self.last_aug_perspective[0], self.last_aug_perspective[1])
            y_hat = unaugment_rotation(y_hat.detach().cpu(), self.last_aug_angle)
            y_hat = unaugment_scale(y_hat, self.last_aug_scale)

            # Translation augmentation
            y_hat += (2 * torch.rand_like(y_hat) - 1) * (MAX_TRANSLATION + TIME_TRANSLATION * self.time_idx)


        self.u_center += y_hat.detach().cpu().numpy().reshape((2,))

    def get_current_gt_displacement(self):
        return self.track_data[self.time_idx * self.index_multiplier, :] - self.track_data[(self.time_idx - 1) * self.index_multiplier, :]
    
    def get_next_gt_displacement(self):
        return self.track_data[(self.time_idx + 1) * self.index_multiplier, :] - self.track_data[self.time_idx * self.index_multiplier, :]

    def get_next(self):
        # Increment time
        self.time_idx += 1

        # gt augment
        if GT_TRANSLATION_MULTIPLE != 0:
            gt_displacement = self.get_current_gt_displacement()
            self.u_center -= GT_TRANSLATION_MULTIPLE * gt_displacement

        if self.time_idx == self.lost_frame:
            self.u_center += (2 * np.random.rand(2) - 1) * np.random.uniform(LOST_TRANSLATION[0], LOST_TRANSLATION[1])

        # Round feature location to accommodate get_patch_voxel
        self.u_center = np.rint(self.u_center)

        # Update gt location
        self.u_center_gt = self.track_data[self.time_idx * self.index_multiplier, :]

        # Update total flow
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)

        # # Update xref (Uncomment if combining frames with events)
        # if self.time_idx % 5 == 0:
        #     frame_idx = self.time_idx // 5
        #     ref_input = read_input(self.frame_paths[frame_idx], 'grayscale')
        #     self.x_ref = get_patch_voxel2(ref_input, self.u_center, self.config.patch_size)

        # Get patch inputs for event representation
        input_1 = read_input(
            self.event_paths[self.time_idx], self.config.representation
        )
        input_1 = augment_input(
            input_1, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )
        x = get_patch_voxel(input_1, self.u_center, self.config.patch_size)
        x = torch.cat([x, self.x_ref], dim=0)

        # Augmentation
        if self.config.augment:
            # Sample rotation and scale
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_scaling,
            ) = augment_scale(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            (
                x[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation(
                x[0 : self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )
            # x[0:self.channels_in_per_patch, :, :], y, self.last_aug_perspective = augment_perspective(x[0:self.channels_in_per_patch, :, :], y,
            #                                                                                           theta=MAX_PERSPECTIVE_THETA)

        # Update center location for next patch
        if self.auto_update_center:
            self.u_center = self.u_center + y.numpy().reshape((2,))

        # Minor Processing Steps
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)

        vis = self.track_vis[self.time_idx * self.index_multiplier]
        vis = torch.from_numpy(vis)
        vis = torch.unsqueeze(vis, 0)

        return x, y, vis
    

    def get_next_2(self):

        # if self.x_ref_2 is None:
        #     self.x_ref_2 = get_patch_voxel(self.ref_input, self.u_center, self.config.patch_size * 2)

        # Increment time
        self.time_idx += 1

        # gt augment
        if GT_TRANSLATION_MULTIPLE != 0:
            gt_displacement = self.get_current_gt_displacement()
            self.u_center -= GT_TRANSLATION_MULTIPLE * gt_displacement

        if self.time_idx == self.lost_frame:
            self.u_center += (2 * np.random.rand(2) - 1) * np.random.uniform(LOST_TRANSLATION[0], LOST_TRANSLATION[1])

        # Round feature location to accommodate get_patch_voxel
        self.u_center = np.rint(self.u_center)

        # Update gt location
        self.u_center_gt = self.track_data[self.time_idx * self.index_multiplier, :]

        # Update total flow
        y = (self.u_center_gt - self.u_center).astype(np.float32)
        y = torch.from_numpy(y)

        # # Update xref (Uncomment if combining frames with events)
        # if self.time_idx % 5 == 0:
        #     frame_idx = self.time_idx // 5
        #     ref_input = read_input(self.frame_paths[frame_idx], 'grayscale')
        #     self.x_ref = get_patch_voxel2(ref_input, self.u_center, self.config.patch_size)

        # Get patch inputs for event representation
        input_1 = read_input(
            self.event_paths[self.time_idx], self.config.representation
        )
        input_1 = augment_input(
            input_1, self.flipped_lr, self.flipped_ud, self.rotation_angle
        )
        x = get_patch_voxel(input_1, self.u_center, self.config.patch_size)
        x = torch.cat([x, self.x_ref], dim=0)
        xx = get_patch_voxel(input_1, self.u_center, self.config.patch_size * 2)
        xx = torch.cat([xx, self.x_ref_2], dim=0)

        # Augmentation
        if self.config.augment:
            # Sample rotation and scale
            (
                x[0 : self.channels_in_per_patch, :, :],
                xx[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_scaling,
            ) = augment_scale_2(
                x[0 : self.channels_in_per_patch, :, :],
                xx[0 : self.channels_in_per_patch, :, :],
                y,
                max_scale_percentage=MAX_SCALE_CHANGE_PERCENTAGE,
            )
            (
                x[0 : self.channels_in_per_patch, :, :],
                xx[0 : self.channels_in_per_patch, :, :],
                y,
                self.last_aug_angle,
            ) = augment_rotation_2(
                x[0 : self.channels_in_per_patch, :, :],
                xx[0 : self.channels_in_per_patch, :, :],
                y,
                max_rotation_deg=MAX_ROTATION_ANGLE,
            )
            # x[0:self.channels_in_per_patch, :, :], y, self.last_aug_perspective = augment_perspective(x[0:self.channels_in_per_patch, :, :], y,
            #                                                                                           theta=MAX_PERSPECTIVE_THETA)

        # Update center location for next patch
        if self.auto_update_center:
            self.u_center = self.u_center + y.numpy().reshape((2,))

        # Minor Processing Steps
        x = torch.unsqueeze(x, 0)
        xx = torch.unsqueeze(xx, 0)
        y = torch.unsqueeze(y, 0)

        vis = self.track_vis[self.time_idx * self.index_multiplier]
        vis = torch.from_numpy(vis)
        vis = torch.unsqueeze(vis, 0)

        return x, xx, y, vis



@dataclass
class TrackDataConfig:
    frame_paths: list
    event_paths: list
    patch_size: int
    representation: str
    track_name: str
    augment: bool


class SubSequenceRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], batch_size=32, test=False) -> None:
        self.indices = indices
        self.batch_size = batch_size
        self.test = test

    def __iter__(self) -> Iterator[int]:
        # n_samples_per_seq = 8
        n_samples_per_seq = self.batch_size
        if not self.test:
            shifted_start = torch.randint(n_samples_per_seq, [1])
            shifted_indices = self.indices[shifted_start:] + self.indices[:shifted_start]
            for i in torch.randperm(math.ceil(len(self.indices) / n_samples_per_seq)):
                i_idx = i * n_samples_per_seq

                for i_yield in range(i_idx, min(i_idx + n_samples_per_seq, self.__len__())):
                    yield shifted_indices[i_yield]
        else:
            for i in range(math.ceil(len(self.indices) / n_samples_per_seq)):
                i_idx = i * n_samples_per_seq

                for i_yield in range(i_idx, min(i_idx + n_samples_per_seq, self.__len__())):
                    yield self.indices[i_yield]


    def __len__(self) -> int:
        return len(self.indices)


class TrackDataset(Dataset):
    """
    Dataloader for a collection of feature tracks. __getitem__ returns an instance of TrackData.
    """

    def __init__(
        self,
        track_tuples,
        get_frame_paths_fn,
        get_event_paths_fn,
        augment=False,
        patch_size=31,
        track_name="shitomasi_custom_v5",
        representation="time_surfaces_v2_5",
    ):
        super(TrackDataset, self).__init__()
        self.track_tuples = track_tuples
        self.get_frame_paths_fn = get_frame_paths_fn
        self.get_event_paths_fn = get_event_paths_fn
        self.patch_size = patch_size
        self.track_name = track_name
        self.representation = representation
        self.augment = augment
        print(f"Initialized recurrent dataset with {len(self.track_tuples)} tracks.")

    def __len__(self):
        return len(self.track_tuples)

    def __getitem__(self, idx_track):
        track_tuple = self.track_tuples[idx_track]
        data_config = TrackDataConfig(
            self.get_frame_paths_fn(track_tuple[0]),
            self.get_event_paths_fn(track_tuple[0], self.representation),
            self.patch_size,
            self.representation,
            self.track_name,
            self.augment,
        )
        return TrackData(track_tuple, data_config)


class MFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        extra_dir,
        dt=0.0100,
        batch_size=16,
        num_workers=4,
        patch_size=31,
        augment=False,
        n_train=20000,
        n_val=2000,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_1",
        mixed_dt=False,
        force_shuffle=False,
        displacement_augment=0,
        time_augment=0,
        lost_augment=None,
        version='origin',
        **kwargs,
    ):
        super(MFDataModule, self).__init__()

        random.seed(1234)

        self.num_workers = num_workers
        self.n_train = n_train
        self.n_val = n_val
        self._has_prepared_data = True

        self.data_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.mixed_dt = mixed_dt
        self.dt = dt
        self.representation = representation
        self.patch_size = patch_size
        self.track_name = track_name

        self.force_shuffle = force_shuffle
        global GT_TRANSLATION_MULTIPLE
        GT_TRANSLATION_MULTIPLE = displacement_augment
        global TIME_TRANSLATION
        TIME_TRANSLATION = time_augment

        if lost_augment is not None:
            global LOST_FRAME, LOST_TRANSLATION
            LOST_FRAME = lost_augment[:2]
            LOST_TRANSLATION = lost_augment[2:]

        assert version in ['origin', 'plus']
        self.version = version

        self.dataset_train, self.dataset_val = None, None

        self.split_track_tuples = {}
        self.split_max_samples = {"train": n_train, "test": n_val}
        for split_name in ["train", "test"]:
            cache_path = (
                self.extra_dir / split_name / ".cache" / f"{track_name}.paths.pkl"
            )
            if cache_path.exists():
                with open(str(cache_path), "rb") as cache_f:
                    track_tuples = pickle.load(cache_f)
            else:
                os.system(f'mkdir -p {self.extra_dir / split_name / ".cache" }')
                track_tuples = retrieve_track_tuples(
                    self.extra_dir / split_name, track_name
                )
                with open(str(cache_path), "wb") as cache_f:
                    pickle.dump(track_tuples, cache_f)

            # Shuffle and trim
            n_tracks = len(track_tuples)
            track_tuples_array = np.asarray(track_tuples)
            track_tuples_array = track_tuples_array[: (n_tracks // 64) * 64, :]
            track_tuples_array = track_tuples_array.reshape([(n_tracks // 64), 64, 2])
            rand_perm = np.random.permutation((n_tracks // 64))
            track_tuples_array = track_tuples_array[rand_perm, :, :].reshape(
                (n_tracks // 64) * 64, 2
            )   # may cut track from same seq
            track_tuples_array[:, 1] = track_tuples_array[:, 1].astype(int)
            track_tuples = []
            for i in range(track_tuples_array.shape[0]):
                track_tuples.append(
                    [track_tuples_array[i, 0], int(track_tuples_array[i, 1])]
                )

            if self.split_max_samples[split_name] < len(track_tuples):
                track_tuples = track_tuples[: self.split_max_samples[split_name]]
            self.split_track_tuples[split_name] = track_tuples


    def get_frame_paths(self, track_path):
        images_dir = Path(
            os.path.split(track_path)[0]
            .replace("_extra", "")
            .replace("tracks", "images")
        )
        if self.version == 'origin':
            return sorted(
                [
                    frame_p
                    for frame_p in glob(str(images_dir / "*.png"))
                    if 400000
                    <= int(os.path.split(frame_p)[1].replace(".png", ""))
                    <= 900000
                ]
            )
        elif self.version == 'plus':
            return sorted(glob(str(images_dir / "*.png")))
        

    def get_event_paths_mixed_dt(self, track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{random.choice([0.0100, 0.0200]):.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        if self.version == 'origin':
            return [
                event_p
                for event_p in event_files
                if 400000 <= float(os.path.split(event_p)[1].replace(".h5", "")) <= 900000
            ]
        elif self.version == 'plus':
            return event_files
        

    def get_event_paths(self, track_path, rep, dt):
        event_files = sorted(
            glob(
                str(
                    Path(os.path.split(track_path)[0].replace("tracks", "events"))
                    / f"{dt:.4f}"
                    / rep
                    / "*.h5"
                )
            )
        )
        if self.version == 'origin':
            return [
                event_p
                for event_p in event_files
                if 400000 <= float(os.path.split(event_p)[1].replace(".h5", "")) <= 900000
            ]
        elif self.version == 'plus':
            return event_files
        

    def setup(self, stage=None):
        # Create train and val splits
        self.dataset_train = TrackDataset(
            self.split_track_tuples["train"],
            self.get_frame_paths,
            partial(self.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(self.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=self.augment,
        )
        self.dataset_val = TrackDataset(
            self.split_track_tuples["test"],
            self.get_frame_paths,
            partial(self.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(self.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=False,
        )

    def train_dataloader(self, strict_order=False):
        if self.force_shuffle:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=recurrent_collate,
                pin_memory=True,
                shuffle=True,
            )

        if strict_order:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_train.__len__())),
            batch_size=self.batch_size,
        )

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )

    def val_dataloader(self, strict_order=False):
        if self.force_shuffle:
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=recurrent_collate,
                pin_memory=True,
                shuffle=True,
            )

        if strict_order:
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        subseq_sampler = SubSequenceRandomSampler(
            list(range(self.dataset_val.__len__())),
            batch_size=self.batch_size,
        )

        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )
    

@hydra.main(config_path="../configs", config_name="train_defaults")
def check_multiflow(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys(cfg)

    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup()
    for batch_dataloaders in data_module.train_dataloader():
        if isinstance(batch_dataloaders[0], list):
            batch_dataloaders = sum(batch_dataloaders, [])
        for batch_dataloader in batch_dataloaders:
            vis_multiflow_track(batch_dataloader)
        return


def vis_multiflow_track(track_loader):
    img_list = []
    for event in track_loader.event_paths:
        event = read_input(event, track_loader.config.representation)
        event = np.transpose(event, (2, 0, 1))
        img_list.append(event_voxel_to_rgb(event))
    
    track_data = track_loader.track_data[::track_loader.index_multiplier]
    track_name = track_loader.track_path.split('/')[-3]
    track_idx = track_loader.track_idx
    track_vis = track_loader.track_vis[::track_loader.index_multiplier]
    print(track_vis.sum())

    img_list = draw_traj(img_list, track_data, vis=track_vis, length=5, thickness=2, color=(255, 0, 0))
    make_video(
        img_list, f'multiflow_{track_name}_{track_idx}_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{time.time()}.mp4', fps=5)


if __name__ == '__main__':
    check_multiflow()