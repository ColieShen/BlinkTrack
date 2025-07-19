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
import hydra
import torch
import numpy as np
import random
import multiprocessing
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import DataLoader

from util.cfg import propagate_keys
from loader.loader_multiflow import MFDataModule, TrackData, TrackDataConfig, TrackDataset, recurrent_collate, vis_multiflow_track


class TrackDataset_batch(TrackDataset):
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
        tracks_per_batch=1,
    ):
        super(TrackDataset_batch, self).__init__(
            track_tuples=track_tuples,
            get_frame_paths_fn=get_frame_paths_fn,
            get_event_paths_fn=get_event_paths_fn,
            augment=augment,
            patch_size=patch_size,
            track_name=track_name,
            representation=representation,
        )
        
        self.tracks_per_batch = tracks_per_batch

        self.idx_offset = multiprocessing.Value('i', 0)  # integer shared attribute
        with self.idx_offset.get_lock():
            self.idx_offset.value = random.randrange(self.tracks_per_batch)
        self.get_count = multiprocessing.Value('i', 0)  # integer shared attribute
        with self.get_count.get_lock():
            self.get_count.value = 0

    def __len__(self):
        return len(self.track_tuples) // self.tracks_per_batch

    def __getitem__(self, idx_track):
        output = []
        for i in range(self.tracks_per_batch):
            output.append(self.get_item(idx_track * self.tracks_per_batch + i))
        self.count_offset()
        return output
    
    def random_offset(self):
        with self.idx_offset.get_lock():
            self.idx_offset.value = random.randrange(self.tracks_per_batch)

    def count_offset(self):
        with self.get_count.get_lock():
            self.get_count.value = self.get_count.value + 1
            if self.get_count.value == self.__len__():
                self.random_offset()
                self.get_count.value = 0
    
    def get_item(self, actual_idx):
        # print(actual_idx, self.idx_offset.value, len(self.track_tuples), (actual_idx + self.idx_offset.value) % len(self.track_tuples))
        actual_idx = (actual_idx + self.idx_offset.value) % len(self.track_tuples)
        track_tuple = self.track_tuples[actual_idx]
        data_config = TrackDataConfig(
            self.get_frame_paths_fn(track_tuple[0]),
            self.get_event_paths_fn(track_tuple[0], self.representation),
            self.patch_size,
            self.representation,
            self.track_name,
            self.augment,
        )
        return TrackData(track_tuple, data_config)


class MFDataModule_batch(MFDataModule):
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
        tracks_per_batch=1,
        displacement_augment=0,
        time_augment=0,
        lost_augment=None,
        version='origin',
        **kwargs,
    ):
        super(MFDataModule_batch, self).__init__(
            data_dir=data_dir,
            extra_dir=extra_dir,
            dt=dt,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_size=patch_size,
            augment=augment,
            n_train=n_train,
            n_val=n_val,
            track_name=track_name,
            representation=representation,
            mixed_dt=mixed_dt,
            displacement_augment=displacement_augment,
            time_augment=time_augment,
            lost_augment=lost_augment,
            version=version,
        )

        self.tracks_per_batch = tracks_per_batch

    def setup(self, stage=None):
        # Create train and val splits
        self.dataset_train = TrackDataset_batch(
            self.split_track_tuples["train"],
            self.get_frame_paths,
            partial(self.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(self.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=self.augment,
            tracks_per_batch=self.tracks_per_batch,
        )
        self.dataset_val = TrackDataset_batch(
            self.split_track_tuples["test"],
            self.get_frame_paths,
            partial(self.get_event_paths_mixed_dt, dt=self.dt)
            if self.mixed_dt
            else partial(self.get_event_paths, dt=self.dt),
            patch_size=self.patch_size,
            track_name=self.track_name,
            representation=self.representation,
            augment=False,
            tracks_per_batch=self.tracks_per_batch,
        )

    def train_dataloader(self, strict_order=False):
        if strict_order:
            return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self, strict_order=False):
        if strict_order:
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=recurrent_collate,
            pin_memory=True,
            shuffle=True,
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
            print(batch_dataloader.track_path, batch_dataloader.track_idx)
            batch_dataloader.auto_update_center = False
            for unroll_idx in range(cfg.max_unrolls):
                x_j, xx_j, y_j = batch_dataloader.get_next_2()
                print(y_j)
                batch_dataloader.accumulate_y_hat(y_j)
            # print(batch_dataloader.track_path, batch_dataloader.track_idx)
            # vis_multiflow_track(batch_dataloader)
            # import ipdb; ipdb.set_trace()
            # return
        break
        # print('========================================')


if __name__ == '__main__':
    check_multiflow()