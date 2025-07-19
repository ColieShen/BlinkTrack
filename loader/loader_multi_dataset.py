import hydra
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule

from util.cfg import propagate_keys
from loader.loader_multiflow import MFDataModule, SubSequenceRandomSampler, recurrent_collate, vis_multiflow_track


class MultiDataset(LightningDataModule):
    def __init__(
        self,
        batch_size=16,
        num_workers=4,
        patch_size=31,
        track_name="shitomasi_custom",
        representation="time_surfaces_v2_1",
        force_shuffle=False,
        dataset_list={},
        **kwargs,
    ):
        super(MultiDataset, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.track_name = track_name
        self.representation = representation
        self.force_shuffle = force_shuffle

        self.dataset_list = []
        for dataset in dataset_list:
            dataset_cfg = dataset_list[dataset]
            dataset_cfg._target_ = MFDataModule
            dataset_cfg.batch_size = batch_size
            dataset_cfg.num_workers = num_workers
            dataset_cfg.patch_size = patch_size
            dataset_cfg.track_name = track_name
            dataset_cfg.representation = representation
            dataset_cfg.force_shuffle = force_shuffle
            self.dataset_list.append(hydra.utils.instantiate(dataset_cfg))


    def setup(self, stage=None):
        for dataset in self.dataset_list:
            dataset.setup()


    def train_dataloader(self, strict_order=False):
        train_dataset = ConcatDataset([dataset.dataset_train for dataset in self.dataset_list])

        if self.force_shuffle:
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=recurrent_collate,
                pin_memory=True, 
                shuffle=True,
            )

        if strict_order:
            return DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        subseq_sampler = SubSequenceRandomSampler(
            list(range(train_dataset.__len__())),
            batch_size=self.batch_size,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=recurrent_collate,
            pin_memory=True,
            sampler=subseq_sampler,
        )

    def val_dataloader(self, strict_order=False):
        val_dataset = ConcatDataset([dataset.dataset_val for dataset in self.dataset_list])

        if self.force_shuffle:
            return DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=recurrent_collate,
                pin_memory=True,
                shuffle=True,
            )

        if strict_order:
            return DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=recurrent_collate,
                pin_memory=True,
            )

        subseq_sampler = SubSequenceRandomSampler(
            list(range(val_dataset.__len__())),
            batch_size=self.batch_size,
        )

        return DataLoader(
            val_dataset,
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
            print(batch_dataloader.track_path, batch_dataloader.track_idx)
            # vis_multiflow_track(batch_dataloader)
        # return
    

if __name__ == '__main__':
    check_multiflow()