import os
import hydra
import torch
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm

from config import CODE_ROOT


def evaluate_mf(model, evaluate_name, cfg):
    pl.seed_everything(1234)

    os.system(f'mkdir -p {CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}')

    data_module = hydra.utils.instantiate(cfg.data)
    data_module.setup()

    evaluate_mf_split(model, data_module.train_dataloader(strict_order=True), evaluate_name, cfg, 'train')
    evaluate_mf_split(model, data_module.val_dataloader(strict_order=True), evaluate_name, cfg, 'test')

    
def evaluate_mf_split(model, dataloader, evaluate_name, cfg, split_name):
    model.eval()
    assert split_name in ['train', 'test']

    if split_name == 'train':
        num_tracks = cfg.data.n_train
    elif split_name == 'test':
        num_tracks = cfg.data.n_val

    with torch.no_grad():
        error_result = torch.zeros((num_tracks, cfg.max_unrolls, 2), dtype=torch.float32, device=model.device)
        error_result[:] = -1

        track_idx = 0
        for batch_dataloaders in tqdm(dataloader):
            if isinstance(batch_dataloaders[0], list):
                batch_dataloaders = sum(batch_dataloaders, [])

            nb = len(batch_dataloaders)
            nt = len(batch_dataloaders)

            # Preparation
            for bl in batch_dataloaders:
                bl.auto_update_center = False
                # print(bl.seq_name, bl.track_idx)

            # Unroll network
            model.reset(nt)

            attn_mask = torch.zeros([nt, nt], device=model.device)
            for i_src in range(nt):
                # src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                src_path = batch_dataloaders[i_src].seq_name
                for i_target in range(nt):
                    # attn_mask[i_src, i_target] = src_path == batch_dataloaders[i_target].track_path.split("/")[-3]
                    attn_mask[i_src, i_target] = src_path == batch_dataloaders[i_target].seq_name
            attn_mask = (1 - attn_mask).bool()

            for i_unroll in range(cfg.max_unrolls):
                # Construct batched x and y for current timestep

                if model.pyramid:
                    x, xx, y = [], [], []
                    for bl in batch_dataloaders:
                        x_j, xx_j, y_j = bl.get_next_2()
                        x.append(x_j)
                        xx.append(xx_j)
                        y.append(y_j)
                    x = torch.cat(x, dim=0).to(model.device)
                    xx = torch.cat(xx, dim=0).to(model.device)
                    y = torch.cat(y, dim=0).to(model.device)

                    # Inference
                    y_hat = model.forward(x, xx, attn_mask)
                else:
                    x, y = [], []
                    for bl in batch_dataloaders:
                        x_j, y_j = bl.get_next()
                        x.append(x_j)
                        y.append(y_j)
                    x = torch.cat(x, dim=0).to(model.device)
                    y = torch.cat(y, dim=0).to(model.device)

                    # Inference
                    y_hat = model.forward(x, attn_mask)

                # Accumulate losses
                error_result[track_idx:track_idx+nt, i_unroll] = torch.abs(y - y_hat)

                # Pass predicted flow to dataloader
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat[j, :])

            track_idx += nt

        # N, T, 2
        np.save(f'{CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}/{split_name}_set.npy', error_result.cpu().numpy())
