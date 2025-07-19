import os
import torch

# THREAD_NUM = 4
# torch.set_num_threads(THREAD_NUM)
# os.environ ['OMP_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['MKL_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_NUM}'
# os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_NUM}'

import hydra
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler
from datetime import datetime
from pytorch_lightning import LightningModule

# from config import args
from model.loss import L2Distance, ReprojectionError
from src.evaluate_ec import evaluate_ec_pl
from src.evaluate_eds import evaluate_eds_pl
from model.kalman_filter import init_filter_state, kalman_filter_4



class Template(LightningModule):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        patch_size=31,
        init_unrolls=4,
        pose_mode=False,
        debug=True,
        eval_freq=10000,
        kalman=False,
        **kwargs,
    ):
        super(Template, self).__init__()
        self.save_hyperparameters()

        # High level model config
        self.representation = representation
        self.patch_size = patch_size
        self.debug = debug
        self.model_type = "non_global"
        self.pose_mode = pose_mode

        # Determine num channels from representation name
        if "grayscale" in representation:
            self.channels_in_per_patch = 1
        else:
            self.channels_in_per_patch = int(representation[-1])

            if "v2" in self.representation:
                self.channels_in_per_patch *= (
                    2  # V2 representations have separate channels for each polarity
                )

        # Loss Function
        self.loss = None
        self.loss_reproj = ReprojectionError(threshold=self.patch_size / 2)

        # Training variables
        self.unrolls = init_unrolls
        self.max_unrolls = max_unrolls
        self.n_vis = n_vis

        # Validation variables
        self.epe_l2_hist = []
        self.l2 = L2Distance()

        self.wandb_logger = None
        self.eval_freq = eval_freq
        if kalman:
            self.kalman_filter = kalman_filter_4()
        else:
            self.kalman_filter = None

    def configure_optimizers(self):
        if not self.debug:
            opt = hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        opt,
                        self.hparams.optimizer.lr,
                        total_steps=1000000,
                        pct_start=0.002,
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                    "name": "lr",
                },
            }
        else:
            return hydra.utils.instantiate(
                self.hparams.optimizer, params=self.parameters()
            )

    def forward(self, x, attn_mask=None):
        return None

    def on_train_epoch_end(self, *args):
        return

    def training_step(self, batch_dataloaders, batch_nb):

        if isinstance(batch_dataloaders[0], list):
            batch_dataloaders = sum(batch_dataloaders, [])

        metric = {}

        if self.global_step != 0 and self.global_step % self.eval_freq == 0:
            torch.cuda.empty_cache()
            self.eval()
            # evaluate_name = f'{self.wandb_logger.name}_step-{self.global_step}'
            evaluate_name = f'{self.wandb_logger.name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

            eval_metric = evaluate_eds_pl(self, None, evaluate_name)    # TODO: fix cfg
            for key in eval_metric.keys():
                metric[f'EDS_{key}'] = eval_metric[key] * self.wandb_logger.sum_freq
                eval_metric[key] = eval_metric[key] * self.wandb_logger.sum_freq
            metric.update(eval_metric)

            eval_metric = evaluate_ec_pl(self, None, evaluate_name)
            for key in eval_metric.keys():
                metric[f'EC_{key}'] = eval_metric[key] * self.wandb_logger.sum_freq

            self.train()

        if self.pose_mode or hasattr(self, 'uncertainty_predictor'):
            # Freeze batchnorm running values for fine-tuning
            self.reference_encoder = self.reference_encoder.eval()
            self.target_encoder = self.target_encoder.eval()
            self.reference_redir = self.reference_redir.eval()
            self.target_redir = self.target_redir.eval()
            self.lstm_predictor = self.lstm_predictor.eval()
            self.predictor = self.predictor.eval()

        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = len(batch_dataloaders)

        # Preparation
        if not self.pose_mode:
            for bl in batch_dataloaders:
                # print(bl.seq_name, bl.track_idx)
                bl.auto_update_center = False
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        loss_uncert_total = 0
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                    i_src : i_src + n_src_tracks, i_src : i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                # src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                src_path = batch_dataloaders[i_src].seq_name
                for i_target in range(nt):
                    # attn_mask[i_src, i_target] = src_path == batch_dataloaders[i_target].track_path.split("/")[-3]
                    attn_mask[i_src, i_target] = src_path == batch_dataloaders[i_target].seq_name
        attn_mask = (1 - attn_mask).bool()

        if self.kalman_filter:
            self.kalman_filter.reset()
            prev_state = {
                'predict': None,
                'update': None,
                'filter_state': init_filter_state(1, nt)
            }
            for key in prev_state['filter_state']:
                prev_state['filter_state'][key] = prev_state['filter_state'][key].to(self.device)
            flow_pred = torch.zeros((1, nt, 2)).to(self.device)

        # random start idx
        # if hasattr(self, 'uncertainty_predictor'):
        #     start_idx = np.random.randint(0, self.max_unrolls - self.unrolls + 1)
        #     for bl in batch_dataloaders:
        #         bl.set_current(start_idx)


        for i_unroll in range(self.unrolls):
            # # kalman predict
            # if self.kalman_filter:
            #     prev_state = self.kalman_filter.detach(prev_state)
            #     flow_pred, prev_state, _ = self.kalman_filter.predict(prev_state, i_unroll) # flow_pred 1, B, 2 from ref

            # load data and run model
            if self.pyramid:
                x, xx, y, vis = [], [], [], []
                for bl_idx, bl in enumerate(batch_dataloaders):
                    x_j, xx_j, y_j, vis_j = bl.get_next_2()
                    if self.kalman_filter:
                        flow_pred_actual = bl.u_center - bl.u_center_init
                        flow_pred[0, bl_idx, 0] = flow_pred_actual[0]  # u_center is changed in get_next
                        flow_pred[0, bl_idx, 1] = flow_pred_actual[1]  # u_center is changed in get_next
                    x.append(x_j)
                    xx.append(xx_j)
                    y.append(y_j)
                    vis.append(vis_j)
                x = torch.cat(x, dim=0).to(self.device)
                xx = torch.cat(xx, dim=0).to(self.device)
                y = torch.cat(y, dim=0).to(self.device)
                if hasattr(self, 'uncertainty_predictor'):
                    vis = torch.cat(vis, dim=0).to(self.device)

                # Inference
                y_hat = self.forward(x, xx, attn_mask)
            else:
                x, y, vis = [], [], []
                for bl_idx, bl in enumerate(batch_dataloaders):
                    x_j, y_j, vis_j = bl.get_next()
                    if self.kalman_filter is not None:
                        flow_pred_actual = bl.u_center - bl.u_center_init
                        flow_pred[0, bl_idx, 0] = flow_pred_actual[0]  # u_center is changed in get_next
                        flow_pred[0, bl_idx, 1] = flow_pred_actual[1]  # u_center is changed in get_next
                    x.append(x_j)
                    y.append(y_j)
                    vis.append(vis_j)
                x = torch.cat(x, dim=0).to(self.device)
                y = torch.cat(y, dim=0).to(self.device)
                if hasattr(self, 'uncertainty_predictor'):
                    vis = torch.cat(vis, dim=0).to(self.device)

                # Inference
                y_hat = self.forward(x, attn_mask)  # B, 2 or 4

            # uncert
            # if hasattr(self, 'uncertainty_predictor'):
            #     uncert = y_hat[..., 2:]

            uncert = y_hat[..., 2:]
            y_hat = y_hat[..., :2]  # B, 2

            # update result with kalman
            if self.kalman_filter:
                R = self.uncert2R(uncert)
                # R = F.softplus(uncert)
                # R = F.softplus(uncert, beta=0.1)
                R = torch.diag_embed(R, dim1=-2, dim2=-1)
                R = R[None] # 1, B, 2, 2
                flow = y_hat[None] + flow_pred  # 1, B, 2
                # flow, prev_state, _, _ = self.kalman_filter.update(flow, R, prev_state, i_unroll)
                flow, prev_state, _, _ = self.kalman_filter.update(flow, R, prev_state, batch_dataloaders[0].time_idx)
                y_hat = (flow - flow_pred)[0]   # B, 2

            # Accumulate losses
            if self.pose_mode:
                u_centers = []
                for bl in batch_dataloaders:
                    u_centers.append(bl.u_centers)
                u_centers = torch.cat(u_centers, dim=0).to(self.device)

                # Reprojection Loss
                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1).to(self.device))

            else:
                loss, loss_mask = self.loss(y, y_hat)
                loss_total += loss
                loss_mask_total += loss_mask

            if hasattr(self, 'uncertainty_predictor'):
                loss_uncert_total += self.uncert_loss(uncert, vis, loss_mask)

            # Pass predicted flow to dataloader
            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat[idx_acc : idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat[j, :])

        # Average out losses (ignoring the masked out steps)
        if not self.pose_mode:
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()
            if hasattr(self, 'uncertainty_predictor'):
                loss_uncert_total /= self.unrolls
                loss_total += loss_uncert_total * 2
                # loss_total = loss_total / 10 + loss_uncert_total
                # loss_total = loss_uncert_total
        else:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)
            loss_total, loss_mask_total = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist
            )

            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)

            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        metric.update({
            'loss_total': loss_total.detach().item()
        })
        if hasattr(self, 'uncertainty_predictor'):
            metric.update({
                'loss_uncert_total': loss_uncert_total.detach().item()
            })
        self.wandb_logger.push(metric)

        return loss_total

    def on_validation_epoch_start(self):
        # Reset distribution monitors
        self.epe_l2_hist = []
        self.track_error_hist = []
        self.feature_age_hist = []

    def validation_step(self, batch_dataloaders, batch_nb):

        if True:
            return 0

        # Determine number of tracks in batch
        nb = len(batch_dataloaders)
        if self.pose_mode:
            nt = 0
            for bl in batch_dataloaders:
                nt += bl.n_tracks
        else:
            nt = nb

        # Flow history visualization for first batch
        if batch_nb == 0:
            x_hat_hist = []
            x_ref_hist = []
            if not self.pose_mode:
                y_hat_total_hist = [
                    torch.zeros((nt, 1, 2), dtype=torch.float32, device="cpu")
                ]
                y_total_hist = [
                    torch.zeros((nt, 1, 2), dtype=torch.float32, device="cpu")
                ]
                x_hist = []
            else:
                loss_hist = []

        # Validation Metrics
        if not self.pose_mode:
            metrics = {
                "feature_age": torch.zeros(nb, dtype=torch.float32, device="cpu"),
                "tracking_error": [[] for _ in range(nb)],
            }

            for bl in batch_dataloaders:
                bl.auto_update_center = False
        else:
            u_centers_init = []
            for bl in batch_dataloaders:
                u_centers_init.append(bl.u_centers)
            u_centers_init = (
                torch.cat(u_centers_init, dim=0).to(self.device).unsqueeze(1)
            )
            u_centers_hist = [u_centers_init]
            projection_matrices_hist = [
                torch.cat(
                    [
                        torch.from_numpy(
                            batch_dataloaders[0].camera_matrix.astype(np.float32)
                        ),
                        torch.zeros((3, 1), dtype=torch.float32),
                    ],
                    dim=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(u_centers_init.size(0), 1, 1, 1)
                .to(self.device)
            ]

        # Unroll network
        loss_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        loss_mask_total = torch.zeros(nt, dtype=torch.float32, device=self.device)
        self.reset(nt)

        if self.pose_mode:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            i_src = 0
            for bl_src in batch_dataloaders:
                n_src_tracks = bl_src.n_tracks
                attn_mask[
                    i_src : i_src + n_src_tracks, i_src : i_src + n_src_tracks
                ] = 1
                i_src += n_src_tracks
        else:
            attn_mask = torch.zeros([nt, nt], device=self.device)
            for i_src in range(nt):
                src_path = batch_dataloaders[i_src].track_path.split("/")[-3]
                for i_target in range(nt):
                    attn_mask[i_src, i_target] = (
                        src_path
                        == batch_dataloaders[i_target].track_path.split("/")[-3]
                    )
        attn_mask = (1 - attn_mask).bool()

        for i_unroll in range(self.unrolls):
            # Construct x and y
            x, y = [], []
            for bl in batch_dataloaders:
                x_j, y_j = bl.get_next()
                x.append(x_j)
                y.append(y_j)
            x = torch.cat(x, dim=0).to(self.device)
            y = torch.cat(y, dim=0).to(self.device)

            # Inference
            y_hat = self.forward(x, attn_mask)

            if self.pose_mode:
                u_centers = []
                for j in range(nb):
                    u_centers.append(batch_dataloaders[j].u_centers)
                u_centers = torch.cat(u_centers, dim=0).to(self.device)

                # Reproj Loss
                u_centers_hist.append(
                    u_centers.unsqueeze(1).detach() + y_hat.unsqueeze(1)
                )
                projection_matrices_hist.append(y.unsqueeze(1).to(self.device))

            else:
                loss, loss_mask = self.loss(y, y_hat)
                loss_total += loss
                loss_mask_total += loss_mask

            # Pass predicted flow to dataloader
            if self.pose_mode:
                idx_acc = 0
                for j in range(nb):
                    n_tracks = batch_dataloaders[j].n_tracks
                    batch_dataloaders[j].accumulate_y_hat(
                        y_hat[idx_acc : idx_acc + n_tracks, :]
                    )
                    idx_acc += n_tracks
            else:
                for j in range(nb):
                    batch_dataloaders[j].accumulate_y_hat(y_hat[j, :])

            # Patch visualizations for first batch
            if batch_nb == 0:
                x_hat_hist.append(
                    torch.max(x[:, :-1, :, :], dim=1, keepdim=True)[0].detach().clone()
                )
                x_ref_hist.append(x[:, -1, :, :].unsqueeze(1).detach().clone())

            # Metrics
            if self.pose_mode is False:
                dist, y_hat_total, y_total = [], [], []
                for j in range(nb):
                    y_hat_total.append(
                        batch_dataloaders[j].u_center
                        - batch_dataloaders[j].u_center_init
                    )
                    y_total.append(
                        batch_dataloaders[j].u_center_gt
                        - batch_dataloaders[j].u_center_init
                    )
                    dist.append(
                        np.linalg.norm(
                            batch_dataloaders[j].u_center_gt
                            - batch_dataloaders[j].u_center
                        )
                    )
                y_total = torch.from_numpy(np.array(y_total))
                y_hat_total = torch.from_numpy(np.array(y_hat_total))
                dist = torch.from_numpy(np.array(dist))

                # Update feature ages
                live_track_idxs = torch.nonzero(dist < self.patch_size)
                for i in live_track_idxs:
                    metrics["feature_age"][i] = (i_unroll + 1) * 0.01
                    if self.representation == "grayscale":
                        metrics["feature_age"] *= 5
                    metrics["tracking_error"][i].append(dist[i].item())

                # Flow history visualization for first batch
                if batch_nb == 0:
                    y_total_hist.append(
                        y_total.detach().unsqueeze(1).clone()
                    )  # Introduce time axis
                    y_hat_total_hist.append(y_hat_total.detach().unsqueeze(1).clone())

        # Log loss for both training modes
        if not self.pose_mode:
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()
        else:
            u_centers_hist = torch.cat(u_centers_hist, dim=1)
            projection_matrices_hist = torch.cat(projection_matrices_hist, dim=1)
            loss_total, loss_mask_total, u_centers_reproj = self.loss_reproj.forward(
                projection_matrices_hist, u_centers_hist, training=False
            )
            loss_hist = loss_total.clone()
            loss_total = loss_total.sum(1)
            loss_mask_total = loss_mask_total.sum(1)
            nonzero_idxs = torch.nonzero(loss_mask_total, as_tuple=True)[0]
            loss_total[nonzero_idxs] /= loss_mask_total[nonzero_idxs]
            loss_total = loss_total.mean()

        return loss_total

    def on_validation_epoch_end(self):
        if self.pose_mode is False:
            pass