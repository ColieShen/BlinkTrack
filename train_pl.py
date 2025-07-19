import os
import torch


# os.environ['NUMEXPR_MAX_THREADS'] = '16'
# os.environ['NUMEXPR_NUM_THREADS'] = '8'


THREAD_NUM = 1
torch.set_num_threads(THREAD_NUM)
os.environ ['OMP_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['MKL_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_NUM}'

import hydra
# import logging
import pytorch_lightning as pl

from datetime import datetime
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, SingleDeviceStrategy

from util.cfg import propagate_keys
from util.logger import Logger
from util.callback import IncreaseSequenceLengthCallback
from model.kalman_filter import kalman_filter_4

# logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="configs", config_name="train_defaults")
def train(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys(cfg)
    # logger.info("\n" + OmegaConf.to_yaml(cfg))

    run_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{cfg.model.name}_{cfg.data.name}_{cfg.experiment}'
    if cfg.checkpoint_path.lower() != "none":
        folder_names = cfg.checkpoint_path.split('/')
        base_folder_idx = folder_names.index('ERK_Tracker')
        assert  base_folder_idx != -1
        run_name += '__' + folder_names[base_folder_idx + 3].replace('-', '').replace('_', '-') + '_' + os.path.basename(cfg.checkpoint_path).split('.')[0]

    # Instantiate model and dataloaders
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    if cfg.checkpoint_path.lower() != "none":
        # Load weights
        model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path, strict=False)

        # Override stuff for fine-tuning
        model.hparams.optimizer.lr = cfg.model.optimizer.lr
        model.hparams.optimizer._target_ = cfg.model.optimizer._target_
        model.debug = True
        model.unrolls = cfg.init_unrolls
        model.max_unrolls = cfg.max_unrolls
        model.pose_mode = cfg.model.pose_mode
        model.eval_freq = cfg.eval_freq
        if cfg.kalman:
            model.kalman_filter = kalman_filter_4()
        else:
            model.kalman_filter = None

    data_module = hydra.utils.instantiate(cfg.data)

    wandb_logger = Logger(run_name, project_name='ERK_Tracker',
                    no_logging= not cfg.wandb_logging,
                    last_steps=0)
    model.wandb_logger = wandb_logger

    # Logging
    if cfg.logging:
        training_logger = pl.loggers.TensorBoardLogger(
            ".", "", "", log_graph=True, default_hp_metric=False
        )
    else:
        training_logger = None

    # Training schedule
    callbacks = [
        IncreaseSequenceLengthCallback(
            unroll_factor=cfg.unroll_factor, schedule=cfg.unroll_schedule
        ),
        # pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.ModelCheckpoint(every_n_train_steps=20000, save_top_k=-1, save_last=True)
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices="auto",
        accelerator="gpu",
        callbacks=callbacks,
        logger=training_logger,
        # strategy='dp',
        # strategy=DDPStrategy(find_unused_parameters=True),
        strategy=DDPStrategy(find_unused_parameters=False),
        # strategy=SingleDeviceStrategy(device=model.device),
        # strategy=DeepSpeedStrategy(stage=3),
        # strategy=DeepSpeedStrategy(
        #     stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        # ),
    )

    # disable pl validation
    trainer.limit_val_batches = 0.0

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
