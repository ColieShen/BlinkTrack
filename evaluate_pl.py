import os
import torch

THREAD_NUM = 16
torch.set_num_threads(THREAD_NUM)
os.environ ['OMP_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['MKL_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_NUM}'
os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_NUM}'

import yaml
import hydra
import torch
import pytorch_lightning as pl

from datetime import datetime
from omegaconf import OmegaConf, open_dict

from config import CODE_ROOT
from util.cfg import propagate_keys
from model.klt import KLTTracker
from util.torch import count_parameters, load_weights
from src.evaluate_mf import evaluate_mf
from src.evaluate_ec import evaluate_ec_pl
from src.evaluate_eds import evaluate_eds_pl
from src.evaluate_for_vis import evaluate_for_vis_pl
from model.blinktrack_image import BlinkTrack_Image


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def evaluate(cfg):
    evaluate_core(cfg)

@torch.no_grad()
def evaluate_core(cfg, seqs=None, result_path=None):

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.representation = cfg.representation
    if cfg.eval_dataset == 'mf':
        propagate_keys(cfg)

    event_model = None
    rgb_model = None

    evaluate_name_list = []
    evaluate_name_list.append(cfg.eval_dataset)
    if cfg.use_event:
        evaluate_name_list.append(cfg.model.name)
        print(cfg.event_weights_path)
    if cfg.use_rgb:
        evaluate_name_list.append(cfg.rgb_model)
    if cfg.use_kalman:
        evaluate_name_list.append('kalman')
    evaluate_name_list.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    evaluate_name = ''
    for eval_name in evaluate_name_list:
        evaluate_name += eval_name + '_'
    evaluate_name = evaluate_name[:-1]
    os.system(f'mkdir -p {CODE_ROOT}/output/evaluate_result/{evaluate_name[:-16]}/{evaluate_name[-15:]}')
    print(evaluate_name)

    if cfg.use_event:
        event_model = hydra.utils.instantiate(cfg.model, _recursive_=False)

        state_dict = torch.load(cfg.event_weights_path, map_location="cuda:0")["state_dict"]
        event_model.load_state_dict(state_dict, strict=False)   # TODO: tmp
        # if torch.cuda.is_available():
        #     event_model = event_model.cuda()
        event_model = event_model.cuda()
        event_model.eval()
        print(f'Event Module parameter count {count_parameters(event_model)}')

    if cfg.use_rgb:
        if cfg.rgb_model == 'blinktrack':
            rgb_model = BlinkTrack_Image()
            rgb_model = load_weights(rgb_model, cfg.image_weights_path)
            rgb_model.cuda()
            rgb_model.eval()
            print(cfg.image_weights_path)
            print(f'RGB Module parameter count {count_parameters(rgb_model)}')
        elif cfg.rgb_model == 'cotracker':
            rgb_model = build_cotracker(cfg.cotracker_weights_path)
            rgb_model.cuda()
            rgb_model.eval()
            print(cfg.cotracker_weights_path)
            print(f'RGB Module parameter count {count_parameters(rgb_model)}')
        elif cfg.rgb_model == 'klt':
            rgb_model = KLTTracker()

    if cfg.eval_dataset == 'eds':
        evaluate_eds_pl(event_model, rgb_model, evaluate_name, use_kalman=cfg.use_kalman, cfg=cfg, seqs=seqs, result_path=result_path)
    elif cfg.eval_dataset == 'ec':
        evaluate_ec_pl(event_model, rgb_model, evaluate_name, use_kalman=cfg.use_kalman, cfg=cfg, seqs=seqs, result_path=result_path)
    elif cfg.eval_dataset == 'both':
        evaluate_ec_pl(event_model, rgb_model, evaluate_name, use_kalman=cfg.use_kalman, cfg=cfg, seqs=seqs, result_path=result_path)
        evaluate_eds_pl(event_model, rgb_model, evaluate_name, use_kalman=cfg.use_kalman, cfg=cfg, seqs=seqs, result_path=result_path)
    elif cfg.eval_dataset == 'mf':
        evaluate_mf(event_model, evaluate_name, cfg)
    elif 'vis' in cfg.eval_dataset:
        evaluate_for_vis_pl(event_model, rgb_model, evaluate_name, use_kalman=cfg.use_kalman, cfg=cfg, seqs=seqs, result_path=result_path)





if __name__ == '__main__':

    pl.seed_everything(1234)
    evaluate()