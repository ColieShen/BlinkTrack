import cv2
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

from config import CODE_ROOT
from util.vis import event_voxel_to_rgb
from src.evaluate_eds import CornerConfig
from loader.loader_ec import ECSubseq
from loader.loader_eds import vis_subseq


class DSECSubseq_vis(ECSubseq):
    # ToDO: Add to config file
    pose_r = 4
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
        image_folder="images_corrected",
        event_folder="events",
        **kwargs,
    ):
        super().__init__(
            root_dir=root_dir,
            sequence_name=sequence_name,
            n_frames=n_frames,
            patch_size=patch_size,
            representation=representation,
            dt=dt,
            corner_config=corner_config,
            image_folder="images_corrected",
            event_folder="events",
            **kwargs,
        )


    def initialize(self, max_keypoints=30):
        self.frame_first = cv2.imread(
            self.first_image_path,
            cv2.IMREAD_GRAYSCALE,
        )
        self.frame_first = cv2.resize(self.frame_first, (640, 480)) # todo
        self.resolution = (self.frame_first.shape[1], self.frame_first.shape[0])
        self.initialize_keypoints(max_keypoints)
        self.initialize_reference_patches()


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
        frame = cv2.resize(frame, self.resolution)
        return self.frame_ts_arr[image_idx], frame
    
    def get_frame_path(self, image_idx):
        return str(
                    self.sequence_dir
                    / self.image_folder
                    / f"{image_idx:06d}.png"
                )
        
    def override_refframe(self, img_frame_idx):
        self.first_frame_idx = img_frame_idx
        self.n_frames -= img_frame_idx
        self.t_init = self.frame_ts_arr[img_frame_idx]
        self.first_event_idx = int((self.t_init - self.frame_ts_arr[0]) // self.dt)
        self.t_now = self.first_event_idx * self.dt + self.frame_ts_arr[0]
        self.first_image_path = self.get_frame_path(self.first_frame_idx)
        self.initialize()

    def override_seqlength(self, seq_length):
        self.n_frames = seq_length + 1
        self.t_end = self.frame_ts_arr[self.first_frame_idx + self.n_frames]
        last_event_idx = int((self.t_end - self.frame_ts_arr[0]) // self.dt)
        self.n_events = last_event_idx - self.first_event_idx + 2


if __name__ == '__main__':

    SEQ_NAME = 'zurich_city_13_a'

    with open(f'{CODE_ROOT}/configs/eval_real_defaults.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    dataset = DSECSubseq_vis(
        f'{cfg.eval_dataset_path}/for_vis',
        SEQ_NAME,
        -1,
        cfg.patch_size,
        cfg.representation,
        0.01,
        corner_config,
        image_folder=cfg.image_folder,
        event_folder=cfg.event_folder,
    )

    if cfg.ref_frame_idx != 'none':
        dataset.override_refframe(cfg.ref_frame_idx)
    if cfg.frame_length != 'none':
        dataset.override_seqlength(cfg.frame_length)

    vis_subseq(dataset)
    