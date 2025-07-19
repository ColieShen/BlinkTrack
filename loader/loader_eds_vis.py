import cv2
import yaml
from omegaconf import OmegaConf

from config import CODE_ROOT
from src.evaluate_eds import CornerConfig
from loader.loader_eds import EDSSubseq, vis_subseq


class EDSSubseq_vis(EDSSubseq):
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
        super().__init__(
            root_dir=root_dir,
            sequence_name=sequence_name,
            n_frames=n_frames,
            patch_size=patch_size,
            representation=representation,
            dt=dt,
            corner_config=corner_config,
            include_prev=include_prev,
            fused=fused,
            grayscale_ref=grayscale_ref,
            use_colmap_poses=use_colmap_poses,
            global_mode=global_mode,
            image_folder=image_folder,
            event_folder=event_folder,
            **kwargs,
        )

    def override_refframe(self, img_frame_idx):
        self.first_frame_idx = img_frame_idx
        self.n_frames -= img_frame_idx
        self.t_init = self.frame_ts_arr[img_frame_idx] * 1e-6
        self.first_event_idx = int((self.t_init - self.frame_ts_arr[0] * 1e-6) // self.dt)
        self.t_now = self.first_event_idx * self.dt + self.frame_ts_arr[0] * 1e-6
        self.first_image_path = self.get_frame_path(self.first_frame_idx)
        self.initialize()

    def override_seqlength(self, seq_length):
        self.n_frames = seq_length + 1
        self.t_end = self.frame_ts_arr[self.first_frame_idx + self.n_frames] * 1e-6
        last_event_idx = int((self.t_end - self.frame_ts_arr[0] * 1e-6) // self.dt)
        self.n_events = last_event_idx - self.first_event_idx + 2


if __name__ == '__main__':

    # SEQ_NAME = '10_office'
    SEQ_NAME = '11_all_characters'
    # SEQ_NAME = '13_airplane'

    with open(f'{CODE_ROOT}/configs/eval_real_defaults.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

    dataset = EDSSubseq_vis(
        f'{cfg.eval_dataset_path}/for_vis',
        SEQ_NAME,
        -1,
        cfg.patch_size,
        cfg.representation,
        0.005,
        corner_config,
        image_folder=cfg.image_folder,
        event_folder=cfg.event_folder,
    )

    if cfg.ref_frame_idx != 'none':
        dataset.override_refframe(cfg.ref_frame_idx)
    if cfg.frame_length != 'none':
        dataset.override_seqlength(cfg.frame_length)

    vis_subseq(dataset)