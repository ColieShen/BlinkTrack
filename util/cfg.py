from omegaconf import OmegaConf, open_dict


def propagate_keys(cfg, testing=False):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.data.representation = cfg.representation
        cfg.data.track_name = cfg.track_name
        cfg.data.patch_size = cfg.patch_size

        cfg.model.representation = cfg.representation
        cfg.model.patch_size = cfg.patch_size

        if not testing:
            cfg.model.n_vis = cfg.n_vis
            cfg.model.init_unrolls = cfg.init_unrolls
            cfg.model.max_unrolls = cfg.max_unrolls
            cfg.model.debug = cfg.debug
            cfg.model.eval_freq = cfg.eval_freq
            cfg.model.kalman = cfg.kalman

        cfg.model.pose_mode = cfg.data.name == "pose"