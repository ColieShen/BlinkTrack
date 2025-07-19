from loader.loader_ec import ECSubseq


class ECSubseq_vis(ECSubseq):
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

        
    def override_refframe(self, img_frame_idx):
        self.first_frame_idx = img_frame_idx
        self.n_frames -= img_frame_idx
        self.t_init = self.frame_ts_arr[img_frame_idx]
        self.first_event_idx = int(self.t_init // self.dt)
        self.t_now = self.first_event_idx * self.dt
        self.first_image_path = self.get_frame_path(self.first_event_idx)
        self.initialize()

    def override_seqlength(self, seq_length):
        self.n_frames = seq_length + 1
        self.t_end = self.frame_ts_arr[self.first_frame_idx + self.n_frames]
        last_event_idx = int(self.t_end // self.dt)
        self.n_events = last_event_idx - self.first_event_idx + 2