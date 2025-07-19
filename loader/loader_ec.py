import os
import cv2
import torch
import numpy as np

from glob import glob
from pathlib import Path

from util.data import array_to_tensor, extract_glimpse, get_patch_voxel, read_input
from loader.loader_eds import SequenceDataset


class ECSubseq(SequenceDataset):
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
        super().__init__()

        # Store config
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.patch_size = patch_size
        self.representation = representation
        self.dt, self.dt_us = dt, dt * 1e6
        self.sequence_dir = self.root_dir / self.sequence_name
        self.corner_config = corner_config
        self.image_folder = image_folder
        self.event_folder = event_folder

        # Determine number of frames
        self.frame_dir = self.sequence_dir / self.image_folder
        # max_frames = len(list(self.frame_dir.iterdir()))
        max_frames = len(glob(str(self.frame_dir / "*.png")))
        if n_frames == -1 or n_frames > max_frames:
            self.n_frames = max_frames
        else:
            self.n_frames = n_frames

        # Check that event representations have been generated for this dt
        if not self.pose_mode:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / self.event_folder
                / f"{dt:.4f}"
                / f"{self.representation}"
            )
        else:
            self.dir_representation = (
                self.root_dir
                / sequence_name
                / self.event_folder
                / f"pose_{self.pose_r:.0f}"
                / f"{self.representation}"
            )

        if not self.dir_representation.exists():
            print(
                f"{self.representation} has not yet been generated for a dt of {self.dt}"
            )
            exit()

        # Read timestamps
        self.frame_ts_arr = np.genfromtxt(str(self.sequence_dir / "images.txt"))

        # Read poses and camera matrix
        # if (self.sequence_dir / "colmap").exists():
        #     pose_data_path = self.sequence_dir / "colmap" / "stamped_groundtruth.txt"
        #     self.pose_data = np.genfromtxt(str(pose_data_path), skip_header=1)
        # else:
        #     self.pose_data = np.genfromtxt(str(self.sequence_dir / "groundtruth.txt"))
        # intrinsics = np.genfromtxt(str(self.sequence_dir / "calib.txt"))
        # self.camera_matrix = np.array(
        #     [
        #         [intrinsics[0], 0, intrinsics[2]],
        #         [0, intrinsics[1], intrinsics[3]],
        #         [0, 0, 1],
        #     ]
        # )

        # Tensor Manipulation
        self.channels_in_per_patch = int(self.representation[-1])
        if "v2" in self.representation:
            self.channels_in_per_patch *= 2
        self.cropping_fn = get_patch_voxel

        # Timing and Indices
        self.t_init = self.frame_ts_arr[0]
        self.t_end = self.frame_ts_arr[-1]
        self.t_now = self.t_init

        # Get counts
        self.n_events = int(np.ceil((self.t_end - self.t_init) / self.dt))

        # Get first imgs
        self.first_image_path = self.get_frame_path(self.first_frame_idx)
        
        # self.event_first = array_to_tensor(read_input(str(self.dir_representation / '0000000.h5'), self.representation))

        # Extract keypoints, store reference patches
        self.initialize()

    def __len__(self):
        return
    
    def override_eventframe(self, img_frame_idx):
        pass

    def reset(self):
        self.t_now = self.t_init
        self.u_centers = self.u_centers_init

    def initialize_reference_patches(self):
        # Store reference patches
        ref_input = (
            torch.from_numpy(self.frame_first.astype(np.float32) / 255)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.x_ref, self.x_ref_2 = self.get_patches(ref_input)

    def globals(self):
        for i in range(self.first_event_idx + 1, self.first_event_idx + self.n_events):
            self.t_now += self.dt
            x = array_to_tensor(
                read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            )

            yield self.t_now, x.unsqueeze(0)

    def events(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            i_start = 0
        else:
            i_start = self.first_event_idx + 1

        for i in range(i_start, self.first_event_idx + self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                if not os.path.exists(str(self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5")):
                    print('===========================================')
                    print(f'{str(self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5")} not exist')
                    print('not enough event! quit!')
                    print('===========================================')
                    break
                input_1 = read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[i], self.representation
                )

                if i > 0 and i % self.pose_r == 0:
                    # print(self.frame_dir / ('frame_' + f'{i // self.pose_r}'.zfill(8) + '.png'))
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / ("frame_" + f"{i//self.pose_r}".zfill(8) + ".png")
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            yield self.t_now, x


    def events_2(self):
        if self.pose_mode:
            self.event_representation_paths = sorted(
                glob(str(self.dir_representation / "*.h5")),
                key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
            )
            self.n_events = len(self.event_representation_paths)
            i_start = 0
        else:
            i_start = self.first_event_idx + 1

        for i in range(i_start, self.first_event_idx + self.n_events):
            # Get patch inputs and set current time
            if not self.pose_mode:
                self.t_now += self.dt
                if not os.path.exists(str(self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5")):
                    print('===========================================')
                    print(f'{str(self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5")} not exist')
                    print('not enough event! quit!')
                    print('===========================================')
                    break
                input_1 = read_input(
                    self.dir_representation / f"{str(int(i * self.dt_us)).zfill(7)}.h5",
                    self.representation,
                )
            else:
                self.t_now = (
                    float(
                        os.path.split(self.event_representation_paths[i])[1].replace(
                            ".h5", ""
                        )
                    )
                    * 1e-6
                )
                input_1 = read_input(
                    self.event_representation_paths[i], self.representation
                )

                if i > 0 and i % self.pose_r == 0:
                    # print(self.frame_dir / ('frame_' + f'{i // self.pose_r}'.zfill(8) + '.png'))
                    ref_input = cv2.imread(
                        str(
                            self.frame_dir
                            / ("frame_" + f"{i//self.pose_r}".zfill(8) + ".png")
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    ref_input = (
                        torch.from_numpy(ref_input.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.x_ref.device)
                    )
                    self.x_ref = extract_glimpse(
                        ref_input.repeat(self.u_centers.size(0), 1, 1, 1),
                        (self.patch_size, self.patch_size),
                        self.u_centers.detach() + 0.5,
                        mode="bilinear",
                    )

            input_1 = np.array(input_1)
            input_1 = np.transpose(input_1, (2, 0, 1))
            input_1 = torch.from_numpy(input_1).unsqueeze(0).to(self.u_centers.device)

            x = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size, self.patch_size),
                self.u_centers.detach() + 0.5,
            )
            x = torch.cat([x, self.x_ref], dim=1)

            xx = extract_glimpse(
                input_1.repeat(self.u_centers.size(0), 1, 1, 1),
                (self.patch_size * 2, self.patch_size * 2),
                self.u_centers.detach() + 0.5,
            )
            xx = torch.cat([xx, self.x_ref_2], dim=1)

            yield self.t_now, x, xx
    

    def full_representation_events(self):
        self.event_representation_paths = sorted(
            glob(str(self.dir_representation / "*.h5")),
            key=lambda k: int(os.path.split(k)[1].replace(".h5", "")),
        )
        self.n_events = len(self.event_representation_paths)
        i_start = 0

        for i in range(i_start, self.n_events):
            self.t_now = (
                float(
                    os.path.split(self.event_representation_paths[i])[1].replace(
                        ".h5", ""
                    )
                )
                * 1e-6
            )
            events_repr = read_input(
                self.event_representation_paths[i], self.representation
            )

            events_repr = np.array(events_repr)
            events_repr = np.transpose(events_repr, (2, 0, 1))

            yield self.t_now, events_repr

    def frames(self):
        for i in range(self.first_frame_idx, self.first_frame_idx + self.n_frames):
            # Update time info
            # self.t_now = self.frame_ts_arr[i]

            # frame = cv2.imread(
            #     self.get_frame_path(i),
            #     cv2.IMREAD_GRAYSCALE,
            # )
            # yield self.t_now, frame
            yield self.get_frame(i)

    def get_next(self):
        pass

    def check_next_rgb_time(self, next_rgb_idx):
        if next_rgb_idx >= self.first_frame_idx + self.n_frames:
            return False

        next_rgb_t = self.frame_ts_arr[next_rgb_idx]
        return self.t_now + self.dt > next_rgb_t

    def get_frame(self, image_idx, grayscale=False):    # TODO
        # Update time info
        # self.t_now = self.frame_ts_arr[idx]

        frame = cv2.imread(
            self.get_frame_path(image_idx),
            cv2.IMREAD_GRAYSCALE,
        )
        return self.frame_ts_arr[image_idx], frame
    
    def get_frame_path(self, image_idx):
        return str(
                    self.sequence_dir
                    / self.image_folder
                    / ("frame_" + f"{image_idx}".zfill(8) + ".png")
                )
