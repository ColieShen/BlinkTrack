import os
import cv2
import yaml
import torch
import numpy as np
import random
import multiprocessing

from tqdm import tqdm
from glob import glob
from datetime import datetime
from omegaconf import OmegaConf
from matplotlib.path import Path
from scipy.interpolate import splprep, splev, interp1d, CubicSpline

from ..util.vis import generate_random_color, instance_image_to_rgb, time_surface_to_rgb
from ..util.data import read_input
from ..util.event import make_events
from superpoint import SuperPoint
from script.vis.vis_evaluate_result import draw_trajs, make_video

THREAD_CORE_NUM = 4
torch.set_num_threads(THREAD_CORE_NUM)
os.environ ['OMP_NUM_THREADS'] = f'{THREAD_CORE_NUM}'
os.environ ['MKL_NUM_THREADS'] = f'{THREAD_CORE_NUM}'
os.environ ['NUMEXPR_NUM_THREADS'] = f'{THREAD_CORE_NUM}'
os.environ ['NUMEXPR_MAX_THREADS'] = f'{THREAD_CORE_NUM}'
os.environ ['OPENBLAS_NUM_THREADS'] = f'{THREAD_CORE_NUM}'
os.environ ['VECLIB_MAXIMUM_THREADS'] = f'{THREAD_CORE_NUM}'


CONFIG_PATH = 'config/multiflowplus.yaml'
THREAD_NUM = 4  # 3090 can hold 30

multiprocessing.set_start_method('spawn', force=True)


def project_points(points, center, pose, mode='project'):
    # points N, 3
    # center (x, y)
    # pose (sclae, rotation, translation_x, translation_y)
    assert mode in ['project', 'reproject']

    scale = pose[0]
    rotation = pose[1]
    translation = pose[2:]

    M = cv2.getRotationMatrix2D((0, 0), rotation, scale)  # rotation is degree
    # M = cv2.getRotationMatrix2D(center, rotation, scale)  # rotation is degree
    M[..., 2] += translation    # (x, y)
    # M[..., 2] += center

    R = M[:, :2]
    if mode == 'reproject':
        R /= scale * scale    # scale -> 1/scale
    t = M[:, 2]

    T = np.zeros((3, 3))
    if mode == 'project':
        T[:2, :2] = R
        T[:2, 2] = t
    elif mode == 'reproject':
        T[:2, :2] = R.T
        T[:2, 2] = -R.T @ t
    T[2, 2] = 1

    # points = T @ points
    # points = points @ T.T
    points = np.einsum('ij,nj->ni', T, points)
    points /= points[..., 2:]

    return points


def get_random_polygon(edges=10):
    points = np.random.rand(edges, 2)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_points = points[np.argsort(angles)]
    polygon = np.append(sorted_points, [sorted_points[0]], axis=0)
    return polygon  # edges, 2


def smooth_polygon(polygon, smoothness=100):
    # Separate the x and y coordinates
    x, y = polygon[:, 0], polygon[:, 1]

    # Fit a B-spline through the polygon points
    tck, u = splprep([x, y], s=0, per=True)

    # Evaluate the B-spline over a dense range of points to create a smooth curve
    u_new = np.linspace(u.min(), u.max(), smoothness)
    x_smooth, y_smooth = splev(u_new, tck)

    return np.stack([x_smooth, y_smooth], axis=1)   # smoothness, 2


def get_polygon_mask(polygon, image_shape):
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    # scale polygon
    polygon[..., 0] *= image_shape[1]
    polygon[..., 1] *= image_shape[0]

    # Create a Path object from the polygon vertices
    path = Path(polygon)

    # Get mask where points are inside the polygon
    grid = path.contains_points(points)
    mask = grid.reshape(image_shape)

    return mask


def get_random_mask(image_shape, repeat=2, smooth_probability=0.5):
    mask = np.zeros(image_shape)
    for repeat_idx in range(repeat):
        polygon = get_random_polygon(edges=8)   # TODO: add edge num to config
        if repeat_idx != 0:
            polygon *= np.random.uniform(0.4, 0.6)  # TODO: add polygon scale to config
        if random.random() < smooth_probability:
            polygon = smooth_polygon(polygon)
        polygon_mask = get_polygon_mask(polygon, image_shape)
        mask += polygon_mask
    mask %= 2
    return mask


def put_image_on_background(image, background_image, 
                            image_mask=None, image_center=None, 
                            scale=1, rotation=0, translation=(0, 0)):
    H, W = image.shape[:2]
    if len(image.shape) == 2:
        image = image[..., None]
    H_back, W_back = background_image.shape[:2] # normally, H_back, W_back should be even
    if image_mask is None:
        image_mask = np.ones_like(image[...,0])[..., None]   # H, W, 1
    if image_center is None:
        image_center = np.array([W / 2, H / 2])   # (x, y)
    if scale is None:
        scale = 1
    if rotation is None:
        rotation = 0
    if translation is None:
        translation = np.array([0, 0])
    
    # TODO: non rigid transformation

    T = cv2.getRotationMatrix2D(image_center, rotation, scale)  # rotation is degree
    T[..., 2] += translation    # (x, y)
    T[..., 2] += np.array((W, H))- image_center

    image_transformed = cv2.warpAffine(image, T, (W + W_back, H + H_back))[H:, W:]
    image_mask_transformed = cv2.warpAffine(image_mask, T, (W + W_back, H + H_back))[H:, W:, None]

    if len(image_transformed.shape) == 2:
        image_transformed = image_transformed[..., None]

    final_image = background_image * (1 - image_mask_transformed) + image_transformed * image_mask_transformed

    return final_image, image_mask_transformed


def put_images_on_background(image_list, background_image, 
                             image_mask_list=None, image_center_list=None, 
                             scale_list=None, rotation_list=None, translation_list=None):
    N = len(image_list)
    if image_mask_list is None:
        image_mask_list = [None] * N
    if image_center_list is None:
        image_center_list = [None] * N
    if scale_list is None:
        scale_list = [None] * N
    if rotation_list is None:
        rotation_list = [None] * N
    if translation_list is None:
        translation_list = [None] * N

    assert (len(image_mask_list) == N
            and len(image_center_list) == N 
            and len(scale_list) == N 
            and len(rotation_list) == N
            and len(translation_list) == N)
    
    instance_image = np.zeros(background_image.shape[:2])
    instance_image[:] = -1
    instance_percent_max = np.ones(background_image.shape[:2])
    
    for image_idx in range(N):
        background_image, instance_mask = put_image_on_background(
            image=image_list[image_idx],
            background_image=background_image,
            image_mask=image_mask_list[image_idx],
            image_center=image_center_list[image_idx],
            scale=scale_list[image_idx],
            rotation=rotation_list[image_idx],
            translation=translation_list[image_idx],)
        instance_mask = instance_mask.reshape(background_image.shape[:2])
        instance_percent_max *= 1 - instance_mask
        # cover_mask = instance_mask >= instance_percent_max    # TODO: two kinds of vis
        cover_mask = instance_mask > 0 # fix bug
        instance_image[cover_mask] = image_idx
        instance_percent_max[cover_mask] = instance_mask[cover_mask]
        
    return background_image, instance_image


def get_forebackground_image(cfg, dataset_path, image_num, seq_path, mode='foreground'):
    assert mode in ['foreground', 'background']

    if mode == 'foreground':
        image_dataset = []
        dataset_path = dataset_path[cfg.dataset_mode]
        for semi_dataset_path in dataset_path:
            image_dataset += (glob(f'{semi_dataset_path}/**/*.jpg', recursive=True) + 
                              glob(f'{semi_dataset_path}/**/*.png', recursive=True))
    elif mode == 'background':
        image_dataset = sorted(glob(f'{dataset_path}/**/*.jpg', recursive=True) + 
                               glob(f'{dataset_path}/**/*.png', recursive=True))
        image_dataset_num = len(image_dataset)
        if cfg.dataset_mode == 'train':
            image_dataset = image_dataset[image_dataset_num//5:]
        elif cfg.dataset_mode == 'test':
            image_dataset = image_dataset[:image_dataset_num//5]

    image_list = []
    for image_idx, image_path in enumerate(random.choices(image_dataset, k=image_num)):
        image = cv2.imread(f'{image_path}', cv2.IMREAD_UNCHANGED)
        if cfg.visualize:
            cv2.imwrite(f'{seq_path}/vis/{mode}_{image_idx}.png', image)
        if mode == 'background':
            image = np.tile(image, (cfg.background_image_repeat[0], cfg.background_image_repeat[1], 1))
        image_list.append(image)

    return image_list


def get_forebackground_pose(cfg, image_num, mode='foreground'): # TODO: add perspective
    assert mode in ['foreground', 'background']

    H, W = cfg.image_size
    if mode == 'foreground':
        init_scale_config = cfg.foreground_init_scale
        scale_config = cfg.foreground_scale
        rotation_config = cfg.foreground_rotation
        translation_config = cfg.foreground_translation
    elif mode == 'background':
        init_scale_config = cfg.background_init_scale
        scale_config = cfg.background_scale
        rotation_config = cfg.background_rotation
        translation_config = cfg.background_translation

    pose_list = []
    for img_idx in range(image_num):
        pose_list_cur = []

        # init
        cur_scale = np.random.uniform(init_scale_config[0], init_scale_config[1])
        cur_rotation = np.random.uniform(-180, 180)
        cur_translation = [np.random.uniform(0, W), np.random.uniform(0, H)]   # (x, y)
        pose_list_cur.append([cur_scale, cur_rotation, cur_translation[0], cur_translation[1]])

        for kf_idx in range(1, cfg.keyframe_num):
            delta_scale = np.random.uniform(scale_config[0], scale_config[1])
            delta_rotation = np.random.uniform(rotation_config[0], rotation_config[1])
            # TODO: background cover all image
            delta_translation_direction = np.random.uniform(-np.pi, np.pi)
            delta_translation_dist = np.random.uniform(translation_config[0], translation_config[1])

            cur_scale *= delta_scale
            cur_rotation += delta_rotation
            cur_translation = [cur_translation[0] + np.sin(delta_translation_direction) * delta_translation_dist,
                               cur_translation[1] + np.cos(delta_translation_direction) * delta_translation_dist]
            
            # when outbound, go toward inner
            if max(max(cur_translation[0] - W, 0 - cur_translation[0]),
                   max(cur_translation[1] - H, 0 - cur_translation[1])) > cfg.foreground_image_outbound_threshold:
                cur_translation = [cur_translation[0] - np.sin(delta_translation_direction) * delta_translation_dist,
                                   cur_translation[1] - np.cos(delta_translation_direction) * delta_translation_dist]
                cur_translation_delta = [np.random.uniform(cfg.foreground_image_outbound_threshold, W - cfg.foreground_image_outbound_threshold),
                                          np.random.uniform(cfg.foreground_image_outbound_threshold, H - cfg.foreground_image_outbound_threshold)]
                cur_translation_delta = np.array(cur_translation_delta) - np.array(cur_translation)
                cur_translation_delta = cur_translation_delta / np.linalg.norm(cur_translation_delta) * delta_translation_dist
                cur_translation = [cur_translation[0] + cur_translation_delta[0],
                                   cur_translation[1] + cur_translation_delta[1]]

            pose_list_cur.append([cur_scale, cur_rotation, cur_translation[0], cur_translation[1]])

        pose_list.append(pose_list_cur)

    return pose_list    # N, L, 4 [scale, rotation, translation_x, translation_y]


def get_random_image_center(cfg, image_list, mode='foreground'):
    assert mode in ['foreground', 'background']
    image_center_list = []
    for image in image_list:
        H, W = image.shape[:2]
        if mode == 'foreground':
            image_center_list.append([np.random.uniform(0, W), np.random.uniform(0, H)])
        elif mode == 'background':
            x_repeat, y_repeat = cfg.background_image_repeat
            image_center_list.append([np.random.uniform(W/x_repeat*(x_repeat//2), W/x_repeat*(x_repeat//2+1)),
                                      np.random.uniform(H/y_repeat*(y_repeat//2), H/y_repeat*(y_repeat//2+1))])
    return image_center_list


def get_pose_list(cfg, keyframe_pose_list, mode='event'):
    assert cfg.animation_mode in ['linear', 'cubic_spline']
    assert mode in ['event', 'frame', 'keyframe']

    if mode == 'event':
        step = 1
    elif mode == 'frame':
        step = cfg.event_per_frame
    elif mode == 'keyframe':
        step = cfg.event_per_frame * cfg.frame_per_keyframe

    num_all_frame = cfg.event_per_frame * cfg.frame_per_keyframe * (cfg.keyframe_num - 1) + 1
    frame_idx = np.arange(0, num_all_frame, step, dtype=np.int32)
    keyframe_idx = np.arange(0, num_all_frame + 1, cfg.event_per_frame * cfg.frame_per_keyframe, dtype=np.int32)
    
    if mode == 'event':
        pose_list = np.zeros((keyframe_pose_list.shape[0], num_all_frame, 4))
    elif mode == 'frame':
        pose_list = np.zeros((keyframe_pose_list.shape[0], cfg.frame_per_keyframe * (cfg.keyframe_num - 1) + 1, 4))
    elif mode == 'keyframe':
        pose_list = np.zeros((keyframe_pose_list.shape[0], cfg.keyframe_num, 4))

    for obj_idx in range(keyframe_pose_list.shape[0]):
        if cfg.animation_mode == 'linear':
            f_scale = interp1d(keyframe_idx, keyframe_pose_list[obj_idx, :, 0])
            f_rotation = interp1d(keyframe_idx, keyframe_pose_list[obj_idx, :, 1])
            f_translation_x = interp1d(keyframe_idx, keyframe_pose_list[obj_idx, :, 2])
            f_translation_y = interp1d(keyframe_idx, keyframe_pose_list[obj_idx, :, 3])
        elif cfg.animation_mode == 'cubic_spline':
            f_scale = CubicSpline(keyframe_idx, keyframe_pose_list[obj_idx, :, 0])
            f_rotation = CubicSpline(keyframe_idx, keyframe_pose_list[obj_idx, :, 1])
            f_translation_x = CubicSpline(keyframe_idx, keyframe_pose_list[obj_idx, :, 2])
            f_translation_y = CubicSpline(keyframe_idx, keyframe_pose_list[obj_idx, :, 3])
        pose_list[obj_idx, :, 0] = f_scale(frame_idx)
        pose_list[obj_idx, :, 1] = f_rotation(frame_idx)
        pose_list[obj_idx, :, 2] = f_translation_x(frame_idx)
        pose_list[obj_idx, :, 3] = f_translation_y(frame_idx)

    return pose_list


def get_keypoints(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray / 255
    img_tensor = torch.tensor(img_gray).unsqueeze(0).unsqueeze(0).float()  # C, H, W shape

    superpoint = SuperPoint({})
    superpoint.net.config['keypoint_threshold'] = 0.4  # TODO: config
    keypoints = superpoint({'image': img_tensor})['keypoints'][0].cpu().numpy().astype(int)
    return keypoints


def load_dense_trajs(seq_path):
    traj_list = []
    for traj_file in sorted(os.listdir(f'{seq_path}/particle/')):
        traj_list.append(np.load(f'{seq_path}/particle/{traj_file}'))
    trajs = np.stack(traj_list, axis=2)
    return trajs


def traj_list_to_tensor(traj_list):
    # traj list N*L, 5
    # traj tensor N, L, 5
    trajs = []
    traj_idxs = np.unique(traj_list[..., 0].astype(int))
    for traj_idx in traj_idxs:
        cur_trajs = traj_list[traj_list[..., 0] == traj_idx]
        time_sort = np.argsort(cur_trajs[..., 1])
        cur_trajs = cur_trajs[time_sort]
        trajs.append(cur_trajs)
    return np.array(trajs)


def traj_tensor_to_list(cfg, trajs):
    traj_list = []
    for frame_idx in range(trajs.shape[1]):
        for track_idx in range(trajs.shape[0]):
            frame_time = frame_idx * cfg.real_time_between_frame
            traj_list.append([track_idx, frame_time, trajs[track_idx, frame_idx, 0], trajs[track_idx, frame_idx, 1], trajs[track_idx, frame_idx, 2]])
    return np.array(traj_list)



def get_traj_txt(cfg, seq_path):
    os.system(f'mkdir -p {seq_path}/tracks')

    init_image = cv2.imread(f'{seq_path}/images/{0:06d}.png', cv2.IMREAD_UNCHANGED)
    keypoints = get_keypoints(init_image)   # N, 2 (x, y)
    np.random.shuffle(keypoints)
    if keypoints.shape[0] > cfg.max_track:
        keypoints = keypoints[:cfg.max_track]
    trajs = load_dense_trajs(seq_path)  # H, W, L, 3
    trajs = trajs[keypoints[..., 1], keypoints[..., 0]] # N, L, 3

    tracks = traj_tensor_to_list(cfg, trajs)

    np.savetxt(f'{seq_path}/tracks/{cfg.track_file_name}.gt.txt', tracks)


def gen_seq_images(cfg, seq_idx, seq_name, seq_path):
    os.system(f'mkdir -p {seq_path}/images')
    os.system(f'mkdir -p {seq_path}/images_high_fps')
    os.system(f'mkdir -p {seq_path}/instance_id')
    if cfg.visualize:
        os.system(f'mkdir -p {seq_path}/vis')
        os.system(f'mkdir -p {seq_path}/instance_vis')
        os.system(f'mkdir -p {seq_path}/background')
        os.system(f'mkdir -p {seq_path}/foreground')
        os.system(f'mkdir -p {seq_path}/foreground_mask')

    # random image num
    foreground_image_num = np.random.randint(cfg.foreground_image_num[0], cfg.foreground_image_num[1] + 1)
    background_image_num = np.random.randint(cfg.background_image_num[0], cfg.background_image_num[1] + 1)

    # get image
    foreground_image_list = get_forebackground_image(cfg, cfg.foreground_image_dataset_path, foreground_image_num, seq_path, mode='foreground')
    background_image_list = get_forebackground_image(cfg, cfg.background_image_dataset_path, background_image_num, seq_path, mode='background')
    forebackground_image_list = background_image_list + foreground_image_list   # background first

    # get random mask
    forebackground_image_mask_list = []
    for img in background_image_list:
        forebackground_image_mask_list.append(None)
    for img in foreground_image_list:
        forebackground_image_mask_list.append(get_random_mask(img.shape[:2]))

    if cfg.visualize:
        for image_idx in range(foreground_image_num + background_image_num):
            if forebackground_image_mask_list[image_idx] is None:
                continue
            img_vis = forebackground_image_list[image_idx].copy()
            alpha_channel = np.zeros((img_vis.shape[0], img_vis.shape[1], 1))
            img_vis = np.concatenate([img_vis, alpha_channel], axis=2)
            img_vis[forebackground_image_mask_list[image_idx].astype(bool), -1] = 255
            cv2.imwrite(f'{seq_path}/vis/image_{image_idx}.png', img_vis)

    # get random image center
    foreground_image_center_list = get_random_image_center(cfg, foreground_image_list, mode='foreground')
    background_image_center_list = get_random_image_center(cfg, background_image_list, mode='background')
    forebackground_image_center_list = background_image_center_list + foreground_image_center_list  # background first
    forebackground_image_center_list = np.array(forebackground_image_center_list)

    # get random pose
    foreground_pose_list = get_forebackground_pose(cfg, foreground_image_num, mode='foreground')
    background_pose_list = get_forebackground_pose(cfg, background_image_num, mode='background')
    forebackground_pose_list = background_pose_list + foreground_pose_list  # background first
    forebackground_pose_list = np.array(forebackground_pose_list)

    np.save(f'{seq_path}/object_center.npy', forebackground_image_center_list)
    np.save(f'{seq_path}/object_pose.npy', forebackground_pose_list)

    H, W = cfg.image_size
    num_all_frame = cfg.event_per_frame * cfg.frame_per_keyframe * (cfg.keyframe_num - 1) + 1
    pose_list = get_pose_list(cfg, forebackground_pose_list, mode='event') # N, num_all_frame, 4

    if cfg.visualize:
        instance_color = generate_random_color(np.arange(len(forebackground_image_list) + 2) - 2)
        instance_color[0] = np.array([180, 170, 170])

    # compose each frame
    # with tqdm(total=num_all_frame) as speed:
    for frame_id in range(num_all_frame):
        if not cfg.gen_event:
            if frame_id % cfg.event_per_frame != 0:
                continue
        image = np.zeros((H, W, 3))
        image, instance_image = put_images_on_background(
            image_list=forebackground_image_list,
            background_image=image,
            image_mask_list=forebackground_image_mask_list,
            image_center_list=forebackground_image_center_list,
            scale_list=pose_list[:, frame_id, 0],
            rotation_list=pose_list[:, frame_id, 1],
            translation_list=pose_list[:, frame_id, 2:])
        image = image.astype(np.uint8)
        if cfg.gen_event:
            cv2.imwrite(f'{seq_path}/images_high_fps/{frame_id:06d}.png', image)
        if frame_id % cfg.event_per_frame == 0:
            cv2.imwrite(f'{seq_path}/images/{frame_id//cfg.event_per_frame:06d}.png', image)
            np.save(f'{seq_path}/instance_id/{frame_id//cfg.event_per_frame:06d}.npy', instance_image)
            if cfg.visualize:
                seg_vis = instance_image_to_rgb(instance_image, instance_color)
                cv2.imwrite(f'{seq_path}/instance_vis/{frame_id//cfg.event_per_frame:06d}.png', seg_vis)
                foreground_mask = instance_image > 0
                foreground_mask_img = np.tile(foreground_mask[..., None], (1, 1, 4)) * 255
                foreground_mask_img[..., :3] = 255 - foreground_mask_img[..., :3]
                cv2.imwrite(f'{seq_path}/foreground_mask/{frame_id//cfg.event_per_frame:06d}.png', foreground_mask_img)
                foreground_image = image.copy()
                foreground_image = np.concatenate([foreground_image, foreground_mask_img[..., -1:]], axis=-1)
                cv2.imwrite(f'{seq_path}/foreground/{frame_id//cfg.event_per_frame:06d}.png', foreground_image)
                background_image = np.zeros((H, W, 3))
                background_image, _ = put_images_on_background(
                    image_list=forebackground_image_list[:1],
                    background_image=background_image,
                    image_mask_list=forebackground_image_mask_list[:1],
                    image_center_list=forebackground_image_center_list[:1],
                    scale_list=pose_list[:1, frame_id, 0],
                    rotation_list=pose_list[:1, frame_id, 1],
                    translation_list=pose_list[:1, frame_id, 2:])
                cv2.imwrite(f'{seq_path}/background/{frame_id//cfg.event_per_frame:06d}.png', background_image)
                

        # speed.update(1)
        frame_id += 1


def gen_seq_trajs(cfg, seq_idx, seq_name, seq_path):

    os.system(f'mkdir -p {seq_path}/particle')
    
    H, W = cfg.image_size
    object_center = np.load(f'{seq_path}/object_center.npy')  # N, 2  (x, y)
    object_pose = np.load(f'{seq_path}/object_pose.npy')  # N, num_kf, 4  (scale, rotation, translation_x, translation_y)
    N = object_center.shape[0]

    x, y = np.meshgrid(np.arange(cfg.image_size[1]), np.arange(cfg.image_size[0]))
    coord_obj = np.stack([x, y], axis=2)    # H, W, 2 (x, y)
    coord_obj = np.concatenate([coord_obj, np.ones((H, W, 1))], axis=2)
    np.save(f'{seq_path}/particle/{0:06d}.npy', coord_obj)

    # reproject init coord to obj coord
    init_instance_map = np.load(f'{seq_path}/instance_id/{0:06d}.npy')
    nocast_mask = init_instance_map == -1
    for obj_idx in range(N):
        if obj_idx == -1:   # no cast
            continue
        mask = init_instance_map == obj_idx
        if mask.sum() == 0:
            continue

        coord_obj[mask] = project_points(coord_obj[mask], object_center[obj_idx], object_pose[obj_idx, 0], mode='reproject')

    # project obj coord to each frame
    frame_pose_list = get_pose_list(cfg, object_pose, mode='frame')
    num_frame = cfg.frame_per_keyframe * (cfg.keyframe_num - 1) + 1
    for frame_id in range(1, num_frame):
        coord = np.copy(coord_obj)
        for obj_idx in range(N):
            if obj_idx == -1:   # no cast
                continue
            mask = init_instance_map == obj_idx
            if mask.sum() == 0:
                continue

            coord[mask] = project_points(coord[mask], object_center[obj_idx], frame_pose_list[obj_idx, frame_id], mode='project')

        # check vis, occ, out
        cur_instance_map = np.load(f'{seq_path}/instance_id/{frame_id:06d}.npy')

        # coord_xy = np.around(coord[...,:2]).astype(int)
        # coord_xy[..., 0] = np.clip(coord_xy[..., 0], 0, W - 1)
        # coord_xy[..., 1] = np.clip(coord_xy[..., 1], 0, H - 1)
        # coord_xy = coord_xy.reshape((-1, 2))
        # coord_instance_id = cur_instance_map[coord_xy[:,1], coord_xy[:,0]].reshape((H, W))
        # coord[..., 2] = init_instance_map == coord_instance_id  # different instance id means occ

        coord_floor = np.floor(coord[...,:2]).astype(int)
        coord_ceil = np.ceil(coord[...,:2]).astype(int)
        coord_floor[..., 0] = np.clip(coord_floor[..., 0], 0, W - 1)
        coord_floor[..., 1] = np.clip(coord_floor[..., 1], 0, H - 1)
        coord_ceil[..., 0] = np.clip(coord_ceil[..., 0], 0, W - 1)
        coord_ceil[..., 1] = np.clip(coord_ceil[..., 1], 0, H - 1)
        coord_floor = coord_floor.reshape((-1, 2))
        coord_ceil = coord_ceil.reshape((-1, 2))
        coord_instance_id_leftup = cur_instance_map[coord_floor[:, 1], coord_floor[:, 0]].reshape((H, W))
        coord_instance_id_leftdown = cur_instance_map[coord_ceil[:, 1], coord_floor[:, 0]].reshape((H, W))
        coord_instance_id_rightup = cur_instance_map[coord_floor[:, 1], coord_ceil[:, 0]].reshape((H, W))
        coord_instance_id_rightdown = cur_instance_map[coord_ceil[:, 1], coord_ceil[:, 0]].reshape((H, W))
        coord[..., 2] = np.logical_or(
            np.logical_or(init_instance_map >= coord_instance_id_leftup, init_instance_map >= coord_instance_id_rightup),
            np.logical_or(init_instance_map >= coord_instance_id_leftdown, init_instance_map >= coord_instance_id_rightdown)    # TODO: == or >= ?
        )  # any nearby vis

        outbound_mask = np.logical_or(
            np.logical_or(coord[..., 0] < 0, coord[..., 0] >= W),
            np.logical_or(coord[..., 1] < 0, coord[..., 1] >= H),
        )
        coord[outbound_mask, 2] = -1 # outbound
        coord[nocast_mask, 2] = -2  # nocast
        np.save(f'{seq_path}/particle/{frame_id:06d}.npy', coord)

        '''
        vis 1
        occ 0
        out -1
        nocast -2
        '''
    
    get_traj_txt(cfg, seq_path)


def gen_seq_events(cfg, seq_idx, seq_name, seq_path):
    os.system(f'mkdir -p {seq_path}/events')

    # events.h5
    num_all_frame = cfg.event_per_frame * cfg.frame_per_keyframe * (cfg.keyframe_num - 1) + 1
    real_time_between_event_frame = cfg.real_time_between_frame / cfg.event_per_frame

    interpTimes = np.arange(num_all_frame) * real_time_between_event_frame
    interpTimes = interpTimes.tolist()
    
    make_events(input_dir=f'{seq_path}/images_high_fps',
                output_dir=f'{seq_path}/events',
                time_list=interpTimes,
                simulator_name=cfg.event_simulator_name)
    
    # generate_time_surface_single(seq_path, cfg.output_path, visualize=False, n_bins=5, dt=0.01,
    #                              start_time=0, 
    #                              end_time=cfg.frame_per_keyframe * (cfg.keyframe_num - 1) * cfg.real_time_between_frame * 1e6,
    #                              H=cfg.image_size[0], 
    #                              W=cfg.image_size[1])    # TODO: config


def vis_seq(cfg, seq_name, seq_path, target_path=None):
    # vis
    image_list = []
    for img_file in sorted(os.listdir(f'{seq_path}/images/')):
        image_list.append(cv2.imread(f'{seq_path}/images/{img_file}', cv2.IMREAD_UNCHANGED))

    # traj_list = []
    # for traj_file in sorted(os.listdir(f'{seq_path}/particle/')):
    #     traj_list.append(np.load(f'{seq_path}/particle/{traj_file}'))
    # trajs = np.stack(traj_list, axis=2)

    # vis_list = []
    # vis_color_map = generate_random_color([-2, -1, 0, 1])
    # fo r frame_idx in range(trajs.shape[2]):
    #     vis_list.append(instance_image_to_rgb(trajs[:, :, frame_idx, 2], color_map=vis_color_map))

    # # superpoint sample
    # init_image = cv2.imread(f'{seq_path}/images/{0:06d}.png', cv2.IMREAD_UNCHANGED)
    # keypoints = get_keypoints(init_image)
    # viss = trajs[keypoints[..., 1], keypoints[..., 0], :, 2]
    # trajs = trajs[keypoints[..., 1], keypoints[..., 0], :, :2]

    # # grid sample
    # viss = trajs[::60, ::60, :, 2]  # TODO: config
    # trajs = trajs[::60, ::60, :, :2]

    event_list = []
    for evt_file in sorted(os.listdir(f'{seq_path}/events/0.0100/time_surfaces_v2_5')): # TODO: config
        event_repr = read_input(f'{seq_path}/events/0.0100/time_surfaces_v2_5/{evt_file}', 'time_surfaces_v2_5')
        event_repr = np.transpose(event_repr, (2, 0, 1))
        event_list.append(time_surface_to_rgb(event_repr))

    trajs = np.genfromtxt(f'{seq_path}/tracks/{cfg.track_file_name}.gt.txt')
    trajs = traj_list_to_tensor(trajs)  # N, L, 5
    trajs = trajs[..., 2:]

    viss = trajs[..., 2]
    trajs = trajs[..., :2]

    viss = viss.reshape((-1, viss.shape[-1]))
    trajs = trajs.reshape((-1, trajs.shape[-2], trajs.shape[-1]))
    # image_list = draw_trajs(image_list, trajs, viss=viss, length=5, thickness=2, color=(255, 0, 0))
    image_list = draw_trajs(event_list, trajs, viss=viss, length=5, thickness=2, color=(255, 0, 0))

    # for frame_idx in range(len(image_list)):
    #     image_list[frame_idx] = np.concatenate([image_list[frame_idx], vis_list[frame_idx]], axis=0)
    # for frame_idx in range(len(image_list)):
    #     image_list[frame_idx] = np.concatenate([image_list[frame_idx], event_list[frame_idx]], axis=0)

    # image_high_fps_list = []
    # make_video(image_high_fps_list, f'{seq_path}/images_high_fps.mp4', fps=5)
    # instance_image = instance_image_to_rgb(instance_image)
    # cv2.imwrite(f'{seq_path}/instance_id/{frame_id//cfg.event_per_frame:06d}.png', instance_image)

    if target_path is None:
        make_video(image_list, f'{seq_path}/images_{seq_name}.mp4', fps=5)
    else:
        make_video(image_list, f'{target_path}/images_{seq_name}.mp4', fps=5)


def clear_seq(seq_path):
    os.system(f'rm {seq_path}/particle -r')
    os.system(f'rm {seq_path}/images_high_fps -r')
    os.system(f'rm {seq_path}/instance_id -r')
    os.system(f'rm {seq_path}/object_center.npy')
    os.system(f'rm {seq_path}/object_pose.npy')


def gen_seq(input_data):    # cfg, seq_idx

    cfg, seq_idx = input_data

    random.seed(seq_idx)
    np.random.seed(seq_idx)

    seq_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{seq_idx}'
    seq_path = f'{cfg.output_path}/{cfg.dataset_mode}/{seq_name}'
    os.system(f'mkdir -p {seq_path}')

    # Convert the configuration to a YAML string
    cfg_yaml = OmegaConf.to_yaml(cfg)
    # Write the YAML string to a file
    with open(f'{seq_path}/config.yaml', 'w') as file:
        file.write(cfg_yaml)

    # gen images
    print("images start")
    gen_seq_images(cfg, seq_idx, seq_name, seq_path)
    # print("images end")

    # gen tracks
    print("trajs start")
    gen_seq_trajs(cfg, seq_idx, seq_name, seq_path)
    # print("trajs end")

    # gen events
    if cfg.gen_event:
        print("events start")
        gen_seq_events(cfg, seq_idx, seq_name, seq_path)
        # print("events end")

    if cfg.visualize:
        # vis seq
        print("vis start")
        vis_seq(cfg, seq_name, seq_path)
        # print("vis end")

    if not cfg.visualize:
        # clear tmp
        # print("clear start")
        clear_seq(seq_path)
        # print("clear end")

    return True


if __name__ == '__main__':

    with open(f'{CONFIG_PATH}', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    tasks = []
    with multiprocessing.Pool(THREAD_NUM) as thread_pool:
        for seq_idx in range(cfg.seq_range[0], cfg.seq_range[1] + 1):
            tasks.append((cfg, seq_idx))
            # gen_seq((cfg, seq_idx))

        list(tqdm(thread_pool.imap(gen_seq, tasks), total=len(tasks)))

