import os, h5py, cv2, torch, sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, '/home/xianr/works/FlyingThings22')
from flying_v7.v2e import view_events, events_scatter_on_image_pos_vs_neg
# from flying_v7.v2e import view_events
from flying_v7.utils import img2video, stack_imgs, get_scatter
# from flying_v7.blender.visHdf5Files import flow_to_rgb


def warp_event_np(evt, flow, t_aim, size, byFrameOrSpeed=True, time_scale=1.0, fps=0.0):
    evt_img = np.zeros(size)
    if byFrameOrSpeed:
        x_warp = evt[:, 1] + flow[:, 0]
        y_warp = evt[:, 2] + flow[:, 1]
    else:
        # import pdb; pdb.set_trace()
        x_warp = evt[:, 1] + flow[:, 0] * (t_aim - evt[:, 0]) * time_scale
        y_warp = evt[:, 2] + flow[:, 1] * (t_aim - evt[:, 0]) * time_scale
    x_warp = x_warp.round().clip(0, size[1] - 1)
    y_warp = y_warp.round().clip(0, size[0] - 1)
    evt_warped = np.stack((evt[:,0], x_warp, y_warp, evt[:,3]), axis=1)
    return evt_warped

def step_by_warp(evt_combo_flow, t_aim, size, time_scale=1.0, fps=0.0, dt=0.1):
    time_slices = []; t_a = evt_combo_flow[:, 0].min(); t_max = evt_combo_flow[:, 0].max()
    while t_a <= t_max:
        t_b = t_a + dt
        evt_mask = np.where((evt_combo_flow[:,0]>=t_a)&(evt_combo_flow[:,0]<t_b))
        combo_step = evt_combo_flow[evt_mask]
        evt_step, evt_flow_step = combo_step[:, :4], combo_step[:, 4:] * fps
        warped_step = warp_event_np(evt_step, evt_flow_step, t_b, size, False, 1.0, fps)
        warped_step[:, 0] = t_b
        potential_combo = evt_combo_flow[(evt_combo_flow[:,0]>=t_b)&(evt_combo_flow[:,0]<t_b+5/fps)]
        new_combo = np.zeros_like(combo_step)
        new_combo[:, :4] = warped_step
        for i, w_e in enumerate(warped_step):
            # warped_step events 的光流 照搬 evt_combo_flow 里距离最近的 events的 光流
            distance = np.linalg.norm(potential_combo[:, (1,2)] - w_e[[1,2]], axis=1)
            distance_norm = (distance - distance.min()) / (distance - distance.max()+1e-9)
            dealt_time = np.abs(potential_combo[:, 0] - w_e[0])
            norm_3d = (dealt_time-dealt_time.min())/(dealt_time-dealt_time.max()+1e-9)+distance_norm
            fit_idx = norm_3d.argmin()
            new_combo[i, 4:] = potential_combo[fit_idx, [4,5]]
        # import pdb; pdb.set_trace()
        evt_combo_flow[evt_mask] = new_combo

        time_slices.append(t_a); t_a+=dt
        if evt_combo_flow[:, 0].min() < t_a: import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
    return evt_combo_flow[:,:4]

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    https://github.com/xinntao/EDVR/blob/master/basicsr/models/archs/arch_util.py
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = torch.nn.functional.grid_sample(
        x, vgrid_scaled, mode=interp_mode,
        padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


# def make_evt_flow_BasedOnFlow(output_dir, fps, size, num_frames, events=None):
#     def warp_mask(flow):
#         h, w = flow.shape[:2]
#         warp_x = (flow[:,:,0] + np.arange(w)).round().astype(np.int32).clip(0,h-1)
#         warp_y = (flow[:,:,1] + np.arange(h)[:,None]).round().astype(np.int32).clip(0,w-1)
#         warped_yx = np.stack((warp_y, warp_x), 2).reshape(-1, 2)
#         # valid_map = np.zeros((h, w), dtype=np.int32)
#         # for y, x in warped_yx:
#         #     valid_map[y, x] += 1
#         valid_map2 = np.zeros((h, w), dtype=np.int32)
#         np.add.at(valid_map2, tuple(warped_yx.transpose().tolist()), 1)
#         return valid_map2

#     frame_dir = f"{output_dir}/frames"
#     if events is None: events = np.load(f"{output_dir}/events.npy")
#     evt_flow_list = []
#     for f_i in range(num_frames):
#         t0, t1 = f_i / fps, (f_i+1) / fps
#         evt_i = events[(events[:,0]>=t0) * (events[:,0]<t1)]
#         with h5py.File(f"{output_dir}/hdf5/{f_i}.hdf5", 'r') as hdf5_i_t0:
#             t0_fwd_flow = np.array(hdf5_i_t0['forward_flow'])
#         evt_flow_map = t0_fwd_flow.copy()
#         if f_i+1 < num_frames:
#             with h5py.File(f"{output_dir}/hdf5/{f_i+1}.hdf5", 'r') as hdf5_i_t1:
#                 t1_fwd_flow = np.array(hdf5_i_t1['forward_flow'])
#             # 在是《光流warp终点》的地方使用下一帧的光流，因为他们是在终点处产生的events
#             valid_mask = warp_mask(t0_fwd_flow)
#             evt_flow_map[valid_mask > 0] = t1_fwd_flow[valid_mask > 0]
#         plt.imsave(f'{frame_dir}/{f_i}_evt_flow_map.png', flow_to_rgb(evt_flow_map).clip(0,1))
#         evt_i_flow = evt_flow_map[evt_i[:,2].astype(np.int32), evt_i[:,1].astype(np.int32)]
#         evt_flow_list.append(evt_i_flow)
#         # evt_i_flow_img = np.zeros_like(t0_fwd_flow)
#         # plt.imsave(f'{frame_dir}/{f_i}_evt_flow_map.png', flow_to_rgb(evt_flow_map))
#     events_flow = np.concatenate(evt_flow_list, 0)
#     np.save(f'{output_dir}/events_flow.npy', events_flow)

#     return events_flow

# def make_evt_flow_BasedOnDepth(output_dir, fps, size, num_frames, events=None):
#     frame_dir = f"{output_dir}/frames"
#     if events is None: events = np.load(f"{output_dir}/events.npy")
#     evt_flow_list = []
#     combo_evt_flow_list = []
#     t0_flow_maps = []
#     for f_i in range(num_frames):
#         t0, t1 = f_i / fps, (f_i+1) / fps
#         evt_i = events[(events[:,0]>=t0) * (events[:,0]<t1)]
#         with h5py.File(f"{output_dir}/hdf5/{f_i}.hdf5", 'r') as hdf5_i_t0:
#             t0_fwd_flow = np.array(hdf5_i_t0['forward_flow'])
#             t0_depth = np.array(hdf5_i_t0['depth'])
#         evt_flow_map = t0_fwd_flow.copy()
#         t0_flow_maps.append(t0_fwd_flow)
#         if f_i+1 < num_frames:
#             with h5py.File(f"{output_dir}/hdf5/{f_i+1}.hdf5", 'r') as hdf5_i_t1:
#                 t1_fwd_flow = np.array(hdf5_i_t1['forward_flow'])
#                 t1_depth = np.array(hdf5_i_t1['depth'])
#             # 在《下一帧深度比上一帧浅》的地方使用下一帧的光流，因为他们是新来的events
#             evt_flow_map[t1_depth < t0_depth] = t1_fwd_flow[t1_depth < t0_depth]
#         plt.imsave(f'{frame_dir}/{f_i}_evt_flow_map.png', flow_to_rgb(evt_flow_map).clip(0,1))
#         evt_i_flow = evt_flow_map[evt_i[:,2].astype(np.int32), evt_i[:,1].astype(np.int32)]
#         # evt_flow_list.append(evt_i_flow)
#         combo_evt_flow_list.append(np.concatenate((evt_i, evt_i_flow), 1))
#     events_flow = np.concatenate(combo_evt_flow_list, 0)


#     is_debug = True
#     if is_debug:
#         # 画出光流分布的散点图
#         plt.cla()
#         plt.scatter(events_flow[:,4], events_flow[:,5])
#         plt.grid()
#         plt.savefig(f"{output_dir}/evt_flow_scatter.png")

#         np.save(f'{output_dir}/events_combo_flow.npy', events_flow)   # delta pixels per frame
#         # 当前events_flow的单位是(delta pixels), 用于多帧warp时，(pixels/s)的速度概念更合适
#         events_flow_speed = events_flow * fps
#         np.save(f'{output_dir}/events_flow_speed.npy', events_flow_speed)   # pixels/s


#         # 对rgb flow的分布和追溯
#         t0_flow_all = np.stack(t0_flow_maps).reshape(-1, 2)
#         # 画出光流分布的散点图
#         plt.imsave(f"{output_dir}/t0_flow_all_scatter.png", get_scatter(t0_flow_all))
#         split_x = [0, 0.5, 2, 3, 4, 4.6, 5.1, 6]
#         xlim = (t0_flow_all[:,0].min()-1, t0_flow_all[:,0].max()+1)
#         ylim = (t0_flow_all[:,1].min()-1, t0_flow_all[:,1].max()+1)
        
#         for i, t0_flow in enumerate(t0_flow_maps):
#             # import pdb; pdb.set_trace()
#             flow_dist = np.zeros(size+(3,), np.uint8)
#             flow_dist[t0_flow[:,:,0]<0.5] = (255, 0, 0)
#             flow_dist[(t0_flow[:,:,0]>=0.5)*(t0_flow[:,:,0]<2)] = (0, 255, 0)
#             flow_dist[(t0_flow[:,:,0]>=2)*(t0_flow[:,:,0]<3)] = (0, 0, 255)
#             flow_dist[(t0_flow[:,:,0]>=3)*(t0_flow[:,:,0]<4)] = (255, 0, 255)
#             flow_dist[(t0_flow[:,:,0]>=4)*(t0_flow[:,:,0]<4.6)] = (0, 255, 255)
#             flow_dist[(t0_flow[:,:,0]>=4.6)*(t0_flow[:,:,0]<5.1)] = (0, 255, 255)
#             flow_dist[t0_flow[:,:,0]>=5.1] = (255, 255, 255)
#             t0_flow_plain = t0_flow.reshape(-1, 2)
#             scater_i = get_scatter(t0_flow_plain, xlim, ylim)
#             # flow_dstribution = stack_imgs([[flow_dist, scater_i]])

#             # import pdb; pdb.set_trace()
#             # 对照events和其depth光流map
#             evt_d_flow_combo = stack_imgs([
#                 [flow_dist, scater_i, None,
#                  plt.imread(f"{output_dir}/frames/{i}_blur.png")],
#                 [plt.imread(f"{output_dir}/frames/{i}_event_image_pos_vs_neg.png"),
#                  plt.imread(f"{output_dir}/frames/{i}_evt_flow_map.png"),
#                  plt.imread(f"{output_dir}/frames/{i}_forward_flow.png"),
#                  plt.imread(f"{output_dir}/frames/{i+1}_forward_flow.png") if i+1<num_frames else None],
#             ])
#             plt.imsave(f"{output_dir}/frames/{i}_evt_d_flow_combo.png", evt_d_flow_combo)

#         img2video(f"{output_dir}/frames", output_dir, num_frames, '_evt_d_flow_combo', 
#                     'evt_d_flow_combo.mp4', 1)


#     return events_flow


def eval_evt_flow(item_dir, fps, size, total_frames, events_combo_flow=None):
    if events_combo_flow is None: events_combo_flow = np.load(f"{item_dir}/events_combo_flow.npy")
    events, evt_flow = np.split(events_combo_flow, [4],1)

    idx_start, idx_end = 0, total_frames-1
    # idx_start, idx_end = 0, total_frames//4
    # idx_start, idx_end = 0, 3
    im0 = plt.imread(f"{item_dir}/frames/{idx_start}_blur.png")[:,:,:3]
    im1 = plt.imread(f"{item_dir}/frames/{idx_end}_blur.png")[:,:,:3]
    ts, te = idx_start / fps, idx_end / fps 
    tm = (ts + te) / 2
    # import pdb; pdb.set_trace()
    events_range = events[(events[:,0] >= ts) & (events[:,0] < te)]

    evt_flow_speed_range = evt_flow[(events[:,0] >= ts) & (events[:,0] < te)] * fps
    # evt_flow_speed_range = evt_flow[(events[:,0] >= ts) & (events[:,0] < te)][:,4:] * fps

    evt_im_range = events_scatter_on_image_pos_vs_neg(events_range, size)
    events_range_warped = warp_event_np(events_range, evt_flow_speed_range, tm, size, False, 1.0, fps)
    evt_im_warp = events_scatter_on_image_pos_vs_neg(events_range_warped, size)
    # events_step_warped = step_by_warp(events_combo_flow, tm, size, 1.0, fps, 0.1)
    # evt_im_step = events_scatter_on_image_pos_vs_neg(events_step_warped, size)

    img_combo = np.hstack((np.vstack((im0, im1)), np.vstack((evt_im_range/255, evt_im_warp/255))))
    # img_combo = stack_imgs([[im0, evt_im_range/255], [im1, evt_im_warp/255, evt_im_step/255]])
    plt.imsave(f'{item_dir}/eval_event_combo_range.png', img_combo)
    # import pdb; pdb.set_trace()



def test():
    item_dir = '/home/xianr/works/FlyingThings22/outputs/Flying_v6_74'
    # evt_flow = make_evt_flow_BasedOnFlow(item_dir, 60, (512,512), 40)
    # evt_flow = make_evt_flow_BasedOnDepth(item_dir, 60, (512,512), 400)
    # import pdb; pdb.set_trace()
    # eval_evt_flow(item_dir, 60, (512,512), 400, evt_flow=evt_flow)
    # eval_evt_flow(item_dir, 60, (512,512), 400)

if __name__=='__main__':
    test()
    