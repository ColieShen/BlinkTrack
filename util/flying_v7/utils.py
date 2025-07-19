import os, glob, random
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_dir(output_base="outputs/default"):
    dir_base = os.path.abspath(output_base)
    output_dir_exists = glob.glob(f"{output_base}_*")
    output_dir_count = max([int(name.split(f"{output_base}_")[-1]) for name in output_dir_exists]) if len(output_dir_exists) else 0
    dir_id = str(output_dir_count+1)
    # dir_id = 0
    output_dir = f"{output_base}_{dir_id}"
    if not os.path.exists(output_dir):
        os.system(f'mkdir -p {output_dir}')
        # os.makedirs(output_dir)

    return output_dir

    
def sample_shape_and_bgimg(base_dir, shape_dir, bg_img_dir, shape_num=1):
    saved_shape_p = f'{os.path.split(base_dir)[0]}/detected_shapes_p.txt'
    if os.path.exists(saved_shape_p):
        with open(saved_shape_p, 'rt') as f_shape:
            all_shapes_p = [s.strip() for s in f_shape.readlines()]
    else:
        all_shapes_p = glob.glob(f"{shape_dir}*/*/models/model_normalized.obj")
        if not all_shapes_p:
            import pdb; pdb.set_trace()
        with open(saved_shape_p, 'wt') as f_shape:
            f_shape.write('\n'.join(all_shapes_p))
    shapes_p = random.sample(all_shapes_p, shape_num)
    # synset_id, source_id = shape_p[0].split('/')[-2:]

    saved_bg_img_p = f'{os.path.split(base_dir)[0]}/detected_bg_img_p.txt'
    if os.path.exists(saved_bg_img_p):
        with open(saved_bg_img_p, 'rt') as f_shape:
            all_bg_img_p = [s.strip() for s in f_shape.readlines()]
    else:
        # all_bg_img_p = glob.glob(f"/home/xianr/data/datasets/ADE20K/images/ADE/training/*/*/*.jpg")
        all_bg_img_p = glob.glob(f"{bg_img_dir}images/ADE/training/*/*/*.jpg")
        if not all_bg_img_p:
            import pdb; pdb.set_trace()
        with open(saved_bg_img_p, 'wt') as f_shape:
            f_shape.write('\n'.join(all_bg_img_p))
    bg_img_p = random.choice(all_bg_img_p)

    with open(f"{base_dir}/image_sourse_info.txt", 'wt') as isi_f:
        content = "Background image:\n- {bg_img_p}\nShape objects:\n- "
        content += "\n- ".join(shapes_p)
        isi_f.write(content)

    return shapes_p, bg_img_p


def get_scatter(t0_flow_plain, xlim=(-1, 10), ylim=(-1, 1)):
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    # _ = plt.hist(pixel_probs_k.cpu().detach().view(-1).numpy(), 100)
    plt.scatter(t0_flow_plain[:,0], t0_flow_plain[:,1])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.grid()
    plt.savefig(buffer, format='png')
    plt.clf()
    buffer.seek(0)
    im = np.array(Image.open(buffer))
    return im

def stack_imgs(img_mats):
    '''
    把任意的numpy 以括弧组织起来，能输出stack到一起的大图，大大方便了对比看图
    img_mat:[[im1, im2,],
                [None, im3,],
                [im4]]
        im: h*w*3
    '''
    h_num = len(img_mats)
    row_lens = [len(row) for row in img_mats]
    w_num = max(row_lens)
    shapes = np.array([[img_mats[i][j].shape[:2] if (
        j<len(img_mats[i]) and img_mats[i][j] is not None
        ) else [0,0] for j in range(w_num)] for i in range(h_num)])
    grille_size = 3 # dark-bright-dark
    Hs, Ws = shapes[:,:,0].max(1), shapes[:,:,1].max(0)
    # import pdb; pdb.set_trace()
    H_canvas, W_canvas = (Hs.sum()+grille_size*(h_num-1)+1)//2*2, (Ws.sum()+grille_size*(w_num-1)+1)//2*2
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    for k in range(1, len(Hs)):
        canvas[Hs[:k].sum()+grille_size*k-grille_size//2-1] = 255
    for k in range(1, len(Ws)):
        canvas[:,Ws[:k].sum()+grille_size*k-grille_size//2-1] = 255
    for i in range(h_num):
        for j in range(w_num):
            if j>=len(img_mats[i]) or img_mats[i][j] is None:
                continue
            img = img_mats[i][j]
            if isinstance(img, np.ndarray):
                H_bias = Hs[:i].sum()+grille_size*i
                W_bias = Ws[:j].sum()+grille_size*j
                H_im, W_im = img.shape[:2]
                if len(img.shape) == 3:
                    if img.dtype==np.float32 and img.max()<=1: 
                        img = (img*255).astype(np.uint8)
                    canvas[H_bias:H_bias+H_im,W_bias:W_bias+W_im] = img[:,:,:3]
                elif len(img.shape) == 2:
                    import pdb; pdb.set_trace()
                    norm = plt.Normalize(vmin=img.min(), vmax=img.max())
                    # map the normalized data to colors
                    # image is now RGBA (512x512x4)
                    img3d = plt.cm.jet(norm(img))
                    canvas[H_bias:H_bias+H_im,W_bias:W_bias+W_im] = img3d[:,:,:3]
                else:
                    import pdb; pdb.set_trace()
            else:
                print('Not numpy array!')
                import pdb; pdb.set_trace()

    return canvas
  

def img2video(input_dir, output_dir, num_f, subfix='_colors', save_name='colors.mp4', duration = 0.1):
    ffmpeg_input = ""
    for i in range(num_f):
        p = f"{input_dir}/{i}{subfix}.png"
        ffmpeg_input += f'file {os.path.abspath(p)}\n'
        ffmpeg_input += f'duration {duration}\n'
    # ffmpeg_input += f'file {os.path.abspath(p)}'

    with open(f"{input_dir}/ffmpeg_input.txt", 'wt') as f:
        f.write(ffmpeg_input)

    os.system(f"ffmpeg -nostdin -f concat -safe 0 -i {input_dir}/ffmpeg_input.txt"
        f" -vsync vfr -pix_fmt yuv420p {output_dir}/{save_name}")


def combo_frame(output_dir, num_frames):
    frame_dir = f"{output_dir}/frames"
    for i in range(num_frames):
        colors = (plt.imread(f'{frame_dir}/{i}_blur.png')*255).astype(np.uint8)
        hdr = (plt.imread(f'{frame_dir}/{i}_hdr.png')*255).astype(np.uint8)
        depth = (plt.imread(f'{frame_dir}/{i}_depth.png')*255).astype(np.uint8)
        fwd_flow = (plt.imread(f'{frame_dir}/{i}_forward_flow.png')*255).astype(np.uint8)
        evt_add = (plt.imread(f'{frame_dir}/{i}_event_image_add.png')*255).astype(np.uint8)
        evt_abs = (plt.imread(f'{frame_dir}/{i}_event_image_abs.png')*255).astype(np.uint8)
        evt_pvsn = (plt.imread(f'{frame_dir}/{i}_event_image_pos_vs_neg.png')*255).astype(np.uint8)
        combo = stack_imgs([
            [colors, depth, fwd_flow],
            [hdr, evt_abs, evt_pvsn]
        ])
        plt.imsave(f'{frame_dir}/{i}_combo_2x3.png', combo)
    img2video(frame_dir, output_dir, num_frames, '_combo_2x3', 'combo_2x3.mp4')


def clean_tmp_files(output_dir):
    os.system(f'rm -r {output_dir}/hdf5')
    
def check_blender_result(output_dir, config):
    slow_num = len(os.listdir(f'{output_dir}/hdf5/slow'))
    fast_num = len(os.listdir(f'{output_dir}/hdf5/fast'))
    return slow_num > 0 and (fast_num > 0 or not config['render_event'])

def clean_unfinished(output_dir):
    print(f'removing {output_dir}')
    os.system(f'rm -r {output_dir}')

def clean_unfinished_all(data_root):
    for scene_name in sorted(os.listdir(f'{data_root}')):
        for seq_name in (sorted(os.listdir(f'{data_root}/{scene_name}'))):
            flag1 = os.path.exists(f'{data_root}/{scene_name}/{seq_name}/events_voxel')
            flag2 = os.path.exists(f'{data_root}/{scene_name}/{seq_name}/events_left')
            if not flag1 and not flag2:
                print(f'removing {data_root}/{scene_name}/{seq_name}')
                os.system(f'rm -r {data_root}/{scene_name}/{seq_name}')

def make_video(images, outvid=None, fps=10, size=None, is_color=True, format="mp4v"):
    if len(images) < 1:
        return
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except:
        pass

    try:
        import bpy
        bpy.context.scene.last_seed = seed
    except:
        pass

if __name__ == '__main__':
    clean_unfinished_all('/mnt/nas_8/datasets/eflyingthings/train')