import cv2
import numpy as np
import random
from collections import Counter
from matplotlib import pyplot as plt


def generate_track_colors_y(points):
    # points N, 2
    y = points[..., 1]
    colormap = apply_colormap(y, colormap=cv2.COLORMAP_RAINBOW).reshape((-1, 3)) / 255.0
    colormap_list = []
    for cm in colormap:
        colormap_list.append(tuple(cm))
    return colormap_list


def generate_track_colors(n_tracks):
    track_colors = []
    for i_track in range(n_tracks):
        track_colors.append(
            (
                random.randint(0, 255) / 255.0,
                random.randint(0, 255) / 255.0,
                random.randint(0, 255) / 255.0,
            )
        )
    return track_colors


def render_pred_tracks(pred_track_interpolator, t, img, track_colors, dt_track=0.0025):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img, cmap="gray")
    ax.autoscale(False)

    for track_id in range(pred_track_interpolator.n_corners):
        pred_track_data_curr = pred_track_interpolator.interpolate(track_id, t)

        if not isinstance(pred_track_data_curr, type(None)):
            pred_track_data_hist = pred_track_interpolator.history(
                track_id, t, dt_track
            )

            # ToDo: Change back
            pred_track_data_hist = np.concatenate(
                [pred_track_data_hist, pred_track_data_curr[None, :]], axis=0
            )

            ax.plot(
                pred_track_data_hist[:, 0],
                pred_track_data_hist[:, 1],
                color=track_colors[track_id],
                alpha=0.5,
                linewidth=8,
                linestyle="solid",
            )

    for track_id in range(pred_track_interpolator.n_corners):
        pred_track_data_curr = pred_track_interpolator.interpolate(track_id, t)

        if not isinstance(pred_track_data_curr, type(None)):

            ax.scatter(
                pred_track_data_curr[0],
                pred_track_data_curr[1],
                color=track_colors[track_id],
                alpha=1,
                linewidth=6,
                s=40,
                marker="o",
            )

    ax.axis("off")
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    fig_array = fig_to_img(fig)
    plt.close(fig)
    return fig_array



def fig_to_img(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    w, h = fig.canvas.get_width_height()
    return data.reshape((h, w, 3))


def event_voxel_to_rgb(event_voxel):
    H, W = event_voxel.shape[-2:]
    img = np.full((H,W,3), fill_value=255,dtype='uint8')

    event_image = np.sum(event_voxel, axis=0)

    # assume the most frequent value corresponds to no event
    a_list = event_image.ravel().tolist()
    counts = Counter(a_list)
    most_frequent = counts.most_common(1)
    most_frequent_element = most_frequent[0][0]
    event_image = event_image - most_frequent_element

    # assume the mean value corresponds to no event
    # event_image = event_image - np.mean(event_image)

    max_v = np.max(np.abs(event_image))
    event_image = event_image / max_v

    magnitude = np.abs(event_image) ** 0.5
    base = 0.2
    color_mag = ((1-base) * 255 * magnitude).astype(np.uint8)
    color_full = np.ones_like(color_mag) * 255
    img[event_image > 0] = np.stack([color_full, 255-color_mag, 255-color_mag], axis=-1)[event_image > 0]
    img[event_image < 0] = np.stack([1-color_mag, 255-color_mag, color_full], axis=-1)[event_image < 0]

    return img


def time_surface_to_rgb(event_voxel):
    C, H, W = event_voxel.shape[-3:]
    img = np.full((H,W,3), fill_value=255,dtype='uint8')

    event_voxel[:C//2] = -event_voxel[:C//2]
    event_image = np.sum(event_voxel, axis=0)

    # assume the most frequent value corresponds to no event
    a_list = event_image.ravel().tolist()
    counts = Counter(a_list)
    most_frequent = counts.most_common(1)
    most_frequent_element = most_frequent[0][0]
    event_image = event_image - most_frequent_element

    # assume the mean value corresponds to no event
    # event_image = event_image - np.mean(event_image)

    max_v = np.max(np.abs(event_image))
    event_image = event_image / max_v

    magnitude = np.abs(event_image) ** 0.25
    base = 0.2
    color_mag = ((1-base) * 255 * magnitude).astype(np.uint8)
    color_full = np.ones_like(color_mag) * 255
    img[event_image > 0] = np.stack([color_full, 255-color_mag, 255-color_mag], axis=-1)[event_image > 0]
    img[event_image < 0] = np.stack([255-color_mag, 255-color_mag, color_full], axis=-1)[event_image < 0]

    return img


# generate random color for each instance
def generate_random_color(ids):
    num = len(ids)
    colors = np.random.randint(0, 255, (num, 3))
    color_map = {}
    for index, _id in enumerate(ids):
        color_map[_id] = colors[index]

    return color_map


def instance_image_to_rgb(instance_image, color_map=None):
    object_ids = np.unique(instance_image)
    if color_map is None:
        color_map = generate_random_color(object_ids.tolist())
    seg_vis = np.zeros(list(instance_image.shape) + [3], dtype=np.uint8)
    for s in object_ids:
        seg_vis[instance_image == s] = color_map[s]
    return seg_vis