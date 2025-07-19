import torch
import numpy as np
import random
import torch.nn.functional as F


from math import pi
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, rotate


def augment_rotation(x, y, max_rotation_deg=15, rotation_deg=None):
    """
    Augment a target patch by rotating it. One of [max_rotation_deg, rotation_deg] must be given.
    If max_rotation_deg is given, an angle is sampled from [-max, max]
    If rotation_deg is is given, that angle is directly applied.
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (m, 2) tensor of the gt displacement
    :param max_rotation_deg: int, max rotation angle (+/-) in degrees
    :param rotation_deg: int, rotation angle in degrees to apply
    :return: x_aug, y_aug
    """

    if not isinstance(max_rotation_deg, type(None)):
        angle = random.randint(-max_rotation_deg, max_rotation_deg)
    else:
        angle = rotation_deg

    x_aug = rotate(x, angle, interpolation=InterpolationMode.NEAREST)
    phi = torch.tensor(-angle * pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    y_aug = torch.reshape((rot @ torch.reshape(y, (2, 1))), (2,))
    return x_aug, y_aug, angle


def unaugment_rotation(y, rotation_deg=None):
    """
    Augment a target patch by rotating it. One of [max_rotation_deg, rotation_deg] must be given.
    If max_rotation_deg is given, an angle is sampled from [-max, max]
    If rotation_deg is is given, that angle is directly applied.
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (m, 2) tensor of the gt displacement
    :param max_rotation_deg: int, max rotation angle (+/-) in degrees
    :param rotation_deg: int, rotation angle in degrees to apply
    :return: x_aug, y_aug
    """

    angle = -rotation_deg
    phi = torch.tensor(-angle * pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    y_aug = torch.reshape((rot @ torch.reshape(y, (2, 1))), (2,))
    return y_aug


def augment_scale(x, y, max_scale_percentage=10, scale_percentage=None):
    """
    Augment a target patch by scaling it. Scale percentage is uniformly sampled from [-MAX, MAX]
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (2,) tensor of the gt displacement
    :param max_scale_percentage: int, max scale change (+/-) in percentage
    :return: x_aug, y_aug
    """
    _, patch_size_old, _ = x.shape
    if not isinstance(max_scale_percentage, type(None)):
        scaling = (
            1.0
            + float(random.randint(-max_scale_percentage, max_scale_percentage)) / 100.0
        )
        patch_size_new = int(round(patch_size_old * scaling))

        # Enforce odd patch size
        if patch_size_new % 2 == 0:
            patch_size_new += 1

        scaling = patch_size_new / patch_size_old
    else:
        scaling = scale_percentage
        patch_size_new = int(patch_size_old * scaling)

    x_aug = resize(x, [patch_size_new], interpolation=InterpolationMode.NEAREST)

    if scaling < 1.0:
        # Pad with zeros
        padding = patch_size_old // 2 - patch_size_new // 2
        x_aug = F.pad(x_aug, (padding, padding, padding, padding))

    elif scaling > 1.0:
        # Center crop
        x_aug = x_aug[
            :,
            patch_size_new // 2
            - patch_size_old // 2 : patch_size_new // 2
            + patch_size_old // 2
            + 1,
            patch_size_new // 2
            - patch_size_old // 2 : patch_size_new // 2
            + patch_size_old // 2
            + 1,
        ]
    y_aug = y * scaling

    return x_aug, y_aug, scaling


def unaugment_scale(y, scale_percentage):
    """
    Augment a target patch by scaling it. Scale percentage is uniformly sampled from [-MAX, MAX]
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (2,) tensor of the gt displacement
    :param max_scale_percentage: int, max scale change (+/-) in percentage
    :return: x_aug, y_aug
    """
    scaling = 1.0 / scale_percentage
    y_aug = y * scaling
    return y_aug


def augment_track(track_data, flipped_lr, flipped_ud, rotation_angle, image_size):
    """
    Augment tracks by flipped LR, UP, then rotating
    :param track_data: Nx2 array of feature locations over time with time increasing in row dimension
    :param flipped_lr: bool
    :param flipped_ud: bool
    :param rotation_angle: numeric
    :param image_size: (W, H)
    :return: augmented_track_data: Nx2 array of augmented feature locs
    """
    image_center = ((image_size[0] - 1.0) / 2.0, (image_size[1] - 1.0) / 2.0)

    # Offset the track data wrt center of image
    track_data_aug = np.copy(track_data)
    track_data_aug[:, 0] -= image_center[0]
    track_data_aug[:, 1] -= image_center[1]

    # Apply augs
    if flipped_lr:
        track_data_aug[:, 0] *= -1
    if flipped_ud:
        track_data_aug[:, 1] *= -1
    if rotation_angle > 0:
        pass

    # Restore coordinate frame
    track_data_aug[:, 0] += image_center[0]
    track_data_aug[:, 1] += image_center[1]
    return track_data_aug


def augment_input(input, flipped_lr, flipped_ud, rotation_angle):
    """
    :param input: array-like of shape (H, W), or (H, W, C)
    :param flipped_lr:
    :param flipped_ud:
    :param rotation_angle:
    :return:
    """

    if flipped_lr:
        input = np.fliplr(input)
    if flipped_ud:
        input = np.flipud(input)
    if rotation_angle > 0:
        pass

    return input


def augment_scale_2(x, xx, y, max_scale_percentage=10, scale_percentage=None):
    """
    Augment a target patch by scaling it. Scale percentage is uniformly sampled from [-MAX, MAX]
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (2,) tensor of the gt displacement
    :param max_scale_percentage: int, max scale change (+/-) in percentage
    :return: x_aug, y_aug
    """
    _, patch_size_old, _ = x.shape
    _, patch_size_old_2, _ = xx.shape
    if not isinstance(max_scale_percentage, type(None)):
        scaling = (
            1.0
            + float(random.randint(-max_scale_percentage, max_scale_percentage)) / 100.0
        )
        patch_size_new = int(round(patch_size_old * scaling))

        # Enforce odd patch size
        if patch_size_new % 2 == 0:
            patch_size_new += 1
        patch_size_new_2 = patch_size_new * 2

        scaling = patch_size_new / patch_size_old
    else:
        scaling = scale_percentage
        patch_size_new = int(patch_size_old * scaling)

    x_aug = resize(x, [patch_size_new], interpolation=InterpolationMode.NEAREST)
    xx_aug = resize(xx, [patch_size_new_2], interpolation=InterpolationMode.NEAREST)

    if scaling < 1.0:
        # Pad with zeros
        padding = patch_size_old // 2 - patch_size_new // 2
        x_aug = F.pad(x_aug, (padding, padding, padding, padding))
        padding_2 = patch_size_old_2 // 2 - patch_size_new_2 // 2
        xx_aug = F.pad(xx_aug, (padding_2, padding_2, padding_2, padding_2))

    elif scaling > 1.0:
        # Center crop
        x_aug = x_aug[
            :,
            patch_size_new // 2
            - patch_size_old // 2 : patch_size_new // 2
            + patch_size_old // 2
            + 1,
            patch_size_new // 2
            - patch_size_old // 2 : patch_size_new // 2
            + patch_size_old // 2
            + 1,
        ]
        xx_aug = xx_aug[
            :,
            patch_size_new_2 // 2
            - patch_size_old_2 // 2: patch_size_new_2 // 2
            + patch_size_old_2 // 2,
            patch_size_new_2 // 2
            - patch_size_old_2 // 2: patch_size_new_2 // 2
            + patch_size_old_2 // 2,
        ]
    y_aug = y * scaling

    return x_aug, xx_aug, y_aug, scaling


def augment_rotation_2(x, xx, y, max_rotation_deg=15, rotation_deg=None):
    """
    Augment a target patch by rotating it. One of [max_rotation_deg, rotation_deg] must be given.
    If max_rotation_deg is given, an angle is sampled from [-max, max]
    If rotation_deg is is given, that angle is directly applied.
    :param x: (C, P, P) tensor of the event representation (patch)
    :param y: (m, 2) tensor of the gt displacement
    :param max_rotation_deg: int, max rotation angle (+/-) in degrees
    :param rotation_deg: int, rotation angle in degrees to apply
    :return: x_aug, y_aug
    """

    if not isinstance(max_rotation_deg, type(None)):
        angle = random.randint(-max_rotation_deg, max_rotation_deg)
    else:
        angle = rotation_deg

    x_aug = rotate(x, angle, interpolation=InterpolationMode.NEAREST)
    xx_aug = rotate(xx, angle, interpolation=InterpolationMode.NEAREST)
    phi = torch.tensor(-angle * pi / 180)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    y_aug = torch.reshape((rot @ torch.reshape(y, (2, 1))), (2,))
    return x_aug, xx_aug, y_aug, angle