import math
import ast

import numpy as np
from PIL import Image

from .utils import convert2rgb


def build_overlay(patches, size, patch_ids, slide_dim, overlay_rgb, background='black'):
    """
    (c) modified from https://github.com/hense96/patho-preprocessing
    Args:
        patches: the dataframe containing the metadata of the patches of this slide
        size: the desired size of the overlay
        patch_ids: the patch IDs for which we want to build the overlay and have the overlay RGB values
        slide_dim: slide.dimensions if slide is the openslide object
        overlay_rgb: [n-patch x 3] The RGB values of the patches in patch_ids
        background: background color for the overlay ['black', 'white']

    Returns: The PIL image of the overlay

    """
    if background == 'black':
        overlay_image = np.zeros((size[0], size[1], 3))
    elif background == 'white':
        overlay_image = np.ones((size[0], size[1], 3))
    else:
        raise ValueError(f"Unsupported background color for overlay: {background}")
    for i, id_ in enumerate(patch_ids):
        this_patch = patches[patches['patch_id'] == id_]
        x_coord, y_coord = ast.literal_eval(this_patch['position_abs'].item())
        patch_size = this_patch['patch_size_abs'].item()
        ds_x_coord = int(x_coord * (size[0] / slide_dim[0]))
        ds_y_coord = int(y_coord * (size[1] / slide_dim[1]))
        ds_patch_size_x = int(math.ceil(patch_size * (size[0] / slide_dim[0])))
        ds_patch_size_y = int(math.ceil(patch_size * (size[1] / slide_dim[1])))
        overlay_image[ds_x_coord:(ds_x_coord + ds_patch_size_x), ds_y_coord:(ds_y_coord + ds_patch_size_y), :] = \
            overlay_rgb[i, :]
    return Image.fromarray(np.uint8(np.transpose(overlay_image, (1, 0, 2)) * 255))


def heatmap_PIL(patches, size, patch_ids, slide_dim, score_values, cmap_name='coolwarm', background='black',
                zero_centered=True):
    """
    builds the PIL image of the attention values.
    Args:
        patches: the dataframe containing the metadata of the patches of this slide
        size: the desired size of the heatmap
        patch_ids: the patch IDs for which we want to build the overlay and have the overlay RGB values
        slide_dim: slide.dimensions if slide is the openslide object
        score_values: The attention values to be converted to a PIL image
        cmap_name: colormap
        background: background color for the overlay ['black', 'white']
        zero_centered: if True, the heatmap colors will be centered at score 0.0

    Returns: The PIL image of the attention image and the RGB values corresponding to the attention values

    """
    scores_rgb = convert2rgb(score_values, cmap_name=cmap_name, zero_centered=zero_centered)
    img = build_overlay(patches, size, patch_ids, slide_dim, scores_rgb, background)
    return img, scores_rgb
