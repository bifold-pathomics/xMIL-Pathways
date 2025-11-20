from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cbook import boxplot_stats


def clean_outliers_fliers(data, return_idxs=False):
    data_clean = deepcopy(data)
    stat = boxplot_stats(data)[0]
    data_clean[data < stat['whislo']] = stat['whislo']
    data_clean[data > stat['whishi']] = stat['whishi']
    filter_idxs = np.logical_and(data >= stat['whislo'], data <= stat['whishi'])
    data_no_outlier = data[filter_idxs]
    if return_idxs:
        return data_clean, data_no_outlier, filter_idxs
    return data_clean, data_no_outlier


def convert2rgb(mat, cmap_name='coolwarm', zero_centered=True):
    """
    converts the given matrix to RGB values
    """
    cmap = plt.get_cmap(cmap_name)
    if zero_centered:
        max_scalar = np.max(np.abs(mat))
        rgb_values = (max_scalar + mat) / (2 * max_scalar)
    else:
        min_scalar = np.min(mat)
        max_scalar = np.max(mat)
        rgb_values = (mat - min_scalar) / (max_scalar - min_scalar)
    return cmap(rgb_values).squeeze()[:, :-1]
