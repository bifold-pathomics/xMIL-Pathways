"""
(C) This code belongs to https://github.com/bifold-pathomics/xMIL-Pathways
Please see the citation and copyright instructions in the above-mentioned repository.
"""

import warnings
import json
import bisect
import argparse
import os
import numpy as np
import pandas as pd
import openslide
import cv2
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from visualization_utils import heatmap_PIL, plot_PIL

# Suppress all warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient-id', type=str, required=True)
    parser.add_argument('--marker', type=str, required=True, help="The name of the IHC marker")
    parser.add_argument('--tiles-dirs', type=str, nargs='+', required=True,
                        help='The directory of the extracted patches from the HE slide.')
    parser.add_argument('--registration-dir', type=str, required=True, help="The directory of registered slides")
    parser.add_argument('--qupath-measurement-dir', type=str, required=True,
                        help="directory of the QuPath measurements.")
    parser.add_argument('--predictions-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--activation-marker', type=str, required=True)

    parser.add_argument('--overwrite-results', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(json.dumps(vars(args), indent=4))

    if args.marker.lower() == 'cd34':
        pathway = 'VEGF'
    elif args.marker.lower() == 'pstat3':
        pathway = 'JAK-STAT'
    elif args.marker.lower() == 'tnfa':
        pathway = 'TNFa'
    print('*************patient ID=', args.patient_id)

    # define directories

    cell_act_dir = os.path.join(args.qupath_measurement_dir, args.marker)
    aggregated_heatmaps_dirs = [os.path.join(p, pathway) for p in args.predictions_dirs]
    reg_patient_dir = os.path.join(args.registration_dir, args.patient_id)
    reg_mask_path = str(list(Path(reg_patient_dir).rglob('**/*_non_rigid_mask.png'))[0])
    reg_slides_dir = os.path.join(reg_patient_dir, 'registered_slides')

    results_dir = os.path.join(args.results_dir, args.patient_id)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, args.marker)
    os.makedirs(results_dir, exist_ok=args.overwrite_results)
    with open(os.path.join(results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # file names

    cell_act_path = [os.path.join(cell_act_dir, s) for s in os.listdir(cell_act_dir)
                     if args.patient_id in s and (args.marker.upper() in s or args.marker in s)][0]  # the qupath file

    print('QuPath measurement file name = ', cell_act_path)
    HE_slide_name = [f for f in os.listdir(reg_slides_dir) if 'HE' in f][0]
    HE_slide_name = HE_slide_name[:HE_slide_name.index('.ome.tiff')]
    print('HE slide name =', HE_slide_name)

    for aggregated_heatmaps_dir in aggregated_heatmaps_dirs:
        folder = [p for p in os.listdir(aggregated_heatmaps_dir) if HE_slide_name in p]
        if len(folder) == 0:
            folder = [p for p in os.listdir(aggregated_heatmaps_dir) if args.patient_id in p and 'HE' in p]

        if len(folder) > 0:
            heatmap_path = os.path.join(aggregated_heatmaps_dir, folder[0])
            break

    print('heatmap path = ', heatmap_path)

    for tiles_dir in args.tiles_dirs:
        folder = [p for p in os.listdir(tiles_dir) if HE_slide_name in p]
        if len(folder) == 0:
            folder = [p for p in os.listdir(tiles_dir) if args.patient_id in p and 'HE' in p]

        if len(folder) > 0:
            tiles_metadata_path = os.path.join(tiles_dir, folder[0], 'metadata')
            break
    print('tiles metadata path = ', tiles_metadata_path)

    # read metadata and slides

    cell_act_info = pd.read_csv(cell_act_path)
    slide_name = cell_act_info['Image'].iloc[0]
    print('cell activation calculated for: ', slide_name)

    cell_act_df = cell_act_info[['Centroid X µm', 'Centroid Y µm', args.activation_marker, 'Classification']]

    slide_IHC = openslide.open_slide(os.path.join(reg_slides_dir, slide_name))
    slide_HE = openslide.open_slide(os.path.join(reg_slides_dir, HE_slide_name + '.ome.tiff'))
    assert slide_IHC.dimensions == slide_HE.dimensions

    df_lrp_means = pd.read_csv(os.path.join(heatmap_path, 'lrp_means.csv'))

    tiles_metadata_orig = pd.read_csv(os.path.join(tiles_metadata_path, 'df.csv'), index_col=0)
    print('number of tiles =', len(tiles_metadata_orig))

    tiles_metadata_orig['x_coords'] = tiles_metadata_orig['position_abs'].apply(
        lambda x: json.loads(x.replace('(', '[').replace(')', ']'))[0])
    tiles_metadata_orig['y_coords'] = tiles_metadata_orig['position_abs'].apply(
        lambda x: json.loads(x.replace('(', '[').replace(')', ']'))[1])

    # find the tissue mask -- this step can be skipped if the preprocessing pipeline has a tissue detection step
    # we do the following to exclude the tiles which are not in the main tissue

    image = cv2.imread(reg_mask_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # green in HSV
    lower_green = np.array([50, 100, 100])  # Lower bound of green
    upper_green = np.array([70, 255, 255])  # Upper bound of green

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    plt.imshow(mask)
    plt.savefig(os.path.join(results_dir, 'tissue_rectangle.png'), format='png', dpi=300)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # bounding box of the rectangle
    x, y, w, h = cv2.boundingRect(contour)
    print(f"Thumbnail Rectangle: Top-left ({x}, {y}), Width {w}, Height {h}")

    wsi_width, wsi_height = slide_HE.dimensions

    thumbnail_width, thumbnail_height = image.shape[1], image.shape[0]

    # scaling factors wsi->thumbnail
    scale_x = wsi_width / thumbnail_width
    scale_y = wsi_height / thumbnail_height

    # thumbnail coordinates --> WSI coordinates
    x_wsi = int(x * scale_x)
    y_wsi = int(y * scale_y)
    w_wsi = int(w * scale_x)
    h_wsi = int(h * scale_y)

    print(f"WSI Rectangle: Top-left ({x_wsi}, {y_wsi}), Width {w_wsi}, Height {h_wsi}")

    # Rectangle boundaries on the WSI
    rect_x0, rect_y0 = x_wsi, y_wsi
    rect_x1 = rect_x0 + w_wsi
    rect_y1 = rect_y0 + h_wsi

    tile_size = 680
    tiles_metadata = pd.DataFrame()
    ind_drop = []
    for i_tile, tile in tiles_metadata_orig.iterrows():
        tile_x0, tile_y0 = json.loads(tile['position_abs'].replace('(', '[').replace(')', ']'))
        tile_x1 = tile_x0 + tile_size
        tile_y1 = tile_y0 + tile_size

        if not (tile_x0 >= rect_x0 and tile_y0 >= rect_y0 and
                tile_x1 <= rect_x1 and tile_y1 <= rect_y1):
            ind_drop.append(i_tile)

    tiles_metadata = tiles_metadata_orig.drop(index=ind_drop)

    print('number of tiles =', len(tiles_metadata))

    mpp_x = tiles_metadata.iloc[0]['slide_mpp']
    tile_size = tiles_metadata.iloc[0]['patch_size_abs']
    print('mpp_x=', mpp_x)
    print('tile_size=', tile_size)

    cell_act_df['x-coord'] = cell_act_df['Centroid X µm'].apply(lambda x: x / mpp_x)
    cell_act_df['y-coord'] = cell_act_df['Centroid Y µm'].apply(lambda x: x / mpp_x)

    df_tiles_cells_lrps = tiles_metadata[['patch_id', 'position_abs']]
    df_tiles_cells_lrps['mean_lrp'] = df_lrp_means['mean_scores']
    df_tiles_cells_lrps['mask'] = df_lrp_means['mask']

    x_coords_all_patches = tiles_metadata['x_coords'].tolist()
    y_coords_all_patches = tiles_metadata['y_coords'].tolist()

    # match the tiles and the activations from qupath

    tiles2cell_cont = defaultdict(list)
    tiles2cell_classification = defaultdict(list)

    for i_cell, cell in cell_act_df.iterrows():

        ind2x = bisect.bisect_left(x_coords_all_patches, cell['x-coord']) - 1
        ind1x = bisect.bisect_left(x_coords_all_patches, x_coords_all_patches[ind2x])

        if x_coords_all_patches[ind1x] == x_coords_all_patches[ind2x] and \
                x_coords_all_patches[ind1x] <= cell['x-coord'] < x_coords_all_patches[ind1x] + tile_size:

            y_coords_in1x_ind2x = y_coords_all_patches[ind1x:ind2x + 1]
            indy = bisect.bisect_left(y_coords_in1x_ind2x, cell['y-coord']) - 1 + ind1x

            if y_coords_all_patches[indy] <= cell['y-coord'] < y_coords_all_patches[indy] + tile_size:
                tiles2cell_cont[indy].append(cell[args.activation_marker])
                tiles2cell_classification[indy].append(cell['Classification'])

    print('tile2cell_classification: ', len(tiles2cell_classification.keys()))

    # the keys are the patch row no in the tiles_metadata, and the values are the sum of activations in the very patch
    teil2cell_sum = {i: 0 for i in range(len(tiles_metadata))}
    teil2cell_sum_binary = {i: 0 for i in range(len(tiles_metadata))}

    for teil_id, vals in tiles2cell_cont.items():
        teil2cell_sum[teil_id] = np.sum(vals)

    for teil_id, vals in tiles2cell_classification.items():
        vals_ = [t == 'Positive' for t in vals]
        teil2cell_sum_binary[teil_id] = np.sum(vals_)

    patch_scores = np.array(list(teil2cell_sum.values()))
    patch_scores_binary = np.array(list(teil2cell_sum_binary.values()))

    df_tiles_cells_lrps['sum_cell_act_cont'] = patch_scores
    df_tiles_cells_lrps['sum_cell_act_binary'] = patch_scores_binary

    df_tiles_cells_lrps_masked = df_tiles_cells_lrps[df_tiles_cells_lrps['mask'] == True]
    # df_tiles_cells_lrps_masked = df_tiles_cells_lrps_masked[df_tiles_cells_lrps_masked['inclusion'] == True]

    lrp_values = np.array(df_tiles_cells_lrps_masked['mean_lrp'].tolist())
    cell_act_values = np.array(df_tiles_cells_lrps_masked['sum_cell_act_cont'].tolist())
    cell_act_values_binary = np.array(df_tiles_cells_lrps_masked['sum_cell_act_binary'].tolist())

    ind_pos = np.where(lrp_values > 0)
    ind_neg = np.where(lrp_values < 0)

    cell_act_pos_binary = cell_act_values_binary[ind_pos]
    cell_act_neg_binary = cell_act_values_binary[ind_neg]

    plt.figure()
    plt.subplot(121)
    sns.boxplot([cell_act_neg_binary, cell_act_pos_binary])
    plt.title('binary')

    cell_act_pos = cell_act_values[ind_pos]
    cell_act_neg = cell_act_values[ind_neg]

    plt.subplot(122)
    sns.boxplot([cell_act_neg, cell_act_pos])
    plt.title('continuous')

    plt.savefig(os.path.join(results_dir, 'boxplots.png'), format='png', dpi=300)

    df_tiles_cells_lrps.to_csv(os.path.join(results_dir, 'df_tiles_cells_lrps.csv'))

    overlay_dims = (slide_HE.dimensions[0] // 32, slide_HE.dimensions[1] // 32)
    patch_ids = np.array(tiles_metadata['patch_id'].tolist())
    cmap_name = 'coolwarm'

    heatmap, _ = heatmap_PIL(
        patches=tiles_metadata, size=overlay_dims, patch_ids=patch_ids, slide_dim=slide_HE.dimensions,
        score_values=patch_scores_binary, cmap_name=cmap_name, zero_centered=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_PIL(ax, heatmap, cmap=cmap_name)
    plt.title('sum of binary')
    plt.savefig(os.path.join(results_dir, 'cell_activations_binary.png'), format='png', dpi=300)
    # -----------
    heatmap, _ = heatmap_PIL(
        patches=tiles_metadata, size=overlay_dims, patch_ids=patch_ids, slide_dim=slide_HE.dimensions,
        score_values=patch_scores, cmap_name=cmap_name, zero_centered=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_PIL(ax, heatmap, cmap=cmap_name)
    plt.title('sum of continuous')

    plt.savefig(os.path.join(results_dir, 'cell_activations_continuous.png'), format='png', dpi=300)

    df_tiles_cells_lrps['mean_lrp_masked'] = df_tiles_cells_lrps['mean_lrp']
    df_tiles_cells_lrps.loc[~df_tiles_cells_lrps['mask'], 'mean_lrp_masked'] = 0

    masked_mean_lrp = df_tiles_cells_lrps['mean_lrp_masked'].tolist()

    heatmap, _ = heatmap_PIL(
        patches=tiles_metadata, size=overlay_dims, patch_ids=patch_ids, slide_dim=slide_HE.dimensions,
        score_values=masked_mean_lrp, cmap_name=cmap_name, zero_centered=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_PIL(ax, heatmap, cmap=cmap_name)
    plt.title('heatmap in the rectangle')

    plt.savefig(os.path.join(results_dir, 'heatmap.png'), format='png', dpi=300)


if __name__ == '__main__':
    main()
