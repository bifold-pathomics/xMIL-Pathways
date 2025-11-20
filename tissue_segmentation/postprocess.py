import os
import json
import argparse
import ast
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import openslide
import PIL
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from scipy.spatial.distance import cdist

from visualization.utils import heatmap_PIL, overlay

PIL.Image.MAX_IMAGE_PIXELS = 196455024 * 10


def load_slide_data(slides_dir, annot_dir, patches_dir, slide_names, openslide_files=False, verbose=True):
    slides, annots, metadata = [], [], []
    slide_names_iter = tqdm(slide_names) if verbose else slide_names
    for slide_name in slide_names_iter:
        if openslide_files:
            if os.path.isdir(os.path.join(slides_dir, slide_name)):
                slide_file = [
                    s_file for s_file in os.listdir(os.path.join(slides_dir, slide_name)) if s_file.endswith(".dcm")
                ][0]
                slides.append(openslide.open_slide(os.path.join(slides_dir, slide_name, slide_file)))
            else:
                slides.append(openslide.open_slide(os.path.join(slides_dir, f'{slide_name}.svs')))
            if annot_dir is not None and os.path.exists(os.path.join(annot_dir, f'{slide_name}.png')):
                annots.append(openslide.open_slide(os.path.join(annot_dir, f'{slide_name}.png')))
            else:
                annots.append(None)
        metadata_ = pd.read_csv(os.path.join(patches_dir, slide_name, 'metadata/df.csv'), index_col=0)
        if annot_dir is not None and os.path.exists(os.path.join(annot_dir, f'{slide_name}.png')):
            metadata_ = pd.concat(
                [metadata_, metadata_['annotation_classes'].apply(json.loads).apply(pd.Series)], axis=1)
        metadata_ = metadata_.sort_values('patch_id')
        metadata_.insert(0, 'slide_id', slide_name)
        metadata_.insert(1, 'feature_idx', range(len(metadata_)))
        metadata.append(metadata_)
    return slides, annots, metadata


def neighborhood_smoother(patch_preds, patch_metadata, life_thresholds=(3, 5), max_iterations=100, verbose=False):
    # Create data structure for smoothing
    patch_positions = patch_metadata['position_rel'].apply(ast.literal_eval).values
    all_x, all_y = list(zip(*patch_positions))
    min_x, max_x, min_y, max_y = min(all_x) - 1, max(all_x) + 2, min(all_y) - 1, max(all_y) + 2
    score_matrix = np.zeros((max_x - min_x, max_y - min_y))
    for patch_pred, (patch_x, patch_y) in zip(patch_preds, patch_positions):
        score_matrix[patch_x - min_x, patch_y - min_y] = patch_pred
    # Apply smoothing
    iterations, num_changes = 0, 1
    while num_changes > 0 and iterations < max_iterations:
        num_changes = 0
        updated_matrix = score_matrix.copy()
        for x_pos, y_pos in product(range(score_matrix.shape[0]), range(score_matrix.shape[1])):
            score = score_matrix[x_pos, y_pos]
            neighbor_score = score_matrix[x_pos-1:x_pos+2, y_pos-1:y_pos+2].sum()
            if neighbor_score <= life_thresholds[0]:
                updated_matrix[x_pos, y_pos] = 0
                num_changes += 1
            elif neighbor_score >= life_thresholds[1]:
                updated_matrix[x_pos, y_pos] = 1
                num_changes += 1
        score_matrix = updated_matrix
        if verbose:
            print(f"Number of changes: {num_changes}")
        iterations += 1
    # Translate results back into scores
    res_preds = patch_preds.copy()
    for idx, (patch_x, patch_y) in enumerate(patch_positions):
        res_preds[idx] = updated_matrix[patch_x - min_x, patch_y - min_y]
    return res_preds


def create_border_class(patch_preds, metadata_df, border_width: int = 1):
    """
    Create a third class for patches within border_width of tumor regions

    Args:
        patch_preds: Binary predictions (0=normal, 1=tumor)
        metadata_df: DataFrame with patch coordinates
        border_width: Distance in pixels to define border region

    Returns:
        numpy array with labels: 0=normal, 1=tumor, 2=border
    """
    # Convert to numpy array if needed
    preds = np.array(patch_preds)

    # Get coordinates of all patches
    coords = metadata_df['position_rel'].values
    coords = np.array([np.array(ast.literal_eval(x)) for x in coords])

    # Find tumor patch indices
    tumor_indices = np.where(preds == 1)[0]

    if len(tumor_indices) == 0:
        # No tumor patches, return original predictions
        return preds

    # Get coordinates of tumor patches
    tumor_coords = coords[tumor_indices]

    # Calculate distances from all patches to all tumor patches
    distances = cdist(coords, tumor_coords)

    # Find minimum distance to any tumor patch for each patch
    min_distances = np.min(distances, axis=1)

    # Create border class: patches within border_width but not tumor themselves
    border_mask = (min_distances <= border_width) & (preds == 0)

    # Create final labels
    labels = preds.copy()
    labels[border_mask] = 2

    return labels


def heatmap_plot(slide, patches, patch_ids, patch_scores_list, annotation=None, size=(2048, 2048), alpha=64,
                title=None):
    # Data generation
    assert len(patch_scores_list) == 2
    slide_thumbnail = slide.get_thumbnail(size)
    if annotation is not None:
        annot_thumbnail = overlay(slide_thumbnail, annotation.get_thumbnail(slide_thumbnail.size), alpha)
    else:
        annot_thumbnail = None
    heatmaps = []
    for patch_scores in patch_scores_list:
        heatmap, _ = heatmap_PIL(
            patches, slide_thumbnail.size, patch_ids, slide.dimensions, patch_scores,
            cmap_name='viridis', background='black', zero_centered=False
        )
        heatmaps.append(overlay(slide_thumbnail, heatmap, alpha))
    # Plotting
    num_subplots = 1 + len(patch_scores_list)
    num_subplots = num_subplots if annotation is None else num_subplots + 1
    fig, axs = plt.subplots(num_subplots, figsize=(num_subplots * 4, 12), sharex=True)
    axs[0].imshow(slide_thumbnail)
    ax_idx = 1
    if annot_thumbnail is not None:
        axs[ax_idx].imshow(annot_thumbnail)
        ax_idx += 1
    for idx, heatmap in enumerate(heatmaps):
        axs[ax_idx].imshow(heatmap)
        ax_idx += 1
    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def get_heatmap(slide, patches, patch_ids, patch_scores):
    overlay_dims = (slide.dimensions[0] // 32, slide.dimensions[1] // 32)
    heatmap_img, _ = heatmap_PIL(
        patches=patches, size=overlay_dims, patch_ids=patch_ids,
        slide_dim=slide.dimensions, score_values=patch_scores, cmap_name='viridis', background='black',
        zero_centered=False)
    return heatmap_img


def get_args():
    parser = argparse.ArgumentParser()

    # Loading and saving
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--patches-dir', type=str, required=True)
    parser.add_argument('--slides-dir', type=str, required=True)
    parser.add_argument('--annotations-dir', type=str, default=None)
    parser.add_argument('--results-dir', type=str, required=True)

    # Visualization params
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--prediction-threshold', type=float, default=0.5)
    parser.add_argument('--neighborhood-smoothing', action='store_true')
    parser.add_argument('--border-width', type=int, default=None)

    # Parse all args
    args = parser.parse_args()

    return args


def main():
    # Process and save input args
    args = get_args()
    print(json.dumps(vars(args), indent=4))
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=False)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    overlay_dir_base = os.path.join(save_dir, 'overlays_base')
    overlay_dir_smooth = os.path.join(save_dir, 'overlays_smooth')
    overlay_dir_border = os.path.join(save_dir, 'overlays_border')
    os.makedirs(overlay_dir_base)
    if args.neighborhood_smoothing:
        os.makedirs(overlay_dir_smooth)
    if args.border_width is not None:
        os.makedirs(overlay_dir_border)

    # Read prediction data
    test_preds = pd.read_csv(os.path.join(args.test_dir, "test_predictions.csv"))

    slide_score_df = pd.DataFrame()

    for slide_id in tqdm(test_preds['slide_id'].unique()):

        # Load slide data
        slide_preds = test_preds[test_preds['slide_id'] == slide_id]
        slide, annot, metadata_df = load_slide_data(
            args.slides_dir, args.annotations_dir, args.patches_dir, [slide_id], openslide_files=True, verbose=False
        )
        slide, annot, metadata_df = slide[0], annot[0], metadata_df[0]
        base_preds = (slide_preds[f'prediction_score{args.target}'] >= args.prediction_threshold).astype(int)

        # Neighborhood smoothing
        if args.neighborhood_smoothing:
            smooth_preds = neighborhood_smoother(
                patch_preds=slide_preds[f'prediction_score{args.target}'].to_numpy(),
                patch_metadata=metadata_df,
                life_thresholds=(3, 5),
                max_iterations=100,
            )
            test_preds.loc[
                test_preds['slide_id'] == slide_id, f'prediction_score{args.target}_smooth'
            ] = smooth_preds.tolist()
            smooth_preds = (smooth_preds >= args.prediction_threshold).astype(int)
        else:
            smooth_preds = None

        # Border prediction
        if args.border_width is not None:
            if smooth_preds is not None:
                border_preds = create_border_class(smooth_preds, metadata_df, args.border_width)
            else:
                border_preds = create_border_class(base_preds, metadata_df, args.border_width)
            test_preds.loc[
                test_preds['slide_id'] == slide_id, f'prediction_score{args.target}_border'
            ] = border_preds.tolist()
        else:
            border_preds = None

        # Compute metrics
        slide_labels = slide_preds[args.target].unique()
        if 0 in slide_labels and 1 in slide_labels and len(slide_labels) == 2:
            metrics = {
                'base_auroc': [roc_auc_score(slide_preds[args.target], base_preds)],
                'base_acc': [accuracy_score(slide_preds[args.target], base_preds)],
                'base_bacc': [balanced_accuracy_score(slide_preds[args.target], base_preds)],
            }
            if smooth_preds is not None:
                metrics.update({
                    'smooth_auroc': [roc_auc_score(slide_preds[args.target], smooth_preds)],
                    'smooth_acc': [accuracy_score(
                        slide_preds[args.target], (smooth_preds >= args.prediction_threshold).astype(int))],
                    'smooth_bacc': [balanced_accuracy_score(
                        slide_preds[args.target], (smooth_preds >= args.prediction_threshold).astype(int))],
                })
            slide_score_df = pd.concat([slide_score_df, pd.DataFrame({**{'slide_id': [slide_id]}, **metrics})])

        base_heatmap = get_heatmap(slide, metadata_df, metadata_df['patch_id'], base_preds)
        base_heatmap.save(os.path.join(overlay_dir_base, f"{slide_id}.png"), "PNG")

        if smooth_preds is not None:
            smooth_heatmap = get_heatmap(slide, metadata_df, metadata_df['patch_id'], smooth_preds)
            smooth_heatmap.save(os.path.join(overlay_dir_smooth, f"{slide_id}.png"), "PNG")

        if border_preds is not None:
            smooth_heatmap = get_heatmap(slide, metadata_df, metadata_df['patch_id'], border_preds)
            smooth_heatmap.save(os.path.join(overlay_dir_border, f"{slide_id}.png"), "PNG")

    slide_score_df.to_csv(os.path.join(save_dir, 'metrics.csv'))
    if args.neighborhood_smoothing:
        test_preds.to_csv(os.path.join(save_dir, 'test_predictions.csv'))


if __name__ == '__main__':
    main()
