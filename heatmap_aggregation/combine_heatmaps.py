import argparse
import ast
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm

from visualization.slideshow import heatmap_PIL
from visualization.utils import clean_outliers_fliers


def get_results_dirs(preds_dir: str) -> List[str]:
    """
    Construct list of directories with results for different seeds, i.e., location to retrieve explanation data.
    """
    results_dirs = sorted([
        str(res_dir) for res_dir in Path(preds_dir).iterdir()
        if res_dir.is_dir() and (res_dir / "args.json").is_file() and (res_dir / "test_predictions.csv").is_file()
    ])
    return results_dirs


def get_explanations_slide(
        results_dirs: List[str],
        slide_id: str,
        target: str = None
) -> (np.ndarray, np.ndarray, np.ndarray, int, List[float]):
    """
    Retrieve explanation scores for a given slide from multiple seeds.
    """
    pred_scores = []
    patch_scores_all = []
    labels = []
    list_patch_ids = []

    for results_dir in results_dirs:
        res_df = pd.read_csv(os.path.join(results_dir, 'test_predictions.csv'), index_col=0)
        sel_idx = res_df.loc[res_df['slide_id'] == slide_id].index[0]

        # Extract prediction information
        if target is None:
            pred_cols = [col for col in res_df.columns if col.startswith('prediction_score')]
            if len(pred_cols) == 1:
                pred_col = pred_cols[0]
            else:
                raise ValueError(f"Could not determine 'prediction_score' column. Please provide a 'target' argument.")
            label_cols = [col for col in res_df.columns if col.startswith('label')]
            if len(pred_cols) == 1:
                label_col = label_cols[0]
            else:
                raise ValueError(f"Could not determine 'label' column. Please provide a 'target' argument.")
        else:
            pred_col, label_col = f"prediction_score_{target}", f"label_{target}"
        pred_score = res_df.loc[sel_idx, pred_col]
        pred_scores.append(pred_score)
        label = res_df.loc[sel_idx, label_col]
        labels.append(label)

        # Read explanation scores
        patch_ids = np.asarray(ast.literal_eval(res_df.loc[sel_idx, 'patch_ids']))
        patch_ids = patch_ids[0] if len(patch_ids.shape) > 1 else patch_ids
        list_patch_ids.append(patch_ids)
        patch_scores = np.asarray(ast.literal_eval(res_df.loc[sel_idx, 'patch_scores_lrp']))
        patch_scores = patch_scores[0] if len(patch_scores.shape) > 1 else patch_scores

        patch_scores_all.append(patch_scores)

    # Slide label
    label = np.unique(np.array(labels))
    if len(label) > 1:
        raise ValueError("Multiple labels detected in slide")
    label = label[0]
    # Patch IDs of the slide
    first_patch_ids = list_patch_ids[0]
    if not all([np.all(patch_ids == first_patch_ids) for patch_ids in list_patch_ids]):
        raise ValueError("Patch IDs do not match across seeds")
    patch_ids = first_patch_ids
    # Create array of patch scores
    patch_scores_array = np.array(patch_scores_all)
    return patch_scores_all, patch_scores_array, patch_ids, label, pred_scores


def get_mean_and_std_scores(patch_scores_array: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute the mean of the signed patch scores, the mean patch scores and the standard deviation of the patch scores.
    """
    mean_sign = np.sign(patch_scores_array).mean(axis=0)
    mean_patch_scores = patch_scores_array.mean(axis=0)
    std_patch_scores = patch_scores_array.std(axis=0)
    return mean_sign, mean_patch_scores, std_patch_scores


def get_patches_with_high_std(std_scores, quantile=0.995):
    quantile_std = np.quantile(std_scores, quantile)
    if quantile_std > 0:
        keep_patches = np.where(std_scores <= quantile_std)[0]
        filter_patches = np.where(std_scores > quantile_std)[0]
    else:
        keep_patches = np.arange(len(std_scores))
        filter_patches = np.array([])
    return keep_patches, filter_patches, quantile_std


def get_lrp_means_n_mask(
        mean_sign: np.ndarray,
        mean_scores: np.ndarray,
        std_scores: np.ndarray,
        filter_patches: np.ndarray,
        verbose: bool
) -> pd.DataFrame:
    """
    Create a DataFrame with the mean and standard deviation of the patch scores and the sign of the mean scores.
    Compute a mask to filter out patches with a sign flip over seeds or with a high standard deviation.
    """
    lrp_means = pd.DataFrame({
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'sign(mean)': np.sign(mean_scores),
        'mean(sign)': mean_sign,
    })
    lrp_means['mask'] = lrp_means.apply(lambda x: x['sign(mean)'] * x['mean(sign)'] > 0,
                                        axis=1)  # Filter based on sign flip
    if len(filter_patches) > 0:
        lrp_means.loc[lrp_means.iloc[filter_patches].index, 'mask'] = False
    if verbose:
        print(f"{lrp_means['mask'].value_counts()=}")
    return lrp_means


def store_outlier_fliers_vs_masking(
        mean_scores: np.ndarray,
        patches_mask: np.ndarray,
        storing_path: str
) -> np.ndarray:
    """
    Get outliers of mean scores based on distribution details. Store cross table of outliers vs masking.
    Return the cleaned mean scores.
    """
    cleaned_data, _, outliers = clean_outliers_fliers(mean_scores, return_idxs=True)
    res = pd.DataFrame({'outliers': outliers, 'patches_mask': patches_mask})
    cross_table = pd.crosstab(index=res['outliers'], columns=res['patches_mask'])
    cross_table.to_csv(os.path.join(storing_path, 'cross_table_outliers_vs_masking.csv'))
    return cleaned_data


def load_slide(slides_dir: str, slide_id: str) -> openslide.OpenSlide:
    if slides_dir is None:
        return None
    slide_path = [
        os.path.join(slides_dir, slide)
        for slide in os.listdir(slides_dir) if slide.startswith(slide_id)
    ]
    if len(slide_path) > 1:
        raise ValueError(f"Could not identify slide: {slide_id} (found: {slide_path}).")
    elif len(slide_path) == 0:
        return None
    slide_path = slide_path[0]
    if os.path.isdir(slide_path):
        slide_file = [
            s_file for s_file in os.listdir(slide_path)
            if os.path.splitext(os.path.join(slide_path, s_file))[1] in [".dcm"]
        ][0]
        slide_path = os.path.join(slide_path, slide_file)
    slide = openslide.open_slide(slide_path) if os.path.exists(slide_path) else None
    return slide


def load_patch_metadata(patches_dir, slide_id):
    patches_in_slide = pd.read_csv(os.path.join(patches_dir, slide_id, 'metadata/df.csv'), index_col=0)
    return patches_in_slide


def mask_patch_information(
        mask: np.ndarray,
        patches: pd.DataFrame,
        patch_ids: np.ndarray,
        patch_scores: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Filter patches based on the mask."""
    masked_patch_ids = patch_ids[mask]
    masked_patch_scores = patch_scores[mask]
    masked_patches = patches[patches.patch_id.isin(masked_patch_ids)].copy()
    return masked_patches, masked_patch_ids, masked_patch_scores


def plot_overlay_slide(
        slide_id: str,
        slide: openslide.OpenSlide,
        patches: pd.DataFrame,
        patch_ids: np.ndarray,
        patch_scores: np.ndarray,
        patches_mask: np.ndarray,
        storing_path: Union[str, Path],
        verbose: bool
):
    """
    Plot overlay for QuPath and storing the image.
    """
    cmap_name = 'coolwarm'
    zero_centered = True

    overlay_dims = (slide.dimensions[0] // 32, slide.dimensions[1] // 32)

    for mask in [True, False]:
        curr_patches, curr_patch_ids, curr_patch_scores = (
            mask_patch_information(patches_mask, patches, patch_ids, patch_scores)
            if mask
            else (patches, patch_ids, patch_scores)
        )

        # overlay for QuPath
        heatmap, _ = heatmap_PIL(
            patches=curr_patches,
            size=overlay_dims,
            patch_ids=curr_patch_ids,
            slide_dim=slide.dimensions,
            score_values=curr_patch_scores,
            cmap_name=cmap_name,
            zero_centered=zero_centered
        )
        fn = f"{slide_id}"
        if mask:
            fn += '_masked'
        heatmap.save(os.path.join(storing_path, f"{fn}.png"), "PNG")
        if verbose:
            print(f"Stored {fn}.png")


def _as_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, list):
        return x
    else:
        return [x]


def main(args):

    # Get data directories
    results_dirs = get_results_dirs(args.preds_dir)

    if args.slide_ids is not None:
        slide_ids = args.slide_ids
    elif args.metadata_dir is not None:
        slide_ids = pd.read_csv(Path(args.metadata_dir) / "slide_metadata.csv")["slide_id"].tolist()
    else:
        raise ValueError(f"Either slide_ids or metadata_dir must be provided.")

    # Process each slide
    for slide_id in tqdm(slide_ids):
        if args.verbose:
            print(f"Processing slide {slide_id}...")

        # Prepare output directories
        save_dir_preds = os.path.join(args.target_dir, "predictions", slide_id)
        save_dir_overlays = os.path.join(args.target_dir, "overlays")
        os.makedirs(save_dir_preds, exist_ok=True)
        os.makedirs(save_dir_overlays, exist_ok=True)

        # Get explanations for the slide
        (_, patch_scores_array, patch_ids, _, pred_scores) = get_explanations_slide(
            results_dirs, slide_id, args.target
        )
        # Get mean of the signed explanations and mean and std of patch the original explanations
        mean_sign, mean_patch_scores, std_patch_scores = get_mean_and_std_scores(patch_scores_array)

        # Get patches with high std
        keep_patches, filter_patches, quant_99 = get_patches_with_high_std(std_patch_scores, quantile=args.std_quantile)

        # Mask the patches with high std or where sign(mean lrp) * mean(sign lrp) is <= 0 (i.e., patches with sign flip over seeds)
        lrp_means = get_lrp_means_n_mask(mean_sign, mean_patch_scores, std_patch_scores, filter_patches, args.verbose)

        # Compute outlier fliers on mean scores and store the cross table with masking
        lrp_means['cleaned_mean_scores'] = store_outlier_fliers_vs_masking(
            lrp_means['mean_scores'].values, lrp_means['mask'].values, save_dir_preds
        )
        # Also store original lrp values to lrp_means for further analysis
        for i in range(patch_scores_array.shape[0]):
            lrp_means[f'lrp_{i}'] = patch_scores_array[i,:]

        # Store the explanation scores and the mask
        lrp_means.to_csv(os.path.join(save_dir_preds, 'lrp_means.csv'))
        pd.Series(pred_scores).to_csv(os.path.join(save_dir_preds, 'pred_scores.csv'))

        # Plotting the aggregated slide heatmaps (mean explanation scores) with and without masking
        slide = load_slide(args.slides_dir, slide_id)
        if slide is None:
            continue

        patches_metadata = load_patch_metadata(args.patches_dir, slide_id)

        plot_overlay_slide(
            slide_id=slide_id,
            slide=slide,
            patches=patches_metadata,
            patch_ids=patch_ids,
            patch_scores=lrp_means['cleaned_mean_scores'].values,
            patches_mask=lrp_means['mask'].values,
            storing_path=save_dir_overlays,
            verbose=args.verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--patches_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, default=None)
    parser.add_argument("--slide_ids", nargs='+', type=str, default=None)
    parser.add_argument("--slides_dir", type=str, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--std_quantile", type=float, default=0.995)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Combine heatmaps for {args.preds_dir} and writing to {args.target_dir}...")
    main(args)
    print("Done!")
