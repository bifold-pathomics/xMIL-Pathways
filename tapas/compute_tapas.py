import os
import ast
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datasets.data_handler import MetadataHandler


ALL_PATHWAYS = [
    "VEGF", "TNFa", "JAK-STAT",
    "EGFR", "Estrogen", "MAPK", "Trail", "WNT", "Hypoxia", "p53", "PI3K", "Androgen", "NFkB", "TGFb"
]


def load_slide_ids(split_path, metadata_dirs, subsets):
    split_metadata = MetadataHandler.load_split_metadata(
        split_path, metadata_dirs, subsets, label_cols=[], modalities=['slide']
    )
    return split_metadata["slide_id"].sort_values().unique().tolist()


def load_predictions(
        heatmaps_dir, slide, pathways, tumor_seg_preds, segmentation_smoothing=True,
):
    res = {"slide_preds": {}, "patch_preds": {}}

    # 1) Tumor segmentations
    tumor_seg_preds = tumor_seg_preds[tumor_seg_preds["slide_id"] == slide].reset_index(drop=True)
    pred_score_col = "prediction_score1_smooth" if segmentation_smoothing else "prediction_score1"
    tumor_seg_preds = tumor_seg_preds[
        ["slide_id", "patch_id", pred_score_col]
    ].rename({pred_score_col: "patch_score_tumor"}, axis=1)
    res["patch_preds"]["tumor_segmentation"] = tumor_seg_preds

    # 2) Pathway predictions
    for pathway in pathways:
        pathway_path = heatmaps_dir / pathway

        # -- predictions
        pathway_preds = pd.read_csv(pathway_path / f"aggregated/predictions/{slide}/pred_scores.csv", index_col=0)
        pathway_preds.insert(0, "slide_id", slide)
        pathway_preds.insert(1, "model_id", [f"model_{idx}" for idx in range(5)])
        pathway_preds = pathway_preds.rename({"0": "prediction_score"}, axis=1)
        res["slide_preds"][pathway] = pathway_preds

        # -- LRP scores
        pathway_lrp = pd.read_csv(pathway_path / f"aggregated/predictions/{slide}/lrp_means.csv", index_col=0)
        pathway_model_0_df = pd.read_csv(pathway_path / f"model_0/test_predictions.csv")
        pathway_model_0_df = pathway_model_0_df[pathway_model_0_df["slide_id"] == slide].reset_index(drop=True)
        pathway_patch_ids = ast.literal_eval(pathway_model_0_df["patch_ids"].iloc[0])[0]
        pathway_lrp.insert(0, "slide_id", slide)
        pathway_lrp.insert(1, "patch_id", pathway_patch_ids)
        pathway_lrp = pathway_lrp.rename({
            **{col: f"lrp_{col}" for col in pathway_lrp.columns[2:8]},
            **{col: f"lrp_model_{col[-1:]}" for col in pathway_lrp.columns[8:]}
        }, axis=1)
        res["patch_preds"][pathway] = pathway_lrp

    # 3) Merge results
    slide_preds_all = pd.DataFrame()
    for key, val in res["slide_preds"].items():
        if len(slide_preds_all) == 0:
            slide_preds_all = val.rename({"prediction_score": f"prediction_score_{key}"}, axis=1)
        else:
            slide_preds_all = pd.merge(
                slide_preds_all, val, on=["slide_id", "model_id"]
            ).rename({"prediction_score": f"prediction_score_{key}"}, axis=1)

    patch_preds_all = res["patch_preds"]["tumor_segmentation"].copy()
    for key, val in res["patch_preds"].items():
        if key != "tumor_segmentation":
            merge_val = val.rename(
                {col: f"{col}_{key}" for col in val.columns if col not in ["slide_id", "patch_id"]}, axis=1
            )
            patch_preds_all = pd.merge(patch_preds_all, merge_val, on=["slide_id", "patch_id"])

    return slide_preds_all, patch_preds_all


def compute_slide_estimates(heatmaps_dir, segmentation_dir, slide_ids, pathways):

    heatmaps_dir = Path(heatmaps_dir)
    segmentation_dir = Path(segmentation_dir)

    tumor_seg_preds = pd.read_csv(
        segmentation_dir / "test_predictions.csv"
    )

    all_model_preds, all_lrp_preds = pd.DataFrame(), pd.DataFrame()

    for sel_slide in tqdm(slide_ids):

        # Load predictions
        try:
            slide_preds, patch_preds = load_predictions(
                heatmaps_dir, sel_slide, pathways, tumor_seg_preds=tumor_seg_preds, segmentation_smoothing=True,
            )
        except FileNotFoundError as err:
            print(err)
            print("Skipping slide: ", sel_slide)
            continue

        # Filter out non-tumor patches and apply LRP score masking (masked = 0)
        patch_preds = patch_preds[patch_preds["patch_score_tumor"] > 0]

        # Apply LRP score masking
        for pathway in pathways:
            col_idx = patch_preds.columns.tolist().index(f"lrp_mean_scores_{pathway}")
            patch_preds.insert(
                col_idx + 1, f"lrp_mean_scores_masked_{pathway}",
                patch_preds[f"lrp_mask_{pathway}"].astype(int) * patch_preds[f"lrp_mean_scores_{pathway}"]
            )
        patch_preds = patch_preds.reset_index(drop=True)

        # Derive LRP positive tumor patches from masked LRP scores
        patch_preds_bin = patch_preds[['slide_id', 'patch_id']].copy()
        patch_preds_bin.insert(
            1, "case_id", patch_preds_bin["slide_id"].apply(lambda x: ".".join(x.split(".")[:3]))
        )
        for pathway in pathways:
            patch_preds_bin.insert(
                len(patch_preds_bin.columns), f"lrp_pos_{pathway}",
                patch_preds[f"lrp_mean_scores_masked_{pathway}"].apply(lambda x: 1 if x > 0 else 0)
            )

        # Aggregate to slide level
        if len(patch_preds_bin) > 0:
            pathway_cols = [f"lrp_pos_{pathway}" for pathway in pathways]
            mean_df = patch_preds_bin.groupby(["case_id", "slide_id"])[["patch_id"] + pathway_cols].agg(
                {**{"patch_id": "count"}, **{col: "mean" for col in pathway_cols}}).reset_index(drop=False)
            lrp_pos_dist = pd.DataFrame()
            for pathway in pathways:
                sub_df = mean_df[["case_id", "slide_id", "patch_id", f"lrp_pos_{pathway}"]].rename(
                    {"patch_id": "num_tumor_patches", f"lrp_pos_{pathway}": "TAPAS"}, axis=1)
                sub_df.insert(2, "pathway", pathway)
                lrp_pos_dist = pd.concat([lrp_pos_dist, sub_df], ignore_index=True)
            lrp_pos_dist = lrp_pos_dist.sort_values(["slide_id", "pathway"]).reset_index(drop=True)
        else:
            lrp_pos_dist = pd.DataFrame.from_dict({
                "case_id": [".".join(sel_slide.split(".")[:3])] * len(pathways),
                "slide_id": [sel_slide] * len(pathways),
                "pathway": pathways,
                "num_tumor_patches": [0] * len(pathways),
                "TAPAS": [0.0] * len(pathways),
            })

        # Collect all predictions
        all_model_preds = pd.concat([all_model_preds, slide_preds], axis=0, ignore_index=True)
        all_lrp_preds = pd.concat([all_lrp_preds, lrp_pos_dist], axis=0, ignore_index=True)

    return all_model_preds, all_lrp_preds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heatmaps-dir', type=str, required=True)
    parser.add_argument('--segmentation-dir', type=str, required=True)
    parser.add_argument('--split-path', type=str, required=True)
    parser.add_argument('--metadata-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--subsets', type=str, nargs='+', required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--pathways', type=str, nargs="+", default=ALL_PATHWAYS)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Set the save directory
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    slide_ids = load_slide_ids(args.split_path, args.metadata_dirs, args.subsets)

    all_model_preds, all_lrp_preds = compute_slide_estimates(
        args.heatmaps_dir, args.segmentation_dir, slide_ids, args.pathways
    )

    all_model_preds.to_csv(os.path.join(save_dir, "model_preds.csv"), index=None)
    all_lrp_preds.to_csv(os.path.join(save_dir, "lrp_preds.csv"), index=None)


if __name__ == '__main__':
    main()
