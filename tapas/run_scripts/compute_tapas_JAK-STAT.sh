#!/bin/bash -l


zenodo_sample_path=/path/to/zenodo_sample

pathway=JAK-STAT
heatmaps_dir=${zenodo_sample_path}/heatmaps
segmentation_dir=${zenodo_sample_path}/results/segmentation/preds/postprocessed
results_dir=${zenodo_sample_path}/results/tapas


python3 compute_tapas.py \
--heatmaps-dir ${heatmaps_dir} \
--segmentation-dir ${segmentation_dir} \
--split-path ${zenodo_sample_path}/metadata/split.csv \
--metadata-dirs ${zenodo_sample_path}/metadata \
--subsets 0 \
--pathways ${pathway} \
--results-dir ${results_dir}
