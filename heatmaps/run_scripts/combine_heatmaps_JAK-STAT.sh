#!/bin/bash


zenodo_sample_path=/path/to/zenodo_sample

pathway="JAK-STAT"
preds_dir=${zenodo_sample_path}/heatmaps/${pathway}
target_dir=${zenodo_sample_path}/results/heatmaps/${pathway}/aggregated


python3 combine_heatmaps.py \
--preds_dir ${preds_dir} \
--target_dir ${target_dir} \
--patches_dir ${zenodo_sample_path}/patches \
--metadata_dir ${zenodo_sample_path}/metadata \
--slides_dir ${zenodo_sample_path}/slides \
--target ${pathway}
