#!/bin/bash -l


zenodo_sample_path=/path/to/zenodo_sample

model_dir=${results_dir}/segmentation/models/2024_07_12__17_10_48
results_dir=${zenodo_sample_path}/results/segmentation/preds

python3 test.py \
--model-dir ${model_dir} \
--split-path ${zenodo_sample_path}/metadata/split.csv \
--metadata-dirs ${zenodo_sample_path}/metadata \
--patches-dirs ${zenodo_sample_path}/patches \
--features-dirs ${zenodo_sample_path}/features/rudolfv \
--results-dir ${results_dir} \
\
--test-subsets 0 \
--drop-duplicates sample \
--val-batch-size 256 \
--include-unlabeled-samples \
--annotation-thresholds 0.0 0.0 \
\
--device cpu
