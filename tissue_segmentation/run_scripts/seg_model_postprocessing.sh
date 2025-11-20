#!/bin/bash -l


zenodo_sample_path=/path/to/zenodo_sample

preds_dir=${res_dir}/results/segmentation/preds

python3 postprocess.py \
\
--test-dir ${preds_dir} \
--patches-dir ${zenodo_sample_path}/patches \
--slides-dir ${zenodo_sample_path}/slides \
--results-dir ${preds_dir}/postprocessed \
\
--target 1 \
--prediction-threshold 0.5 \
--neighborhood-smoothing \
--border-width 3
