#!/bin/bash -l

patient_id=P6
marker=pSTAT3
activation_marker='Nucleus: DAB OD mean'

zenodo_sample_dir=/path/to/zenodo_sample
tiles_dirs=( "$zenodo_sample_dir"/patches )
reg_dir="$zenodo_sample_dir"/results/registration
qupath_measurements_dir="$zenodo_sample_dir"/qupath_measurements
predictions_dirs=( "$zenodo_sample_dir"/predictions )
results_dir="$zenodo_sample_dir"/results/ihc_he_analysis
mkdir -p $results_dir


python3 cell_activation_analyze.py \
--tiles-dirs "${tiles_dirs[@]}" \
--registration-dir $reg_dir \
--qupath-measurement-dir $qupath_measurements_dir \
--predictions-dirs "${predictions_dirs[@]}" \
--results-dir $results_dir \
--marker "$marker" \
--activation-marker "$activation_marker" \
--patient-id "$patient_id" \
--overwrite-results