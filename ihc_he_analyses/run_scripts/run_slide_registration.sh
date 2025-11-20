#!/bin/bash


zenodo_sample_dir=path/to/zenodo_sample
slides_dir="$zenodo_sample_dir"/slides
results_dir="$zenodo_sample_dir"/results/registration
mkdir -p $results_dir

python3 slide_registration.py \
--results-dir $results_dir \
--patient-slides-dir $slides_dir \
--patient-id P6
#--do-not-save

