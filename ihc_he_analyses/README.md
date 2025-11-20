Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer
==========

![version](https://img.shields.io/badge/version-0.1-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)

<details>
<summary>
  <b>Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer</b>, arXiv, 2024.
  <br><em>Julius Hense*, Mina Jamshidi Idaji*, Laure Ciernik, Jonas Dippel,
Fatma Ersan, Maximilian Knebel, Ada Pusztai, Andrea Sendelhofert,
Oliver Buchstab, Stefan Fr¨ohling, Sven Otto, Jochen Hess, Paris Liokatis,
Frederick Klauschen, Klaus-Robert M¨uller, Andreas Mock </em></br>
* Equal contribution

:octocat: https://github.com/tubml-pathology/xMIL-Pathways

</summary>

```bash
citation bibtex
```

</details>

## IHC-HE analysis

For this part, you would need to have created the aggregated heatmap and then follow the below steps:

1. **Do the IHC-HE registration**

    We register IHC slide(s) on the H&E slide using [VALIS](https://valis.readthedocs.io/en/latest/). 
    The `slide_registration.py` does the registration, and you can run the pipeline for patient P6 using `run_scripts/run_slide_registration.sh`.
2. **Compute the QuPath measurements from the IHCs**

    Please see the manuscript for the descriptions. The analysis should result in a dataframe similar to 
    `zenodo_sample/qupath_measurements/pSTAT3/P6.pSTAT3.tsv`.

3. **Overlap the cell activations and the H&E heatmaps**
    
    We need to aggregate the cell activations within the patches of the H&E slide and the aggregatd heatmap. `cell_activation_analyze.py` does this by using the metadata of the extracted patches.
    You can run the pipeline for patient P6 using `run_scripts/run_cell_activation.sh`.
    The output will be a dataframe that assigns a heatmap score and an IHC activation value to each patch, and whether the patch is included in the aggregated heatmap.
4. **Do the tissue compartment-resolved overlap of heatmap and IHC cell activations**
    Use the notebooks in this directory to perform this analysis. First run `analysis_cell_act_different_regions_pstat3.ipynb` for saving the dataframes.
    Then use `plotting_cell_activation_box_plots.ipynb` to plot the boxplots and do the statistical tests.
    Note that these notebooks are tailored to run for sample patient P6. You can tailor them to your whole cohort with minimal adjustments.