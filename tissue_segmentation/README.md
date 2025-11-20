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

## Tissue Segmentation

1. ```cd tissue_segmentation```
2. Run ```run_scripts/seg_model_training.sh``` to train the segmentation model. Note: This is not generally functional as we cannot publish the underlying data. 
   The resulting model checkpoint is provided at the Zenodo repository: ```zenodo_sample/results/segmentation/models/2024_07_12__17_10_48```.
3. Run ```run_scripts/seg_model_inference.sh``` to apply the segmentation model to the example slide.
4. Run ```run_scripts/seg_model_postprocessing.sh``` to run neighborhood smoothing and border estimation and to generate visualizations.
