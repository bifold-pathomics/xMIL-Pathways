Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer
==========

![version](https://img.shields.io/badge/version-0.1-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)

<details>
<summary>
  <b>Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer</b>.
  <br><em>Julius Hense*, Mina Jamshidi Idaji*, Laure Ciernik, Jonas Dippel,
Fatma Ersan, Maximilian Knebel, Ada Pusztai, Andrea Sendelhofert,
Oliver Buchstab, Stefan Fröhling, Sven Otto, Jochen Hess, Paris Liokatis,
Frederick Klauschen, Klaus-Robert Müller, Andreas Mock </em></br>
* Equal contribution

:octocat: https://github.com/bifold-pathomics/xMIL-Pathways
</summary>

```bash
@article {Hense2025.11.24.689710,
	author = {Hense, Julius and Idaji, Mina Jamshidi and Ciernik, Laure and Dippel, Jonas and Ersan, Fatma and Knebel, Maximilian and Pusztai, Ada and Sendelhofert, Andrea and Buchstab, Oliver and Fr{\"o}hling, Stefan and Otto, Sven and Hess, Jochen and Liokatis, Paris and Klauschen, Frederick and M{\"u}ller, Klaus-Robert and Mock, Andreas},
	title = {Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer},
	year = {2025},
	doi = {10.1101/2025.11.24.689710},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

</details>

## Tissue Segmentation

1. ```cd tissue_segmentation```
2. Run ```run_scripts/seg_model_training.sh``` to train the segmentation model. Note: This is not generally functional as we cannot publish the underlying data. 
   The resulting model checkpoint is provided at the Zenodo repository: ```zenodo_sample/results/segmentation/models/2024_07_12__17_10_48```.
3. Run ```run_scripts/seg_model_inference.sh``` to apply the segmentation model to the example slide.
4. Run ```run_scripts/seg_model_postprocessing.sh``` to run neighborhood smoothing and border estimation and to generate visualizations.
