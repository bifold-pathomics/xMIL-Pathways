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

## Heatmap Aggregation

You can find the LRP heatmaps from the five JAK-STAT prediction models for the example slide (P6..HE-PRE) at: `/path/to/zenodo_sample/heatmaps/JAK-STAT/model{0-4}/overlays`. They were created using the [xMIL](https://github.com/bifold-pathomics/xMIL) repository.
Run the following code to combine those heatmaps:

1. ```cd heatmap_aggregation```
2. Run ```run_scripts/combine_heatmaps_JAK-STAT.sh``` to combine the heatmaps according to the strategy described in the paper. The results will be written to the specified folder. The ground truth results are also provided at: `/path/to/zenodo_sample/heatmaps/JAK-STAT/aggregated`.

Requirements
```
openslide
numpy
pandas
tqdm
matplotlib
PIL
```
