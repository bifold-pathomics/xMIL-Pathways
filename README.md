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

Dear all,

please see the todo list below and push your codes to a branch and open a PR. you may remove `__init__.py` from your folder.

**Please add a description of your code in this readme file so that we can after collecting all info organize it. 
your description doesn't need to be very detailed, but it should be detailed enough for a user to be able to 
orient themselves in the repo.**

- [ ] constructing the readme [Mina/Julius, all]
- [x] IHC-HE registration [Mina] --> folder:  IHC_HE_registration
- [x] computing cell activation sums within patches [Mina] --> folder: cell_activations_analyses
- [x] plotting and statistical analysis for cell activation [Mina/Laure] --> folder: cell_activations_analyses
- [ ] heatmap aggregation [Julius] (this is already contributed by Laure, as Julius has used the code to reproduce virchow2, I think he is familiar now where is what\M)
- [ ] TAPAS computations [Julius]
- [ ] tissue segmentation (tumor segmentation and tumor border detection) [Julius] --> folder: tumor_segmentation
- [ ] R-codes for all the analyses of Figure 3 [Andy] --> folder: patient_stratification
