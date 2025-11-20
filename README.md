Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer
==========

# TODO
- [ ] check all the referenced links work
- [ ] the `pip_requirement.txt` should be updated with the versions


![version](https://img.shields.io/badge/version-0.1-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)

<details>
<summary>
  <b>Digital Spatial Pathway Mapping Reveals Prognostic Tumor States in Head and Neck Cancer</b>.
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


**Summary**: In this work, we infer transcriptome-derived signaling pathway activities 
directly from routine H&E slides. We propose  to use the MIL heatmaps for stratifying the patients. 
In this regard we propose spatial activity metric (TAPAS) quantifying intratumoral heterogeneity based on XAI heatmaps. 
In this repository, we share the codes for this manuscript. We have also shared data of an exemplar patient at Zenodo. 
We have prepared the pipelines in the way that they work fluently with the shared data.


## Usage

The workflow of this work is as the following:

1. Train a (Transformer-based) MIL model to predict the biomarker from H&E slide and create heatmaps for the test set.
    For this step, we used a code-base developed by us, publicly available at [xMIL](https://github.com/tubml-pathology/xMIL).
    Please also see our [NeurIPS 2024 publication](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f9e0309d8a947ca44463a9b7e8b6a3f-Abstract-Conference.html) about explaining MIL models.
2. If you have multiple models (e.g., from your cross-validation training), aggregated the heatmaps.
    For this step, you can use the code at folder [heatmap_aggregation](https://github.com/tubml-pathology/xMIL-Pathways/heatmap_aggregation).
3. Perform tissue segmentation: for our subsequent analyses we segment the H&E slide into tumor, non-tumor, and border.
    For this step, you can use the code at folder [tissue_segmentation](https://github.com/tubml-pathology/xMIL-Pathways/tissue_segmentation).
    Please see 
   [tissue_segmentation/README.md](https://github.com/tubml-pathology/xMIL-Pathways/tissue_segmentation/README.md) for detailed info.
4. From the tumor area and the generated (aggregated) heatmap, you can compute TAPAS score.
   For this step, you can use the code at folder [computing_tapas](https://github.com/tubml-pathology/xMIL-Pathways/computing_tapas).
5. You can perform the IHC-H&E analyses using the code at folder [ihc_he_analyses](https://github.com/tubml-pathology/xMIL-Pathways/ihc_he_analyses). 
   This includes IHC-H&E registration, aggregating the IHC activations within the patches of the H&E slide and overlapping 
   the heatmap and IHC activations. Please see 
   [ihc_he_analyses/README.md](https://github.com/tubml-pathology/xMIL-Pathways/ihc_he_analyses/README.md) for detailed info.
6. You can find the code for analyses doing patient stratification using TAPAS score and clinical metadata at
    [patient_stratification](https://github.com/tubml-pathology/xMIL-Pathways/patient_stratification)


## Contact us
If you face issues using our codes, you can open an issue in this repository, or contact us: 

:email: [Julius Hense](https://github.com/hense96) and [Mina Jamshidi](https://github.com/minajamshidi)

## License and citation
If you find our codes useful in your work, please cite us:
```bash
citation bibtex
```

:copyright: This code is provided under CC BY-NC-ND 4.0. 
Please refer to the license file for details.
