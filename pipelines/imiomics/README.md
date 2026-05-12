# Pipelines for implementing Imiomics with GLM on the lumbar spine

The Imiomics workflow consists of the following pipelines:

1. Segmentation of all images using TotalSpineSeg. See pipeline [totalspineseg_inference.py](../totalspineseg_inference/totalspineseg_inference.py).
2. Registration of all images to a common coordinate system: [register_with_straightening_to_single_image.py](./register_with_straightening_to_single_image.py) [^1]
   The pipeline consists of multiple steps:
   1. Pre-processing of images including: straightening of the spine, masking and scaling of signal intensities to CSF.
   2. Non-rigid registration using the software [deform](https://github.com/simeks/deform) [^2] [^3]
3. Intensity normalization: [intensity_normalization.py](./intensity_normalization.py)
4. GLM analysis [glm_analysis.py](./glm_analysis.py)
   Perform general linear modelling together with non-parametric inference using nilearn [^4]

An additional pipeline exists [estimate_dsca.py](./estimate_dsca.py) to estimate the dura cross sectional area for each spinal based on the segmented spinal canal.

## Configuration files

The folder [configuration](./configuration/) includes example configuration files for running the Imiomics pipeline.


[^1]: Warszawer, Yehuda & Molinier, Nathan & Valosek, Jan & Benveniste, Pierre-Louis & Bédard, Sandrine & Shirbint, Emanuel & Mohamed, Feroze & Tsagkas, Charidimos & Kolind, Shannon & Lynd, Larry & Oh, Jiwon & Prat, Alexandre & Tam, Roger & Traboulsee, Anthony & Patten, Scott & Lee, Lisa Eunyoung & Achiron, Anat & Cohen-Adad, Julien. (2025). TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI. [10.13140/RG.2.2.31318.56649](https://doi.org/10.13140/RG.2.2.31318.56649).

[^2]: S. Ekström, F. Malmberg, H. Ahlström, J. Kullberg, and R. Strand, “Fast graph-cut based optimization for practical dense deformable registration of volume images,” Computerized Medical Imaging and Graphics, vol. 84, p. 101745, Sep. 2020, doi: 10.1016/j.compmedimag.2020.101745.

[^3]: S. Ekström, M. Pilia, J. Kullberg, H. Ahlström, R. Strand, and F. Malmberg, “Faster dense deformable image registration by utilizing both CPU and GPU,” J. Med. Imag., vol. 8, no. 01, Feb. 2021, doi: 10.1117/1.JMI.8.1.014002.

[^4]: Nilearn contributors, Chamma, A., Frau-Pascual, A., Rothberg, A., Abadie, A., Abraham, A., Gramfort, A., Savio, A., Cionca, A., Sayal, A., Thual, A., Kodibagkar, A., Kanaan, A., Pinho, A. L., Joshi, A., Idrobo, A. H., Kieslinger, A.-S., Kumari, A., Rokem, A., … Nájera, Ó. (2025). nilearn (0.12.1). Zenodo. https://doi.org/10.5281/zenodo.17043133
