# spinemira

**spinemira** (<ins>SPINE</ins> <ins>M</ins>R<ins>I</ins> Time T<ins>r</ins>acking and <ins>A</ins>nalysis) is a Python toolkit for voxel-wise analysis of spine MRI data, with current focus on Imiomics-based analysis of the lumbar spine.

The software is developed as part of the PhD project: *Advanced MRI methods for tracking of tissue changes in lumbar spinal stenosis: Insights into monitoring and outcome prediction* and Sahlgrenska University Hospital and University of Gothenburg.

Currently, spinemira implements methods for voxel-wise analysis using Imiomics, enabling the study of spatial relationships between MRI data and clinical variables. The framework is presently applied to lumbar spinal stenosis (LSS) cohorts, including analyses of associations between imaging data and clinical outcome measures such as the Oswestry Disability Index (ODI) prior to surgery.

Support for longitudinal analysis and time-resolved tracking of tissue changes is planned and will be developed as part of the ongoing research.

## Dataset structure

spinemira uses a BIDS like dataset structure following the standard BIDS specification, but with some extensions:

- An extra suffix for all files are allowed to distinguish between segmentation or other derived data which is dependent on a contrast. E.g. for an image of `T1w` contrast and with segmentation a filename such as `sub-001_T1w_dseg.nii.gz` is suitable.
