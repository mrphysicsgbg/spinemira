# Pipelines for implementing Imiomics on the lumbar spine

The Imiomics workflow consists of the following pipelines:

1. Segmentation of all images using TotalSpineSeg. See pipeline [totalspineseg_inference.py](../totalspineseg_inference/totalspineseg_inference.py).
2. Creation of spine templates. This is fulfilled by two pipelines:
   1. [register_with_straightening_to_single_image.py](./register_with_straightening_to_single_image.py)
   2. [make_template.py](./make_template.py)
3. Registration of all images to be included in the analysis to the templates created in step 2 [register_with_straightening_to_template.py](./register_with_straightening_to_template.py).
4. Intensity normalization of registered images [intensity_normalization.py](./intensity_normalization.py)
