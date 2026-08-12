# Multiple Regions Histogram Normalization Workflow

Multiple Regions Histogram Normalization Workflow is a pipeline for normalizing signal intensities of images by using histogram matching of specific segmented tissues.

This pipeline requires a dataset to contain segmentation of the input images. E.g. using the [TotalSpineSeg](../../totalspineseg/README.md) workflow to segment images.

## Example

```bash
uv run python -m spinemira.workflows.normalization.histogram --config ./src/spinemira/workflows/normalization/histogram/example.yaml --reference-image ./examples/spider_dataset/derivatives/scale_normalized/sub-001/anat/sub-001_T2w.nii.gz --reference-label-map ./examples/spider_dataset/derivatives/totalspineseg/sub-001/anat/sub-001_T2w_dseg.nii.gz
```
