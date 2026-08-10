# Scale Normalize Workflow

Scale Normalize Workflow is a pipeline for normalizing signal intensities of images by scaling to the mode of some segmented tissue. For example scaling signal intensities to the mode of the spinal canal which typically correspond to CSF.

This pipeline requires a dataset to contain segmentation of the input images. E.g. using the [TotalSpineSeg](../../totalspineseg/README.md) workflow to segment images.

## Example

```bash
uv run python -m spinemira.workflows.normalization.scale --config ./src/spinemira/workflows/normalization/scale/example.yaml --label 2
```
