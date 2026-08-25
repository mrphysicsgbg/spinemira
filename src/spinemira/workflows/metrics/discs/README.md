# Disc Metrics Workflow

The Disc Metrics Workflow is a pipeline for extracting signal intensity metrics from segmented discs in the lumbar spine. **Currently, it only supports disc signal profile analysis.**

This workflow requires a dataset to contain segmentation of the discs. For example, you can use the [TotalSpineSeg](../../totalspineseg/README.md) workflow to segment the images before running this pipeline.

## Example

```bash
uv run python -m spinemira.workflows.metrics.discs --config ./src/spinemira/workflows/metrics/discs/example.yaml
```

## Features

- [x] **Disc Signal Profile Analysis**: Computes signal intensity statistics (mean, median, IQR, min, max) for discs split into sub-regions along a specified direction (e.g., anterior to posterior).
- [ ] Pfirrman analysis based on histogram peaks (Not yet implemented)

## Requirements

- Segmented discs (e.g., using TotalSpineSeg or another segmentation tool).
