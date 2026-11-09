# Segment workflows

The two workflows segment a MIDS dataset into separate derivatives. Both publish a `dseg` image and a JSON label sidecar. Existing outputs are reused unless `--overwrite` is supplied. Use `--device cuda` for GPU inference. Both commands accept YAML configuration with `--config`.

TotalSpineSeg:

```bash
python -m spinemira.workflows.segment.totalspineseg --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"' --totalspineseg-data-dir /path/to/models
```

Add `--sagittal-first` to segment sagittal scans before related axial scans, using the sagittal segmentations as localizers. `--sagittal-query` and `--axial-query` select orientations. Axial images without exactly one matching sagittal image in the same subject and session are skipped.

MuscleMap:

```bash
python -m spinemira.workflows.segment.musclemap --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"'
```

The default derivative names are `totalspineseg` and `musclemap`; use `--output-derivative-name` to override them. The installed commands are `spinemira-segment-totalspineseg` and `spinemira-segment-musclemap`.
