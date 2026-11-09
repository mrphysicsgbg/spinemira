# Segment workflows

The workflows segment a MIDS dataset into separate derivatives. All publish a `dseg` image and a JSON label sidecar. Existing outputs are reused unless `--overwrite` is supplied. Use `--device cuda` for GPU inference. All commands accept YAML configuration with `--config`.

TotalSpineSeg:

```bash
python -m spinemira.workflows.segmentation.totalspineseg --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"' --totalspineseg-data-dir /path/to/models
```

Add `--sagittal-first` to segment sagittal scans before related axial scans, using the sagittal segmentations as localizers. `--sagittal-query` and `--axial-query` select orientations. Axial images without exactly one matching sagittal image in the same subject and session are skipped.

MuscleMap:

```bash
python -m spinemira.workflows.segmentation.musclemap --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"'
```

The default derivative names are `totalspineseg` and `musclemap`; use `--output-derivative-name` to override them. The installed commands are `spinemira-segment-totalspineseg` and `spinemira-segment-musclemap`.

SCT spinal canal segmentation (T2-weighted images):

```bash
python -m spinemira.workflows.segmentation.sct_canal --dataset-root /path/to/dataset \
  --image-query '`source` == "raw" and `suffix` == "T2w"'
```

Install Spinal Cord Toolbox and its `sc_canal_t2` model first (`sct_deepseg sc_canal_t2 -install`); the workflow checks the executable but does not install or verify the model. Use `--sct-deepseg-executable` to specify its path. The default derivative is `sct_canal_t2`; the installed command is `spinemira-segment-sct-canal`. The model outputs a binary mask of the spinal canal (dural sac). See the [SCT model documentation](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/sc_canal_t2.html).
