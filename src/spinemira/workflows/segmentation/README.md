# Combined segmentation workflow

Runs TotalSpineSeg and MuscleMap on each image selected from a MIDS dataset.
Each tool is used only when enabled and available. Commands (`totalspineseg`
and `mm_segment`) are found on `PATH`, or supplied using `--totalspineseg-executable` and `--musclemap-executable`
with absolute paths. Disable tools with `--no-use-totalspineseg` or
`--no-use-musclemap` (YAML: `use_totalspineseg: false` or `use_musclemap: false`).

Alternatively, set `TOTALSPINESEG_EXECUTABLE` and `MUSCLEMAP_EXECUTABLE` to
the respective executable paths. These variables override the default command
names; a different executable supplied through the CLI or YAML takes priority.
If neither tool can run, processing is skipped.

```bash
python -m spinemira.workflows.segmentation \
  --config src/spinemira/workflows/segmentation/example.yaml
```

Outputs are saved under `derivatives/<output_derivative_name>/`:

- `*_dseg.nii.gz`: labels from the available enabled tool, or merged labels
  when both tools run, retaining each tool's original label values.
  TotalSpineSeg takes priority wherever its label is nonzero; MuscleMap fills
  the background. MuscleMap is resampled to the TotalSpineSeg grid with
  nearest-neighbor interpolation if their grids differ.
- `*_levels.nii.gz`: the TotalSpineSeg levels map, only when TotalSpineSeg is used.
  The workflow returns `None` for levels when only MuscleMap runs.

Existing outputs are reused unless `--overwrite` is set; the levels file is
required for reuse only when TotalSpineSeg is used. Use `--overwrite` when changing
tool selection for an existing derivative. `totalspineseg_data_dir` is required
only when TotalSpineSeg runs.
Intermediate segmentations are retained in the Pydra task cache.
For GPU execution, set `--device cuda`.
MuscleMap options include `--musclemap-region`, `--musclemap-model`, and
`--musclemap-model-version`.
