# spinemira

**spinemira** (<ins>SPINE</ins> <ins>M</ins>R<ins>I</ins> Time T<ins>r</ins>acking and <ins>A</ins>nalysis) is a Python toolkit for voxel-wise analysis of spine MRI data, with current focus on Imiomics-based analysis of the lumbar spine.

The software is developed as part of the PhD project: *Advanced MRI methods for tracking of tissue changes in lumbar spinal stenosis: Insights into monitoring and outcome prediction* at Sahlgrenska University Hospital and University of Gothenburg.

spinemira uses a [ORMIR-MIDS](https://ormir-mids.github.io/)-like dataset structure and includes multiple different workflows which processes original data stored in MIDS-dataset to compute derivative data.

## Architecture

spinemira is built on top of **[Pydra](https://pydra.readthedocs.io/)** and uses primarily Numpy, Pandas and SimpleITK as core libraries.

The project follows a modular architecture organized into:

```
spinemira/
├── core/                    # Core functionality and utilities
│   ├── analysis/            # Analysis algorithms and metrics
│   ├── morphology/          # Morphological operations (e.g., spine straightening)
│   ├── registration/        # Image registration utilities
│   ├── segmentation/        # Segmentation tools and label definitions
│   └── logging.py           # Logging configuration
│
├── io/                      # Input/Output operations
│   └── mids.py              # MIDS dataset support
│
├── pipelines/               # Pipeline configuration and utilities
│   ├── config.py            # CLI configuration decorator with Pydra integration
│   └── imiomics/            # Imiomics-based analysis pipelines
│                               # (Ongoing project: being rewritten as Pydra tasks and workflows)
│
├── tasks/                   # Atomic processing tasks (Pydra tasks)
│   ├── discs.py             # Disc-specific operations
│   ├── filters.py           # Image filtering tasks
│   ├── image.py             # General image processing
│   ├── logging.py           # Task logging utilities
│   ├── mids.py              # MIDS-related tasks (query, index, etc.)
│   ├── segmentation.py      # Segmentation and label metadata tasks
│   └── utils.py             # Utility tasks
│
├── plotting/                # Visualization utilities
│   └── plot.py              # Plotting functions
│
├── workflows/               # Pre-built Pydra workflows
│   ├── demo/                # Demo/example workflow
│   ├── metrics/             # Metric calculation workflows
│   │   ├── collect/         # Collect data from dataset into a CSV
│   │   └── discs/           # Disc metrics workflows
│   ├── normalization/       # Image normalization workflows
│   │   ├── histogram/       # Histogram matching normalization
│   │   └── scale/           # Scale normalization
│   ├── segmentation/        # TotalSpineSeg, MuscleMap, and spinal canal workflows
│   └── totalspineseg/       # Earlier TotalSpineSeg workflow
│
└── tests/                   # Unit and integration tests
```

## Command Line Interface

spinemira provides the following command-line scripts that can be executed as modules or via pip installation:

### Installed Scripts (via `pip install`)

| Command                         | Description                                      |
|---------------------------------|--------------------------------------------------|
| `spinemira-disc-metrics`        | Run the disc analysis workflow                   |
| `spinemira-collect-metrics`     | Run the collect metrics workflow                 |
| `spinemira-normalize-histogram` | Run the normalize by histogram workflow          |
| `spinemira-normalize-scale`     | Run the normalize by scaling workflow            |
| `spinemira-totalspineseg`       | Run the earlier TotalSpineSeg workflow           |
| `spinemira-segment-totalspineseg` | Segment spine structures with TotalSpineSeg      |
| `spinemira-segment-musclemap`   | Segment muscles with MuscleMap                   |
| `spinemira-segment-sct-canal`   | Segment the spinal canal in T2-weighted images   |

### Segmentation

The segmentation workflows query a MIDS dataset and publish a `dseg` image and a JSON label sidecar under a derivative directory. They support TotalSpineSeg, MuscleMap, and SCT spinal canal segmentation. Complete outputs are reused unless `--overwrite` is supplied; if a sidecar is missing, the image and sidecar are regenerated.

```bash
spinemira-segment-totalspineseg --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"' --totalspineseg-data-dir /path/to/models

spinemira-segment-musclemap --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"'

spinemira-segment-sct-canal --dataset-root /path/to/dataset \
  --image-query '`source` == "raw" and `suffix` == "T2w"'
```

Each workflow requires its corresponding external segmentation tool. TotalSpineSeg also needs its model data directory; SCT requires Spinal Cord Toolbox and the `sc_canal_t2` model (`sct_deepseg sc_canal_t2 -install`). The included `install_totalspineseg.py` and `install_musclemap.py` scripts can install those tools in separate environments; pass the resulting executable path with `--totalspineseg-executable` or `--musclemap-executable`. Use `--device cuda` for TotalSpineSeg or MuscleMap GPU inference, and `--output-derivative-name` to change the output directory name.

TotalSpineSeg also supports `--sagittal-first`, which uses sagittal segmentations as localizers for related axial images. See the [segmentation workflow guide](./src/spinemira/workflows/segmentation/README.md) for details and options.

### Module Execution (Python -m)

All workflows can also be executed directly as Python modules:

```bash
# Demo workflow
python -m spinemira.workflows.demo

# Segmentation workflows
python -m spinemira.workflows.segmentation.totalspineseg
python -m spinemira.workflows.segmentation.musclemap
python -m spinemira.workflows.segmentation.sct_canal

# Earlier TotalSpineSeg workflow
python -m spinemira.workflows.totalspineseg

# Disc metrics calculation
python -m spinemira.workflows.metrics.discs

# Histogram matching normalization
python -m spinemira.workflows.normalization.histogram

# Scale normalization
python -m spinemira.workflows.normalization.scale

# Collect metrics
python -m spinemira.workflows.metrics.collect
```

### Common Options

All CLI scripts support the following Pydra execution options:

- `--worker`: Execution backend (`debug`, `cf` for concurrent futures, or distributed workers)
- `--rerun`: Force Pydra tasks to rerun; use `--overwrite` to replace existing segmentation outputs
- `--config`: Path to YAML configuration file for parameter defaults
- Individual workflow parameters can be specified as command-line flags (e.g., `--dataset-root`, `--image-query`)

## Current Functionality

### Pydra-based Workflows

1. **[Segmentation Workflows](./src/spinemira/workflows/segmentation/README.md)** (`spinemira.workflows.segmentation`)
   - Segment spine structures, muscles, or the spinal canal and publish label metadata
   - Support dataset queries, configurable derivative names, and output reuse

2. **[Demo Workflow](./src/spinemira/workflows/demo/README.md)** (`spinemira.workflows.demo`)
   - Demonstrates basic Pydra workflow structure
   - Reorients images to LPS coordinate system
   - Illustrates parallel execution with artificial delays
   - Useful for testing and understanding the framework

3. **[TotalSpineSeg Workflow](./src/spinemira/workflows/totalspineseg/README.md)** (`spinemira.workflows.totalspineseg`)
   - Integrates [TotalSpineSeg](github.com/neuropoly/totalspineseg) for automatic spine segmentation
   - Processes entire datasets with support for automatic spine and intervertebral disc segmentation, label map generation, and vertebral level identification
   - Supports CPU and CUDA execution
   - Requires the external TotalSpineSeg package and model data

4. **[Disc Metrics Workflow](./src/spinemira/workflows/metrics/discs/README.md)** (`spinemira.workflows.metrics.discs`)
   - Calculates disc-specific metrics from segmented MRI data
   - Features signal profile analysis across discs, delta-mu calculation, multi-part disc segmentation, and custom disc label definitions

5. **[Histogram Matching Normalization](./src/spinemira/workflows/normalization/histogram/README.md)** (`spinemira.workflows.normalization.histogram`)
   - Multi-region histogram matching for intensity normalization
   - Region-specific normalization based on segmentation labels with configurable histogram bins and background weighting

6. **[Scale Normalization](./src/spinemira/workflows/normalization/scale/README.md)** (`spinemira.workflows.normalization.scale`)
   - Scale-based intensity normalization based on selected segmentation regions
   - Support for rescaling to positive values

7. **[Collect metrics](./src/spinemira/workflows/metrics/collect/README.md)** (`spinemira.workflows.metrics.collect`)
   - Collects data from sidecars, path entities and participants.tsv
   - Assembles specified data into a table

### Legacy and In-Development

- **Imiomics Pipelines**: Existing Imiomics-based analysis pipelines are defined in the `pipelines/imiomics/` directory. These are currently being rewritten as Pydra tasks and workflows to integrate with the modern architecture.
