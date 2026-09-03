# Collect Metrics Workflow

The Collect Metrics Workflow is a pipeline for querying a MIDS dataset and collecting sidecar data from the main participants files, as well as original and derivative sidecar data, and assembling these into a Pandas DataFrame.

## Usage

To run the workflow, use the following command with a configuration file:

```bash
uv run python -m spinemira.workflows.metrics.collect --config path/to/your/config.yaml --output-file path/to/output.csv
```

## Configuration

The workflow requires a YAML configuration file with the following fields:

- **`dataset_root`**: Path to the root directory of the MIDS dataset.
- **`raw_query`**: Query string to filter raw files (e.g., `"`source` == \"raw\""`).
- **`derivative_queries`**: Dictionary of derivative queries, where keys are derivative names and values are query strings (e.g., `"`pipeline` == \"disc_metrics\""`).
- **`columns`**: Dictionary defining how to extract data for each column. Keys are the column names in the output, and values are JSON pointers to the field in the extracted data. Use `/entities/` to access entities extracted from file paths (e.g., `/entities/participant_id`) and `/sidecar/` to access data from sidecar files (e.g., `/sidecar/statistics/DISC_L5_S/mean`).
