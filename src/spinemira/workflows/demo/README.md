# Demo Workflow

The Demo Workflow is a pipeline for querying a BIDS/MIDS dataset and reorienting images to LPS orientation, storing them as derivatives. This workflow is designed to illustrate the use of Pydra for parallel and concurrent execution, including a sleep step to highlight the effects of parallelization.

## Example

The configuration file [example.yaml](./example.yaml) provides an example configuration for reorienting images in a dataset. The pipeline can be executed with:

```bash
uv run python -m spinemira.workflows.demo --config ./src/spinemira/workflows/demo/example.yaml
```

## Features

- Queries a MIDS dataset using a custom query.
- Reorients images to LPS orientation using SimpleITK.
- Stores the reoriented images as derivatives in the dataset.
- Includes a sleep step to demonstrate the difference between concurrent and parallel execution.
