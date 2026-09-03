from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.metrics.collect.workflow import collect_metrics_workflow


@with_cli_config()
def run_collect_metrics_workflow(
    dataset_root: Path,
    raw_query: str,
    derivative_queries: dict[str, str],
    columns: dict[str, str],
    output_file: Path,
    load_sidecars: bool = False,
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    """
    Run the Collect Metrics Workflow to extract and compile metrics from a MIDS dataset.

    This function initializes and executes the workflow for collecting metrics from raw and
    derivative files in a MIDS dataset, based on user-defined queries and column definitions.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the MIDS dataset.
    raw_query : str
        Query string to filter raw files (e.g., '`source` == "raw"').
    derivative_queries : dict[str, str]
        Dictionary mapping derivative names to query strings for filtering derivatives
        (e.g., {"disc_metrics": '`pipeline` == "disc_metrics"'}).
    columns : dict[str, str]
        Dictionary defining how to extract data for each column. Keys are the column names
        in the output, and values are JSON pointers to the field in the extracted data.
        Use `/entities/` for path-extracted data and `/sidecar/` for sidecar file data.
    output_file : Path
        Path to the output CSV file where the collected metrics will be saved.
    load_sidecars : bool, optional
        Whether to load sidecar files when indexing the dataset, by default False.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists, by default False.
    rerun : bool, optional
        Whether to rerun the workflow if outputs already exist, by default False.
    worker : str, optional
        Worker type for executing the workflow (e.g., "debug", "serial", "parallel"),
        by default "debug".
    """
    dataset_root = Path(dataset_root).resolve()
    output_file = Path(output_file).resolve()

    wf = collect_metrics_workflow(
        dataset_root=dataset_root,
        raw_query=raw_query,
        derivative_queries=derivative_queries,
        columns=columns,
        output_file=output_file,
        load_sidecars=load_sidecars,
        overwrite=overwrite,
    )

    wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    """
    Main entry point for the Collect Metrics Workflow.

    This function sets up logging and invokes the workflow execution.
    """
    setup_logging(filename="collect_metrics_workflow.log")
    run_collect_metrics_workflow()
