from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.totalspineseg.workflow import segment_dataset_workflow


@with_cli_config()
def run_segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str,
    totalspineseg_data_dir: Path,
    overwrite: bool = False,
    device: str = "cpu",
    totalspineseg_quiet: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    """
    Execute the TotalSpineSeg segmentation workflow.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    image_query : str
        Base query string to filter images.
    output_derivative_name : str
        Name of the output derivative folder.
    totalspineseg_data_dir : Path
        Directory containing TotalSpineSeg data and models.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.
    device : str, optional
        Device to run inference on (e.g., "cpu", "cuda"), by default "cpu".
    totalspineseg_quiet : bool, optional
        Whether to suppress TotalSpineSeg logging output, by default False.
    rerun : bool, optional
        Whether to rerun the workflow even if outputs exist, by default False.
    worker : str, optional
        Which Pydra worker to use for execution, by default "debug". Use e.g "cf" for concurrent execution.
    """

    dataset_root = Path(dataset_root).resolve()
    totalspineseg_data_dir = Path(totalspineseg_data_dir).resolve()

    wf = segment_dataset_workflow(
        dataset_root=dataset_root,
        image_query=image_query,
        output_derivative_name=output_derivative_name,
        totalspineseg_data_dir=totalspineseg_data_dir,
        overwrite=overwrite,
        device=device,
        totalspineseg_quiet=totalspineseg_quiet,
    )

    wf(worker=worker, rerun=rerun)


def main():
    setup_logging(filename="segmentation_pipeline.log")
    run_segment_dataset_workflow()


if __name__ == "__main__":
    main()
