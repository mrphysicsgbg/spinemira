from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.pipelines.config import with_cli_config
from .workflow import scale_normalize_dataset_workflow


@with_cli_config()
def run_scale_normalize_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    label: int = TotalSpineSegLabels.SPINAL_CANAL.value,
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    """
    Execute the scale normalization workflow.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    image_query : str
        Base query string to filter images.
    label_map_query : str
        Query to find segmented images.
    output_derivative_name : str
        Name of the output derivative folder.
    label : int, optional
        Label in
    rerun : bool, optional
        Whether to rerun the workflow even if outputs exist, by default False.
    worker : str, optional
        Which Pydra worker to use for execution, by default "debug". Use e.g "cf" for concurrent execution.
    """

    dataset_root = Path(dataset_root).resolve()

    wf = scale_normalize_dataset_workflow(
        dataset_root=dataset_root,
        image_query=image_query,
        label_map_query=label_map_query,
        output_derivative_name=output_derivative_name,
        label=label,
        overwrite=overwrite,
    )

    wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="scale_normalize_dataset.log")
    run_scale_normalize_dataset_workflow()


if __name__ == "__main__":
    main()
