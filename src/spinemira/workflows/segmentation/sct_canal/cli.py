from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.segmentation.sct_canal.workflow import segment_dataset_workflow


@with_cli_config()
def run_segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str = "sct_canal_t2",
    sct_deepseg_executable: str = "sct_deepseg",
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    """Segment T2-weighted images using SCT sc_canal_t2."""

    wf = segment_dataset_workflow(
        dataset_root=Path(dataset_root).resolve(),
        image_query=image_query,
        output_derivative_name=output_derivative_name,
        sct_deepseg_executable=sct_deepseg_executable,
        overwrite=overwrite,
    )
    return wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="sct_canal_segmentation.log")
    run_segment_dataset_workflow()
