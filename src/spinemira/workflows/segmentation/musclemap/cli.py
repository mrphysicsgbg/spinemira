from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.segmentation.musclemap.workflow import segment_dataset_workflow


@with_cli_config()
def run_segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str = "musclemap",
    device: str = "cpu",
    musclemap_executable: str = "mm_segment",
    musclemap_region: str = "wholebody",
    musclemap_model: str | None = None,
    musclemap_model_version: str | None = None,
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    """Segment a dataset with MuscleMap."""
    wf = segment_dataset_workflow(
        dataset_root=Path(dataset_root).resolve(),
        image_query=image_query,
        output_derivative_name=output_derivative_name,
        device=device,
        musclemap_executable=musclemap_executable,
        musclemap_region=musclemap_region,
        musclemap_model=musclemap_model,
        musclemap_model_version=musclemap_model_version,
        overwrite=overwrite,
    )
    return wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="musclemap_segment.log")
    run_segment_dataset_workflow()
