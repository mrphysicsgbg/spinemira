from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.segmentation.workflow import segment_dataset_workflow


@with_cli_config()
def run_segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str,
    totalspineseg_data_dir: Path | None = None,
    overwrite: bool = False,
    device: str = "cpu",
    totalspineseg_quiet: bool = False,
    totalspineseg_executable: str = "totalspineseg",
    musclemap_executable: str = "mm_segment",
    musclemap_region: str = "wholebody",
    musclemap_model: str | None = None,
    musclemap_model_version: str | None = None,
    use_totalspineseg: bool = True,
    use_musclemap: bool = True,
    rerun: bool = False,
    worker: str = "debug",
):
    """Run TotalSpineSeg and MuscleMap and publish merged segmentation labels.

    Disabled tools and unavailable executables are skipped.
    Use device cuda for GPU execution.
    """
    wf = segment_dataset_workflow(
        dataset_root=Path(dataset_root).resolve(),
        image_query=image_query,
        output_derivative_name=output_derivative_name,
        totalspineseg_data_dir=(
            Path(totalspineseg_data_dir).resolve()
            if totalspineseg_data_dir is not None
            else None
        ),
        overwrite=overwrite,
        device=device,
        totalspineseg_quiet=totalspineseg_quiet,
        totalspineseg_executable=totalspineseg_executable,
        musclemap_executable=musclemap_executable,
        musclemap_region=musclemap_region,
        musclemap_model=musclemap_model,
        musclemap_model_version=musclemap_model_version,
        use_totalspineseg=use_totalspineseg,
        use_musclemap=use_musclemap,
    )
    return wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="combined_segmentation_pipeline.log")
    run_segment_dataset_workflow()


if __name__ == "__main__":
    main()
