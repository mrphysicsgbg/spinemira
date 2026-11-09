from pathlib import Path

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.segment.totalspineseg.workflow import segment_dataset_workflow


@with_cli_config()
def run_segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    totalspineseg_data_dir: Path,
    output_derivative_name: str = "totalspineseg",
    device: str = "cpu",
    totalspineseg_executable: str = "totalspineseg",
    totalspineseg_quiet: bool = False,
    overwrite: bool = False,
    sagittal_first: bool = False,
    sagittal_query: str = '`vp` == "sag"',
    axial_query: str = '`vp` == "ax"',
    rerun: bool = False,
    worker: str = "debug",
):
    """Segment a dataset with TotalSpineSeg."""
    wf = segment_dataset_workflow(
        dataset_root=Path(dataset_root).resolve(),
        image_query=image_query,
        totalspineseg_data_dir=Path(totalspineseg_data_dir).resolve(),
        output_derivative_name=output_derivative_name,
        device=device,
        totalspineseg_executable=totalspineseg_executable,
        totalspineseg_quiet=totalspineseg_quiet,
        overwrite=overwrite,
        sagittal_first=sagittal_first,
        sagittal_query=sagittal_query,
        axial_query=axial_query,
    )
    return wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="totalspineseg_segment.log")
    run_segment_dataset_workflow()
