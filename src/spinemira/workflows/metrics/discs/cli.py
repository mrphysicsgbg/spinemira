from pathlib import Path


from spinemira.core.directions import LPS
from spinemira.core.logging import setup_logging
from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.metrics.discs.workflow import (
    calc_disc_metrics_dataset_workflow,
)


_DEFAULT_DISC_LABELS = {
    "DISC_C2_C3": TotalSpineSegLabels.DISC_C2_C3.value,
    "DISC_C3_C4": TotalSpineSegLabels.DISC_C3_C4.value,
    "DISC_C4_C5": TotalSpineSegLabels.DISC_C4_C5.value,
    "DISC_C5_C6": TotalSpineSegLabels.DISC_C5_C6.value,
    "DISC_C6_C7": TotalSpineSegLabels.DISC_C6_C7.value,
    "DISC_C7_T1": TotalSpineSegLabels.DISC_C7_T1.value,
    "DISC_T1_T2": TotalSpineSegLabels.DISC_T1_T2.value,
    "DISC_T2_T3": TotalSpineSegLabels.DISC_T2_T3.value,
    "DISC_T3_T4": TotalSpineSegLabels.DISC_T3_T4.value,
    "DISC_T4_T5": TotalSpineSegLabels.DISC_T4_T5.value,
    "DISC_T5_T6": TotalSpineSegLabels.DISC_T5_T6.value,
    "DISC_T6_T7": TotalSpineSegLabels.DISC_T6_T7.value,
    "DISC_T7_T8": TotalSpineSegLabels.DISC_T7_T8.value,
    "DISC_T8_T9": TotalSpineSegLabels.DISC_T8_T9.value,
    "DISC_T9_T10": TotalSpineSegLabels.DISC_T9_T10.value,
    "DISC_T10_T11": TotalSpineSegLabels.DISC_T10_T11.value,
    "DISC_T12_L1": TotalSpineSegLabels.DISC_T12_L1.value,
    "DISC_L1_L2": TotalSpineSegLabels.DISC_L1_L2.value,
    "DISC_L2_L3": TotalSpineSegLabels.DISC_L2_L3.value,
    "DISC_L3_L4": TotalSpineSegLabels.DISC_L3_L4.value,
    "DISC_L4_L5": TotalSpineSegLabels.DISC_L4_L5.value,
    "DISC_L5_S": TotalSpineSegLabels.DISC_L5_S.value,
}


@with_cli_config()
def run_calculate_disc_metrics_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    disc_labels: dict[str, int] = _DEFAULT_DISC_LABELS,
    load_sidecars: bool = False,
    n_parts: int = 5,
    split_direction: tuple[int, int, int] = LPS.ANTERIOR_TO_POSTERIOR.value,
    split_plane_normal: tuple[int, int, int] = LPS.LEFT_TO_RIGHT.value,
    max_plane_distance: float | None = 5,
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):

    dataset_root = Path(dataset_root).resolve()

    wf = calc_disc_metrics_dataset_workflow(
        dataset_root=dataset_root,
        image_query=image_query,
        label_map_query=label_map_query,
        output_derivative_name=output_derivative_name,
        disc_labels=disc_labels,
        load_sidecars=load_sidecars,
        n_parts=n_parts,
        split_direction=split_direction,
        split_plane_normal=split_plane_normal,
        max_plane_distance=max_plane_distance,
        overwrite=overwrite,
    )

    wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="calculate_disc_metrics_dataset_workflow.log")
    run_calculate_disc_metrics_dataset_workflow()
