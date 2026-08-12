from pathlib import Path

from fileformats.medimage import NiftiGz

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from .workflow import multiple_regions_histogram_matching_normalize_dataset_workflow


@with_cli_config()
def run_multiple_regions_histogram_matching_normalize_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    segmentation_label_intervals: dict,
    reference_image: Path,
    reference_label_map: NiftiGz,
    mask_query: str | None = None,
    reference_image_mask: NiftiGz | None = None,
    load_sidecars: bool = False,
    overwrite: bool = False,
    background_rel_weight: float = 0.1,
    num_bins: int = 512,
    rerun: bool = False,
    worker: str = "debug",
):
    """
    Execute the multiple regions histogram matching normalization workflow.

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
    segmentation_label_intervals : dict
        Dictionary of label intervals for segmentation (e.g., `spinal_canal: (2, 2)`).
    reference_image : Path
        Path to the reference image for histogram matching.
    reference_label_map : NiftiGz
        Path to the reference label map for histogram matching.
    mask_query : str | None, optional
        Query to find masks for the images, by default None.
    reference_image_mask : NiftiGz | None, optional
        Path to the reference image mask, by default None.
    load_sidecars : bool, optional
        Whether to load sidecar files, by default False.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.
    background_rel_weight : float, optional
        Relative weight for background in histogram matching, by default 0.1.
    num_bins : int, optional
        Number of bins for histogram matching, by default 512.
    rerun : bool, optional
        Whether to rerun the workflow even if outputs exist, by default False.
    worker : str, optional
        Which Pydra worker to use for execution, by default "debug". Use e.g "cf" for concurrent execution.
    """

    dataset_root = Path(dataset_root).resolve()

    reference_image_parsed = NiftiGz(reference_image)
    reference_label_map_parsed = NiftiGz(reference_label_map)

    if reference_image_mask is not None:
        reference_image_mask_parsed = NiftiGz(reference_image_mask)
    else:
        reference_image_mask_parsed = None

    segmentation_label_intervals_parsed = [
        tuple(eval(v)) for k, v in segmentation_label_intervals.items()
    ]

    wf = multiple_regions_histogram_matching_normalize_dataset_workflow(
        dataset_root=dataset_root,
        image_query=image_query,
        label_map_query=label_map_query,
        output_derivative_name=output_derivative_name,
        segmentation_label_intervals=segmentation_label_intervals_parsed,
        reference_image=reference_image_parsed,
        reference_label_map=reference_label_map_parsed,
        mask_query=mask_query,
        reference_image_mask=reference_image_mask_parsed,
        load_sidecars=load_sidecars,
        overwrite=overwrite,
        background_rel_weight=background_rel_weight,
        num_bins=num_bins,
    )

    wf(worker=worker, rerun=rerun, propagate_rerun=True)


def main():
    setup_logging(filename="histogram_normalize_dataset.log")
    run_multiple_regions_histogram_matching_normalize_dataset_workflow()


if __name__ == "__main__":
    main()
