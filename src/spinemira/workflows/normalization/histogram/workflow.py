from importlib.metadata import version
import logging
from pathlib import Path

from fileformats.medimage import NiftiGz
from pydra.compose import workflow

import spinemira
from spinemira.tasks.mids import (
    find_indexed_derivative,
    index,
    initialize_derivative,
    query_mids,
    resolve_derivative,
    publish_derivative,
)
from spinemira.tasks.filters import (
    reduce_label_map,
    multiple_regions_histogram_matching,
)


logger = logging.getLogger(__name__)


@workflow.define(outputs=["file"])
def multiple_regions_histogram_matching_single_image_workflow(
    image: NiftiGz,
    label_map: NiftiGz,
    ref_image: NiftiGz,
    ref_label_map: NiftiGz,
    label_intervals: list[tuple[int, int]],
    output_image: Path,
    mask: NiftiGz | None = None,
    ref_mask: NiftiGz | None = None,
    overwrite: bool = False,
    background_rel_weight: float = 0.1,
    num_bins: int = 512,
) -> NiftiGz:
    """
    Normalize a single image using multiple regions histogram matching.

    Parameters
    ----------
    image : NiftiGz
        Input image to normalize.
    label_map : NiftiGz
        Label map for the input image.
    ref_image : NiftiGz
        Reference image for histogram matching.
    ref_label_map : NiftiGz
        Reference label map for histogram matching.
    label_intervals : list[tuple[int, int]]
        List of label intervals for segmentation.
    output_image : Path
        Path to save the normalized image.
    mask : NiftiGz | None, optional
        Mask for the input image, by default None.
    ref_mask : NiftiGz | None, optional
        Mask for the reference image, by default None.
    overwrite : bool, optional
        Whether to overwrite the output file if it exists, by default False.
    background_rel_weight : float, optional
        Relative weight for background in histogram matching, by default 0.1.
    num_bins : int, optional
        Number of bins for histogram matching, by default 512.

    Returns
    -------
    NiftiGz
        Normalized image.
    """

    if output_image.exists() and not overwrite:
        logger.info(f"Output file exists {output_image}. Reusing.")
        return NiftiGz(output_image)

    logger.info(f"Processing: {image}")

    reduced_label_map = workflow.add(
        reduce_label_map(label_map=label_map, intervals=label_intervals),
        name="reduce_label_map",
    )

    reduced_ref_label_map = workflow.add(
        reduce_label_map(label_map=ref_label_map, intervals=label_intervals),
        name="reduced_reference_label_map",
    )

    normalized = workflow.add(
        multiple_regions_histogram_matching(
            src_image=image,
            src_label_map=reduced_label_map.file,
            ref_image=ref_image,
            ref_label_map=reduced_ref_label_map.file,
            src_mask=mask,
            ref_mask=ref_mask,
            bg_rel_weight=background_rel_weight,
            num_bins=num_bins,
        ),
        name="multiple_regions_histogram_matching",
    )

    published = workflow.add(
        publish_derivative(
            file=normalized.file, destination=output_image, overwrite=True
        ),
        name="Publish_filtered_image",
    )

    return published.file


@workflow.define()
def multiple_regions_histogram_matching_normalize_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    segmentation_label_intervals: list[tuple[int, int]],
    reference_image: NiftiGz,
    reference_label_map: NiftiGz,
    mask_query: str | None = None,
    reference_image_mask: NiftiGz | None = None,
    load_sidecars: bool = False,
    overwrite: bool = False,
    background_rel_weight: float = 0.1,
    num_bins: int = 512,
) -> list[NiftiGz]:
    """
    Normalize a dataset using multiple regions histogram matching.

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
    segmentation_label_intervals : list[tuple[int, int]]
        List of label intervals for segmentation.
    reference_image : NiftiGz
        Reference image for histogram matching.
    reference_label_map : NiftiGz
        Reference label map for histogram matching.
    mask_query : str | None, optional
        Query to find masks for the images, by default None.
    reference_image_mask : NiftiGz | None, optional
        Mask for the reference image, by default None.
    load_sidecars : bool, optional
        Whether to load sidecar files, by default False.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.
    background_rel_weight : float, optional
        Relative weight for background in histogram matching, by default 0.1.
    num_bins : int, optional
        Number of bins for histogram matching, by default 512.

    Returns
    -------
    list[NiftiGz]
        List of normalized images.
    """

    mids_index = workflow.add(
        index(
            dataset_root=dataset_root,
            include_derivative=True,
            load_sidecars=load_sidecars,
        ),
        name="index_mids",
    )

    initialized_derivative = workflow.add(
        initialize_derivative(
            dataset_root=dataset_root,
            derivative_name=output_derivative_name,
            pipeline_name="Scale Normalized - spinemira",
            pipeline_version=version(spinemira.__package__),
        ),
        name="initialize_derivative",
    )

    images = workflow.add(
        query_mids(
            dataset_root=dataset_root,
            query=image_query,
            mids_index=mids_index.file,
        ),
        name="query_dataset",
    )

    label_maps = workflow.add(
        find_indexed_derivative(
            dataset_root=dataset_root,
            mids_index=mids_index.file,
            flt=label_map_query,
        )
        .split(file=images.files)
        .combine("file"),
        name="find_label_maps",
    )

    if mask_query is not None:
        masks = workflow.add(
            find_indexed_derivative(
                dataset_root=dataset_root,
                mids_index=mids_index,
                flt=mask_query,
            )
            .split(file=images.files)
            .combine("file"),
            name="find_masks",
        )
    else:
        masks = None

    out_images = workflow.add(
        resolve_derivative(derivative_folder=initialized_derivative.path)
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_paths",
    )

    # Prepare split arguments dynamically to handle optional mask
    split_vars = ["image", "label_map", "output_image"]
    split_args = {
        "image": images.files,
        "label_map": label_maps.file,
        "output_image": out_images.path,
    }

    if masks is not None:
        split_vars.append("mask")
        split_args["mask"] = masks.file

    normalized = workflow.add(
        multiple_regions_histogram_matching_single_image_workflow(
            ref_image=reference_image,
            ref_label_map=reference_label_map,
            ref_mask=reference_image_mask,
            label_intervals=segmentation_label_intervals,
            overwrite=overwrite,
            background_rel_weight=background_rel_weight,
            num_bins=num_bins,
        )
        .split(tuple(split_vars), **split_args)
        .combine("image"),
        name="process_single_image",
    )

    return normalized.file
