from importlib.metadata import version
import logging
from pathlib import Path

from fileformats.medimage import NiftiGz
from pydra.compose import workflow

import spinemira
from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.tasks.filters import normalize_to_label_intensity_mode
from spinemira.tasks.mids import (
    find_indexed_derivative,
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
    resolve_derivative,
)


logger = logging.getLogger(__name__)


@workflow.define()
def scale_normalize_single_image_workflow(
    image: NiftiGz,
    label_map: NiftiGz,
    label: int,
    output_image: Path,
    nbins: int,
    overwrite: bool,
) -> NiftiGz:

    if output_image.exists() and not overwrite:
        logger.info(f"Output file exists {output_image}. Reusing.")
        return NiftiGz(output_image)

    logger.info(f"Processing: {image}")

    normalized = workflow.add(
        normalize_to_label_intensity_mode(
            image=image,
            label_map=label_map,
            label=label,
            nbins=nbins,
        ),
        name="normalize_to_label_intensity_mode",
    )

    published_image = workflow.add(
        publish_derivative(
            file=normalized.file, destination=output_image, overwrite=True
        ),
        name="publish_filtered_image",
    )

    return published_image.file


@workflow.define()
def scale_normalize_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    overwrite: bool = False,
    label: int = TotalSpineSegLabels.SPINAL_CANAL,
    nbins: int = 1024,
) -> list[NiftiGz]:
    """
    Create a workflow for scale normalize a dataset.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    image_query : str
        Base query string to filter images.
    output_derivative_name : str
        Name of the output derivative folder.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.

    Return
    ------
    list[NiftiGz]
        List of normalized image.
    """

    mids_index = workflow.add(
        index(
            dataset_root=dataset_root,
            include_derivative=True,
            load_sidecars=True,
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

    out_images = workflow.add(
        resolve_derivative(derivative_folder=initialized_derivative.path)
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_paths",
    )

    normalized = workflow.add(
        scale_normalize_single_image_workflow(
            label=label, overwrite=overwrite, nbins=nbins
        )
        .split(
            ("image", "label_map", "output_image"),
            image=images.files,
            label_map=label_maps.file,
            output_image=out_images.path,
        )
        .combine("image"),
        name="process_single_image",
    )

    return normalized.out
