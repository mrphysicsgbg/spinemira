from importlib.metadata import version
import logging
from pathlib import Path

from fileformats.medimage import NiftiGz
from pydra.compose import workflow

import spinemira
from spinemira.tasks.mids import (
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
    resolve_derivative,
)
from spinemira.tasks.segmentation import run_totalspineseg


logger = logging.getLogger(__name__)


@workflow.define(outputs=["label_map", "levels"])
def segment_single_image_workflow(
    image: NiftiGz,
    output_label_map: Path,
    output_levels: Path,
    totalspineseg_data_dir: Path,
    overwrite: bool,
    device: str,
    totalspineseg_quiet: bool,
) -> tuple[NiftiGz, NiftiGz]:

    if output_label_map.exists() and output_levels.exists() and not overwrite:
        logger.info(
            f"Output files exits {output_label_map} and {output_levels}. Reusing."
        )
        return NiftiGz(output_label_map), NiftiGz(output_levels)

    segmented = workflow.add(
        run_totalspineseg(
            image=image,
            data_dir=totalspineseg_data_dir,
            device=device,
            quiet=totalspineseg_quiet,
        ),
        name="run_segmentation",
    )

    published_label_map = workflow.add(
        publish_derivative(
            file=segmented.label_map, destination=output_label_map, overwrite=True
        ),
        name="publish_label_map",
    )

    published_levels = workflow.add(
        publish_derivative(
            file=segmented.levels, destination=output_levels, overwrite=True
        ),
        name="publish_levels",
    )

    return published_label_map.file, published_levels.file


@workflow.define(outputs=["label_map", "levels"])
def segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str,
    totalspineseg_data_dir: Path,
    overwrite: bool = False,
    device: str = "cpu",
    totalspineseg_quiet: bool = False,
):
    """
    Create a workflow for segmenting a dataset using TotalSpineSeg.

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

    Returns
    -------
    tuple[Any, Any]
        A tuple containing the label map and levels outputs from the workflow.
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
            pipeline_name="TotalSpineSeg - spinemira",
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

    out_label_maps = workflow.add(
        resolve_derivative(derivative_folder=initialized_derivative.path, suffix="dseg")
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_label_maps",
    )

    out_levels = workflow.add(
        resolve_derivative(
            derivative_folder=initialized_derivative.path, suffix="levels"
        )
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_levels",
    )

    segmented = workflow.add(
        segment_single_image_workflow(
            totalspineseg_data_dir=totalspineseg_data_dir,
            totalspineseg_quiet=totalspineseg_quiet,
            device=device,
            overwrite=overwrite,
        )
        .split(
            ("image", "output_label_map", "output_levels"),
            image=images.files,
            output_label_map=out_label_maps.path,
            output_levels=out_levels.path,
        )
        .combine("image"),
        name="process_single_image",
    )

    return segmented.label_map, segmented.levels
