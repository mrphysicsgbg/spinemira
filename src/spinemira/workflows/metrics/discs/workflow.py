from importlib.metadata import version
import logging
from pathlib import Path

from fileformats.application import Json
from fileformats.medimage import NiftiGz
from pydra.compose import workflow

import spinemira
from spinemira.core.directions import LPS
from spinemira.tasks.discs import calc_disc_signal_intensity_profile, calc_disc_delta_mu
from spinemira.tasks.mids import (
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
    find_indexed_derivative,
    resolve_derivative,
)
from spinemira.tasks.utils import make_merge_json_files_task


logger = logging.getLogger(__name__)


@workflow.define(outputs=["file"])
def calc_disc_metrics_single_image_workflow(
    image: NiftiGz,
    label_map: NiftiGz,
    disc_labels: dict[str, int],
    output_json: Path,
    calc_signal_profile: bool = True,
    calc_delta_mu: bool = True,
    n_parts: int = 5,
    split_direction: tuple[float, float, float] = LPS.ANTERIOR_TO_POSTERIOR.value,
    split_plane_normal: tuple[float, float, float] = LPS.LEFT_TO_RIGHT.value,
    max_plane_distance: float | None = 5,
    overwrite: bool = False,
) -> Json:
    """
    Create a workflow for calculating disc signal intensity metrics for a single image.

    This workflow computes signal intensity statistics (mean, median, IQR, min, max) for each disc
    in the provided label map. Each disc is split into `n_parts` sub-regions along a specified direction.
    The delta mu metric is based on the methodology described in [1]_.

    Parameters
    ----------
    image : NiftiGz
        Input image file from which intensity values are extracted.
    label_map : NiftiGz
        Input label map file containing labeled discs.
    disc_labels : dict[str, int]
        Dictionary mapping disc names (e.g., "L1_L2") to their corresponding integer labels in the label map.
    output_json : Path
        Output path for the JSON file containing the computed statistics.
    calc_signal_profile : bool, by default True.
        Wether to calculate disc signal profile.
    calc_delta_mu : bool, by default True.
        Wether to calculate delta mu.
    n_parts : int, optional
        Number of sub-regions to split each disc into, by default 5.
    split_direction : tuple[float, float, float], optional
        Direction vector (x, y, z) in physical SimpleITK coordinates to split the disc along.
        Defaults to LPS.ANTERIOR_TO_POSTERIOR (i.e., [0, 1, 0]).
    split_plane_normal : tuple[float, float, float], optional
        Normal vector (x, y, z) of the plane in physical SimpleITK coordinates.
        Defaults to LPS.LEFT_TO_RIGHT (i.e., [-1, 0, 0]).
    max_plane_distance : float | None, optional
        Maximum distance from the principal axis to consider for splitting, in physical units.
        If None, no distance limit is applied. Defaults to 5.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.

    Returns
    -------
    Json
        Path to the JSON file containing the computed disc signal intensity statistics.

    References
    ----------
    .. [1] Waldenberg, Christian & Hebelka, Hanna & Brisby, Helena & Lagerstrand, Kerstin.
        (2018). MRI histogram analysis enables objective and continuous classification of
        intervertebral disc degeneration. European Spine Journal. 27.
        10.1007/s00586-017-5264-7.
    """

    if output_json.exists() and not overwrite:
        logger.info(f"Output file exists {output_json}. Reusing.")
        return Json(output_json)

    logger.info(f"Processing: {image}")

    json_outputs_to_merge = {}

    if calc_signal_profile:
        calculated_disc_signal_intensity_profile = workflow.add(
            calc_disc_signal_intensity_profile(
                image=image,
                label_map=label_map,
                disc_labels=disc_labels,
                n_parts=n_parts,
                split_direction=split_direction,
                split_plane_normal=split_plane_normal,
                max_plane_distance=max_plane_distance,
            ),
            name="calc_disc_signal_intensity_profile",
        )
        json_outputs_to_merge["calc_disc_signal_profile"] = (
            calculated_disc_signal_intensity_profile.json
        )

    if calc_delta_mu:
        calculated_disc_delta_mu = workflow.add(
            calc_disc_delta_mu(
                image=image, label_map=label_map, disc_labels=disc_labels
            ),
            name="calc_disc_delta_mu",
        )
        json_outputs_to_merge["delta_mu"] = calculated_disc_delta_mu.json

    merge_json_files_task = make_merge_json_files_task(*json_outputs_to_merge.keys())

    merged_json_files = workflow.add(
        merge_json_files_task(**json_outputs_to_merge), name="merge_json_files"
    )

    published = workflow.add(
        publish_derivative(
            file=merged_json_files.json,
            destination=output_json,
            overwrite=True,
        ),
        name="publish_calculated_disc_metrics",
    )

    return published.file


@workflow.define(outputs=["file"])
def calc_disc_metrics_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    label_map_query: str,
    output_derivative_name: str,
    disc_labels: dict[str, int],
    load_sidecars: bool = False,
    calc_signal_profile: bool = True,
    calc_delta_mu: bool = True,
    n_parts: int = 5,
    split_direction: tuple[float, float, float] = LPS.ANTERIOR_TO_POSTERIOR.value,
    split_plane_normal: tuple[float, float, float] = LPS.LEFT_TO_RIGHT.value,
    max_plane_distance: float | None = 5,
    overwrite: bool = False,
) -> list[Json]:
    """
    Create a workflow for calculating disc signal intensity metrics for a dataset.

    This workflow processes all images matching the provided `image_query` in the dataset,
    computes signal intensity statistics for each disc in the corresponding label maps, and
    saves the results as JSON files in a MIDS derivative folder.
    The delta mu metric is based on the methodology described in [1]_.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    image_query : str
        Base query string to filter images for processing.
    label_map_query : str
        Query string to filter label maps corresponding to the images.
    output_derivative_name : str
        Name of the output derivative folder where results will be saved.
    disc_labels : dict[str, int]
        Dictionary mapping disc names (e.g., "L1_L2") to their corresponding integer labels in the label map.
    load_sidecars : bool, optional
        Whether to load sidecar files when indexing the dataset, by default False.
    calc_signal_profile : bool, by default True.
        Wether to calculate disc signal profile.
    calc_delta_mu : bool, by default True.
        Wether to calculate delta mu.
    n_parts : int, optional
        Number of sub-regions to split each disc into, by default 5.
    split_direction : tuple[float, float, float], optional
        Direction vector (x, y, z) in physical SimpleITK coordinates to split the disc along.
        Defaults to LPS.ANTERIOR_TO_POSTERIOR (i.e., [0, 1, 0]).
    split_plane_normal : tuple[float, float, float], optional
        Normal vector (x, y, z) of the plane in physical SimpleITK coordinates.
        Defaults to LPS.LEFT_TO_RIGHT (i.e., [-1, 0, 0]).
    max_plane_distance : float | None, optional
        Maximum distance from the principal axis to consider for splitting, in physical units.
        If None, no distance limit is applied. Defaults to 5.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.

    Returns
    -------
    list[Json]
        List of paths to the JSON files containing the computed disc signal intensity statistics for each image.

    References
    ----------
    .. [1] Waldenberg, Christian & Hebelka, Hanna & Brisby, Helena & Lagerstrand, Kerstin.
        (2018). MRI histogram analysis enables objective and continuous classification of
        intervertebral disc degeneration. European Spine Journal. 27.
        10.1007/s00586-017-5264-7.
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
            pipeline_name="Disc metrics - spinemira",
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

    out_metrics = workflow.add(
        resolve_derivative(derivative_folder=initialized_derivative.path)
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_paths",
    )

    calculated_metrics = workflow.add(
        calc_disc_metrics_single_image_workflow(
            overwrite=overwrite,
            disc_labels=disc_labels,
            calc_signal_profile=calc_signal_profile,
            calc_delta_mu=calc_delta_mu,
            n_parts=n_parts,
            split_direction=split_direction,
            split_plane_normal=split_plane_normal,
            max_plane_distance=max_plane_distance,
        )
        .split(
            ("image", "label_map", "output_json"),
            image=images.files,
            label_map=label_maps.file,
            output_json=out_metrics.path,
        )
        .combine("image"),
        name="calc_disc_metrics_single_image_workflow",
    )

    return calculated_metrics.file
