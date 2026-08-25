from pathlib import Path
from typing import TypedDict

from fileformats.application import Json
from fileformats.medimage import NiftiGz
import numpy as np
from pydra.compose import python
import SimpleITK as sitk
import json

from spinemira.core.analysis.segmentation import (
    multiple_labels_intensity_statistics,
)
from spinemira.core.directions import LPS
from spinemira.core.filters import filter_label_map
from spinemira.core.segmentation.utils import split_label_along_principal_axis


class DiscSignalIntensityStatistics(TypedDict):
    """
    Container for signal intensity statistics of a single disc.

    Attributes
    ----------
    mean : list[float]
        List of mean intensity values for each sub-region of the disc.
    median : list[float]
        List of median intensity values for each sub-region of the disc.
    iqr : list[float]
        List of interquartile range (IQR) values for each sub-region of the disc.
    min : list[float]
        List of minimum intensity values for each sub-region of the disc.
    max : list[float]
        List of maximum intensity values for each sub-region of the disc.
    """

    mean: list[float]
    median: list[float]
    iqr: list[float]
    min: list[float]
    max: list[float]


class DiscsSignalIntensityStatistics(TypedDict):
    """
    Container for signal intensity statistics of multiple discs.

    Attributes
    ----------
    split_direction : tuple[float, float, float]
        Direction vector (x, y, z) in physical SimpleITK coordinates used for splitting.
    split_plane_normal : tuple[float, float, float]
        Normal vector (x, y, z) of the plane in physical SimpleITK coordinates used for splitting.
    max_plane_distance : float | None
        Maximum distance from the principal axis to consider for splitting, or None for no limit.
    statistics : dict[str, DiscSignalIntensityStatistics]
        Dictionary mapping disc labels to their respective signal intensity statistics.
    """

    split_direction: tuple[float, float, float]
    split_plane_normal: tuple[float, float, float]
    max_plane_distance: float | None
    statistics: dict[str, DiscSignalIntensityStatistics]


@python.define(outputs=["json"])
def calc_disc_signal_intensity_profile(
    image: NiftiGz,
    label_map: NiftiGz,
    disc_labels: dict[str, int],
    n_parts: int,
    split_direction: tuple[float, float, float] = LPS.ANTERIOR_TO_POSTERIOR.value,
    split_plane_normal: tuple[float, float, float] = LPS.LEFT_TO_RIGHT.value,
    max_plane_distance: float | None = 5,
) -> Json:
    """
    Calculate signal intensity statistics for discs split into sub-regions along a specified direction.

    This function is a Pydra task wrapper that computes intensity statistics for each disc in the provided
    label map. Each disc is split into `n_parts` sub-regions along the principal inertia axis closest to
    `split_direction`, and statistics (mean, median, IQR, min, max) are calculated for each sub-region.

    Parameters
    ----------
    image : NiftiGz
        Input image file (NIfTI format, gzipped) from which intensity values are extracted.
    label_map : NiftiGz
        Input label map file (NIfTI format, gzipped) containing labeled discs.
    disc_labels : dict[str, int]
        Dictionary mapping disc names (e.g., "L1_L2") to their corresponding integer labels in the label map.
    n_parts : int
        Number of sub-regions to split each disc into.
    split_direction : tuple[float, float, float], optional
        Direction vector (x, y, z) in physical SimpleITK coordinates to split the disc along.
        Defaults to LPS.ANTERIOR_TO_POSTERIOR (i.e., [0, 1, 0]).
        For a standard LPS-oriented SimpleITK image:
            Left -> Right      = [-1,  0,  0]
            Right -> Left      = [ 1,  0,  0]
            Posterior -> Anterior = [0, -1,  0]
            Anterior -> Posterior = [0,  1,  0]
            Inferior -> Superior = [0,  0,  1]
            Superior -> Inferior = [0,  0, -1]
    split_plane_normal : tuple[float, float, float], optional
        Normal vector (x, y, z) of the plane in physical SimpleITK coordinates.
        If specified, the principal axis for splitting is calculated within this plane.
        Defaults to LPS.LEFT_TO_RIGHT (i.e., [-1, 0, 0]).
    max_plane_distance : float | None, optional
        Maximum distance from the principal axis to consider for splitting, in physical units.
        If None, no distance limit is applied. Defaults to 5.

    Returns
    -------
    Json
        Path to a JSON file containing the computed statistics for all discs.
        The JSON structure follows the `DiscsSignalIntensityStatistics` schema.

    See Also
    --------
    spinemira.core.segmentation.utils.split_label_along_principal_axis : Core function for splitting labels.
    spinemira.core.analysis.segmentation.multiple_labels_intensity_statistics : Core function for computing statistics.
    """

    image_sitk = sitk.ReadImage(image.fspath)
    label_map_sitk = sitk.ReadImage(label_map.fspath)

    sub_region_labels = list(range(1, n_parts + 1))
    statistics: dict[str, DiscSignalIntensityStatistics] = {}

    for label_name, label in disc_labels.items():
        disc_label_map_sitk = filter_label_map(
            label_map=label_map_sitk, labels=set([label])
        )

        if not np.any(sitk.GetArrayViewFromImage(disc_label_map_sitk)):
            continue

        splitted_label_map, _ = split_label_along_principal_axis(
            label_map=disc_label_map_sitk,
            n_parts=n_parts,
            target_direction=split_direction,
            plane_normal=split_plane_normal,
            max_plane_distance=max_plane_distance,
        )

        # For each region in the splitted label map, calculate statistics.
        stat = multiple_labels_intensity_statistics(
            image=image_sitk,
            label_map=splitted_label_map,
            labels=sub_region_labels,
        )

        statistics[label_name] = {
            "mean": [stat[label]["mean"] for label in sub_region_labels],
            "median": [stat[label]["median"] for label in sub_region_labels],
            "iqr": [stat[label]["iqr"] for label in sub_region_labels],
            "min": [stat[label]["min"] for label in sub_region_labels],
            "max": [stat[label]["max"] for label in sub_region_labels],
        }

    discs_signal_intensity_statistics = DiscsSignalIntensityStatistics(
        split_direction=split_direction,
        split_plane_normal=split_plane_normal,
        max_plane_distance=max_plane_distance,
        statistics=statistics,
    )

    result_file = Path.cwd() / "discs_signal_intensity_statistics.json"
    result_file.write_text(
        json.dumps(discs_signal_intensity_statistics, indent=2), encoding="utf-8"
    )

    return Json(result_file)
