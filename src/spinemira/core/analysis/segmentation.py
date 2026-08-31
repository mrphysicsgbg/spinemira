from typing import TypedDict

import numpy as np
import SimpleITK as sitk


class IntensityStatistics(TypedDict):
    mean: float
    median: float
    iqr: float
    min: float
    max: float


def multiple_labels_intensity_statistics(
    image: sitk.Image, label_map: sitk.Image, labels: list[int] | dict[str, int]
) -> dict[str | int, IntensityStatistics]:
    """Calculate intensity statistics for multiple segmented structures.

    Parameters
    ----------
    image : sitk.Image
        The input image containing intensity values.
    label_map : sitk.Image
        The label map image where segmented structures are labeled.
    labels : list[int] | dict[str, int]
        Either a list of integer label values, or a dictionary mapping label names to label values.
        If a list is provided, the output dictionary keys will be the label values.
        If a dictionary is provided, the output dictionary keys will be the label names.

    Returns
    -------
    dict[str | int, IntensityStatistics]
        A dictionary mapping label identifiers to their corresponding IntensityStatistics.
        The keys are either label names (if dict input) or label values (if list input).
    """
    result: dict[str | int, IntensityStatistics] = {}

    if isinstance(labels, dict):
        # Dictionary input: use label names as keys
        for label_name, label_value in labels.items():
            stat = intensity_statistics(image, label_map, label_value)
            if stat is not None:
                result[label_name] = stat
    else:
        # List input: use label values as keys
        for label_value in labels:
            stat = intensity_statistics(image, label_map, label_value)
            if stat is not None:
                result[label_value] = stat

    return result


def intensity_statistics(
    image: sitk.Image, label_map: sitk.Image, label: int
) -> IntensityStatistics | None:
    """Calculate intensity statistics for a segmented structure.

    Parameters
    ----------
    image : sitk.Image
        The input image containing intensity values.
    label_map : sitk.Image
        The label map image where segmented structures are labeled.
    label : int
        The label value identifying the structure of interest.

    Returns
    -------
    IntensityStatistics | None
        A dataclass containing the calculated statistics:
        - mean: Mean intensity of the segmented structure
        - median: Median intensity of the segmented structure
        - iqr: Interquartile range (75th percentile - 25th percentile)
        - min: Minimum intensity value in the segmented structure
        - max: Maximum intensity value in the segmented structure
        If the label is not present in the label map, then None is returned.
    """
    image_array = sitk.GetArrayFromImage(image)
    label_array = sitk.GetArrayFromImage(label_map)

    # Extract intensities where label matches
    intensities = image_array[label_array == label]

    if len(intensities) == 0:
        return None

    # Calculate statistics
    mean = float(np.mean(intensities))
    median = float(np.median(intensities))
    q75, q25 = np.percentile(intensities, [75, 25])
    iqr = float(q75 - q25)
    min_val = float(np.min(intensities))
    max_val = float(np.max(intensities))

    return IntensityStatistics(
        mean=mean, median=median, iqr=iqr, min=min_val, max=max_val
    )
