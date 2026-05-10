import SimpleITK as sitk
import numpy as np


def dice(
    target_label_map: sitk.Image,
    source_label_map: sitk.Image,
    labels_to_evaluate: set[int] | None = None,
) -> dict[str, float]:
    """Calculate DICE scores for specified label maps

    Parameters
    ----------
    target_label_map : sitk.Image
        Target label map
    source_label_map : sitk.Image
        Source target map
    labels_to_evaluate : set[int]
        Optional set of labels to evaluate

    Returns
    -------
    dict[str, float]
        Dictionary with labels as keys and DICE score as value
    """

    def dsc(source: np.ndarray, target: np.ndarray) -> float:
        return 2 * np.logical_and(source, target).sum() / (source.sum() + target.sum())

    target_array = sitk.GetArrayFromImage(target_label_map)
    source_array = sitk.GetArrayFromImage(source_label_map)

    score = {}

    labels_in_target_array = np.unique(target_array)
    labels_in_source_array = np.unique(source_array)
    labels_common = set(np.intersect1d(labels_in_target_array, labels_in_source_array))

    if 0 in labels_common:
        labels_common.remove(0)

    if labels_to_evaluate:
        labels_common = labels_common.intersection(labels_to_evaluate)

    weighted_dice_sum = 0
    total_weight = 0

    for label in labels_common:
        target = target_array == label
        source = source_array == label

        dice_score = dsc(source, target)
        voxel_count = target.sum()

        score[str(label)] = dice_score
        weighted_dice_sum += dice_score * voxel_count
        total_weight += voxel_count

    target = np.isin(target_array, list(labels_common))
    source = np.isin(source_array, list(labels_common))

    score["overall"] = weighted_dice_sum / total_weight if total_weight > 0 else 1.0

    return score
