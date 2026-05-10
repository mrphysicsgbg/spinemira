import SimpleITK as sitk
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def get_common_labels(label_map_1: sitk.Image, label_map_2: sitk.Image) -> set[int]:
    """
    Find common non-zero labels between two label maps.

    Parameters
    ----------
    label_map_1 : sitk.Image
        First label map.
    label_map_2 : sitk.Image
        Second label map.

    Returns
    -------
    set[int]
        Set of common labels excluding background (0).
    """
    labels1 = set(np.unique(sitk.GetArrayViewFromImage(label_map_1)).tolist())
    labels2 = set(np.unique(sitk.GetArrayViewFromImage(label_map_2)).tolist())
    common_labels = labels1 & labels2
    common_labels.discard(0)
    return common_labels


def get_centroid_coordinates(
    label_map: sitk.Image, labels: set[int] | None = None
) -> list[tuple[float, float, float]]:
    """Get coordinates of levels in the label map

    Parameters
    ----------
    label_map : sitk.Image
        Levels label map
    labels : set[int]
        Set of labels to get coordinates for

    Returns
    -------
    list[tuple[float, float, float]]
        List of coordinates for each label
    """
    label_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_image_filter.Execute(label_map)

    coordinates = []

    for label in label_image_filter.GetLabels():
        if labels and label not in labels:
            continue

        coordinates.append(label_image_filter.GetCentroid(label))

    return coordinates


def get_level_coordinates(
    levels_label_map: sitk.Image, labels: set[int] | None = None
) -> list[tuple[float, float, float]]:
    """Get coordinates of levels in the label map

    Parameters
    ----------
    levels_label_map : sitk.Image
        Levels label map
    labels : set[int]
        Set of labels to get coordinates for

    Returns
    -------
    list[tuple[float, float, float]]
        List of coordinates for each label
    """
    label_image_filter = sitk.LabelShapeStatisticsImageFilter()
    label_image_filter.Execute(levels_label_map)

    coordinates = []

    for label in label_image_filter.GetLabels():
        if labels and label not in labels:
            continue

        coordinates.append(label_image_filter.GetCentroid(label))

    return coordinates


def get_roi_mask(
    label_mask, dilate_radius: tuple[int, ...] | None = None
) -> sitk.Image:
    """Calculates ROI using Convex Hull.

    Parameters
    ----------
    label_mask : sitk.Image
        Label mask
    dilate_radius : tuple[int, ...] | None
        Number of voxels to dilate mask with, default None

    Returns
    -------
    sitk.Image
        ROI mask
    """
    label_mask_array = sitk.GetArrayViewFromImage(label_mask)

    # Get the coordinates of the non-zero points in the binary mask
    points = np.argwhere(label_mask_array > 0)

    # Compute the convex hull
    hull = ConvexHull(points)

    # Use Delaunay triangulation for point-in-hull testing
    delaunay = Delaunay(points[hull.vertices])

    roi_mask_arr = np.zeros_like(label_mask_array, dtype=bool)

    # Generate all possible points within the mask's bounding box
    x, y, z = np.indices(label_mask_array.shape)
    grid_points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)

    # Check which points are inside the convex hull
    inside_hull = delaunay.find_simplex(grid_points) >= 0

    # Map the results back into the 3D space
    roi_mask_arr[
        grid_points[inside_hull, 0],
        grid_points[inside_hull, 1],
        grid_points[inside_hull, 2],
    ] = True

    roi_mask = sitk.GetImageFromArray(roi_mask_arr.astype(np.uint8))
    roi_mask.CopyInformation(label_mask)

    if dilate_radius:
        roi_mask = sitk.BinaryDilate(roi_mask, dilate_radius)

    return sitk.Cast(roi_mask, sitk.sitkFloat32)


def filter_label_map(label_map: sitk.Image, labels: set[int]) -> sitk.Image:
    """
    Filter a mask by retaining only the specified labels.

    Parameters
    ----------
    mask : sitk.Image
        Input label map.
    labels : set[int]
        Labels to retain.

    Returns
    -------
    sitk.Image
        Filtered mask containing only the specified labels.
    """
    filtered = sitk.Image(label_map.GetSize(), label_map.GetPixelID())
    filtered.CopyInformation(label_map)

    for label in labels:
        binary = sitk.BinaryThreshold(label_map, label, label, label, 0)
        filtered = sitk.Add(filtered, binary)

    return filtered
