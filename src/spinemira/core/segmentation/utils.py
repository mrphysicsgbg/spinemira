import SimpleITK as sitk
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import inertia_tensor


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


def split_label_along_principal_axis(
    label_map: sitk.Image,
    n_parts: int,
    target_direction: np.typing.ArrayLike,
    plane_normal: np.typing.ArrayLike | None = None,
    max_plane_distance: float | None = None,
) -> tuple[sitk.Image, np.ndarray]:
    """
    Split the non-zero region of a 3D SimpleITK image into n equal-length
    parts along the principal inertia axis closest to target_direction.

    Parameters
    ----------
    label_map : sitk.Image
        SimpleITK image. All non-zero voxels are treated as one region.

    n_parts : int
        Number of sections.

    target_direction :  np.typing.ArrayLike, shape (3,)
        Direction in physical SimpleITK coordinates, i.e. (x, y, z).

        For a standard LPS-oriented SimpleITK image:
            Left -> Right      = [-1,  0,  0]
            Right -> Left      = [ 1,  0,  0]
            Posterior -> Anterior = [0, -1,  0]
            Anterior -> Posterior = [0,  1,  0]
            Inferior -> Superior = [0,  0,  1]
            Superior -> Inferior = [0,  0, -1]

    plane_normal : np.typing.ArrayLike | None, optional
        Optional vector (3, ) defining the normal of a plane in physical SimpleIKT coordinates, i.e. (x, y, z).
        If specified, then the principal axis to perform the split along will  be calculated in the plane.

    max_plane_distance : float | None, optional
        If specified along with plane_normal, voxels more than this many physical units away from the normal plane
        will be masked.

    Returns
    -------
    result : sitk.Image
        Integer image with:
            0 = background
            1 ... n_parts = sections

    axis : np.ndarray, shape (3,)
        Selected principal axis in physical (x, y, z) coordinates.
    """
    if label_map.GetDimension() != 3:
        raise ValueError("label_map must be 3D.")

    if n_parts < 1 or n_parts > 255:
        raise ValueError("n_parts must be >= 1 and <= 255.")

    # Normalize input vectors
    target = np.asarray(target_direction, dtype=float)

    if target.shape != (3,):
        raise ValueError("target_direction must have length 3.")

    target /= np.linalg.norm(target)

    if plane_normal is not None:
        plane_normal = np.asarray(plane_normal, dtype=float)

        if plane_normal.shape != (3,):
            raise ValueError("plane_normal must have length 3.")

        plane_normal /= np.linalg.norm(plane_normal)

    # SimpleITK -> NumPy is [z, y, x]
    label_map_arr = sitk.GetArrayFromImage(label_map)
    segmentation = label_map_arr != 0

    if not np.any(segmentation):
        raise ValueError("label_image contains no non-zero voxels.")

    # Compute inertia tensor in physical-length units
    sx, sy, sz = label_map.GetSpacing()
    spacing_zyx = (sz, sy, sx)

    inertia_tensor_zyx = inertia_tensor(
        segmentation.astype(np.float32), spacing=spacing_zyx
    )

    _, eigenvectors_zyx = np.linalg.eigh(inertia_tensor_zyx)

    # Convert [z,y,x] -> [x,y,z], eigenvectors are stored as columns
    eigenvectors_xyz = eigenvectors_zyx[::-1, :]

    # Convert index-coordinate directions into physical SimpleITK directions
    direction_matrix = np.asarray(label_map.GetDirection()).reshape(3, 3)

    eigenvectors_phys = direction_matrix @ eigenvectors_xyz
    eigenvectors_phys /= np.linalg.norm(eigenvectors_phys, axis=0, keepdims=True)

    alignment = np.abs(eigenvectors_phys.T @ target)
    best_idx = np.argmax(alignment)
    chosen_axis = eigenvectors_phys[:, best_idx]

    # Eigenvectors have arbitrary sign.
    # Orient it along the requested direction.
    if (chosen_axis @ target) < 0:
        chosen_axis = -chosen_axis

    # Check if axis should be calculated in plane
    if plane_normal is not None:
        plane_normal = np.asarray(plane_normal, dtype=float)

        if plane_normal.shape != (3,):
            raise ValueError("plane_normal must have length 3.")

        chosen_axis = chosen_axis - (plane_normal * (chosen_axis @ plane_normal))

    # Get physical coordinates of every voxel in the structure
    zyx = np.argwhere(segmentation)

    # Convert [z,y,x] -> [x,y,z]
    xyz_index = zyx[:, ::-1].astype(float)

    spacing_xyz = np.asarray(label_map.GetSpacing())
    origin = np.asarray(label_map.GetOrigin())

    physical_points = origin + (direction_matrix @ (xyz_index * spacing_xyz).T).T
    centroid = physical_points.mean(axis=0)

    physical_points_vector = physical_points - centroid

    # Check if region should be masked around the plane
    if plane_normal is not None and max_plane_distance is not None:
        distance_to_plane = np.abs(physical_points_vector @ plane_normal)
        mask = distance_to_plane <= max_plane_distance

        physical_points_vector = physical_points_vector[mask]
        zyx = zyx[mask]

    # Project all points on the in plane axis
    projection = physical_points_vector @ chosen_axis

    # Calculate edges for the different sectors
    edges = np.linspace(
        projection.min(),
        projection.max(),
        n_parts + 1,
    )

    part_ids = np.digitize(projection, edges[1:-1]) + 1

    # Create divided segmentation by assigning label ids for each region
    result_arr = np.zeros(segmentation.shape, dtype=np.uint16)
    result_arr[zyx[:, 0], zyx[:, 1], zyx[:, 2]] = part_ids

    result = sitk.GetImageFromArray(result_arr)
    result = sitk.Cast(result, sitk.sitkUInt8)
    result.CopyInformation(label_map)

    return result, chosen_axis


def roi_inner_most(roi: sitk.Image, axis: int = 2, count: int = 3) -> sitk.Image:
    """Get mask of the inner most slices along axis.

    Number of slices returned will be equal around the center. Example if count = 3, but the range is 10 slices,
    then four non-zero slices will be returned.

    Parameters
    ----------
    roi : sitk.Image
        N-dimensional input image
    axis : int, optional
        Axis to return slices along, by default 2
    count : int, optional
        Number of slices to be included, by default 3

    Returns
    -------
    sitk.Image
        Output N-dimensional image
    """

    roi_arr = sitk.GetArrayViewFromImage(roi)

    roi_rolled = np.moveaxis(roi_arr, axis, 0)

    nonzero = np.argwhere(roi_rolled)
    if nonzero.size == 0:
        return roi  # Return original if there's no nonzero region

    bounds_min, bounds_max = nonzero[:, 0].min(), nonzero[:, 0].max()
    bounds_range = bounds_max - bounds_min + 1  # Inclusive range

    # Ensure count isn't larger than bounds_range
    count = min(count, bounds_range)

    padding = (bounds_range - count) // 2
    start, end = (
        bounds_min + padding,
        bounds_max - padding + 1,
    )  # +1 for inclusive slicing

    # Safety check for valid index range
    start, end = np.clip([start, end], 0, roi_rolled.shape[0])

    # Create the new array and update only the inner region
    roi_inner_most_rolled = np.zeros_like(roi_rolled)
    roi_inner_most_rolled[start:end] = roi_rolled[start:end]

    roi_inner_most = np.moveaxis(roi_inner_most_rolled, 0, axis)

    roi_inner_most_image = sitk.GetImageFromArray(roi_inner_most)
    roi_inner_most_image.CopyInformation(roi)

    return roi_inner_most_image
