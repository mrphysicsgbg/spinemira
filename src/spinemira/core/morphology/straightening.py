import SimpleITK as sitk
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import splprep, splev


def straighten_coord(
    curve_points_physical,
    images,
    orders=None,
    si_padding_mm=0.0,
    target_curve_point_spacing_mm=None,
):
    """
    Straightens a 3D image loaded with LPS orientation (SimpleITK) so that a curved centerline becomes vertical
    (along Superior-Inferior axis), keeping the full original Left–Right and Posterior–Anterior size.

    TODO: Currently there is a problem with straighten of label maps with labels only consisting of single voxels.

    Parameters
    ----------
    curve_points_physical : array_like, shape (N, 3)
        3D points representing the curved centerline as physical points.
    images : sitk.Image or list of sitk.Image
        One or more 3D SimpleITK images to be straightened.
    orders : int or list of int, optional
        Interpolation order(s) for `scipy.ndimage.map_coordinates`. Defaults to 1 (linear).
    si_padding_mm: float, optional
        Extra length in mm to add before and after the centerline path. Default is 0 (no padding).
    target_curve_point_spacing_mm : float, optional
        If given, forces the input curve points to map to positions spaced
        this distance apart along the straightened superior–inferior axis.
        Output voxel spacing is unchanged.

    Returns
    -------
    list of sitk.Image
        The straightened image(s), with the curved centerline mapped to the Superior-Inferior axis.
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    if not isinstance(orders, (list, tuple)):
        orders = [orders] * len(images) if orders is not None else [1] * len(images)

    image_arrs = [sitk.GetArrayViewFromImage(image) for image in images]
    S, P, L = image_arrs[0].shape  # Superior-Inferior, Posterior-Anterior, Left-Right
    spacing = images[0].GetSpacing()

    # Convert physical to voxel indices
    curve_points = np.array(
        [
            images[0].TransformPhysicalPointToContinuousIndex(p)[::-1]
            for p in curve_points_physical
        ]
    )

    # Sort curve points along S-direction
    curve_points = curve_points[curve_points[:, 0].argsort()]

    # Fit spline in voxel space
    tck, u = splprep(curve_points.T, s=0)

    # Compute cumulative arc length (in voxel units)
    diffs = np.diff(curve_points, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate(([0], np.cumsum(seglen)))
    total_arc_len = arc_lengths[-1] * spacing[2]  # mm

    # Decide how many SI samples
    spacing_si = spacing[2]

    if target_curve_point_spacing_mm is not None:
        # TODO: It would be good to scale the output spacing in the SI direction.

        output_height = int(
            round((len(curve_points) - 1) * target_curve_point_spacing_mm / spacing_si)
        )

        # Target positions for each original curve point
        target_positions = np.arange(len(arc_lengths)) * target_curve_point_spacing_mm

        # Map straightened SI coordinates (mm) -> original arc length (mm)
        si_coords = np.arange(output_height) * spacing_si
        arc_coords_mm = np.interp(
            si_coords,
            target_positions,
            arc_lengths * spacing[2],
            left=arc_lengths[0] * spacing[2],
            right=arc_lengths[-1] * spacing[2],
        )

        # Convert back to normalized spline parameter u
        u_for_output = np.interp(arc_coords_mm, arc_lengths * spacing[2], u)
    else:
        output_height = int(total_arc_len / spacing_si)

        # Default: uniform SI spacing along arc length
        u_for_output = np.linspace(0, 1, output_height)

    # Interpolated curve and tangents
    curve_interp = np.vstack(splev(u_for_output, tck)).T
    tangents = np.vstack(splev(u_for_output, tck, der=1)).T
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    # Check if padding should be added
    pad_len_voxels = int(round(si_padding_mm / spacing_si))
    if pad_len_voxels > 0:
        # TODO: The padded tangents seems to yield strange morphological changes. Probably need to
        # pad using interpolation of several curve points (not only using the last or first tangent).

        head_point, head_tangent = curve_interp[0], tangents[0]
        pad_start = [
            head_point - head_tangent * spacing_si * i
            for i in reversed(range(1, pad_len_voxels + 1))
        ]

        tail_point, tail_tangent = curve_interp[-1], tangents[-1]
        pad_end = [
            tail_point + tail_tangent * spacing_si * i
            for i in range(1, pad_len_voxels + 1)
        ]

        padded_curve = np.vstack(pad_start + [*curve_interp] + pad_end)
        padded_tangents = np.vstack(
            [head_tangent] * pad_len_voxels
            + [*tangents]
            + [tail_tangent] * pad_len_voxels
        )
    else:
        padded_curve = curve_interp
        padded_tangents = tangents

    # Sampling grid in PA–LR plane
    pa_coords, lr_coords = np.meshgrid(np.arange(P), np.arange(L), indexing="ij")
    pa_offsets = -(pa_coords - P // 2)  # flip PA
    lr_offsets = lr_coords - L // 2

    straightened_arrs = [
        np.zeros((len(padded_curve), P, L), dtype=image_arr.dtype)
        for image_arr in image_arrs
    ]

    for i in range(len(padded_curve)):
        center = padded_curve[i]
        tangent = padded_tangents[i]

        ref = np.array([0, 0, 1]) if abs(tangent[2]) < 0.9 else np.array([0, 1, 0])
        normal = np.cross(tangent, ref)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)

        coords = (
            center[:, None, None]
            + normal[:, None, None] * pa_offsets
            + binormal[:, None, None] * lr_offsets
        )

        zz, yy, xx = coords
        for j in range(len(images)):
            straightened_arrs[j][i] = map_coordinates(
                image_arrs[j], [zz, yy, xx], order=orders[j], mode="constant", cval=0.0
            )

    straightened_images = []
    for i, straightened_arr in enumerate(straightened_arrs):
        straightened = sitk.GetImageFromArray(straightened_arr)
        straightened.SetSpacing(images[i].GetSpacing())  # keep original spacing
        straightened_images.append(straightened)

    return straightened_images
