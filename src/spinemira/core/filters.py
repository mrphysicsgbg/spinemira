import logging
import SimpleITK as sitk
import numpy as np
from scipy.stats.mstats import winsorize as winsorize_scipy


logger = logging.getLogger(__name__)


def histogram_matching(
    src_image: sitk.Image,
    ref_image: sitk.Image,
    num_bins: int = 256,
    num_match_points=10,
) -> sitk.Image:
    """Match histogram of image to reference image

    Wraps sitk.HistogramMatchingImageFilter.

    Parameters
    ----------
    src_image : sitk.Image
        Source image
    ref_image : sitk.Image
        Reference image to match histogram against
    num_bins : int, optional
        Number of bins used when creating histogram, by default 256
    num_match_points : int, optional
        Number of quantile values to be matched, by default 10

    Returns
    -------
    sitk.Image
        Filtered image
    """

    filter = sitk.HistogramMatchingImageFilter()
    filter.SetNumberOfHistogramLevels(num_bins)
    filter.SetNumberOfMatchPoints(num_match_points)
    return filter.Execute(src_image, ref_image)


def multiple_regions_histogram_matching(
    src_image: sitk.Image,
    src_label_map: sitk.Image,
    ref_image: sitk.Image,
    ref_label_map: sitk.Image,
    src_mask: sitk.Image | None = None,
    ref_mask: sitk.Image | None = None,
    labels: set[int] | None = None,
    num_bins: int = 512,
    bg_rel_weight: float | None = 0.1,
) -> sitk.Image:
    """Match image by matching histogram of multiple regions.

    Parameters
    ----------
    src_image : sitk.Image
        Source image
    src_label_map : sitk.Image
        Label map of source image
    ref_image : sitk.Image
        Reference image
    ref_label_map : sitk.Image
        Reference label map
    src_mask : sitk.Image | None, optional
        Mask for source image, if unspecified no masking is performed
    ref_mask : sitk.Image | None, optional
        Mask for reference image, if unspecified no masking is performed
    labels : set[int] | None, optional
        Labels to label maps to use for calculating regions, if unspecified, then all common
        labels present in both the source and reference label map are used
    num_bins : int, optional
        Number of bins, by default 512
    bg_rel_weight : float | None, optional
        Weight for including histogram of background in matching, by default 0.1

    Returns
    -------
    sitk.Image
        Image matched to histogram of source

    Raises
    ------
    ValueError
        Raised if input images contains no finite values
    """

    # Convert inputs to Numpy arrays
    src_arr = sitk.GetArrayFromImage(src_image).astype(np.float32)
    ref_arr = sitk.GetArrayFromImage(ref_image).astype(np.float32)
    src_label_map_arr = sitk.GetArrayViewFromImage(src_label_map).astype(np.uint8)
    ref_label_map_arr = sitk.GetArrayViewFromImage(ref_label_map).astype(np.uint8)

    src_mask_arr = (
        sitk.GetArrayViewFromImage(src_mask).astype(np.uint8)
        if src_mask is not None
        else np.ones(src_arr.shape, dtype=np.uint8)
    )

    ref_mask_arr = (
        sitk.GetArrayViewFromImage(ref_mask).astype(np.uint8)
        if ref_mask is not None
        else np.ones(ref_arr.shape, dtype=np.uint8)
    )

    if labels is None:
        # If labels are unspecified, use all unique and common labels in the label maps
        labels = set(np.unique(src_label_map_arr)).union(
            set(np.unique(ref_label_map_arr))
        )

        # Remove background from labels
        labels.discard(0)

    src_finite_vals = src_arr[np.isfinite(src_arr)]
    ref_finite_vals = ref_arr[np.isfinite(ref_arr)]

    if src_finite_vals.size == 0:
        raise ValueError("Source image must contain finite values.")

    if ref_finite_vals.size == 0:
        raise ValueError("Reference image must contain finite values.")

    # Rescale source to match reference range
    src_min, src_max = np.nanmin(src_finite_vals), np.nanmax(src_finite_vals)
    ref_min, ref_max = np.nanmin(ref_finite_vals), np.nanmax(ref_finite_vals)

    src_arr = (src_arr - src_min) / (src_max - src_min)
    src_arr = src_arr * (ref_max - ref_min) + ref_min

    # Create global bins
    bins = np.linspace(ref_min, ref_max, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    region_mappings = []
    region_weights = []

    def _compute_cdf(values, bins):
        hist, _ = np.histogram(values, bins=bins, density=False)
        cdf = np.cumsum(hist).astype(np.float64)
        cdf /= cdf[-1]
        return cdf

    def _compute_region_mapping(label):
        src_mask_i = np.logical_and(src_label_map_arr == label, src_mask_arr)
        ref_mask_i = np.logical_and(ref_label_map_arr == label, ref_mask_arr)

        # Ensure no empty regions are included
        if np.count_nonzero(src_mask_i) == 0:
            logger.info(f"Skipping label {label} due to empty region in source image")
            return None, None

        if np.count_nonzero(ref_mask_i) == 0:
            logger.info(
                f"Skipping label {label} due to empty region in reference image"
            )
            return None, None

        src_vals_i = src_arr[src_mask_i]
        ref_vals_i = ref_arr[ref_mask_i]

        # Compute CDF of source and reference regions
        src_cdf_i = _compute_cdf(src_vals_i, bins)
        ref_cdf_i = _compute_cdf(ref_vals_i, bins)

        # Get unique values to enforce monotonicity
        ref_cdf_i_u, idx = np.unique(ref_cdf_i, return_index=True)
        centers_i_u = centers[idx]

        map_vals_i = np.interp(src_cdf_i, ref_cdf_i_u, centers_i_u)

        return map_vals_i, src_vals_i.size

    # Create region mapping for each label
    for label in labels:
        region_mapping, region_size = _compute_region_mapping(label)

        if region_mapping is not None:
            region_mappings.append(region_mapping)
            region_weights.append(region_size)

    # Check if background should be added
    if bg_rel_weight is not None and bg_rel_weight > 0:
        region_mapping, _ = _compute_region_mapping(0)

        if region_mapping is not None:
            region_mappings.append(region_mapping)
            region_weights.append(
                np.sum(region_weights) * bg_rel_weight if region_weights else 1.0
            )

    # Combine region mappings into a global mapping
    weights = np.array(region_weights, dtype=np.float64)
    weights /= weights.sum()
    global_mapping = np.average(region_mappings, axis=0, weights=weights)

    # Get unique values to enforce monotonicity
    global_mapping_u, idx = np.unique(global_mapping, return_index=True)
    centers_u = centers[idx]

    # Create matched image
    matched_arr = np.interp(src_arr, centers_u, global_mapping_u)

    # Ensure mapped values are within the dynamic range of the reference image
    matched_arr = np.clip(matched_arr, ref_min, ref_max)

    matched_image = sitk.GetImageFromArray(matched_arr)
    matched_image.CopyInformation(src_image)

    return matched_image


def winsorize(
    image: sitk.Image, lower: float, upper: float, mask: sitk.Image | None = None
) -> sitk.Image:
    """Winsorize (clip) image intensities using SciPy's winsorize.

    Winsorization limits extreme values by replacing values below/above
    given quantile limits with the corresponding boundary values.

    Parameters
    ----------
    image : sitk.Image
        Input image
    lower : float
        Lower limit
    upper : float
        Upper limit
    mask : sitk.Image | None, optional
        Optional mask to set which region to be filtered

    Returns
    -------
    sitk.Image
        Filtered image
    """

    if mask is not None:
        image_arr = sitk.GetArrayFromImage(image)
        mask_bool_arr = sitk.GetArrayViewFromImage(mask).astype(bool)
        image_arr[mask_bool_arr] = winsorize_scipy(
            image_arr[mask_bool_arr], limits=(lower, upper)
        )
    else:
        image_arr = sitk.GetArrayViewFromImage(image)
        image_arr = winsorize_scipy(image_arr, limits=(lower, upper))

    image_filtered = sitk.GetImageFromArray(image_arr)
    image_filtered.CopyInformation(image)

    return image_filtered


def filter_mask(mask: sitk.Image, labels: set[float]) -> sitk.Image:
    """
    Filter a mask by retaining only the specified labels.

    Parameters
    ----------
    mask : sitk.Image
        Input label map.
    labels : set[float]
        Labels to retain.

    Returns
    -------
    sitk.Image
        Filtered mask containing only the specified labels.
    """
    filtered = sitk.Image(mask.GetSize(), mask.GetPixelID())
    filtered.CopyInformation(mask)

    for label in labels:
        binary = sitk.BinaryThreshold(mask, label, label, label, 0)
        filtered = sitk.Add(filtered, binary)

    return filtered


def reduce_label_map(
    label_map: sitk.Image, intervals: list[tuple[int, int]], new_labels: list[int]
) -> sitk.Image:
    label_map_reduced = sitk.Image(label_map.GetSize(), sitk.sitkUInt8)
    label_map_reduced.CopyInformation(label_map)

    for (min_val, max_val), new_label in zip(intervals, new_labels):
        mask = sitk.And(
            sitk.GreaterEqual(label_map, min_val), sitk.LessEqual(label_map, max_val)
        )
        label_map_reduced[mask] = new_label

    return label_map_reduced


def normalize_to_label_intensity_mode(
    image: sitk.Image, label_map: sitk.Image, label: int, nbins: int = 1024
) -> sitk.Image:
    """
    Normalize an image by the modal intensity within a labeled region.

    The function estimates the mode of the intensity distribution inside the region defined by `label` in `label_map`
    using a histogram-based approximation, and scales the entire image by this value.

    Parameters
    ----------
    image : sitk.Image
        Input intensity image.
    label_map : sitk.Image
        Label image defining regions of interest. Must have the same size as
        `image`.
    label : int
        Label value identifying the region used to compute the intensity mode.
    nbins : int, optional
        Number of histogram bins used to approximate the mode, by default 1024.

    Returns
    -------
    sitk.Image
        Image normalized by the estimated modal intensity of the labeled region.

    Raises
    ------
    ValueError
        If the label is not present in the label map, or if the estimated mode
        is zero.
    """

    image_arr = sitk.GetArrayViewFromImage(image)
    label_arr = sitk.GetArrayViewFromImage(label_map)

    # Extract intensities within the label
    mask = label_arr == label
    if not np.any(mask):
        raise ValueError(f"Label {label} not found in label_map.")

    values = image_arr[mask]

    # Estimate mode via histogram
    hist, bin_edges = np.histogram(values, bins=nbins)

    max_bin_idx = np.argmax(hist)
    mode = 0.5 * (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1])

    if mode == 0:
        raise ValueError("Estimated mode is zero; normalization is undefined.")

    # Normalize image
    normalized_image = image / float(mode)

    return normalized_image
