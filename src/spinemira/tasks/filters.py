from pathlib import Path
import SimpleITK as sitk
from fileformats.medimage import NiftiGz
from pydra.compose import python

from spinemira.core import filters
from spinemira.core.io import load_image, load_label_map


@python.define()
def histogram_matching(
    src_image: NiftiGz,
    ref_image: NiftiGz,
    num_bins: int = 256,
    num_match_points: int = 10,
) -> NiftiGz:
    """
    Match histogram of image to reference image.

    This function is a Pydra task wrapper for the core function `histogram_matching`
    from `spinemira.core.filters`. It matches the histogram of a source image to a reference image.

    Parameters
    ----------
    src_image : NiftiGz
        Source image file (NIfTI format, gzipped)
    ref_image : NiftiGz
        Reference image file to match histogram against (NIfTI format, gzipped)
    num_bins : int, optional
        Number of bins used when creating histogram, by default 256
    num_match_points : int, optional
        Number of quantile values to be matched, by default 10

    Returns
    -------
    NiftiGz
        Filtered image file (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.histogram_matching : Core implementation
    """

    src_image_sitk = load_image(src_image.fspath, pixel_type=sitk.sitkUnknown)
    ref_image_sitk = load_image(ref_image.fspath, pixel_type=sitk.sitkUnknown)

    filtered_sitk = filters.histogram_matching(
        src_image=src_image_sitk,
        ref_image=ref_image_sitk,
        num_bins=num_bins,
        num_match_points=num_match_points,
    )

    output_path = Path.cwd() / "histogram_matched.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)


@python.define()
def multiple_regions_histogram_matching(
    src_image: NiftiGz,
    src_label_map: NiftiGz,
    ref_image: NiftiGz,
    ref_label_map: NiftiGz,
    src_mask: NiftiGz | None = None,
    ref_mask: NiftiGz | None = None,
    labels: set[int] | None = None,
    num_bins: int = 512,
    bg_rel_weight: float | None = 0.1,
) -> NiftiGz:
    """
    Match image by matching histogram of multiple regions.

    This function is a Pydra task wrapper for the core function `multiple_regions_histogram_matching`
    from `spinemira.core.filters`. It matches an image by matching the histogram of multiple regions.

    Parameters
    ----------
    src_image : NiftiGz
        Source image file (NIfTI format, gzipped)
    src_label_map : NiftiGz
        Label map of source image (NIfTI format, gzipped)
    ref_image : NiftiGz
        Reference image file (NIfTI format, gzipped)
    ref_label_map : NiftiGz
        Reference label map (NIfTI format, gzipped)
    src_mask : NiftiGz | None, optional
        Mask for source image, if unspecified no masking is performed
    ref_mask : NiftiGz | None, optional
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
    NiftiGz
        Image matched to histogram of source (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.multiple_regions_histogram_matching : Core implementation
    """

    src_image_sitk = load_image(src_image.fspath, pixel_type=sitk.sitkUnknown)
    src_label_map_sitk = load_label_map(src_label_map.fspath)
    ref_image_sitk = load_image(ref_image.fspath, pixel_type=sitk.sitkUnknown)
    ref_label_map_sitk = load_label_map(ref_label_map.fspath)

    src_mask_sitk = (
        load_image(src_mask.fspath, pixel_type=sitk.sitkUInt8) if src_mask else None
    )
    ref_mask_sitk = (
        load_image(ref_mask.fspath, pixel_type=sitk.sitkUInt8) if ref_mask else None
    )

    filtered_sitk = filters.multiple_regions_histogram_matching(
        src_image=src_image_sitk,
        src_label_map=src_label_map_sitk,
        ref_image=ref_image_sitk,
        ref_label_map=ref_label_map_sitk,
        src_mask=src_mask_sitk,
        ref_mask=ref_mask_sitk,
        labels=labels,
        num_bins=num_bins,
        bg_rel_weight=bg_rel_weight,
    )

    output_path = Path.cwd() / "multi_region_histogram_matched.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)


@python.define()
def winsorize(
    image: NiftiGz, lower: float, upper: float, mask: NiftiGz | None = None
) -> NiftiGz:
    """
    Winsorize (clip) image intensities using SciPy's winsorize (Pydra task wrapper).

    This function is a Pydra task wrapper for the core function `winsorize`
    from `spinemira.core.filters`. It limits extreme values by replacing values below/above
    given quantile limits with the corresponding boundary values.

    Parameters
    ----------
    image : NiftiGz
        Input image file (NIfTI format, gzipped)
    lower : float
        Lower limit
    upper : float
        Upper limit
    mask : NiftiGz | None, optional
        Optional mask to set which region to be filtered

    Returns
    -------
    NiftiGz
        Filtered image file (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.winsorize : Core implementation
    """

    image_sitk = load_image(image.fspath, pixel_type=sitk.sitkUnknown)
    mask_sitk = load_image(mask.fspath, pixel_type=sitk.sitkUInt8) if mask else None

    filtered_sitk = filters.winsorize(
        image=image_sitk, lower=lower, upper=upper, mask=mask_sitk
    )

    output_path = Path.cwd() / "winsorized.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)


@python.define()
def filter_mask(mask: NiftiGz, labels: set[float]) -> NiftiGz:
    """
    Filter a mask by retaining only the specified labels.

    This function is a Pydra task wrapper for the core function `filter_mask`
    from `spinemira.core.filters`. It filters a mask by retaining only the specified labels.

    Parameters
    ----------
    mask : NiftiGz
        Input label map file (NIfTI format, gzipped)
    labels : set[float]
        Labels to retain

    Returns
    -------
    NiftiGz
        Filtered mask containing only the specified labels (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.filter_mask : Core implementation
    """

    mask_sitk = load_label_map(mask.fspath)

    filtered_sitk = filters.filter_mask(mask=mask_sitk, labels=labels)

    output_path = Path.cwd() / "filtered_mask.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)


@python.define()
def reduce_label_map(
    label_map: NiftiGz, intervals: list[tuple[int, int]], new_labels: list[int]
) -> NiftiGz:
    """
    Reduce label map by mapping intervals to new labels.

    This function is a Pydra task wrapper for the core function `reduce_label_map`
    from `spinemira.core.filters`. It reduces a label map by mapping intervals to new labels.

    Parameters
    ----------
    label_map : NiftiGz
        Input label map file (NIfTI format, gzipped)
    intervals : list[tuple[int, int]]
        List of intervals (min_val, max_val) to map
    new_labels : list[int]
        List of new labels corresponding to each interval

    Returns
    -------
    NiftiGz
        Reduced label map file (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.reduce_label_map : Core implementation
    """

    label_map_sitk = load_label_map(label_map.fspath)

    filtered_sitk = filters.reduce_label_map(
        label_map=label_map_sitk, intervals=intervals, new_labels=new_labels
    )

    output_path = Path.cwd() / "reduced_label_map.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)


@python.define()
def normalize_to_label_intensity_mode(
    image: NiftiGz,
    label_map: NiftiGz,
    label: int,
    nbins: int = 1024,
) -> NiftiGz:
    """
    Normalize an image by the modal intensity within a labeled region.

    This function is a Pydra task wrapper for the core function `normalize_to_label_intensity_mode`
    from `spinemira.core.filters`. It normalizes an input image by estimating the mode of the
    intensity distribution within a specific labeled region and scaling the entire image by this value.

    Parameters
    ----------
    image : NiftiGz
        Input intensity image file (NIfTI format, gzipped)
    label_map : NiftiGz
        Label image file defining regions of interest (NIfTI format, gzipped)
    label : int
        Label value identifying the region used to compute the intensity mode
    nbins : int, optional
        Number of histogram bins used to approximate the mode, by default 1024

    Returns
    -------
    NiftiGz
        Normalized image file (NIfTI format, gzipped)

    See Also
    --------
    spinemira.core.filters.normalize_to_label_intensity_mode : Core implementation
    """

    image_sitk = load_image(image.fspath, pixel_type=sitk.sitkUnknown)
    label_map_sitk = load_label_map(label_map.fspath)

    filtered_sitk = filters.normalize_to_label_intensity_mode(
        image=image_sitk, label_map=label_map_sitk, label=label, nbins=nbins
    )

    output_path = Path.cwd() / "filtered.nii.gz"
    sitk.WriteImage(filtered_sitk, output_path)

    return NiftiGz(output_path)
