from pathlib import Path

import SimpleITK as sitk


def load_image(path: Path | str, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    """Load an image.

    Parameters
    ----------
    path : Path | str
        Path to image
    pixel_type : int, optional
        Pixel type, by default sitk.sitkFloat32

    Returns
    -------
    sitk.Image
        Loaded image
    """
    image = sitk.ReadImage(path, outputPixelType=pixel_type)
    image = sitk.DICOMOrient(image, "LPS")
    return image


def load_label_map(path: Path | str, pixel_type: int = sitk.sitkUInt8) -> sitk.Image:
    """Load a label map.

    Parameters
    ----------
    path : Path | str
        Path to label map
    pixel_type : int, optional
        Pixel type, by default sitk.sitkUInt8

    Returns
    -------
    sitk.Image
        Loaded label map
    """
    label_map = sitk.ReadImage(path, outputPixelType=pixel_type)
    label_map = sitk.DICOMOrient(label_map, "LPS")
    return label_map
