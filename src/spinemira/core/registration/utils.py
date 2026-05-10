import SimpleITK as sitk
import numpy as np


def get_affine_transform_from_landmarks(
    fixed_landmarks: list[tuple[float, float, float]],
    moving_landmarks: list[tuple[float, float, float]],
    reference_image: sitk.Image,
):
    """Get affine transform from landmarks

    Parameters
    ----------
    fixed_landmarks : list[tuple[float, float, float]]
        List of fixed landmarks
    moving_landmarks : list[tuple[float, float, float]]
        List of moving landmarks
    reference_image : sitk.Image
        Reference image

    Returns
    -------
    sitk.AffineTransform
        Affine transform
    """
    fixed_landmarks_flatten = [x for xs in fixed_landmarks for x in xs]
    moving_landmarks_flatten = [x for xs in moving_landmarks for x in xs]

    similarity_transform = sitk.LandmarkBasedTransformInitializer(
        sitk.Similarity3DTransform(),
        fixed_landmarks_flatten,
        moving_landmarks_flatten,
        referenceImage=reference_image,
        numberOfControlPoints=len(fixed_landmarks),
    )
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetMatrix(similarity_transform.GetMatrix())
    affine_transform.SetTranslation(similarity_transform.GetTranslation())
    affine_transform.SetCenter(similarity_transform.GetCenter())

    return affine_transform


def get_translation_transform_from_landmarks(
    fixed_landmarks: list[tuple[float, float, float]],
    moving_landmarks: list[tuple[float, float, float]],
) -> sitk.AffineTransform:
    """
    Get an affine transform only affecting translation from landmarks.

    Notes
    -----
    An affine transform is returned, but the transform is purely translation.

    Parameters
    ----------
    fixed_landmarks : list[tuple[float, float, float]]
        List of fixed landmarks
    moving_landmarks : list[tuple[float, float, float]]
        List of moving landmarks

    Returns
    -------
    sitk.AffineTransform
        Affine transform
    """

    fixed_centroid = np.array(fixed_landmarks).mean(axis=0)
    moving_centroid = np.array(moving_landmarks).mean(axis=0)

    translation_vector = moving_centroid - fixed_centroid

    affine_transform = sitk.AffineTransform(3)
    # Identity matrix to only affect translation
    affine_transform.SetMatrix(np.eye(3).flatten())
    affine_transform.SetTranslation(translation_vector)

    return affine_transform


def get_image_center(image: sitk.Image) -> sitk.VectorDouble:
    """Get physical center of input image.

    Parameters
    ----------
    image : sitk.Image
        Input image

    Returns
    -------
    sitk.VectorDouble
        Physical center
    """

    size = image.GetSize()
    center_index = [int(size[i] / 2) for i in range(len(size))]
    return image.TransformIndexToPhysicalPoint(center_index)


def get_center_alignment_transform(
    moving_image: sitk.Image, fixed_image: sitk.Image
) -> sitk.AffineTransform:
    """Get a transform which centers moving image one fixed image.

    Notes
    -----
    An affine transform is returned, but the transform is purely translation.

    Parameters
    ----------
    moving_image : sitk.Image
        Moving image
    fixed_image : sitk.Image
        Fixed image

    Returns
    -------
    sitk.AffineTransform
        Transform which centers moving image.
    """

    if moving_image.GetDimension() != fixed_image.GetDimension():
        raise ValueError(
            "Moving image and fixed image needs to be of same number of dimensions"
        )

    moving_image_center = get_image_center(moving_image)
    fixed_image_center = get_image_center(fixed_image)

    translation = np.asarray(moving_image_center) - np.asarray(fixed_image_center)

    transform = sitk.AffineTransform(moving_image.GetDimension())
    transform.SetTranslation(translation)

    return transform


def harmonize_directions(
    images: list[sitk.Image], tol: float = 1e-3
) -> list[sitk.Image]:
    """
    Ensures that all images in the list have the same direction as the first image.
    If the direction differs within the specified tolerance, it is overridden.

    Parameters
    ----------
    images : list of sitk.Image
        List of SimpleITK images to harmonize.
    tol : float, optional
        Absolute tolerance for direction comparison (default is 1e-3).

    Returns
    -------
    list of sitk.Image
        List of images with harmonized directions.
    """

    ref_dir = np.array(images[0].GetDirection())
    harmonized = [images[0]]

    for idx, image in enumerate(images[1:], start=1):
        current_dir = np.array(image.GetDirection())

        if not np.allclose(current_dir, ref_dir, atol=tol):
            max_diff = np.max(np.abs(current_dir - ref_dir))
            raise ValueError(
                f"Direction mismatch for image at index {idx} exceeds tolerance {tol:.1e} "
                f"(max difference: {max_diff:.2e}). Cannot harmonize directions automatically."
            )
        else:
            img_copy = sitk.Image(image)
            img_copy.SetDirection(tuple(ref_dir))
            harmonized.append(img_copy)

    return harmonized
