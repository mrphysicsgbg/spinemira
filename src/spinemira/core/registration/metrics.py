import SimpleITK as sitk


def negative_jacobian_count(deform_field: sitk.Image) -> int:
    """Calculate the sum negative jacobian entries

    Parameters
    ----------
    deform_field : sitk.Image
        Deform field

    Returns
    -------
    float
        Sum of negative entries in the jacobian determinant
    """
    jacobian_determinant = sitk.DisplacementFieldJacobianDeterminant(deform_field)
    negative_jacobian_determinant = sitk.GetArrayFromImage(jacobian_determinant) < 0
    return int(negative_jacobian_determinant.sum(dtype=int))
