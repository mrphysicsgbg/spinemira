from pathlib import Path
import SimpleITK as sitk

from fileformats.medimage.nifti import NiftiGz
from pydra.compose import python


@python.define()
def reorient_lps(
    image: NiftiGz, orientation: str = "LPS", output_path: Path | None = None
) -> NiftiGz:

    input_path = Path(image)

    if output_path is None:
        output_path = Path.cwd() / "reoriented_image.nii.gz"
    else:
        output_path = Path(output_path)

    loaded = sitk.ReadImage(input_path)
    loaded = sitk.DICOMOrient(loaded, orientation)

    sitk.WriteImage(loaded, output_path)

    return NiftiGz(output_path)
