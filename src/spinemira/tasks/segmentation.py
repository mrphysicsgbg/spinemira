import json
from pathlib import Path

import SimpleITK as sitk
from fileformats.application import Json
from fileformats.core import FileSet
from fileformats.medimage import NiftiGz
import numpy as np
from pydra.compose import python

from spinemira.core.segmentation import segmentation as core_segmentation
from spinemira.core.segmentation import utils as core_utils
from spinemira.core.segmentation.segmentation import (
    is_muscle_map_segment_available as is_muscle_map_segment_available,
    is_sct_deepseg_available as is_sct_deepseg_available,
    is_totalspineseg_available as is_totalspineseg_available,
)
from spinemira.io.mids import resolve_derivative


@python.define(outputs=["file"])
def split_label_along_principal_axis(
    label_map: NiftiGz,
    n_parts: int,
    target_direction: tuple[int, int, int],
    plane_normal: tuple[int, int, int],
) -> NiftiGz:
    """
    Split the non-zero region of a 3D label map into n equal-length parts along the principal inertia axis closest to target_direction.

    This function is a Pydra task wrapper for the core function `split_label_along_principal_axis`
    from `spinemira.core.segmentation.utils`. It splits a label map into multiple parts along the principal
    inertia axis that best aligns with the target direction.

    Parameters
    ----------
    label_map : NiftiGz
        Input label map file (NIfTI format, gzipped). All non-zero voxels are treated as one region.
    n_parts : int
        Number of sections to split the label map into.
    target_direction : tuple[int, int, int]
        Direction in physical SimpleITK coordinates, i.e. (x, y, z).

        For a standard LPS-oriented SimpleITK image:
            Left -> Right      = [-1,  0,  0]
            Right -> Left      = [ 1,  0,  0]
            Posterior -> Anterior = [0, -1,  0]
            Anterior -> Posterior = [0,  1,  0]
            Inferior -> Superior = [0,  0,  1]
            Superior -> Inferior = [0,  0, -1]
    plane_normal : tuple[int, int, int]
        Vector (3,) defining the normal of a plane in physical SimpleITK coordinates, i.e. (x, y, z).
        If specified, then the principal axis to perform the split along will be calculated in the plane.

    Returns
    -------
    NiftiGz
        Label map file with the divided segmentation (NIfTI format, gzipped).
        Integer image with:
            0 = background
            1 ... n_parts = sections

    See Also
    --------
    spinemira.core.segmentation.utils.split_label_along_principal_axis : Core implementation
    """

    # Load label map
    label_map_sitk = sitk.ReadImage(label_map.fspath)

    # Split label map
    splitted_sitk, _ = core_utils.split_label_along_principal_axis(
        label_map=label_map_sitk,
        n_parts=n_parts,
        target_direction=target_direction,
        plane_normal=plane_normal,
    )

    # Save result
    output_path = Path.cwd() / "label_map_splitted.nii.gz"
    sitk.WriteImage(splitted_sitk, output_path)

    return NiftiGz(output_path)


@python.define(outputs=["label_map", "levels"])
def run_totalspineseg(
    image: NiftiGz,
    data_dir: Path,
    device: str = "cpu",
    localizer: NiftiGz | None = None,
    iso: bool = False,
    no_stalling: bool = False,
    max_workers: int | None = None,
    max_workers_nnunet: int | None = None,
    quiet: bool = False,
    executable: str = "totalspineseg",
    collect_levels: bool = True,
) -> tuple[NiftiGz, NiftiGz | None]:
    """
    Run TotalSpineSeg segmentation on a single image.

    Parameters
    ----------
    image : NiftiGz
        Input image file to be segmented.
    data_dir : Path
        Directory containing TotalSpineSeg data and models.
    device : str, optional
        Device to run inference on (e.g., "cpu", "cuda"), by default "cpu".
    localizer : NiftiGz
        Input localizer segmentation to use to detect first vertebrae, by default None.
    iso : bool, optional
        Whether to use isotropic resampling, by default False.
    no_stalling : bool, optional
        Whether to disable stalling behavior, by default False.
    max_workers : int | None, optional
        Maximum number of workers for parallel processing, by default None.
    max_workers_nnunet : int | None, optional
        Maximum number of workers for nnUNet, by default None.
    quiet : bool, optional
        Whether to suppress logging output, by default False.
    executable : str, optional
        Executable name or path. The default uses TOTALSPINESEG_EXECUTABLE
        when set, otherwise "totalspineseg" on PATH.
    collect_levels : bool, optional
        Collect the levels map alongside the segmentation, by default True.

    Returns
    -------
    tuple[NiftiGz, NiftiGz | None]
        The output label map and, when requested, the levels map.

    Raises
    ------
    subprocess.CalledProcessError
        If the TotalSpineSeg process fails with non-zero exit code.
    FileNotFoundError
        If the expected output files are not generated by TotalSpineSeg.
    """

    label_map, levels = core_segmentation.run_totalspineseg(
        image=Path(image.fspath),
        data_dir=data_dir,
        device=device,
        localizer=Path(localizer.fspath) if localizer is not None else None,
        iso=iso,
        no_stalling=no_stalling,
        max_workers=max_workers,
        max_workers_nnunet=max_workers_nnunet,
        quiet=quiet,
        executable=executable,
        collect_levels=collect_levels,
    )
    return NiftiGz(label_map), NiftiGz(levels) if levels is not None else None


@python.define(outputs=["label_map"])
def run_muscle_map(
    image: NiftiGz,
    region: str = "wholebody",
    model: str | None = None,
    model_version: str | None = None,
    use_gpu: bool = True,
    overlap: int = 90,
    chunk_size: int | str = "auto",
    suffix: str = "dseg",
    executable: str = "mm_segment",
) -> NiftiGz:
    """Run MuscleMap segmentation on a single image.

    Parameters
    ----------
    image : NiftiGz
        Input image file to segment.
    region : str, optional
        Anatomical region: wholebody, abdomen, pelvis, thigh, or leg.
        Defaults to wholebody.
    model : str | None, optional
        Alternative model to use. Defaults to the CLI's model selection.
    model_version : str | None, optional
        Model version, e.g. "1.3". None selects the latest available version.
    use_gpu : bool, optional
        Whether to use a GPU if available. Defaults to True.
    overlap : int, optional
        Percent spatial overlap during sliding window inference. Defaults to 90.
    chunk_size : int | str, optional
        Axial slices per chunk, or "auto" to size from available memory.
    suffix : str, optional
        Output filename suffix without a leading underscore. Defaults to dseg.
    executable : str, optional
        Executable name or path. The default uses MUSCLEMAP_EXECUTABLE
        when set, otherwise "mm_segment" on PATH.

    Returns
    -------
    NiftiGz
        Segmentation label map in the task's working directory.

    Raises
    ------
    subprocess.CalledProcessError
        If MuscleMap exits with a non-zero status.
    FileNotFoundError
        If MuscleMap does not generate the expected label map.
    """
    label_map = core_segmentation.run_muscle_map(
        image=Path(image.fspath),
        region=region,
        model=model,
        model_version=model_version,
        use_gpu=use_gpu,
        overlap=overlap,
        chunk_size=chunk_size,
        suffix=suffix,
        executable=executable,
    )
    return NiftiGz(label_map)


@python.define(outputs=["file"])
def create_label_sidecar(
    label_map: NiftiGz, labels: dict[int, list[dict[str, str]]]
) -> Json:
    values = sorted(
        int(v)
        for v in np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(label_map))))
        if v != 0
    )
    sidecar_labels = {str(value): labels.get(value, []) for value in values}
    path = Path.cwd() / "dseg.json"
    path.write_text(json.dumps({"Labels": sidecar_labels}, indent=2) + "\n")
    return Json(path)


@python.define(outputs=["label_maps"])
def _resolve_output_paths(
    files: list[FileSet],
    derivative_folder: Path,
) -> list[Path]:
    """Resolve destinations together to preserve paths for identical images.

    FileSet hashes are based on contents, so resolving each file in a separate
    cached task can reuse another subject's destination for identical images.
    """
    label_maps = []
    for file in files:
        images = [path for path in file.fspaths if NiftiGz.matches(path)]
        if len(images) != 1:
            raise ValueError(f"Expected one NIfTI image in {file}, found {len(images)}")
        label_maps.append(
            resolve_derivative(
                original=images[0],
                derivative_folder=derivative_folder,
                suffix="dseg",
            )
        )
    return label_maps


def label_metadata(source, enum):
    """Build sidecar label metadata for a segmentation tool."""
    return {member.value: [{"source": source, "name": member.name}] for member in enum}


@python.define(outputs=["label_map"])
def run_sct_deepseg(
    image: NiftiGz,
    model: str = "sc_canal_t2",
    executable: str = "sct_deepseg",
) -> NiftiGz:
    """Run an installed SCT deepseg model producing one NIfTI segmentation."""
    return NiftiGz(
        core_segmentation.run_sct_deepseg(
            image=Path(image.fspath),
            model=model,
            executable=executable,
        )
    )
