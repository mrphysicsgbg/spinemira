import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import SimpleITK as sitk
from fileformats.medimage import NiftiGz
from pydra.compose import python

from spinemira.core.segmentation import utils as segmentationUtils


logger = logging.getLogger(__name__)


def _segmentation_executable(executable: str, default: str, variable: str) -> str:
    """Use the configured environment path when the default command is selected."""
    if executable == default:
        return os.environ.get(variable) or executable
    return executable


def is_totalspineseg_available(executable: str = "totalspineseg") -> bool:
    """
    Check if TotalSpineSeg command is installed.

    Parameters
    ----------
    executable : str, by default totalspineseg
        Name of executable

    Returns
    -------
    Boolean indicating if totalspineseg was found
    """
    return (
        shutil.which(
            _segmentation_executable(
                executable, "totalspineseg", "TOTALSPINESEG_EXECUTABLE"
            )
        )
        is not None
    )


def is_muscle_map_segment_available(executable: str = "mm_segment") -> bool:
    """
    Check if MuscleMap segmentation command is installed.

    Parameters
    ----------
    executable : str, by default mm_segment
        Name of executable

    Returns
    -------
    Boolean indicating if MuscleMap was found
    """
    return (
        shutil.which(
            _segmentation_executable(executable, "mm_segment", "MUSCLEMAP_EXECUTABLE")
        )
        is not None
    )


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
    splitted_sitk, _ = segmentationUtils.split_label_along_principal_axis(
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
    iso: bool = False,
    no_stalling: bool = False,
    max_workers: int | None = None,
    max_workers_nnunet: int | None = None,
    quiet: bool = False,
    executable: str = "totalspineseg",
) -> tuple[NiftiGz, NiftiGz]:
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

    Returns
    -------
    tuple[NiftiGz, NiftiGz]
        A tuple containing the output label map file and levels map file.

    Raises
    ------
    subprocess.CalledProcessError
        If the TotalSpineSeg process fails with non-zero exit code.
    FileNotFoundError
        If the expected output files are not generated by TotalSpineSeg.
    """

    with tempfile.TemporaryDirectory(prefix="totalspineseg-") as tempdir:
        intermediate_output = Path(tempdir)

        cmd = [
            _segmentation_executable(
                executable, "totalspineseg", "TOTALSPINESEG_EXECUTABLE"
            ),
            str(Path(image)),
            str(intermediate_output),
        ]

        if iso:
            cmd.append("--iso")

        cmd.extend(
            [
                "--data-dir",
                str(data_dir),
            ]
        )

        if no_stalling:
            cmd.append("--no-stalling")

        if max_workers is not None:
            cmd.extend(
                [
                    "--max-workers",
                    str(max_workers),
                ]
            )

        if max_workers_nnunet is not None:
            cmd.extend(
                [
                    "--max-workers-nnunet",
                    str(max_workers_nnunet),
                ]
            )

        if device is not None:
            cmd.extend(
                [
                    "--device",
                    device,
                ]
            )

        if quiet:
            cmd.append("--quiet")

        logger.info("Running: %s", " ".join(cmd))

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is not None:
                for line in process.stdout:
                    if not quiet:
                        logger.info(line.rstrip())

            returncode = process.wait()

        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                cmd,
            )

        # Copy output files from TotalSpineSeg to output paths

        generated_label_map = NiftiGz(intermediate_output / "step2_output" / image.name)
        generated_levels_map = NiftiGz(
            intermediate_output / "step1_levels" / image.name
        )

        if not generated_label_map.is_file():
            raise FileNotFoundError(
                f"TotalSpineSeg did not generate the expected label map: {str(generated_label_map)}"
            )

        if not generated_levels_map.is_file():
            raise FileNotFoundError(
                f"TotalSpineSeg did not generate the expected levels map: {str(generated_levels_map)}"
            )

        label_map = generated_label_map.move(
            dest_dir=Path.cwd(), stem_suffix="dseg", overwrite=True
        )
        levels = generated_levels_map.move(
            dest_dir=Path.cwd(), stem_suffix="levels", overwrite=True
        )

        logger.info(f"File: {str(label_map)} created")
        logger.info(f"File: {str(levels)} created")

        return label_map, levels


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
    with tempfile.TemporaryDirectory(prefix="muscle-map-") as tempdir:
        intermediate_output = Path(tempdir)
        cmd = [
            _segmentation_executable(executable, "mm_segment", "MUSCLEMAP_EXECUTABLE"),
            "--input_image",
            str(Path(image)),
            "--region",
            region,
            "--output_dir",
            str(intermediate_output),
            "--use_GPU",
            "Y" if use_gpu else "N",
            "--overlap",
            str(overlap),
            "--chunk_size",
            str(chunk_size),
            "--suffix",
            suffix,
        ]
        if model is not None:
            cmd.extend(["--model", model])
        if model_version is not None:
            cmd.extend(["--model_version", model_version])

        logger.info("Running: %s", " ".join(cmd))
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is not None:
                for line in process.stdout:
                    logger.info(line.rstrip())
            returncode = process.wait()

        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

        output_name = f"{image.name.removesuffix('.nii.gz')}_{suffix}.nii.gz"
        generated_label_map = intermediate_output / output_name
        if not generated_label_map.is_file():
            raise FileNotFoundError(
                f"MuscleMap did not generate the expected label map: {generated_label_map}"
            )

        label_map = NiftiGz(generated_label_map).move(
            dest_dir=Path.cwd(), overwrite=True
        )
        logger.info("File: %s created", label_map)

        return label_map
