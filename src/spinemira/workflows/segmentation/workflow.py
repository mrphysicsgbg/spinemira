from importlib.metadata import version
import logging
from pathlib import Path

from fileformats.core import FileSet
from fileformats.medimage import NiftiGz
from pydra.compose import python, workflow

import spinemira
from spinemira.io.mids import resolve_derivative as resolve_derivative_path
from spinemira.tasks.filters import make_merge_label_maps_task
from spinemira.tasks.mids import (
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
)
from spinemira.tasks.segmentation import (
    is_muscle_map_segment_available,
    is_totalspineseg_available,
    run_muscle_map,
    run_totalspineseg,
)


logger = logging.getLogger(__name__)


merge_label_maps = make_merge_label_maps_task(
    "totalspineseg_label_map", "musclemap_label_map"
)


def _enabled_tools(
    use_totalspineseg: bool,
    use_musclemap: bool,
    totalspineseg_executable: str,
    musclemap_executable: str,
) -> tuple[bool, bool]:
    if use_totalspineseg and not is_totalspineseg_available(totalspineseg_executable):
        logger.warning(
            "Skipping unavailable TotalSpineSeg: %s", totalspineseg_executable
        )
        use_totalspineseg = False
    if use_musclemap and not is_muscle_map_segment_available(musclemap_executable):
        logger.warning("Skipping unavailable MuscleMap: %s", musclemap_executable)
        use_musclemap = False
    return use_totalspineseg, use_musclemap


@python.define(outputs=["label_maps", "levels"])
def _resolve_output_paths(
    files: list[FileSet],
    derivative_folder: Path,
) -> tuple[list[Path], list[Path]]:
    """Resolve destinations together to preserve paths for identical images.

    FileSet hashes are based on contents, so resolving each file in a separate
    cached task can reuse another subject's destination for identical images.
    """
    label_maps = []
    levels = []
    for file in files:
        images = [path for path in file.fspaths if NiftiGz.matches(path)]
        if len(images) != 1:
            raise ValueError(f"Expected one NIfTI image in {file}, found {len(images)}")
        label_maps.append(
            resolve_derivative_path(
                original=images[0],
                derivative_folder=derivative_folder,
                suffix="dseg",
            )
        )
        levels.append(
            resolve_derivative_path(
                original=images[0],
                derivative_folder=derivative_folder,
                suffix="levels",
            )
        )
    return label_maps, levels


@workflow.define(outputs=["label_map", "levels"])
def segment_single_image_workflow(
    image: NiftiGz,
    output_label_map: Path,
    output_levels: Path,
    totalspineseg_data_dir: Path | None = None,
    overwrite: bool = False,
    device: str = "cpu",
    totalspineseg_quiet: bool = False,
    totalspineseg_executable: str = "totalspineseg",
    musclemap_executable: str = "mm_segment",
    musclemap_region: str = "wholebody",
    musclemap_model: str | None = None,
    musclemap_model_version: str | None = None,
    use_totalspineseg: bool = True,
    use_musclemap: bool = True,
) -> tuple[NiftiGz | None, NiftiGz | None]:
    """Run enabled, available tools and publish their labels.

    Merge when both tools run, giving TotalSpineSeg priority. Levels are None
    without TotalSpineSeg; both outputs are None when no tool can run.
    Existing outputs are reused unless overwrite is true. Set overwrite when
    changing tool selection for an existing derivative.
    """
    use_totalspineseg, use_musclemap = _enabled_tools(
        use_totalspineseg,
        use_musclemap,
        totalspineseg_executable,
        musclemap_executable,
    )
    if not (use_totalspineseg or use_musclemap):
        logger.info("No segmentation tools enabled and available. Skipping %s", image)
        return None, None

    if (
        output_label_map.exists()
        and (not use_totalspineseg or output_levels.exists())
        and not overwrite
    ):
        logger.info("Reusing existing segmentation: %s", output_label_map)
        return (
            NiftiGz(output_label_map),
            NiftiGz(output_levels) if use_totalspineseg else None,
        )

    totalspineseg_output = None
    musclemap_output = None
    if use_totalspineseg:
        if totalspineseg_data_dir is None:
            raise ValueError(
                "totalspineseg_data_dir is required when using TotalSpineSeg"
            )
        totalspineseg_output = workflow.add(
            run_totalspineseg(
                image=image,
                data_dir=totalspineseg_data_dir,
                device=device,
                quiet=totalspineseg_quiet,
                executable=totalspineseg_executable,
            ),
            name="run_totalspineseg",
        )

    if use_musclemap:
        musclemap_output = workflow.add(
            run_muscle_map(
                image=image,
                region=musclemap_region,
                model=musclemap_model,
                model_version=musclemap_model_version,
                use_gpu=device == "cuda",
                executable=musclemap_executable,
            ),
            name="run_musclemap",
        )

    if totalspineseg_output is not None and musclemap_output is not None:
        merged = workflow.add(
            merge_label_maps(
                totalspineseg_label_map=totalspineseg_output.label_map,
                musclemap_label_map=musclemap_output.label_map,
            ),
            name="merge_label_maps",
        )
        label_map = merged.label_map
    elif totalspineseg_output is not None:
        label_map = totalspineseg_output.label_map
    else:
        assert musclemap_output is not None
        label_map = musclemap_output.label_map

    published_label_map = workflow.add(
        publish_derivative(
            file=label_map, destination=output_label_map, overwrite=True
        ),
        name="publish_label_map",
    )

    levels = None
    if totalspineseg_output is not None:
        published_levels = workflow.add(
            publish_derivative(
                file=totalspineseg_output.levels,
                destination=output_levels,
                overwrite=True,
            ),
            name="publish_levels",
        )
        levels = published_levels.file

    return published_label_map.file, levels


@workflow.define(outputs=["label_map", "levels"])
def segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str,
    totalspineseg_data_dir: Path | None = None,
    overwrite: bool = False,
    device: str = "cpu",
    totalspineseg_quiet: bool = False,
    totalspineseg_executable: str = "totalspineseg",
    musclemap_executable: str = "mm_segment",
    musclemap_region: str = "wholebody",
    musclemap_model: str | None = None,
    musclemap_model_version: str | None = None,
    use_totalspineseg: bool = True,
    use_musclemap: bool = True,
):
    """
    Segment a dataset using TotalSpineSeg and MuscleMap, merging their labels.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    image_query : str
        Base query string to filter images.
    output_derivative_name : str
        Name of the output derivative folder.
    totalspineseg_data_dir : Path
        Directory containing TotalSpineSeg data and models. Required only when
        TotalSpineSeg runs.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.
    device : str, optional
        Device to run inference on (e.g., "cpu", "cuda"), by default "cpu".
    totalspineseg_quiet : bool, optional
        Whether to suppress TotalSpineSeg logging output, by default False.

    totalspineseg_executable : str, optional
        TotalSpineSeg command name or absolute executable path.
    musclemap_executable : str, optional
        MuscleMap command name or absolute executable path.
    musclemap_region : str, optional
        MuscleMap anatomical region, by default wholebody.
    musclemap_model : str | None, optional
        Alternative MuscleMap model; None uses the CLI default.
    musclemap_model_version : str | None, optional
        MuscleMap model version; None selects the CLI default.

    use_totalspineseg : bool, optional
        Enable TotalSpineSeg if its executable is available, by default True.
    use_musclemap : bool, optional
        Enable MuscleMap if its executable is available, by default True.

    Returns
    -------
    tuple[Any, Any]
        Label maps and optional TotalSpineSeg levels in query order. Empty lists
        when neither tool is enabled and available.
    """

    use_totalspineseg, use_musclemap = _enabled_tools(
        use_totalspineseg,
        use_musclemap,
        totalspineseg_executable,
        musclemap_executable,
    )
    if not (use_totalspineseg or use_musclemap):
        logger.info("No segmentation tools enabled and available. Skipping dataset.")
        return [], []

    pipeline_name = (
        " + ".join(
            name
            for name, enabled in (
                ("TotalSpineSeg", use_totalspineseg),
                ("MuscleMap", use_musclemap),
            )
            if enabled
        )
        + " - spinemira"
    )

    mids_index = workflow.add(
        index(
            dataset_root=dataset_root,
            include_derivative=True,
            load_sidecars=True,
        ),
        name="index_mids",
    )

    initialized_derivative = workflow.add(
        initialize_derivative(
            dataset_root=dataset_root,
            derivative_name=output_derivative_name,
            pipeline_name=pipeline_name,
            pipeline_version=version(spinemira.__package__),
        ),
        name="initialize_derivative",
    )

    images = workflow.add(
        query_mids(
            dataset_root=dataset_root,
            query=image_query,
            mids_index=mids_index.file,
        ),
        name="query_dataset",
    )

    destinations = workflow.add(
        _resolve_output_paths(
            files=images.files,
            derivative_folder=initialized_derivative.path,
        ),
        name="resolve_destinations",
    )

    segmented = workflow.add(
        segment_single_image_workflow(
            totalspineseg_data_dir=totalspineseg_data_dir,
            totalspineseg_quiet=totalspineseg_quiet,
            totalspineseg_executable=totalspineseg_executable,
            musclemap_executable=musclemap_executable,
            musclemap_region=musclemap_region,
            musclemap_model=musclemap_model,
            musclemap_model_version=musclemap_model_version,
            use_totalspineseg=use_totalspineseg,
            use_musclemap=use_musclemap,
            device=device,
            overwrite=overwrite,
        )
        .split(
            ("image", "output_label_map", "output_levels"),
            image=images.files,
            output_label_map=destinations.label_maps,
            output_levels=destinations.levels,
        )
        .combine("image"),
        name="process_single_image",
    )

    return segmented.label_map, segmented.levels
