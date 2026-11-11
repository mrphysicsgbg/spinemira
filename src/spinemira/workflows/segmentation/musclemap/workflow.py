from importlib.metadata import version
from pathlib import Path

from fileformats.medimage import NiftiGz, NiftiGzX
from pydra.compose import workflow
import spinemira
from spinemira.io.mids import get_stem
from spinemira.tasks.mids import (
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
)
from spinemira.tasks.segmentation import (
    _resolve_output_paths,
    label_metadata,
    attach_sidecar_data,
)

from spinemira.core.segmentation.labels import MuscleMapLabels
from spinemira.tasks.segmentation import is_muscle_map_segment_available, run_muscle_map


@workflow.define(outputs=["label_map"])
def musclemap_workflow(
    image: NiftiGz,
    output_label_map: Path,
    device: str = "cpu",
    executable: str = "mm_segment",
    region: str = "wholebody",
    model: str | None = None,
    model_version: str | None = None,
    overwrite: bool = False,
):
    """Segment one image with MuscleMap and publish its label sidecar."""
    sidecar_path = output_label_map.with_name(get_stem(output_label_map) + ".json")
    if output_label_map.exists() and sidecar_path.exists() and not overwrite:
        return NiftiGzX(output_label_map, sidecar_path)

    task = run_muscle_map(
        image=image,
        region=region,
        model=model,
        model_version=model_version,
        use_gpu=device == "cuda",
        executable=executable,
    )

    segmented = workflow.add(task, name="run_segmentation")

    label_map_with_sidecar = workflow.add(
        attach_sidecar_data(
            label_map=segmented.label_map,
            labels=label_metadata("MuscleMap", MuscleMapLabels),
        ),
        name="describe_labels",
    )

    result = workflow.add(
        publish_derivative(
            file=label_map_with_sidecar.file,
            destination=output_label_map,
            overwrite=True,
        ),
        name="publish_derivative",
    )
    return result.file


@workflow.define(outputs=["label_map"])
def segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str = "musclemap",
    device: str = "cpu",
    musclemap_executable: str = "mm_segment",
    musclemap_region: str = "wholebody",
    musclemap_model: str | None = None,
    musclemap_model_version: str | None = None,
    overwrite: bool = False,
):
    """Segment a dataset with MuscleMap."""
    if not is_muscle_map_segment_available(musclemap_executable):
        raise FileNotFoundError(
            f"Segmentation executable not available: {musclemap_executable}"
        )
    pipeline_name = "MuscleMap - spinemira"
    mids = workflow.add(
        index(dataset_root=dataset_root, include_derivative=True, load_sidecars=True),
        name="index_mids",
    )
    derivative = workflow.add(
        initialize_derivative(
            dataset_root=dataset_root,
            derivative_name=output_derivative_name,
            pipeline_name=pipeline_name,
            pipeline_version=version(spinemira.__package__),
        ),
        name="initialize_derivative",
    )
    child = musclemap_workflow(
        device=device,
        executable=musclemap_executable,
        region=musclemap_region,
        model=musclemap_model,
        model_version=musclemap_model_version,
        overwrite=overwrite,
    )
    images = workflow.add(
        query_mids(dataset_root=dataset_root, query=image_query, mids_index=mids.file),
        name="query_dataset",
    )
    destinations = workflow.add(
        _resolve_output_paths(files=images.files, derivative_folder=derivative.path),
        name="resolve_destinations",
    )
    result = workflow.add(
        child.split(
            ("image", "output_label_map"),
            image=images.files,
            output_label_map=destinations.label_maps,
        ).combine("image"),
        name="segment_images",
    )
    return result.label_map
