"""SCT sc_canal_t2 segmentation workflows."""

from importlib.metadata import version
from pathlib import Path

from fileformats.medimage import NiftiGz
from pydra.compose import workflow

import spinemira
from spinemira.tasks.mids import (
    index,
    initialize_derivative,
    publish_derivative,
    query_mids,
)
from spinemira.tasks.segmentation import (
    _resolve_output_paths,
    create_label_sidecar,
    is_sct_deepseg_available,
    run_sct_deepseg,
)


@workflow.define(outputs=["label_map"])
def segment_image_workflow(
    image: NiftiGz,
    output_label_map: Path,
    executable: str = "sct_deepseg",
    overwrite: bool = False,
):
    """Segment a T2-weighted image and publish the canal mask and sidecar."""

    if output_label_map.exists() and not overwrite:
        label_map = NiftiGz(output_label_map)
    else:
        segmented = workflow.add(
            run_sct_deepseg(image=image, model="sc_canal_t2", executable=executable),
            name="run_sct_deepseg",
        )
        label_map = segmented.label_map

    sidecar = workflow.add(
        create_label_sidecar(
            label_map=label_map,
            labels={1: [{"source": "SCT sc_canal_t2", "name": "spinal_canal"}]},
        ),
        name="describe_labels",
    )

    published = workflow.add(
        publish_derivative(
            file=label_map,
            destination=output_label_map,
            sidecar=sidecar.file,
            overwrite=True,
        ),
        name="publish_derivative",
    )

    return published.file


@workflow.define(outputs=["label_map"])
def segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str = "sct_canal_t2",
    sct_deepseg_executable: str = "sct_deepseg",
    overwrite: bool = False,
):
    """Segment selected T2-weighted MIDS images using SCT sc_canal_t2."""

    if not is_sct_deepseg_available(sct_deepseg_executable):
        raise FileNotFoundError(
            f"SCT deepseg executable not available: {sct_deepseg_executable}"
        )

    mids = workflow.add(
        index(dataset_root=dataset_root, include_derivative=True, load_sidecars=True),
        name="index_mids",
    )

    derivative = workflow.add(
        initialize_derivative(
            dataset_root=dataset_root,
            derivative_name=output_derivative_name,
            pipeline_name="SCT sc_canal_t2 - spinemira",
            pipeline_version=version(spinemira.__package__),
        ),
        name="initialize_derivative",
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
        segment_image_workflow(executable=sct_deepseg_executable, overwrite=overwrite)
        .split(
            ("image", "output_label_map"),
            image=images.files,
            output_label_map=destinations.label_maps,
        )
        .combine("image"),
        name="segment_images",
    )

    return result.label_map
