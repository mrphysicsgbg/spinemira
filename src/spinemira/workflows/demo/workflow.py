"""
This file illustrates a demo workflow using Pydra in which a BIDS/MIDS dataset is queried and images matching the
query is reoriented into LPS and stored as a derivative.

To highlight effect of parallelization, one of the task includes a sleep.
"""

from importlib.metadata import version
import logging
from pathlib import Path
from time import sleep
import SimpleITK as sitk

from fileformats.medimage import NiftiGz
from pydra.compose import python, workflow

import spinemira
from spinemira.tasks.mids import (
    initialize_derivative,
    query_mids,
    publish_derivative,
    resolve_derivative,
)


logger = logging.getLogger(__name__)
logging.root.setLevel(logging.INFO)


@python.define()
def reorient_lps_slow(
    image: NiftiGz, orientation: str = "LPS", output_path: Path | None = None
) -> NiftiGz:

    input_path = Path(image)

    if output_path is None:
        output_path = Path.cwd() / "reoriented_image.nii.gz"
    else:
        output_path = Path(output_path)

    loaded = sitk.ReadImage(input_path)
    loaded = sitk.DICOMOrient(loaded, orientation)

    sleep(
        10
    )  # Sleep some time to illustrate the difference between concurrent execution and parallel execution

    sitk.WriteImage(loaded, output_path)

    return NiftiGz(output_path)


@workflow.define(outputs=["file"])
def demo_process_single_entry_workflow(
    image: NiftiGz, output_path: Path, overwrite: bool = False
) -> NiftiGz:

    logger.info(f"Processing {image}...")

    if output_path.exists() and not overwrite:
        logger.info(f"Output file {output_path} exists. Reusing.")
        return NiftiGz(output_path)

    reoriented = workflow.add(
        reorient_lps_slow(
            image=image,
        ),
    )

    published = workflow.add(
        publish_derivative(
            file=reoriented.out, destination=output_path, overwrite=True
        ),
    )

    return published.file


@workflow.define(outputs=["files"])
def demo_workflow(
    dataset_root: Path,
    query: str,
    output_derivative_name: str,
    overwrite: bool = False,
) -> list[NiftiGz]:

    initialized_derivative = workflow.add(
        initialize_derivative(
            dataset_root=dataset_root,
            derivative_name=output_derivative_name,
            pipeline_name="Demo - spinemira",
            pipeline_version=version(spinemira.__package__),
        ),
    )

    images = workflow.add(
        query_mids(dataset_root=dataset_root, query=query),
        name="query_dataset",
    )

    output_paths = workflow.add(
        resolve_derivative(
            derivative_folder=initialized_derivative.path,
        )
        .split(original=images.files)
        .combine("original"),
        name="resolve_destination_paths",
    )

    processed = workflow.add(
        demo_process_single_entry_workflow(overwrite=overwrite)
        .split(
            ("image", "output_path"), image=images.files, output_path=output_paths.path
        )
        .combine("image"),
        name="process_single_image",
    )

    return processed.file
