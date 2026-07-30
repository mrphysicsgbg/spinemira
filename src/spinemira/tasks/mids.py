import json
import logging
from pathlib import Path

from fileformats.generic import File, Directory
from pydra.compose import python

from spinemira.io.mids import (
    Layout,
    resolve_derivative as resolve_derivative_path,
)

logger = logging.getLogger(__name__)


@python.define(outputs=["files"])
def query_mids(
    dataset_root: Directory,
    query: str,
    include_derivatives: bool = True,
    load_sidecars: bool = False,
    mids_index: File | None = None,
) -> list[File]:
    """
    Query files in a MIDS dataset.

    This function queries a MIDS dataset using the specified query string and returns
    a list of matching files. It can optionally use a pre-existing index for faster
    performance or create a new index if none is provided.

    Parameters
    ----------
    dataset_root : Directory
        Root directory of the MIDS dataset.
    query : str
        Query string to filter files in the dataset.
    include_derivatives : bool, optional
        Whether to include derivative datasets in the search, by default True.
    load_sidecars : bool, optional
        Whether to load sidecar files (metadata files) when indexing,
        by default False.
    mids_index : File | None, optional
        Path to a pre-existing MIDS index file. If provided and exists,
        the index will be loaded instead of creating a new one, by default None.

    Returns
    -------
    list[File]
        List of File objects representing the paths of files that match the query.
    """

    layout = Layout(
        root=Path(dataset_root),
        include_derivatives=include_derivatives,
    )

    if mids_index is not None and mids_index.exists():
        logger.info(f"Loading MIDS index at {str(mids_index)}")
        layout.load_index(Path(mids_index))
    else:
        layout.index(load_sidecars=load_sidecars)

    matches = layout.query(query)

    for path in matches["path"].to_list():
        logger.info(f"Query result: {path}")

    return [File(path) for path in matches["path"].to_list()]


@python.define(outputs=["folder"])
def initialize_derivative(
    dataset_root: Directory,
    derivative_name: str,
    pipeline_name: str,
    pipeline_version: str,
) -> Path:
    """
    Initializes a derivative.

    Parameters
    ----------
    dataset_root : Directory
        Directory to dataset root
    derivative_name : str
        Name of derivative
    pipeline_name : str
        Name of the pipeline used to generate the derivative
    pipeline_version : str
        Version of pipeline

    Returns
    -------
    Path
        Path to initialized directory
    """

    derivative_root = Path(dataset_root) / "derivatives" / derivative_name
    derivative_root.mkdir(parents=True, exist_ok=True)

    description = {
        "Name": f"{pipeline_name} Outputs",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": pipeline_name,
                "Version": pipeline_version,
            }
        ],
    }

    dataset_description_path = derivative_root / "dataset_description.json"

    dataset_description_path.write_text(json.dumps(description, indent=2) + "\n")

    logger.info(f"Initialized derivative folder: {str(derivative_root)}")

    return derivative_root


@python.define(outputs=["path"])
def resolve_derivative(
    file: File,
    derivative_folder: Directory | None = None,
    derivative_name: str | None = None,
    suffix: str | None = None,
    extension: str | None = None,
) -> Path:
    """
    Resolve MIDS derivative

    Parameters
    ----------
    file : File
        Original file
    derivative_name : str | None, optional
        Name of derivative
    derivative_folder : Directory | None, optional
        Derivative folder
    suffix : str | None, optional
        Suffix, by default None
    extension : str | None, optional
        Extensions, by default None

    Returns
    -------
    Path
        Derivative path corresponding to the input File and options
    """

    if derivative_folder is None and derivative_name is None:
        raise ValueError("Either `derivative_folder` or `derivative_name` is required.")

    derivative_folder_path = (
        Path(derivative_folder) if derivative_folder is not None else None
    )

    derivative = resolve_derivative_path(
        original=Path(file),
        derivative_name=derivative_name,
        derivative_folder=derivative_folder_path,
        suffix=suffix,
        extension=extension,
    )

    return derivative
