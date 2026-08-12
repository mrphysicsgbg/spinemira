import json
import logging
from pathlib import Path

from fileformats.core import FileSet
from fileformats.generic import File, Directory
from pandas import Series
from pydra.compose import python


from spinemira.io.mids import (
    Layout,
    get_stem,
    resolve_derivative as resolve_derivative_path,
    IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def _select_main_path(fileset: FileSet) -> Path:
    matches = [
        path for path in fileset.fspaths if path.name.endswith(tuple(IMAGE_EXTENSIONS))
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one main entry, "
            f"found {len(matches)} in {sorted(fileset.fspaths)}"
        )

    return matches[0]


@python.define(outputs=["file"])
def index(
    dataset_root: Path,
    include_derivative: bool = False,
    load_sidecars: bool = False,
    ignore_encoding_errors: bool = False,
) -> File:
    """
     Parameters
    ----------
    dataset_root : Path
        Root directory of the MIDS dataset.
    include_derivatives : bool, optional
        Whether to include derivative datasets in the search, by default True.
    load_sidecars : bool, optional
        Whether to load sidecar files (metadata files) when indexing, by default False.
    ignore_encoding_errors: bool, optional.
        Whether to ignore encoding errors when saving index, by default False.

    Returns
    -------
    File
        Saved copy of index as CSV-file.
    """

    layout = Layout(root=dataset_root, include_derivatives=include_derivative)

    logger.info(f"Indexing {dataset_root}")

    layout.index(load_sidecars=load_sidecars)

    index_path = Path.cwd() / "index.csv"

    layout.save_index(path=index_path, ignore_encoding_errors=ignore_encoding_errors)

    return File(index_path)


@python.define(outputs=["files"])
def query_mids(
    dataset_root: Path,
    query: str,
    include_derivatives: bool = True,
    load_sidecars: bool = False,
    mids_index: File | None = None,
) -> list[FileSet]:
    """
    Query files in a MIDS dataset.

    This function queries a MIDS dataset using the specified query string and returns
    a list of matching files. It can optionally use a pre-existing index for faster
    performance or create a new index if none is provided.

    Parameters
    ----------
    dataset_root : Path
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
    list[FileSet]
        List of FileSet objects representing groups of files that share the same stem.
    """

    layout = Layout(
        root=dataset_root,
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

    # Use the utility method to group files by stem and directory
    file_groups = layout.get_main_files_with_sidecars(matches)

    # Create FileSet objects for each group
    filesets = []
    for file_group in file_groups:
        files = [Path(path) for path in file_group]
        filesets.append(FileSet(files))

    return filesets


@python.define(outputs=["file"])
def find_indexed_derivative(
    dataset_root: Path,
    file: FileSet,
    flt: str | dict[str, str] | None = None,
    load_sidecars: bool = False,
    mids_index: File | None = None,
) -> File:
    """
    Find indexed derivative for file.

    Parameters
    ----------
    dataset_root : Path,
        Directory to dataset root
    file : FileSet
        Input file to find derivative for.
    flt : str | dict[str, str] | None, optional
        Filter query, either as a string which can be passed to Pandas Query, or as a dictionary
        containing key value pairs to filter on.
    load_sidecars : bool, optional
        Whether to load sidecar files (metadata files) when indexing, by default False.
    mids_index : File | None, optional
        MIDS index. If unspecified, the layout will be indexed automatically.

    Return
    ------
    File for resolved derivative.

    See Also
    --------
    spinemira.core.layout.Layout.find_derivative : Core implementation

    """

    layout = Layout(
        root=dataset_root,
        include_derivatives=True,
    )

    if mids_index is not None and mids_index.exists():
        logger.info(f"Loading MIDS index at {str(mids_index)}")
        layout.load_index(Path(mids_index))
    else:
        logger.info(f"Indexing dataset at {str(dataset_root)}")
        layout.index(load_sidecars=load_sidecars)

    file_path = _select_main_path(file)

    entry = layout.find_derivative(input_data=file_path, flt=flt)

    assert isinstance(entry, Series)
    assert isinstance(entry.path, (str, Path))

    logger.info(f"Found derivative: {file_path} -> {entry.path}")

    return File(entry.path)


@python.define(outputs=["path"])
def initialize_derivative(
    dataset_root: Path,
    derivative_name: str,
    pipeline_name: str,
    pipeline_version: str,
) -> Path:
    """
    Initializes a derivative.

    Parameters
    ----------
    dataset_root : Path
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

    derivative_root = dataset_root / "derivatives" / derivative_name
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
    original: FileSet,
    derivative_folder: Directory | None = None,
    derivative_name: str | None = None,
    suffix: str | None = None,
    extension: str | None = None,
) -> Path:
    """
    Resolve MIDS derivative

    Parameters
    ----------
    original : FileSet
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

    original_path = _select_main_path(original)

    return resolve_derivative_path(
        original=original_path,
        derivative_name=derivative_name,
        derivative_folder=derivative_folder_path,
        suffix=suffix,
        extension=extension,
    )


@python.define(outputs=["file"])
def publish_derivative(
    file: FileSet,
    destination: Path,
    overwrite: bool = False,
) -> FileSet:

    published = file.copy(
        dest_dir=destination.parent,
        new_stem=get_stem(destination),
        mode=FileSet.CopyMode.copy,
        collation=FileSet.CopyCollation.any,
        make_dirs=True,
        overwrite=overwrite,
    )

    logger.info(f"Published {published}")

    return published
