from collections import defaultdict
import json
import logging
from pathlib import Path
import typing

from fileformats.application import Json
from fileformats.core import FileSet, converter
from fileformats.core.exceptions import FormatMismatchError
from fileformats.generic import File, Directory
from fileformats.medimage import NiftiGz
import pandas as pd
from pydra.compose import python


from spinemira.io.mids import (
    Layout,
    get_stem,
    resolve_derivative as resolve_derivative_path,
    default_participants_path,
    read_participants_file,
    extract_entities_from_path,
    IMAGE_EXTENSIONS,
)
from spinemira.tasks.utils import resolve_json_pointer

logger = logging.getLogger(__name__)


def _select_main_path(fileset: FileSet) -> Path:

    # If only one path is present, return that
    if len(fileset.fspaths) == 1:
        return list(fileset.fspaths)[0]

    matches = [
        path for path in fileset.fspaths if path.name.endswith(tuple(IMAGE_EXTENSIONS))
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one main entry, "
            f"found {len(matches)} in {sorted(fileset.fspaths)}"
        )

    return matches[0]


@converter
@python.define(outputs=["out_file"])
def fileset_to_nifti_gz(in_file: FileSet) -> NiftiGz:

    matches = [path for path in in_file.fspaths if NiftiGz.matches(path)]

    if len(matches) != 1:
        raise FormatMismatchError(
            f"Expected exactly one NiftiGz in {in_file}, "
            f"found {len(matches)}: {matches}"
        )

    return NiftiGz(matches[0])


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
) -> FileSet:
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
    FileSet for resolved derivative. Grouped together with any sidecars found by checking for files sharing the same
    stem.

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
    entry = layout.find_derivative(
        input_data=file_path, flt=flt, return_type="dataframe"
    )
    assert isinstance(entry, pd.DataFrame)

    file_groups = layout.get_main_files_with_sidecars(entry)
    file_set = file_groups[0]

    logger.info(f"Found derivative: {file_path} -> {file_set[0]}")

    return FileSet(file_set)


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


@python.define(outputs=["file"])
def get_associated_participant_data(
    file: FileSet,
    dataset_root: str | Path | None = None,
    participants_file: File | None = None,
) -> Json:
    """
    Get associated participant data for an entry as JSON.

    This function extracts participant metadata (e.g., age, sex) for a given file
    by resolving the participant ID from the file's entities and querying the
    participants file.

    Parameters
    ----------
    file : FileSet
        Input file (or group of files) for which to retrieve participant data.
    dataset_root : str | Path | None, optional
        Root directory of the dataset, used to locate the default participants file
        if `participants_file` is not provided, by default None.
    participants_file : File | None, optional
        Path to a custom participants file. If provided, this file is used instead
        of the default participants file, by default None.

    Returns
    -------
    Json
        JSON file containing the participant data as a dictionary.
    """

    # Load participants file
    participants_file_path = (
        participants_file.fspath if participants_file is not None else None
    )
    participants_df = read_participants_file(
        participants_file=participants_file_path, dataset_root=dataset_root
    )

    # Parse subject if from file
    main_file_path = _select_main_path(file)
    entities = extract_entities_from_path(main_file_path)

    participant_data: dict[str, typing.Any] = {}

    if "participant_id" in entities:
        participant_df = participants_df[
            participants_df["participant_id"] == entities["participant_id"]
        ]
        if len(participant_df) == 1:
            participant_data = participant_df.squeeze().to_dict()

    output_json_path = Path.cwd() / "participant.json"

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(participant_data, f, indent=2)

    return Json(output_json_path)


@python.define(outputs=["file"])
def default_participants_file(dataset_root: Path) -> File:
    """
    Get default participants file

    Parameters
    ----------
    dataset_root : Path
        Path to dataset root

    Returns
    -------
    File
        Pointer to participants file
    """
    participants = default_participants_path(dataset_root)
    return File(participants)


@python.define(outputs=["json"])
def collect_entry_data(
    raw: FileSet,
    derivative_files: list[FileSet],
    derivative_names: list[str],
    participant_data: Json,
    columns: dict[str, str],
) -> Json:
    """
    Collect data from raw, derivative, and participant files based on specified column definitions.

    This function aggregates data from multiple sources (raw files, derivatives, and participant
    metadata) and extracts specific fields using JSON pointers defined in the columns dictionary.

    Parameters
    ----------
    raw : FileSet
        Raw file(s) for which to collect data.
    derivative_files : list[FileSet]
        List of derivative files (or groups of files) to collect data from.
    derivative_names : list[str]
        List of names corresponding to the derivative files (e.g., ["disc_metrics", "seg"]).
    participant_data : Json
        JSON file containing participant metadata (e.g., age, sex).
    columns : dict[str, str]
        Dictionary defining how to extract data for each column. Keys are the column names
        in the output, and values are JSON pointers to the field in the extracted data.
        Use `/entities/` for path-extracted data and `/sidecar/` for sidecar file data.

    Returns
    -------
    Json
        JSON file containing the collected data, with keys corresponding to the column names
        in the `columns` dictionary.
    """

    data: dict[str, typing.Any] = defaultdict(dict)

    # Load from participants
    data["participant"] = participant_data.load()

    # Extract entities
    data["raw"]["entities"] = extract_entities_from_path(_select_main_path(raw))

    # Load from raw sidecar
    for file in raw.fspaths:
        if Json.matches(file):
            data["raw"]["sidecar"] = Json(file).load()
            break

    # Load from each derivative
    for name, derivative_file_set in zip(derivative_names, derivative_files):
        # Extract entities
        data[name]["entities"] = extract_entities_from_path(
            _select_main_path(derivative_file_set)
        )

        # Load from sidecar
        for file in derivative_file_set.fspaths:
            if Json.matches(file):
                data[name]["sidecar"] = Json(file).load()
                break

    # Resolve mappings
    mapped_data: dict[str, typing.Any] = {}

    for column_id, json_pointer in columns.items():
        value = resolve_json_pointer(data, json_pointer)
        mapped_data[column_id] = value

    # Save
    output_json_path = Path.cwd() / "collected.json"

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(mapped_data, f, indent=2)

    return Json(output_json_path)
