import logging
from pathlib import Path

from fileformats.text import Csv
import pandas as pd

from fileformats.application import Json
from fileformats.core import FileSet
from fileformats.generic import File
from pydra.compose import python, workflow

from spinemira.tasks.mids import (
    collect_entry_data,
    default_participants_file,
    find_indexed_derivative,
    get_associated_participant_data,
    index,
    query_mids,
)


logger = logging.getLogger(__name__)


@python.define(outputs=["file"])
def copy_file(file: File, dest: Path, overwrite: bool = False) -> File:
    """
    Copy a file to a specified destination.

    Parameters
    ----------
    file : File
        The file to copy.
    dest : Path
        Destination path for the copied file.
    overwrite : bool, optional
        Whether to overwrite the destination file if it exists, by default False.

    Returns
    -------
    File
        The copied file at the destination path.
    """
    return file.copy(
        dest_dir=dest.parent,
        make_dirs=True,
        new_stem=dest.stem,
        overwrite=overwrite,
    )


@python.define(outputs=["file"])
def combine_json_files(files: list[Json]) -> Csv:
    """
    Combine a list of JSON files into a single CSV file.

    This function loads data from multiple JSON files, converts them into a Pandas DataFrame,
    and saves the result as a CSV file.

    Parameters
    ----------
    files : list[Json]
        List of JSON files to combine.

    Returns
    -------
    Csv
        A CSV file containing the combined data from all input JSON files.
    """

    data = [file.load() for file in files]
    df = pd.DataFrame(data)

    output_file = Path.cwd() / "output.csv"
    df.to_csv(output_file, sep=";", encoding="utf-8")
    return Csv(output_file)


@workflow.define(outputs=["json"])
def process_single_entry(
    dataset_root: Path,
    raw: FileSet,
    participants: File,
    derivative_queries: dict[str, str],
    columns: dict[str, str],
    mids_index: File,
) -> Json:
    """
    Process a single raw entry to collect metrics based on the provided column definitions.

    This workflow retrieves participant data, queries for derivative files, and collects
    metrics from raw, derivative, and participant files based on the provided column definitions.

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset.
    raw : FileSet
        Raw file(s) to process.
    participants : File
        Participants file containing metadata.
    derivative_queries : dict[str, str]
        Dictionary mapping derivative names to query strings for filtering derivatives.
    columns : dict[str, str]
        Dictionary defining how to extract data for each column. Keys are the column names
        in the output, and values are JSON pointers to the field in the extracted data.
    mids_index : File
        MIDS index file for querying derivatives.

    Returns
    -------
    Json
        JSON file containing the collected metrics for the processed entry.
    """

    participant_data = workflow.add(
        get_associated_participant_data(file=raw, participants_file=participants),
        name="get_associated_participant_data",
    )

    derivative_queries_lst = [query for _, query in derivative_queries.items()]
    derivative_names_lst = [name for name, _ in derivative_queries.items()]

    derivatives = workflow.add(
        find_indexed_derivative(
            dataset_root=dataset_root, file=raw, mids_index=mids_index
        )
        .split(flt=derivative_queries_lst)
        .combine("flt")
    )

    collected = workflow.add(
        collect_entry_data(
            raw=raw,
            derivative_files=derivatives.file,
            derivative_names=derivative_names_lst,
            participant_data=participant_data.file,
            columns=columns,
        )
    )

    return collected.json


@workflow.define()
def collect_metrics_workflow(
    dataset_root: Path,
    raw_query: str,
    derivative_queries: dict[str, str],
    columns: dict[str, str],
    output_file: Path,
    load_sidecars: bool = False,
    overwrite: bool = False,
) -> Csv:
    """
    Create a workflow for collecting metrics from a MIDS dataset.

    This workflow queries a MIDS dataset for raw and derivative files, processes each entry
    to extract metrics based on user-defined column definitions, and combines the results into a single
    output file (CSV).

    Parameters
    ----------
    dataset_root : Path
        Root directory of the dataset to process.
    raw_query : str
        Query string to filter raw files for processing.
    derivative_queries : dict[str, str]
        Dictionary mapping derivative names to query strings for filtering derivatives.
    columns : dict[str, str]
        Dictionary defining how to extract data for each column. Keys are the column names
        in the output, and values are JSON pointers to the field in the extracted data.
    output_file : Path
        Path to the output CSV file where the collected metrics will be saved.
    load_sidecars : bool, optional
        Whether to load sidecar files when indexing the dataset, by default False.
    overwrite : bool, optional
        Whether to overwrite existing output files, by default False.

    Returns
    -------
    Csv
        The output CSV file containing the collected metrics.
    """

    if output_file.exists() and not overwrite:
        logger.info("Output file exists. Pass `overwrite` to overwrite it.")
        return Csv(output_file)

    mids_index = workflow.add(
        index(
            dataset_root=dataset_root,
            include_derivative=True,
            load_sidecars=load_sidecars,
        ),
        name="mids_index",
    )

    # Resolve subject metadata from participants.tsv
    participants = workflow.add(default_participants_file(dataset_root=dataset_root))

    # Get all raw images
    raw = workflow.add(
        query_mids(
            dataset_root=dataset_root,
            query=raw_query,
            mids_index=mids_index.file,
        ),
        name="query_raw",
    )

    # Process a single raw entry
    processed_single_entry = workflow.add(
        process_single_entry(
            dataset_root=dataset_root,
            participants=participants.file,
            derivative_queries=derivative_queries,
            columns=columns,
            mids_index=mids_index.file,
        )
        .split(raw=raw.files)
        .combine("raw"),
        name="process_entries",
    )

    # Assemble files into a single output DataFrame
    combined_json_files = workflow.add(
        combine_json_files(files=processed_single_entry.json), name="combine_outputs"
    )

    # Copy output
    copied_file = workflow.add(
        copy_file(file=combined_json_files.file, dest=output_file, overwrite=overwrite),
        name="copy_output_file",
    )

    return copied_file.file
