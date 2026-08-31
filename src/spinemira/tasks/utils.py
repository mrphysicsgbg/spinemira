import json
from pathlib import Path
from typing import Any

from fileformats.application import Json
from pydra.compose import python


def _merge_dics(dst: dict, src: dict) -> None:
    """
    Recursively merge two dictionaries.

    If a key exists in both dictionaries and both corresponding values are
    dictionaries, they are merged recursively. Otherwise, the value from the
    source dictionary (`src`) overwrites the value in the destination dictionary
    (`dst`).

    Parameters
    ----------
    dst : dict
        The destination dictionary to merge into (modified in-place).
    src : dict
        The source dictionary to merge from.

    Returns
    -------
    None
        The `dst` dictionary is modified in-place.
    """
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _merge_dics(dst[key], value)
        else:
            dst[key] = value


def make_merge_json_files_task(*input_names: str) -> python.Task:
    """
    Create a Pydra task that merges multiple JSON files into a single output.

    This function generates a Pydra task that takes multiple JSON files as input,
    merges their contents recursively, and outputs a single JSON file with the
    combined data.

    Parameters
    ----------
    *input_names : str
        Variable-length list of input names for the JSON files to be merged.
        Each input name corresponds to a JSON file path.

    Returns
    -------
    python.Task
        A Pydra task that, when executed, merges the input JSON files and
        produces a single merged JSON file.

    Notes
    -----
    The merging process is recursive: if a key exists in multiple input files
    and the corresponding values are dictionaries, they are merged recursively.
    Otherwise, the last occurrence of the key overwrites previous values.
    """

    @python.define(
        inputs={name: Json for name in input_names},
        outputs={"json": Json},
        name="merge_json_files",
    )
    def merge_json_files(**json_files: Json) -> Json:
        """
        Merge multiple JSON files into a single output file.

        Parameters
        ----------
        **json_files : Json
            Keyword arguments where each key is an input name and each value
            is a `Json` object representing a JSON file path.

        Returns
        -------
        Json
            A `Json` object pointing to the output file containing the merged
            JSON data.
        """

        merged: dict[Any, Any] = {}

        for json_file in json_files.values():
            with open(json_file.fspath, "r", encoding="utf-8") as file:
                data = json.load(file)
                _merge_dics(merged, data)

        output_path = Path.cwd() / "json_files_merged.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(merged, file, indent=2)

        return Json(output_path)

    return merge_json_files
