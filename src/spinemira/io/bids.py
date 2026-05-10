import fnmatch
from os import PathLike
from typing import List, Dict, Any
from pathlib import Path
import re
import pandas as pd
import json


ENTITY_PATTERN = re.compile(r"(?P<key>[a-zA-Z0-9]+)-(?P<value>[^_/]+)")
IMAGE_EXTENSIONS = {".nii", ".nii.gz", ".h5", ".hdf5"}
SIDECAR_EXTENSIONS = {".json", ".tsv", ".bval", ".bvec"}
VALID_EXTENSIONS = IMAGE_EXTENSIONS.union(SIDECAR_EXTENSIONS)


class Layout:
    """
    Indexes BIDS-formatted datasets and optionally their derivatives.

    This dataset indexer has extended the BIDS specifications to allow for additional suffixes. Uses Pandas as intermediate storage for managing resolved entries.

    Attributes
    ----------
    _root : Path
        Root directory of the BIDS dataset.
    _include_derivatives : bool
        Whether to index derivatives directories.
    _df : None | pd.DataFrame
        Cached DataFrame of indexed files.
    """

    def __init__(self, root: str | Path, include_derivatives: bool = True):
        """
        Parameters
        ----------
        root : str or Path
            Path to the root of the BIDS dataset.
        include_derivatives : bool, optional
            Whether to index derivatives (default is True).
        """
        self._root = Path(root)
        self._include_derivatives = include_derivatives
        self._df: None | pd.DataFrame = None

    def index(
        self, load_sidecars: bool = False, load_sidecars_max_depth: int = 2
    ) -> pd.DataFrame:
        """
        Recursively index all valid BIDS files in the dataset.

        Parameters
        ----------
        load_sidecars : bool, optional
            Whether to load JSON sidecars (default is False).
        load_sidecars_max_depth : int, optional
            Max depth of JSON keys in sidecars to load(default is 2). Ignored if `load_sidecars` is set to False.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per indexed file, including extracted entities and metadata.
        """
        rows = []

        # Raw BIDS
        rows.extend(
            self._index_directory(
                self._root,
                source="raw",
                pipeline=None,
                load_sidecars=load_sidecars,
                load_sidecars_max_depth=load_sidecars_max_depth,
            )
        )

        # Derivatives
        if self._include_derivatives:
            derivatives_dir = self._root / "derivatives"
            if derivatives_dir.is_dir():
                for pipeline_dir in derivatives_dir.iterdir():
                    if pipeline_dir.is_dir():
                        rows.extend(
                            self._index_directory(
                                pipeline_dir,
                                source="derivative",
                                pipeline=pipeline_dir.name,
                                load_sidecars=load_sidecars,
                                load_sidecars_max_depth=load_sidecars_max_depth,
                            )
                        )

        self._df = pd.DataFrame(rows)
        return self._df

    def _index_directory(
        self,
        base_dir: Path,
        source: str,
        pipeline: None | str,
        load_sidecars: bool = False,
        load_sidecars_max_depth: int = 2,
    ) -> List[Dict[str, str | None]]:
        """
        Index a specific directory, extracting metadata and paths.

        Parameters
        ----------
        base_dir : Path
            The directory to search recursively for files.
        source : str
            Either "raw" or "derivative" to distinguish data sources.
        pipeline : str or None
            Name of the pipeline (for derivatives) or None for raw data.
        load_sidecars : bool, optional
            Whether to load JSON sidecars (default is False).
        load_sidecars_max_depth : int, optional
            Max depth of JSON keys in sidecars to load(default is 2).

        Returns
        -------
        list of dict
            List of dictionaries with extracted metadata for each valid file.
        """
        records = []
        for file in base_dir.rglob("*"):
            if not file.is_file():
                continue
            if source == "raw" and "derivatives" in file.parts:
                continue
            if not is_valid_bids_file(file.name):
                continue

            rel_path = file.relative_to(base_dir)
            entities = extract_entities_from_path(rel_path)
            entities["path"] = str(file.resolve())
            entities["rel_path"] = str(file.relative_to(self._root))
            entities["source"] = source
            entities["pipeline"] = pipeline
            entities["is_sidecar"] = is_sidecar_file(file.name)

            # Load and flatten metadata if JSON
            if load_sidecars and file.suffix == ".json":
                metadata = self._load_json_metadata(file)

                def flatten_metadata(d, parent_key="meta", depth=1, max_depth=2):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{parent_key}_{k}"
                        if isinstance(v, dict) and depth < max_depth:
                            items.update(
                                flatten_metadata(v, new_key, depth + 1, max_depth)
                            )
                        elif isinstance(v, (str, int, float, bool)):
                            items[new_key] = v
                    return items

                flat_metadata = flatten_metadata(
                    metadata, max_depth=load_sidecars_max_depth
                )
                entities.update(flat_metadata)
                entities["json_metadata"] = metadata

            records.append(entities)

        return records

    def _load_json_metadata(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _normalize_input(
        self,
        input_data: str
        | Path
        | pd.Series
        | list[str | Path | pd.Series]
        | pd.DataFrame,
    ):
        items: list[str | Path | pd.Series]

        if isinstance(input_data, pd.DataFrame):
            items = [row for _, row in input_data.iterrows()]
        elif isinstance(input_data, list):
            items = input_data
        else:
            items = [input_data]

        return items

    def _normalize_output(
        self, return_type: str, indices
    ) -> pd.Series | int | list[int] | pd.DataFrame | None:
        assert self._df is not None

        if return_type == "dataframe":
            return self._df.loc[indices]
        elif return_type == "index":
            return indices
        elif return_type == "auto":
            if len(indices) == 0:
                return None
            elif len(indices) == 1:
                return self._df.loc[indices[0]]
            else:
                return self._df.loc[indices]
        else:
            raise ValueError("Unsupported return type.")

    def dataframe(self) -> pd.DataFrame:
        """
        Return the indexed DataFrame.

        Returns
        -------
        pd.DataFrame
            Indexed file DataFrame.

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        """
        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")
        return self._df

    def query(self, expr: str) -> pd.DataFrame:
        """
        Query the indexed DataFrame using a pandas expression.

        Parameters
        ----------
        expr : str
            Query string compatible with pandas.DataFrame.query.

        Returns
        -------
        pd.DataFrame
            Filtered result of the query.

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        """
        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")
        return self._df.query(expr)

    def join_subject_metadata(
        self,
        metadata_df: pd.DataFrame,
        on: str = "sub",
        update_internal: bool = True,
    ) -> pd.DataFrame:
        """
        Join subject-level metadata with the indexed DataFrame.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Metadata to join with.
        on : str, optional
            Column to join on (default is "sub").
        update_internal : bool, optional
            Whether the internal dataframe should be updated with the subject metadata, by default True

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with subject metadata.

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        """
        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")

        df = self._df.merge(metadata_df, on=on, how="left")

        if update_internal:
            self._df = df

        return df

    def save_index(
        self,
        path: str | Path,
        format: str = "csv",
        ignore_encoding_errors: bool = False,
    ):
        """
        Save the indexed DataFrame to disk.

        Parameters
        ----------
        path : str or Path
            File path to save the DataFrame.
        format : str, optional
            File format: "csv" or "parquet" (default is "csv").
        ignore_encoding_errors : bool, optional
            Whatever encoding errors should be ignored, by default False

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        ValueError
            If an unsupported format is specified.
        """
        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")
        path = Path(path)
        format = format.lower()

        df_to_save = self._df.copy()
        if "json_metadata" in df_to_save.columns:
            df_to_save["json_metadata"] = df_to_save["json_metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else ""
            )

        if format == "csv":
            errors = "strict" if not ignore_encoding_errors else "replace"
            df_to_save.to_csv(path_or_buf=path, index=False, errors=errors)  # type: ignore[call-overload]
        elif format == "parquet":
            df_to_save.to_parquet(path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

    def load_index(self, path: PathLike, format: str = "csv") -> pd.DataFrame:
        """
        Load a previously saved index from disk.

        Parameters
        ----------
        path : str or Path
            File path of the saved index.
        format : str, optional
            File format: "csv" or "parquet" (default is "csv").

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        Raises
        ------
        ValueError
            If an unsupported format is specified.
        """
        path = Path(path)
        format = format.lower()

        if format == "csv":
            df = pd.read_csv(path, converters={"sub": str})
        elif format == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

        # Deserialize json_metadata if present
        if "json_metadata" in df.columns:

            def safe_load(x):
                return json.loads(x) if pd.notna(x) and x.strip() != "" else {}

            df["json_metadata"] = df["json_metadata"].apply(safe_load)

        self._df = df

        return self._df

    def find_main(
        self,
        input_data: str
        | Path
        | pd.Series
        | list[str | Path | pd.Series]
        | pd.DataFrame,
        return_type: str = "auto",
    ) -> pd.Series | int | list[int] | pd.DataFrame | None:
        """
        Return the main file(s) associated with one or more sidecar files (e.g., JSON, TSV).

        Parameters
        ----------
        input_data : str, Path, pd.Series, list[str | Path | pd.Series], or pd.DataFrame
            Single or multiple sidecar paths or entries.
        return_type : {'dataframe', 'list', 'auto'}, optional
            Determines the output format:
            - 'dataframe': always return DataFrame of results
            - 'index': always return list of row indices
            - 'auto': single result returns Series, multiple returns DataFrame (default)

        Returns
        -------
        Various formats depending on return_type:
            - pd.Series or pd.DataFrame
            - int or list of int
            - None (if no match found)

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        ValueError
            If a non-sidecar entry is provided or multiple main files match.
        """

        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")

        def resolve_entry(item: str | Path | pd.Series) -> pd.Series | int | None:
            if isinstance(item, (str, Path)):
                entry = self.get_entry_from_path(item)
            elif isinstance(item, pd.Series):
                entry = item
            else:
                raise TypeError("Unsupported input type")

            if self._df is None:
                raise RuntimeError("You must call `.index()` first.")

            if entry is None:
                return None
            if not entry.get("is_sidecar", False):
                raise ValueError(f"File is not a sidecar: {entry.get('path', item)}")

            rel_path = Path(entry["rel_path"])
            parent_dir = rel_path.parent
            stem = rel_path.with_suffix("").stem

            candidates = self._df[
                (~self._df["is_sidecar"])
                & (
                    self._df["rel_path"].apply(
                        lambda p: (
                            Path(p).parent == parent_dir
                            and Path(p).with_suffix("").stem == stem
                        )
                    )
                )
            ]

            if len(candidates) > 1:
                raise ValueError(
                    f"Multiple associated main files found for sidecar: {entry.get('path', '<unknown>')}"
                )
            elif len(candidates) == 1:
                return candidates.index[0]
            else:
                return None

        items = self._normalize_input(input_data)

        indices = [
            resolved
            for item in items
            if item is not None and (resolved := resolve_entry(item)) is not None
        ]

        return self._normalize_output(return_type, indices)

    def find_raw(
        self,
        input_data: str
        | Path
        | pd.Series
        | list[str | Path | pd.Series]
        | pd.DataFrame,
        return_type: str = "auto",
    ) -> pd.Series | int | list[int] | pd.DataFrame | None:
        """
        Return the raw file(s) associated with one or more sidecar files (e.g., JSON, TSV).

        Parameters
        ----------
        input_data : str, Path, pd.Series, list[str | Path | pd.Series], or pd.DataFrame
            Single or multiple sidecar paths or entries.
        return_type : {'dataframe', 'index', 'auto'}, optional
            Determines the output format:
            - 'dataframe': always return DataFrame of results
            - 'index': always return list of row indices
            - 'auto': single result returns Series, multiple returns DataFrame (default)

        Returns
        -------
        Various formats depending on return_type:
            - pd.Series or pd.DataFrame
            - int or list of int
            - None (if no match found)

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        ValueError
            If a non-sidecar entry is provided or multiple raw files match.
        """

        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")

        def resolve_entry(item: str | Path | pd.Series) -> pd.Series | int | None:
            if isinstance(item, (str, Path)):
                entry = self.get_entry_from_path(item)
            elif isinstance(item, pd.Series):
                entry = item
            else:
                raise TypeError("Unsupported input type")

            if self._df is None:
                raise RuntimeError("You must call `.index()` first.")

            if entry is None:
                return None

            match_keys = {
                "sub",
                "ses",
                "task",
                "acq",
                "run",
                "space",
                "desc",
                "suffix",
            }

            keys_to_match = match_keys & set(entry.keys())

            conditions = [
                f"{k!s} == {repr(entry[k])}"
                for k in keys_to_match
                if pd.notna(entry[k])
            ]

            conditions.append("source == 'raw'")
            conditions.append("~is_sidecar")

            query_str = " and ".join(conditions)

            matches = self._df.query(query_str)

            if len(matches) > 1:
                raise ValueError(
                    f"Multiple associated raw files found for: {entry.get('path', '<unknown>')}"
                )
            elif len(matches) == 1:
                return matches.index[0]
            else:
                print(query_str)
                raise ValueError("Couldn't find raw for: " + str(entry))
                # return None

        items = self._normalize_input(input_data)

        indices = [
            resolved
            for item in items
            if item is not None and (resolved := resolve_entry(item)) is not None
        ]

        return self._normalize_output(return_type, indices)

    def find_derivative(
        self,
        input_data: str
        | Path
        | pd.Series
        | list[str | Path | pd.Series]
        | pd.DataFrame,
        return_type: str = "auto",
        flt: str | dict[str, str] | None = None,
    ) -> pd.Series | int | list[int] | pd.DataFrame | None:
        """
        Return the derivatives associated with one or more entries or files.

        Parameters
        ----------
        input_data : str, Path, pd.Series, list[str | Path | pd.Series], or pd.DataFrame
            Single or multiple sidecar paths or entries.
        return_type : {'dataframe', 'index', 'auto'}, optional
            Determines the output format:
            - 'dataframe': always return DataFrame of results
            - 'index': always return list of row indices
            - 'auto': single result returns Series, multiple returns DataFrame (default)
        flt : str dict[str, str] | None, optional
            Filter query, either as a string which can be passed to Pandas Query, or as a dictionary
            containing key value pairs to filter on

        Returns
        -------
        Various formats depending on return_type:
            - pd.Series or pd.DataFrame
            - int or list of int
            - None (if no match found)

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        ValueError
            If a non-sidecar entry is provided or multiple raw files match.
        """

        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")

        def resolve_entry(item: str | Path | pd.Series) -> pd.Series | int | None:
            if isinstance(item, (str, Path)):
                entry = self.get_entry_from_path(item)
            elif isinstance(item, pd.Series):
                entry = item
            else:
                raise TypeError("Unsupported input type")

            if self._df is None:
                raise RuntimeError("You must call `.index()` first.")

            if entry is None:
                return None

            match_keys = {
                "sub",
                "ses",
                "task",
                "acq",
                "run",
                "space",
                "desc",
                "suffix",
            }

            keys_to_match = match_keys & set(entry.keys())

            conditions = [
                f"{k!s} == {repr(entry[k])}"
                for k in keys_to_match
                if pd.notna(entry[k])
            ]

            conditions.append("source == 'derivative'")
            conditions.append("~is_sidecar")

            if isinstance(flt, dict):
                for key, value in flt.items():
                    conditions.append(f"{key!s} == {repr(value)}")
            elif isinstance(flt, str) and flt.strip():
                conditions.append(f"({flt})")

            query_str = " and ".join(conditions)

            matches = self._df.query(query_str)

            if len(matches) > 1:
                raise ValueError(
                    f"Multiple associated derivatives found for: {entry.get('path', '<unknown>')}"
                )
            elif len(matches) == 1:
                return matches.index[0]
            else:
                return None

        items = self._normalize_input(input_data)

        indices = [
            resolved
            for item in items
            if item is not None and (resolved := resolve_entry(item)) is not None
        ]

        return self._normalize_output(return_type, indices)

    def get_entry_from_path(self, path: str | Path) -> None | pd.Series:
        """
        Retrieve the DataFrame entry that matches the given file path.
        Supports absolute and relative paths (relative to root or derivative base).

        Parameters
        ----------
        path : str or Path
            Path to the file.

        Returns
        -------
        pd.Series or None
            Matching row from the DataFrame, or None if not found.

        Raises
        ------
        RuntimeError
            If `.index()` has not been called yet.
        ValueError
            If multiple entries match the given path.
        """
        if self._df is None:
            raise RuntimeError("You must call `.index()` first.")

        path = Path(path)

        # Try resolving path relative to BIDS root
        candidate_paths = [self._root / path]

        # If derivatives are included, try resolving relative to each pipeline base
        if self._include_derivatives:
            derivatives_root = self._root / "derivatives"
            for pipeline in self._df["pipeline"].dropna().unique():
                candidate_paths.append(derivatives_root / pipeline / path)

        # Also include fully resolved path
        candidate_paths.append(path.resolve())

        for candidate in candidate_paths:
            resolved = str(candidate.resolve())
            match = self._df[self._df["path"] == resolved]
            if len(match) > 1:
                raise ValueError(f"Multiple entries found for path: {resolved}")
            if len(match) == 1:
                return match.iloc[0]

        return None


def extract_entities_from_path(rel_path: Path) -> dict[str, Any]:
    """
    Extract BIDS entities and suffix from a relative file path.

    Parameters
    ----------
    rel_path : Path
        Path relative to the dataset root.

    Returns
    -------
    dict
        Dictionary of extracted BIDS entities and metadata.
    """
    entities: dict[str, str] = {}

    # Detect datatype
    parts = rel_path.parts
    for i, part in enumerate(parts):
        if part.startswith("sub-"):
            if i + 2 < len(parts) and parts[i + 1].startswith("ses-"):
                entities["datatype"] = parts[i + 2]
            elif i + 1 < len(parts):
                entities["datatype"] = parts[i + 1]
            break

    # Remove extension
    name = rel_path.name
    ext = ""
    for ext in sorted(VALID_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    if ext:
        entities["file_extension"] = ext

    # Parse entities
    suffix = None
    additional_suffixes = []
    for part in name.split("_"):
        match = ENTITY_PATTERN.fullmatch(part)
        if match:
            entities[match.group("key")] = match.group("value")
        else:
            if not suffix:
                suffix = part
            else:
                additional_suffixes.append(part)

    if suffix:
        entities["suffix"] = suffix

    if additional_suffixes:
        entities["additional_suffixes"] = "_".join(additional_suffixes)

    if "sub" in entities:
        entities["participant_id"] = "sub-" + entities["sub"]

    return entities


def is_valid_bids_file(fname: str) -> bool:
    """
    Check if a file has a valid BIDS extension.

    Parameters
    ----------
    fname : str
        Filename to check.

    Returns
    -------
    bool
        True if the file extension is valid.
    """
    return any(fname.endswith(ext) for ext in VALID_EXTENSIONS)


def is_sidecar_file(fname: str) -> bool:
    """
    Check if a file is a sidecar file.

    Parameters
    ----------
    fname : str
        Filename to check.

    Returns
    -------
    bool
        True if the file is a sidecar.
    """
    return any(fname.endswith(ext) for ext in SIDECAR_EXTENSIONS)


def resolve_data_root(path: Path, resolve_raw_root: bool = True) -> Path:
    """
    Resolve the dataset root directory given a file path by locating the 'sub-*' folder or the parent folder of a derivatives directory.

    Parameters
    ----------
    path : Path
        Path within the dataset directory structure.

    Returns
    -------
    Path
        Path to the dataset root (parent of the matched 'sub-*' folder or parent of 'derivatives' folder).
    """

    raw_dataset_root_or_derivative = next(
        p for p in path.parents if fnmatch.fnmatch(p.name, "sub-*")
    ).parent

    # Check if found a derivative and if the "raw" dataset root should be returned
    if resolve_raw_root and raw_dataset_root_or_derivative.parent.name == "derivatives":
        return raw_dataset_root_or_derivative.parent.parent

    return raw_dataset_root_or_derivative


def resolve_derivative(
    original: Path,
    derivative_name: str,
    suffix: str | None = None,
    extension: str | None = None,
) -> Path:
    """Resolve derivative path

    Parameters
    ----------
    original : Path
        Original path
    derivative_name : str
        Name of derivative
    suffix : str | None, optional
        Suffix, by default None
    extension : str | None, optional
        Extensions, by default None

    Returns
    -------
    Path
        Derivative path corresponding to the input path and options
    """
    dataset_root = resolve_data_root(original)
    local_dataset_root = resolve_data_root(original, resolve_raw_root=False)

    relative_path = original.relative_to(local_dataset_root)

    base_path = dataset_root / "derivatives" / derivative_name / relative_path

    stem, ext = base_path.name.split(".", maxsplit=1)

    if extension:
        ext = extension[1:] if extension.startswith(".") else extension

    suffix_part = f"_{suffix}" if suffix else ""

    derivative_filename = f"{stem}{suffix_part}.{ext}"

    return base_path.parent / derivative_filename


def resolve_sidecar(path: Path, extension: str = ".json") -> Path:
    """Get sidecar from path

    Parameters
    ----------
    path : Path
        Path of file
    extension : str, optional
        Extension of sidecar, by default ".json"

    Returns
    -------
    Path
        Path to sidecar
    """
    stem, _ = path.name.split(".", maxsplit=1)
    return path.parent / f"{stem}{extension}"


def add_suffix_to_path_name(path: Path, suffix: str) -> Path:
    """Adds suffix to a path name

    TODO: This method doesn't handle file names which includes dot in the name which is not part
    of the extensions.

    Parameters
    ----------
    path : Path
        Input path
    suffix : str
        Suffix

    Returns
    -------
    Path
        Path with suffix
    """
    stem, ext = path.name.split(".", maxsplit=1)
    return path.with_name(f"{stem}_{suffix}.{ext}")
