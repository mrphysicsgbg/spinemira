"""Pydra tasks for related-image segmentation and label sidecars."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fileformats.application import Json
from fileformats.generic import File
from fileformats.medimage import NiftiGz, NiftiGzX
from pydra.compose import python

from spinemira.core.segmentation.labels import MuscleMapLabels, TotalSpineSegLabels
from spinemira.io.mids import Layout, resolve_derivative

logger = logging.getLogger(__name__)


@python.define(
    outputs=[
        "sagittal",
        "axial",
        "sources",
        "sagittal_labels",
        "sagittal_levels",
        "axial_labels",
        "axial_levels",
    ]
)
def select_related_images(
    dataset_root: Path,
    mids_index: File,
    image_query: str,
    sagittal_query: str,
    axial_query: str,
    derivative: Path,
):
    layout = Layout(dataset_root, include_derivatives=True)
    layout.load_index(Path(mids_index))
    selected = layout.query(image_query)
    print("Selected")
    print(selected)
    selected = selected[~selected["is_sidecar"]]
    print("Not sidecar:")
    print(selected)
    sagittal = selected.query(sagittal_query)
    axial = selected.query(axial_query)
    if set(sagittal.path) & set(axial.path):
        raise ValueError("Sagittal and axial queries must select different images")

    sag = [Path(p) for p in sagittal.path]
    ax, sources = [], []
    for _, row in axial.iterrows():
        candidates = sagittal[sagittal["sub"].eq(row["sub"])]
        if "ses" in selected.columns:
            candidates = candidates[
                candidates["ses"].isna()
                if pd.isna(row.get("ses"))
                else candidates["ses"].eq(row["ses"])
            ]
        if len(candidates) != 1:
            logger.warning(
                "Skipping %s: expected one sagittal image in the session, found %d",
                row["path"],
                len(candidates),
            )
            continue
        ax.append(Path(row["path"]))
        sources.append(Path(candidates.iloc[0]["path"]))

    def dest(paths, suffix):
        return [
            resolve_derivative(p, derivative_folder=derivative, suffix=suffix)
            for p in paths
        ]

    print("Test query:")
    print(sagittal_query)

    ret = (
        sag,
        ax,
        sources,
        dest(sag, "dseg"),
        dest(sag, "levels"),
        dest(ax, "dseg"),
        dest(ax, "levels"),
    )

    print(ret)

    return ret


@python.define(outputs=["localizers"])
def select_localizers(
    sagittal: list[Path], labels: list[NiftiGz], sources: list[Path]
) -> list[NiftiGz]:
    if len(sagittal) != len(labels) or any(label is None for label in labels):
        raise RuntimeError("Sagittal segmentation failed")
    mapping = dict(zip(sagittal, labels, strict=True))
    return [mapping[source] for source in sources]


@python.define(outputs=["file"])
def create_label_sidecar(
    label_map: NiftiGz, use_totalspineseg: bool, use_musclemap: bool
) -> Json:
    values = sorted(
        int(v)
        for v in np.unique(sitk.GetArrayFromImage(sitk.ReadImage(str(label_map))))
        if v != 0
    )
    names: dict[int, list[dict[str, str]]] = {}
    for enabled, name, enum in (
        (use_totalspineseg, "TotalSpineSeg", TotalSpineSegLabels),
        (use_musclemap, "MuscleMap", MuscleMapLabels),
    ):
        if enabled:
            for member in enum:
                names.setdefault(member.value, []).append(
                    {"source": name, "name": member.name}
                )
    labels = {str(value): names.get(value, []) for value in values}
    path = Path.cwd() / "dseg.json"
    path.write_text(json.dumps({"Labels": labels}, indent=2) + "\n")
    return Json(path)


@python.define(outputs=["label_map"])
def require_sidecar(label_map: NiftiGz, sidecar: Json) -> NiftiGzX:
    """Return the segmentation and its published JSON as one file set."""
    return NiftiGzX(label_map.fspath, sidecar.fspath)


@python.define(outputs=["label_maps", "levels"])
def combine_results(
    sagittal_labels: list[NiftiGzX],
    sagittal_levels: list[NiftiGz],
    axial_labels: list[NiftiGzX],
    axial_levels: list[NiftiGz],
):
    return sagittal_labels + axial_labels, sagittal_levels + axial_levels
