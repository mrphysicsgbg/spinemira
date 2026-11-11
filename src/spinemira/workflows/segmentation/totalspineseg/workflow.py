from importlib.metadata import version
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from fileformats.generic import File
from fileformats.medimage import NiftiGz, NiftiGzX
from pydra.compose import python, workflow
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


from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.io.mids import Layout, resolve_derivative
from spinemira.tasks.segmentation import is_totalspineseg_available, run_totalspineseg

logger = logging.getLogger(__name__)


@python.define(
    outputs=[
        "sagittal",
        "axial",
        "sources",
        "sagittal_labels",
        "axial_labels",
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
    selected = selected[~selected["is_sidecar"]]
    sagittal = selected.query(sagittal_query)
    axial = selected.query(axial_query)
    if set(sagittal.path) & set(axial.path):
        raise ValueError("Sagittal and axial queries must select different images")

    groups: dict[str, dict[str, list[Path]]] = {}
    for _, row in sagittal.iterrows():
        group = groups.setdefault(row["sub"], {"sag": [], "ax": [], "sources": []})
        group["sag"].append(Path(row["path"]))
    for _, row in axial.iterrows():
        candidates = sagittal[sagittal["sub"].eq(row["sub"])]
        if "ses" in selected.columns:
            candidates = candidates[
                candidates["ses"].isna()
                if pd.isna(row.get("ses"))
                else candidates["ses"].eq(row["ses"])
            ]
        if "suffix" in candidates.columns and pd.notna(row.get("suffix")):
            same_weight = candidates[candidates["suffix"].eq(row["suffix"])]
            if not same_weight.empty:
                candidates = same_weight
        if len(candidates) != 1:
            logger.warning(
                "Skipping %s: expected one sagittal image after weight matching, found %d",
                row["path"],
                len(candidates),
            )
            continue
        group = groups[row["sub"]]
        group["ax"].append(Path(row["path"]))
        group["sources"].append(Path(candidates.iloc[0]["path"]))

    def dest(paths, suffix):
        return [
            resolve_derivative(p, derivative_folder=derivative, suffix=suffix)
            for p in paths
        ]

    return (
        [group["sag"] for group in groups.values()],
        [group["ax"] for group in groups.values()],
        [group["sources"] for group in groups.values()],
        [dest(group["sag"], "dseg") for group in groups.values()],
        [dest(group["ax"], "dseg") for group in groups.values()],
    )


@python.define(outputs=["localizers"])
def select_localizers(
    sagittal: list[Path], labels: list[NiftiGz], sources: list[Path]
) -> list[NiftiGz]:
    if len(sagittal) != len(labels) or any(label is None for label in labels):
        raise RuntimeError("Sagittal segmentation failed")
    mapping = dict(zip(sagittal, labels, strict=True))
    return [mapping[source] for source in sources]


@python.define(outputs=["label_map"])
def totalspineseg_localizer(label_map: NiftiGz) -> NiftiGz:
    """Keep only TotalSpineSeg labels in a segmentation used as a localizer."""
    image = sitk.ReadImage(str(label_map))
    values = sitk.GetArrayFromImage(image)
    values[~np.isin(values, [member.value for member in TotalSpineSegLabels])] = 0
    filtered = sitk.GetImageFromArray(values)
    filtered.CopyInformation(image)
    path = Path.cwd() / "totalspineseg_localizer.nii.gz"
    sitk.WriteImage(filtered, str(path))
    return NiftiGz(path)


@python.define(outputs=["label_maps"])
def combine_results(
    sagittal_labels: list[NiftiGzX],
    axial_labels: list[NiftiGzX],
):
    return sagittal_labels + axial_labels


@python.define(outputs=["label_maps"])
def flatten_results(groups: list[list[NiftiGzX]]) -> list[NiftiGzX]:
    return [label for group in groups for label in group]


@workflow.define(outputs=["label_map"])
def totalspineseg_workflow(
    image: NiftiGz,
    output_label_map: Path,
    data_dir: Path,
    device: str = "cpu",
    executable: str = "totalspineseg",
    quiet: bool = False,
    overwrite: bool = False,
    localizer: NiftiGz | None = None,
    no_stalling: bool = True,
):
    """Segment one image with TotalSpineSeg and publish its label sidecar."""
    sidecar_path = output_label_map.with_name(get_stem(output_label_map) + ".json")
    if output_label_map.exists() and sidecar_path.exists() and not overwrite:
        return NiftiGzX(output_label_map, sidecar_path)

    task = run_totalspineseg(
        image=image,
        data_dir=data_dir,
        device=device,
        quiet=quiet,
        executable=executable,
        localizer=localizer,
        no_stalling=no_stalling,
        collect_levels=False,
    )

    segmented = workflow.add(task, name="run_segmentation")

    label_map_with_sidecar = workflow.add(
        attach_sidecar_data(
            label_map=segmented.label_map,
            labels=label_metadata("TotalSpineSeg", TotalSpineSegLabels),
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
def totalspineseglocalizer_workflow(
    sagittal: list[Path],
    axial: list[Path],
    sources: list[Path],
    sagittal_labels: list[Path],
    axial_labels: list[Path],
    data_dir: Path,
    device: str = "cpu",
    executable: str = "totalspineseg",
    quiet: bool = False,
    overwrite: bool = False,
    no_stalling: bool = True,
):
    """Segment sagittal scans first and use them to localize axial scans."""
    options = dict(
        data_dir=data_dir,
        device=device,
        executable=executable,
        quiet=quiet,
        overwrite=overwrite,
        no_stalling=no_stalling,
    )
    sag = workflow.add(
        totalspineseg_workflow(**options)
        .split(
            ("image", "output_label_map"),
            image=sagittal,
            output_label_map=sagittal_labels,
        )
        .combine("image"),
        name="segment_sagittal",
    )
    localizers = workflow.add(
        select_localizers(sagittal=sagittal, labels=sag.label_map, sources=sources),
        name="select_localizers",
    )
    filtered = workflow.add(
        totalspineseg_localizer()
        .split("label_map", label_map=localizers.localizers)
        .combine("label_map"),
        name="filter_localizers",
    )
    ax = workflow.add(
        totalspineseg_workflow(**options)
        .split(
            ("image", "output_label_map", "localizer"),
            image=axial,
            output_label_map=axial_labels,
            localizer=filtered.label_map,
        )
        .combine("image"),
        name="segment_axial",
    )
    result = workflow.add(
        combine_results(sagittal_labels=sag.label_map, axial_labels=ax.label_map),
        name="combine_results",
    )
    return result.label_maps


@workflow.define(outputs=["label_map"])
def segment_dataset_workflow(
    dataset_root: Path,
    image_query: str,
    totalspineseg_data_dir: Path,
    output_derivative_name: str = "totalspineseg",
    device: str = "cpu",
    totalspineseg_executable: str = "totalspineseg",
    totalspineseg_quiet: bool = False,
    overwrite: bool = False,
    sagittal_first: bool = False,
    sagittal_query: str = '`vp` == "sag"',
    axial_query: str = '`vp` == "ax"',
    no_stalling: bool = True,
):
    """Segment a dataset, optionally localizing axial scans with sagittal scans."""
    if not is_totalspineseg_available(totalspineseg_executable):
        raise FileNotFoundError(
            f"Segmentation executable not available: {totalspineseg_executable}"
        )
    pipeline_name = "TotalSpineSeg - spinemira"
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
    if sagittal_first:
        related = workflow.add(
            select_related_images(
                dataset_root=dataset_root,
                mids_index=mids.file,
                image_query=image_query,
                sagittal_query=sagittal_query,
                axial_query=axial_query,
                derivative=derivative.path,
            ),
            name="select_related_images",
        )
        segmented = workflow.add(
            totalspineseglocalizer_workflow(
                data_dir=totalspineseg_data_dir,
                device=device,
                executable=totalspineseg_executable,
                quiet=totalspineseg_quiet,
                overwrite=overwrite,
                no_stalling=no_stalling,
            )
            .split(
                ("sagittal", "axial", "sources", "sagittal_labels", "axial_labels"),
                sagittal=related.sagittal,
                axial=related.axial,
                sources=related.sources,
                sagittal_labels=related.sagittal_labels,
                axial_labels=related.axial_labels,
            )
            .combine("sagittal"),
            name="segment_subjects",
        )
        result = workflow.add(
            flatten_results(groups=segmented.label_map), name="flatten_results"
        )
        return result.label_maps
    child = totalspineseg_workflow(
        data_dir=totalspineseg_data_dir,
        device=device,
        executable=totalspineseg_executable,
        quiet=totalspineseg_quiet,
        overwrite=overwrite,
        no_stalling=no_stalling,
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
