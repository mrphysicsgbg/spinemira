from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import json

import SimpleITK as sitk
from pandas import Series
import pandas as pd
import numpy as np

from spinemira.core.io import load_image, load_label_map
from spinemira.core.logging import setup_logging
from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.io import mids
from spinemira.io.mids import Layout, resolve_derivative
from spinemira.pipelines.config import with_cli_config

from common.common import ImageBundle

logger = logging.getLogger(__name__)

DISC_LABELS_TO_EVALUATE = [
    TotalSpineSegLabels.DISC_T11_T12,
    TotalSpineSegLabels.DISC_T12_L1,
    TotalSpineSegLabels.DISC_L1_L2,
    TotalSpineSegLabels.DISC_L2_L3,
    TotalSpineSegLabels.DISC_L3_L4,
    TotalSpineSegLabels.DISC_L4_L5,
    TotalSpineSegLabels.DISC_L5_S,
]

SPINAL_CANAL_LABELS = [
    TotalSpineSegLabels.SPINAL_CANAL,
    TotalSpineSegLabels.SPINAL_CORD,
]

META_DATA_KEY_PREFIX = "dsca_estimated_"


@dataclass
class EstimateDscaPipelineConfig:
    dataset_root: Path
    output_derivative_name: str
    images_query: str
    segmentation_query: str
    levels_query: str
    load_sidecars: bool
    overwrite_existing: bool
    output_summary_file: Path | None
    output_narrowest_file: Path | None
    output_isolated_narrowest_file: Path | None
    compute_narrowest_level_for_levels: list[str] | None
    compute_narrowest_isolated_level_for_levels: list[str] | None


@dataclass(frozen=True)
class EstimateDscaLevelJob:
    input_image: Path
    input_label_map: Path
    input_levels: Path
    output_file: Path


def _require_derivative(layout: Layout, input_image: Path, flt: str, what: str) -> Path:
    series = layout.find_derivative(input_image, flt=flt)
    if not isinstance(series, Series):
        raise RuntimeError(f"Couldn't find {what} derivative for {input_image}.")
    return Path(series["path"])


def load_image_bundle(job: EstimateDscaLevelJob) -> ImageBundle:
    return ImageBundle(
        image=load_image(job.input_image),
        label_map=load_label_map(job.input_label_map),
        levels=load_label_map(job.input_levels),
    )


def compute_spinal_canal_areas(image_bundle: ImageBundle) -> dict[str, float]:
    straightened_image_bundle = image_bundle.straightened()

    label_map = straightened_image_bundle.label_map

    if label_map is None:
        raise ValueError

    # Find z-indexes for discs

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label_map)

    disc_z_levels = {}
    for label in DISC_LABELS_TO_EVALUATE:
        if label not in label_shape_filter.GetLabels():
            continue

        # Find spinal level based on segmented disc
        disc_centroid = label_map.TransformPhysicalPointToIndex(
            label_shape_filter.GetCentroid(label)
        )[::-1]

        disc_z_levels[label] = disc_centroid[0]

    # Get intervals for which region of the spinal canal is closes to which disc.
    sorted_disc_labels = sorted(
        disc_z_levels.keys(), key=lambda label: disc_z_levels[label]
    )
    sorted_z_indexes = [disc_z_levels[label] for label in sorted_disc_labels]
    z_indexes_mid = [
        int((a + b) / 2) for a, b in zip(sorted_z_indexes, sorted_z_indexes[1:])
    ]
    z_indexes_bounds = [sorted_z_indexes[0]] + z_indexes_mid + [sorted_z_indexes[-1]]

    disc_z_levels_intervals = {
        label: (z_indexes_bounds[i], z_indexes_bounds[i + 1])
        for i, label in enumerate(sorted_disc_labels)
    }

    # Get narrowest area per disc interval

    label_map_arr = sitk.GetArrayViewFromImage(label_map)
    label_map_arr_spinal_canal = np.zeros_like(label_map_arr)
    label_map_arr_spinal_canal[np.isin(label_map_arr, SPINAL_CANAL_LABELS)] = 1

    spinal_canal_cross_sectional_areas = {}
    axial_plane_voxel_size_mm2 = label_map.GetSpacing()[0] * label_map.GetSpacing()[1]

    for label, (z_index_start, z_index_end) in disc_z_levels_intervals.items():
        spinal_canal_narrowest_voxel_area = np.min(
            np.sum(
                label_map_arr_spinal_canal[z_index_start:z_index_end, :, :], axis=(1, 2)
            )
        )
        spinal_canal_cross_sectional_areas[
            META_DATA_KEY_PREFIX + TotalSpineSegLabels(label).name
        ] = spinal_canal_narrowest_voxel_area * axial_plane_voxel_size_mm2

    return spinal_canal_cross_sectional_areas


def get_input_image_paths(
    cnf: EstimateDscaPipelineConfig, layout: Layout
) -> Sequence[Path]:
    logger.info("Resolving input image paths...")

    df_input_images = layout.query(cnf.images_query + " and ~is_sidecar")

    input_image_paths: list[Path] = [Path(p) for p in df_input_images["path"].to_list()]

    return input_image_paths


def get_job(
    cnf: EstimateDscaPipelineConfig,
    layout: Layout,
    input_image_path: Path,
) -> EstimateDscaLevelJob:
    logger.info(f"Resolving paths for {input_image_path}")

    input_label_map_path = _require_derivative(
        layout, input_image_path, cnf.segmentation_query, "segmentation"
    )
    input_levels_path = _require_derivative(
        layout, input_image_path, cnf.levels_query, "levels"
    )
    output_file_path = resolve_derivative(
        input_image_path, cnf.output_derivative_name, extension=".json"
    )

    return EstimateDscaLevelJob(
        input_image=input_image_path,
        input_label_map=input_label_map_path,
        input_levels=input_levels_path,
        output_file=output_file_path,
    )


def get_jobs(
    cnf: EstimateDscaPipelineConfig,
    layout: Layout,
    image_paths: Sequence[Path],
) -> list[EstimateDscaLevelJob]:
    logger.info("Resolving jobs...")

    jobs = [get_job(cnf, layout, image_path) for image_path in image_paths]

    if not cnf.overwrite_existing:
        logger.info("Filtering already completed jobs...")
        jobs = [job for job in jobs if not job.output_file.exists()]

    logger.info(f"Compiled {len(jobs)} jobs to perform.")

    return jobs


def process_jobs(
    cnf: EstimateDscaPipelineConfig,
    jobs: list[EstimateDscaLevelJob],
):
    num_jobs = len(jobs)
    for index, job in enumerate(jobs):
        logger.info(f"Processing job {index + 1}/{num_jobs} ({str(job.input_image)})")

        image_bundle = load_image_bundle(job)
        spinal_canal_cross_sectional_areas = compute_spinal_canal_areas(image_bundle)

        job.output_file.parent.mkdir(parents=True, exist_ok=True)
        job.output_file.write_text(
            json.dumps(spinal_canal_cross_sectional_areas, indent=2)
        )
        logger.info(f"Saved {job.output_file}.")


def process_summary(
    cnf: EstimateDscaPipelineConfig,
    jobs: list[EstimateDscaLevelJob],
):
    logging.info("Processing summary...")

    output_file = cnf.output_summary_file

    if output_file is None:
        return

    cross_sectional_areas = []

    num_jobs = len(jobs)
    for index, job in enumerate(jobs):
        logger.info(f"Processing file {index + 1}/{num_jobs} ({str(job.output_file)})")

        if not job.output_file.exists():
            logger.info("Skipping not existing file.")
            continue

        entities = mids.extract_entities_from_path(job.output_file)
        spinal_canal_cross_sectional_areas: dict[str, float] = json.loads(
            job.output_file.read_text()
        )
        spinal_canal_cross_sectional_areas = {
            k.removeprefix(META_DATA_KEY_PREFIX): v
            for k, v in spinal_canal_cross_sectional_areas.items()
        }

        areas = np.asarray(list(spinal_canal_cross_sectional_areas.values()))

        narrowest_divided_by_widest = np.min(areas) / np.max(areas)

        cross_sectional_areas.append(
            {
                "sub": entities["sub"],
                "suffix": entities["suffix"],
                "acq": entities["acq"],
                "ses": entities["ses"],
                "narrowest_relative_widest": narrowest_divided_by_widest,
            }
            | spinal_canal_cross_sectional_areas
        )

    cross_sectional_areas_df = pd.DataFrame(cross_sectional_areas)
    save_dataframe(cross_sectional_areas_df, output_file)


def process_narrowest_level_per_subject(
    cnf: EstimateDscaPipelineConfig, jobs: list[EstimateDscaLevelJob]
):
    logging.info("Computing isolated narrowest level per subject...")

    output_file = cnf.output_narrowest_file

    if output_file is None:
        return

    if cnf.compute_narrowest_level_for_levels is not None:
        levels_to_evaluate = {
            label.upper().strip() for label in cnf.compute_narrowest_level_for_levels
        }
    else:
        levels_to_evaluate = {label.name for label in DISC_LABELS_TO_EVALUATE}

    narrowest_level_per_subject_and_contrast: dict[str, dict] = {}
    num_jobs = len(jobs)
    for index, job in enumerate(jobs):
        logger.info(f"Processing file {index + 1}/{num_jobs} ({str(job.output_file)})")

        if not job.output_file.exists():
            logger.info("Skipping not existing file.")
            continue

        entities = mids.extract_entities_from_path(job.output_file)

        # Load cross sectional areas
        spinal_canal_cross_sectional_areas: dict[str, float] = json.loads(
            job.output_file.read_text()
        )

        # Filter and normalize keys
        spinal_canal_cross_sectional_areas = {
            k.removeprefix(META_DATA_KEY_PREFIX): v
            for k, v in spinal_canal_cross_sectional_areas.items()
            if k.removeprefix(META_DATA_KEY_PREFIX) in levels_to_evaluate
        }

        # Find smallest level
        narrowest_spinal_level, _ = min(
            spinal_canal_cross_sectional_areas.items(), key=lambda x: x[1]
        )

        if entities["sub"] not in narrowest_level_per_subject_and_contrast:
            narrowest_level_per_subject_and_contrast[entities["sub"]] = {}

        narrowest_level_per_subject_and_contrast[entities["sub"]][
            entities["suffix"]
        ] = narrowest_spinal_level

    narrowest_level_per_subject = []

    for sub in sorted(
        narrowest_level_per_subject_and_contrast.keys(), key=lambda level: int(level)
    ):
        narrowest_level = (
            narrowest_level_per_subject_and_contrast[sub]["T2w"]
            if "T2w" in narrowest_level_per_subject_and_contrast[sub].keys()
            else narrowest_level_per_subject_and_contrast[sub]["T1w"]
        )
        narrowest_level_per_subject.append(
            {"sub": f"sub-{sub}", "level": narrowest_level}
        )

    narrowest_level_per_subject_df = pd.DataFrame(narrowest_level_per_subject)
    save_dataframe(narrowest_level_per_subject_df, output_file)


def process_isolated_narrowest_level_per_subject(
    cnf: EstimateDscaPipelineConfig, jobs: list[EstimateDscaLevelJob]
):
    logging.info("Computing isolated narrowest level per subject...")

    output_file = cnf.output_isolated_narrowest_file

    if output_file is None:
        return

    if cnf.compute_narrowest_isolated_level_for_levels is not None:
        levels_to_evaluate = {
            label.upper().strip()
            for label in cnf.compute_narrowest_isolated_level_for_levels
        }
    else:
        levels_to_evaluate = {label.name for label in DISC_LABELS_TO_EVALUATE}

    narrowest_level_per_subject_and_contrast: dict[str, dict] = {}
    num_jobs = len(jobs)
    for index, job in enumerate(jobs):
        logger.info(f"Processing file {index + 1}/{num_jobs} ({str(job.output_file)})")

        if not job.output_file.exists():
            logger.info("Skipping not existing file.")
            continue

        entities = mids.extract_entities_from_path(job.output_file)

        # Load cross sectional areas
        spinal_canal_cross_sectional_areas: dict[str, float] = json.loads(
            job.output_file.read_text()
        )

        # Filter and normalize keys
        spinal_canal_cross_sectional_areas = {
            k.removeprefix(META_DATA_KEY_PREFIX): v
            for k, v in spinal_canal_cross_sectional_areas.items()
            if k.removeprefix(META_DATA_KEY_PREFIX) in levels_to_evaluate
        }

        # Find isolated smallest level by finding the smallest level and ensuring there are
        # no other levels where the cross sectional area is less than 50 % of the median cross sectional area

        narrowest_spinal_level, smallest_area = min(
            spinal_canal_cross_sectional_areas.items(), key=lambda x: x[1]
        )

        median_area = np.median(
            [area for _, area in spinal_canal_cross_sectional_areas.items()]
        )
        is_isolated_narrowest = True

        for level, area in spinal_canal_cross_sectional_areas.items():
            if level == narrowest_spinal_level:
                continue
            if area < median_area - ((median_area - smallest_area) * 0.5):
                logger.info(
                    f"Level {level} has area: {area}. Median area is {median_area} and smallest area is {smallest_area}"
                )
                is_isolated_narrowest = False

        if entities["sub"] not in narrowest_level_per_subject_and_contrast:
            narrowest_level_per_subject_and_contrast[entities["sub"]] = {}

        narrowest_level_per_subject_and_contrast[entities["sub"]][
            entities["suffix"]
        ] = narrowest_spinal_level if is_isolated_narrowest else None

    narrowest_level_per_subject = []

    for sub in sorted(
        narrowest_level_per_subject_and_contrast.keys(), key=lambda level: int(level)
    ):
        narrowest_level = (
            narrowest_level_per_subject_and_contrast[sub]["T2w"]
            if "T2w" in narrowest_level_per_subject_and_contrast[sub].keys()
            else narrowest_level_per_subject_and_contrast[sub]["T1w"]
        )

        if narrowest_level is None:
            continue

        narrowest_level_per_subject.append(
            {"sub": f"sub-{sub}", "level": narrowest_level}
        )

    narrowest_level_per_subject_df = pd.DataFrame(narrowest_level_per_subject)
    save_dataframe(narrowest_level_per_subject_df, output_file)


def save_dataframe(df: pd.DataFrame, file: str | Path):
    path = Path(file)
    path.parent.mkdir(exist_ok=True, parents=True)

    ext = path.suffix.lower()

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".tsv":
        df.to_csv(path, index=False, sep="\t")
    elif ext == ".json":
        df.to_json(path, indent=2, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.info(f"Saved file {path}")


@with_cli_config(log_fn=logger.info)
def run_pipeline(
    dataset_root: Path,
    images_query: str,
    segmentation_query: str,
    levels_query: str,
    output_derivative_name: str,
    output_summary_file: Path | None = None,
    output_narrowest_file: Path | None = None,
    output_isolated_narrowest_file: Path | None = None,
    overwrite_existing: bool = False,
    load_sidecars: bool = False,
    compute_narrowest_level_for_levels: list[str] | None = None,
    compute_narrowest_isolated_level_for_levels: list[str] | None = None,
):
    cnf = EstimateDscaPipelineConfig(
        dataset_root=Path(dataset_root),
        images_query=images_query,
        segmentation_query=segmentation_query,
        levels_query=levels_query,
        load_sidecars=load_sidecars,
        overwrite_existing=overwrite_existing,
        output_derivative_name=output_derivative_name,
        output_summary_file=output_summary_file,
        output_narrowest_file=output_narrowest_file,
        output_isolated_narrowest_file=output_isolated_narrowest_file,
        compute_narrowest_level_for_levels=compute_narrowest_level_for_levels,
        compute_narrowest_isolated_level_for_levels=compute_narrowest_isolated_level_for_levels,
    )

    # Index dataset
    logging.info("Loading dataset...")
    layout = Layout(root=cnf.dataset_root, include_derivatives=True)
    layout.index(load_sidecars=cnf.load_sidecars)
    logging.info("Dataset indexed.")

    layout = Layout(root=cnf.dataset_root, include_derivatives=True)
    layout.load_index(cnf.dataset_root / "index.csv")

    # Get inputs
    image_paths = get_input_image_paths(cnf, layout)
    jobs = get_jobs(cnf, layout, image_paths)

    # Process inputs / jobs
    process_jobs(cnf, jobs)

    # Collect summary
    if cnf.output_summary_file is not None:
        process_summary(cnf, jobs)

    # Collect narrowest level per subject
    if cnf.output_narrowest_file is not None:
        process_narrowest_level_per_subject(cnf, jobs)

    # Collect isolated narrowest level per subject
    if cnf.output_isolated_narrowest_file is not None:
        process_isolated_narrowest_level_per_subject(cnf, jobs)


if __name__ == "__main__":
    setup_logging(filename="estimate_dsca.log")
    run_pipeline()
