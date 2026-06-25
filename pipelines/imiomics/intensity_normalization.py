from collections.abc import Sequence
from dataclasses import dataclass, replace
import logging
from pathlib import Path

import SimpleITK as sitk
from pandas import Series

from spinemira.core.filters import multiple_regions_histogram_matching, reduce_label_map
from spinemira.core.io import load_image, load_label_map
from spinemira.core.logging import setup_logging
from spinemira.io import mids
from spinemira.io.mids import Layout, resolve_derivative
from spinemira.pipelines.config import with_cli_config

from common.common import ImageBundle, PipelineConfig, build_image_query, index_dataset


logger = logging.getLogger(__name__)


@dataclass
class IntensityNormalizationPipelineConfig(PipelineConfig):
    images_query: str
    segmentation_query: str
    mask_query: str
    load_sidecars: bool
    overwrite_existing: bool
    reference_image_bundle_path: Path
    segmentation_label_intervals: dict[str, tuple[int, int]]
    num_bins: int
    background_rel_weight: float


@dataclass(frozen=True)
class IntensityNormalizationJob:
    input_image: Path
    input_label_map: Path
    input_mask: Path
    output_image: Path


def _require_derivative(layout: Layout, input_image: Path, flt: str, what: str) -> Path:
    series = layout.find_derivative(input_image, flt=flt)
    if not isinstance(series, Series):
        raise RuntimeError(f"Couldn't find {what} derivative for {input_image}.")
    return Path(series["path"])


def get_reference_image_bundle(
    cnf: IntensityNormalizationPipelineConfig,
) -> ImageBundle:
    if cnf.reference_image_bundle_path.exists():
        path = cnf.reference_image_bundle_path
    else:
        path = mids.add_suffix_to_path_name(
            cnf.reference_image_bundle_path, cnf.image_weight
        )

    logger.info(f"Loading reference image bundle {path}")

    return ImageBundle.load_from_same_directory(path)


def load_image_bundle(job: IntensityNormalizationJob) -> ImageBundle:
    return ImageBundle(
        image=load_image(job.input_image),
        label_map=load_label_map(job.input_label_map),
        mask=load_label_map(job.input_mask),
    )


def image_bundle_reduce_label_map(
    cnf: IntensityNormalizationPipelineConfig, image_bundle: ImageBundle
) -> ImageBundle:
    intervals: list[tuple[int, int]] = [
        v for _, v in cnf.segmentation_label_intervals.items()
    ]
    new_labels = list(range(1, len(intervals) + 1))

    label_map = image_bundle.label_map

    if label_map is None:
        raise ValueError("Label map needs to be present.")

    reduced_label_map = reduce_label_map(label_map, intervals, new_labels)

    return replace(image_bundle, label_map=reduced_label_map)


def get_input_image_paths(
    cnf: IntensityNormalizationPipelineConfig, layout: Layout
) -> Sequence[Path]:
    logger.info("Resolving input image paths...")

    df_input_images = layout.query(
        build_image_query(cnf.images_query, cnf.image_weight)
    )

    input_image_paths: list[Path] = [Path(p) for p in df_input_images["path"].to_list()]

    return input_image_paths


def get_job(
    cnf: IntensityNormalizationPipelineConfig, layout: Layout, input_image_path: Path
) -> IntensityNormalizationJob:
    logger.info(f"Resolving paths for {input_image_path}")

    input_label_map_path = _require_derivative(
        layout, input_image_path, cnf.segmentation_query, "segmentation"
    )
    input_mask_path = _require_derivative(
        layout, input_image_path, cnf.mask_query, "segmentation"
    )
    output_image_path = resolve_derivative(input_image_path, cnf.output_derivative_name)

    return IntensityNormalizationJob(
        input_image=input_image_path,
        input_label_map=input_label_map_path,
        input_mask=input_mask_path,
        output_image=output_image_path,
    )


def get_jobs(
    cnf: IntensityNormalizationPipelineConfig,
    layout: Layout,
    image_paths: Sequence[Path],
) -> list[IntensityNormalizationJob]:
    logger.info("Resolving jobs...")

    jobs = [get_job(cnf, layout, image_path) for image_path in image_paths]

    if not cnf.overwrite_existing:
        logger.info("Filtering already completed jobs...")
        jobs = [job for job in jobs if not job.output_image.exists()]

    logger.info(f"Compiled {len(jobs)} jobs to perform.")

    return jobs


def normalize_image_bundle(
    cnf: IntensityNormalizationPipelineConfig,
    image_bundle: ImageBundle,
    reference_image_bundle: ImageBundle,
) -> ImageBundle:
    src_label_map = image_bundle.label_map
    ref_label_map = reference_image_bundle.label_map

    if src_label_map is None or ref_label_map is None:
        raise ValueError(
            "Label maps needs to be present to perform intensity normalization."
        )

    filtered_image = multiple_regions_histogram_matching(
        src_image=image_bundle.image,
        src_label_map=src_label_map,
        ref_image=reference_image_bundle.image,
        ref_label_map=ref_label_map,
        src_mask=image_bundle.mask,
        ref_mask=reference_image_bundle.mask,
        num_bins=cnf.num_bins,
        bg_rel_weight=cnf.background_rel_weight,
    )

    return replace(image_bundle, image=filtered_image)


def process_jobs(
    cnf: IntensityNormalizationPipelineConfig,
    jobs: list[IntensityNormalizationJob],
    reference_image_bundle: ImageBundle,
):
    reference_image_bundle_reduced = image_bundle_reduce_label_map(
        cnf, reference_image_bundle
    )

    num_jobs = len(jobs)
    for index, job in enumerate(jobs):
        logger.info(f"Processing job {index + 1}/{num_jobs}")

        image_bundle = load_image_bundle(job)
        image_bundle_reduced = image_bundle_reduce_label_map(cnf, image_bundle)

        processed_image_bundle = normalize_image_bundle(
            cnf, image_bundle_reduced, reference_image_bundle_reduced
        )

        logger.info(f"Saving filtered image: {job.output_image}.")
        image = sitk.Cast(processed_image_bundle.image, sitk.sitkFloat32)
        job.output_image.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, job.output_image)


@with_cli_config(log_fn=logger.info)
def run_pipeline(
    dataset_root: Path,
    images_query: str,
    segmentation_query: str,
    mask_query: str,
    output_derivative_name: str,
    image_weight: str,
    reference_image_bundle_path: Path,
    segmentation_label_intervals: dict,
    load_sidecars: bool = False,
    overwrite_existing: bool = False,
    background_rel_weight: float = 0.1,
    num_bins: int = 512,
):
    parsed_segmentation_label_intervals: dict[str, tuple] = {
        str(k): tuple(eval(v)) for k, v in segmentation_label_intervals.items()
    }

    cnf = IntensityNormalizationPipelineConfig(
        dataset_root=dataset_root,
        images_query=images_query,
        mask_query=mask_query,
        segmentation_query=segmentation_query,
        output_derivative_name=output_derivative_name,
        image_weight=image_weight,
        segmentation_label_intervals=parsed_segmentation_label_intervals,
        reference_image_bundle_path=Path(reference_image_bundle_path),
        load_sidecars=load_sidecars,
        overwrite_existing=overwrite_existing,
        background_rel_weight=background_rel_weight,
        num_bins=num_bins,
    )

    # Index the dataset
    layout = index_dataset(cnf)

    # Get inputs
    image_paths = get_input_image_paths(cnf, layout)
    jobs = get_jobs(cnf, layout, image_paths)

    # Load reference image
    reference = get_reference_image_bundle(cnf)

    # Process jobs
    process_jobs(cnf, jobs, reference)


if __name__ == "__main__":
    setup_logging(filename="intensity_normalization_pipeline.log")
    run_pipeline()
