from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import SimpleITK as sitk

from pandas import Series
import pandas as pd
import pydeform.sitk_api as pydeform

from spinemira.core.io import load_image, load_label_map
from spinemira.core.registration.metrics import negative_jacobian_count
from spinemira.core.registration.utils import get_translation_transform_from_landmarks
from spinemira.core.segmentation.labels import TotalSpineSegLabels
from spinemira.core.segmentation.metrics import dice
from spinemira.core.segmentation.utils import get_level_coordinates
from spinemira.io import bids
from spinemira.io.bids import (
    Layout,
    add_suffix_to_path_name,
    extract_entities_from_path,
    resolve_derivative,
    resolve_sidecar,
)

from .common import ImageBundle, PipelineConfig, build_image_query


logger = logging.getLogger(__name__)


@dataclass
class RegistrationPipelineConfig(PipelineConfig):
    moving_images_query: str
    segmentation_query: str
    levels_query: str
    pydeform: dict
    use_gpu: bool
    fixed_image_query: str | None
    fixed_image_bundle_path: Path | None


@dataclass(frozen=True)
class RegistrationTask:
    fixed: ImageBundle
    moving: ImageBundle
    settings: dict
    initial_transform: sitk.AffineTransform | None = None

    fixed_images: tuple[sitk.Image, ...] | None = None
    moving_images: tuple[sitk.Image, ...] | None = None

    def with_initial_transform_from_landmarks(self) -> "RegistrationTask":
        logger.info("Calculating initial transform from landmarks.")

        if self.moving.levels is None or self.fixed.levels is None:
            raise ValueError("Moving and fixed levels needs to be present")

        initial_transform = get_translation_transform_from_landmarks(
            get_level_coordinates(
                self.fixed.filtered_to_common_labels(self.moving).levels  # type: ignore[arg-type]
            ),
            get_level_coordinates(
                self.moving.filtered_to_common_labels(self.fixed).levels  # type: ignore[arg-type]
            ),
        )
        return replace(self, initial_transform=initial_transform)

    def with_segmentation_as_input_layers(
        self, intervals: Sequence[tuple[int, int]]
    ) -> "RegistrationTask":
        logger.info("Using intervals from label map as input images.")

        if self.fixed.label_map is None or self.moving.label_map is None:
            raise ValueError("Fixed and moving label maps need to be present.")

        def filter_to_intervals(label_map: sitk.Image) -> list[sitk.Image]:
            return [
                sitk.Cast(
                    sitk.And(label_map >= interval[0], label_map <= interval[1]) > 0,
                    sitk.sitkFloat32,
                )
                for interval in intervals
            ]

        return replace(
            self,
            fixed_images=(self.fixed.image, *filter_to_intervals(self.fixed.label_map)),
            moving_images=(
                self.moving.image,
                *filter_to_intervals(self.moving.label_map),
            ),
        )

    def run(self, use_gpu: bool = False) -> sitk.Image:
        logger.info("Starting registration...")

        moving_images = self.moving_images or (self.moving.image,)
        fixed_images = self.fixed_images or (self.fixed.image)

        # All input image are of the same data type
        data_type = fixed_images[0].GetPixelID()
        fixed_images = tuple([sitk.Cast(image, data_type) for image in fixed_images])
        moving_images = tuple([sitk.Cast(image, data_type) for image in moving_images])

        df = pydeform.register(
            moving_images=moving_images,
            fixed_images=fixed_images,
            moving_mask=self.moving.mask,
            affine_transform=self.initial_transform,
            use_gpu=use_gpu,
            settings=self.settings,
            silent=False,
            log_level=pydeform.LogLevel.Verbose,
        )
        logger.info("Registration done.")

        return df


@dataclass
class RegistrationJob:
    input_image: Path
    input_label_map: Path
    input_levels: Path

    output_image: Path
    output_label_map: Path
    output_levels: Path
    output_mask: Path
    output_df: Path


@dataclass(frozen=True)
class FixedResult:
    bundle: ImageBundle
    image_path: Path | None


def load_image_bundle(job: RegistrationJob) -> ImageBundle:
    logger.info(f"Loading images for input image {job.input_image}")

    return ImageBundle(
        image=load_image(job.input_image),
        label_map=load_label_map(job.input_label_map),
        levels=load_label_map(job.input_levels),
        mask=None,
    )


def save_image_bundle(
    job: RegistrationJob, image_bundle: ImageBundle, sidecar: dict | None = None
) -> None:
    logger.info(f"Saving output images for input image {job.input_image}...")

    job.output_image.parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(image_bundle.image, job.output_image)

    if sidecar:
        resolve_sidecar(job.output_image).write_text(
            json.dumps(sidecar, sort_keys=True, indent=4)
        )

    if image_bundle.label_map is not None:
        job.output_label_map.parent.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(image_bundle.label_map, job.output_label_map)

    if image_bundle.levels is not None:
        job.output_levels.parent.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(image_bundle.levels, job.output_levels)

    if image_bundle.mask is not None:
        job.output_mask.parent.mkdir(exist_ok=True, parents=True)
        sitk.WriteImage(image_bundle.mask, job.output_mask)


def get_fixed_input_image_path(cnf: RegistrationPipelineConfig, layout: Layout) -> Path:
    logger.info("Resolving fixed image...")

    assert cnf.fixed_image_query is not None

    df_fixed_image = layout.query(
        build_image_query(cnf.fixed_image_query, cnf.image_weight)
    )

    if len(df_fixed_image) != 1:
        raise RuntimeError(
            f"Found {len(df_fixed_image)} entries matching fixed image query. Only a single entry is required to be matched."
        )

    return Path(df_fixed_image.iloc[0]["path"])


def get_or_create_fixed(cnf: RegistrationPipelineConfig, layout: Layout) -> FixedResult:
    bundle_path = (
        add_suffix_to_path_name(cnf.fixed_image_bundle_path, cnf.image_weight)
        if cnf.fixed_image_bundle_path is not None
        else None
    )

    if bundle_path is not None and bundle_path.exists():
        logger.info(f"Loading fixed image bundle with main image {bundle_path}.")
        bundle = ImageBundle.load_from_same_directory(bundle_path)
        return FixedResult(bundle=bundle, image_path=None)

    fixed_image_path = get_fixed_input_image_path(cnf, layout)
    fixed_job = get_registration_job(cnf, layout, fixed_image_path)

    bundle = process_fixed(fixed_job)

    if bundle_path is not None:
        logger.info(f"Saving fixed image bundle to {bundle_path}")
        bundle.save_to_same_directory(bundle_path)

    return FixedResult(bundle=bundle, image_path=fixed_image_path)


def get_moving_input_image_paths(
    cnf: RegistrationPipelineConfig, layout: Layout, fixed_image_path: Path | None
) -> Sequence[Path]:
    logger.info("Resolving moving image paths...")

    df_moving_images = layout.query(
        build_image_query(cnf.moving_images_query, cnf.image_weight)
    )

    moving_image_paths: list[Path] = [
        Path(p) for p in df_moving_images["path"].to_list()
    ]

    if fixed_image_path is not None and fixed_image_path in moving_image_paths:
        moving_image_paths.remove(fixed_image_path)

    return moving_image_paths


def _require_derivative(layout: Layout, input_image: Path, flt: str, what: str) -> Path:
    series = layout.find_derivative(input_image, flt=flt)
    if not isinstance(series, Series):
        raise RuntimeError(f"Couldn't find {what} derivative for {input_image}.")
    return Path(series["path"])


def get_registration_job(
    cnf: RegistrationPipelineConfig, layout: Layout, input_image: Path
) -> RegistrationJob:
    logger.info(f"Resolving paths for {input_image}")

    # Inputs
    input_label_map = _require_derivative(
        layout, input_image, cnf.segmentation_query, "segmentation"
    )
    input_levels = _require_derivative(layout, input_image, cnf.levels_query, "levels")

    # Outputs
    output_image = resolve_derivative(input_image, cnf.output_derivative_name)
    output_label_map = resolve_derivative(
        input_image,
        cnf.output_derivative_name,
        suffix=extract_entities_from_path(input_label_map)["additional_suffixes"],
    )
    output_levels_map = resolve_derivative(
        input_image,
        cnf.output_derivative_name,
        suffix=extract_entities_from_path(input_levels)["additional_suffixes"],
    )
    output_mask = resolve_derivative(
        input_image, cnf.output_derivative_name, suffix="mask"
    )
    output_df = resolve_derivative(
        input_image, cnf.output_derivative_name, suffix="df", extension="h5"
    )

    return RegistrationJob(
        input_image=input_image,
        input_label_map=input_label_map,
        input_levels=input_levels,
        output_image=output_image,
        output_label_map=output_label_map,
        output_levels=output_levels_map,
        output_mask=output_mask,
        output_df=output_df,
    )


def get_moving_images_jobs(
    cnf: RegistrationPipelineConfig, layout: Layout, moving_image_paths: Sequence[Path]
) -> list[RegistrationJob]:
    logger.info("Resolving jobs for moving images...")

    jobs = [
        get_registration_job(cnf, layout, moving_image_path)
        for moving_image_path in moving_image_paths
    ]

    if not cnf.overwrite_existing:
        logger.info("Filtering already completed jobs...")
        jobs = [job for job in jobs if not job.output_image.exists()]

    logger.info(f"Compiled {len(jobs)} jobs to perform.")

    return jobs


def make_zero_displacement_field(ref: sitk.Image) -> sitk.Image:
    df = sitk.Image(ref.GetSize(), sitk.sitkVectorFloat64, ref.GetDimension())
    df.CopyInformation(ref)
    return df


def save_displacement_field(job: RegistrationJob, df: sitk.Image):
    sitk.WriteTransform(
        sitk.DisplacementFieldTransform(sitk.Cast(df, sitk.sitkVectorFloat64)),
        job.output_df,
    )


def evaluate_registration(
    fixed: ImageBundle,
    moving: ImageBundle,
    df: sitk.Image,
    filter_labels: set[int] | None = None,
) -> dict:
    logger.info("Evaluating registration...")

    if fixed.label_map is None or moving.label_map is None:
        raise ValueError("Fixed and moving label maps are required to be present.")

    dice_scores = dice(
        source_label_map=fixed.label_map,
        target_label_map=moving.label_map,
        labels_to_evaluate=filter_labels,
    )

    # Exchange label values to label names
    dice_scores = {
        TotalSpineSegLabels(int(k)).name if k.isnumeric() else k.upper(): v
        for k, v in dice_scores.items()
    }

    dice_table_lines = ["Label:".ljust(15) + " | DICE", "-" * 15 + "-|--------"]

    for k, v in dice_scores.items():
        dice_table_lines.append(f"{k.ljust(15)} | {v:.2f}")

    logger.info("Computed DICE scores:\n" + "\n".join(dice_table_lines))

    negative_jacobian = negative_jacobian_count(df)

    logger.info(f"Negative jacobian sum: {negative_jacobian}")

    return {
        "dice": dice_scores,
        "negative_jacobian_sum": negative_jacobian,
    }


def preprocess_image_bundle(
    bundle: ImageBundle, reference_bundle: ImageBundle | None = None
) -> ImageBundle:
    logger.info("Preprocessing image bundle...")
    bundle = (
        bundle.straightened()
        .with_harmonized_directions()
        .with_roi_mask()
        .normalized_to_csf(TotalSpineSegLabels.SPINAL_CANAL)
    )

    if reference_bundle is not None:
        bundle = bundle.filtered_to_common_labels(reference_bundle)

    return bundle


def run_registration(
    cnf: RegistrationPipelineConfig, fixed: ImageBundle, moving: ImageBundle
) -> sitk.Image:
    return (
        RegistrationTask(fixed, moving, settings=cnf.pydeform)
        .with_initial_transform_from_landmarks()
        .with_segmentation_as_input_layers(
            intervals=(
                (TotalSpineSegLabels.VERTEBRAE_T12, TotalSpineSegLabels.SACRUM),
                (TotalSpineSegLabels.DISC_T12_L1, TotalSpineSegLabels.DISC_L5_S),
            )
        )
        .run(cnf.use_gpu)
    )


def process_fixed(job: RegistrationJob) -> ImageBundle:
    fixed = load_image_bundle(job)
    fixed = preprocess_image_bundle(fixed)

    save_image_bundle(job, fixed)

    zero_df = make_zero_displacement_field(fixed.image)
    save_displacement_field(job, zero_df)

    return fixed


def collect_registration_metrics(
    cnf: RegistrationPipelineConfig, layout: Layout
) -> pd.DataFrame:
    logger.info("Collecting registration metrics...")

    data_rows = []

    df = layout._df

    if df is None:
        raise RuntimeError("Layout not indexed.")

    registration_sidecars = df[
        (df["pipeline"] == cnf.output_derivative_name)
        & (df["additional_suffixes"].isna())
        & (df["file_extension"] == ".json")
    ]["path"]

    for registration_sidecar in registration_sidecars:
        registration_sidecar_path = Path(registration_sidecar)

        with open(registration_sidecar, "r") as f:
            data = json.load(f)

        fields = bids.extract_entities_from_path(registration_sidecar_path)

        dice = {f"dice_{key}": value for key, value in data["dice"].items()}

        data = {
            "participant_id": fields["participant_id"],
            "suffix": fields["suffix"],
            "negative_jacobian_sum": data["negative_jacobian_sum"],
        }

        data.update(dice)
        data.update(fields)

        data_rows.append(data)

    return pd.DataFrame(data_rows)


def describe_dice(df: pd.DataFrame) -> pd.DataFrame:
    def _rename_dice_columns(name: str) -> str:
        if "dice" not in name:
            return name
        return name.split("_", maxsplit=1)[1]

    dice_columns = [name for name in df.columns.values if name.startswith("dice_")]
    df_dice = (
        df[
            [
                "suffix",
            ]
            + dice_columns
        ]
        .rename(_rename_dice_columns, axis="columns")
        .groupby("suffix")
    )

    df_described_long = (
        df_dice.describe()
        .stack(level=0)
        .reset_index()
        .rename(
            columns={
                "level_1": "label",
            }
        )
    )

    logger.info(f"DICE:\n\n{df_described_long.to_markdown(index=False)}")

    return df_described_long


def save_registration_metrics_to_derivative(
    cnf: RegistrationPipelineConfig, df: pd.DataFrame
):
    output_file = Path(
        cnf.dataset_root, "derivatives", cnf.output_derivative_name, "dice.tsv"
    )

    logger.info(f"Writing registration metrics to {output_file}")

    df.to_csv(output_file, sep="\t")
