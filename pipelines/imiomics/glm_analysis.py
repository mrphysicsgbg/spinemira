from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage

from spinemira.core.logging import setup_logging
from spinemira.io.bids import Layout
from spinemira.pipelines.config import with_cli_config
from common.common import PipelineConfig, build_image_query, index_dataset

logger = logging.getLogger(__name__)


@dataclass
class GlmAnalysisPipelineConfig(PipelineConfig):
    images_query: str
    regressors: list[str]
    categorical_regressor_mappings: dict[str, dict[str, int]]
    output_dir: Path
    mask_path: Path | None
    two_sided_tests: bool
    reindex_dataset: bool
    n_jobs: int
    skip_contrast: bool
    n_permutations: int | None
    tfce: bool


def get_input(
    cnf: GlmAnalysisPipelineConfig, layout: Layout
) -> tuple[Sequence[Path], pd.DataFrame]:
    logger.info("Resolving input image paths...")

    if layout._df is None:
        raise ValueError("Database not loaded")

    df = layout._df.copy()

    # Fill metadata
    meta_cols = [column for column in df.columns if column.startswith("meta_")]
    keys = ["sub", "acq", "suffix"]

    df.update(df.groupby(keys)[meta_cols].transform(lambda g: g.ffill().bfill()))

    df_input_images = df.query(build_image_query(cnf.images_query, cnf.image_weight))

    df_input_images = df_input_images[cnf.regressors + ["path"]]

    # Filter out any images missing values for regressors
    df_nan = df_input_images[pd.isna(df_input_images).any(axis=1)]

    if len(df_nan):
        print(
            f"Skipping {len(df_nan)} images missing values for regressors: {', '.join([str(p) for p in df_nan['path'].to_list()])}"
        )

    df_input_images = df_input_images.drop(df_nan.index)
    input_image_paths: list[Path] = [Path(p) for p in df_input_images["path"].to_list()]

    logger.info(f"Found {len(input_image_paths)} to include.")

    return input_image_paths, df_input_images[cnf.regressors]


def create_design_matrix(
    cnf: GlmAnalysisPipelineConfig, df_regressors: pd.DataFrame, intercept: bool = True
) -> pd.DataFrame:
    # Create design matrix
    design_matrix = pd.DataFrame(df_regressors)

    if intercept:
        design_matrix["intercept"] = 1

    # Map categorical regressors
    for regressor, mapping in cnf.categorical_regressor_mappings.items():
        logger.info(f"Remapping regressor '{regressor}'")
        if regressor not in design_matrix.columns:
            raise ValueError(f"Regressor {regressor} is not present in design matrix.")
        design_matrix[regressor] = design_matrix[regressor].map(mapping)

    return design_matrix


def save_design_matrix(
    cnf: GlmAnalysisPipelineConfig,
    design_matrix: pd.DataFrame,
    list_images: Sequence[Path],
):
    design_matrix_with_paths = design_matrix.assign(image=[str(p) for p in list_images])
    design_matrix_out = cnf.output_dir / f"design_matrix_{cnf.image_weight}.csv"
    design_matrix_with_paths.to_csv(design_matrix_out, index=False)
    logger.info(f"Saved design matrix to {design_matrix_out}.")


def compute_contrast(
    cnf: GlmAnalysisPipelineConfig,
    list_images: Sequence[Path],
    design_matrix: pd.DataFrame,
    mask=FileBasedImage,
):
    logger.info("Fitting GLM...")

    # Fit GLM
    model = SecondLevelModel(
        n_jobs=cnf.n_jobs,
        mask_img=mask,
        verbose=2,
    )

    fitted_model = model.fit(list_images, design_matrix=design_matrix)

    # Compute contrasts
    for regressor in cnf.regressors:
        logger.info(f"Computing contrast for {regressor}")
        contrast = fitted_model.compute_contrast(
            second_level_contrast=regressor, output_type="all"
        )

        for key, image in contrast.items():
            filename = f"{regressor}_{key}_{cnf.image_weight}.nii.gz"
            image.to_filename(cnf.output_dir / filename)


def perform_non_parametric_inference(
    cnf: GlmAnalysisPipelineConfig,
    list_images: Sequence[Path],
    design_matrix: pd.DataFrame,
    mask=FileBasedImage,
):
    # Perform non parametric testing
    for regressor in cnf.regressors:
        logger.info(f"Performing non parametric inference for {regressor}")

        result = non_parametric_inference(
            list_images,
            design_matrix=design_matrix,
            second_level_contrast=regressor,
            mask=mask,
            model_intercept=True,
            two_sided_test=cnf.two_sided_tests,
            n_jobs=cnf.n_jobs,
            n_perm=cnf.n_permutations,
            tfce=cnf.tfce,
        )

        if isinstance(result, dict):
            for key, stat in result.items():
                filename = f"{regressor}_{key}_{cnf.image_weight}.nii.gz"
                stat.to_filename(cnf.output_dir / filename)
                logger.info(f"Saved {filename}")
        else:
            filename = f"{regressor}_neg_log10_fwer_pvals_{cnf.image_weight}.nii.gz"
            result.to_filename(cnf.output_dir / filename)
            logger.info(f"Saved {filename}")


@with_cli_config(log_fn=logger.info)
def run_pipeline(
    dataset_root: Path | str,
    image_weight: str,
    images_query: str,
    regressors: str,
    categorical_regressor_mappings: dict[str, dict[str, int]],
    output_dir: Path | str,
    mask_path: Path | str | None = None,
    two_sided_tests: bool = True,
    overwrite_existing: bool = False,
    load_sidecars: bool = False,
    reindex_dataset: bool = False,
    n_jobs: int = 1,
    n_permutations: int = 10000,
    tfce: bool = True,
    skip_contrast: bool = False,
):
    cnf = GlmAnalysisPipelineConfig(
        dataset_root=Path(dataset_root),
        output_derivative_name="",
        image_weight=image_weight,
        images_query=images_query,
        regressors=regressors.split(";"),
        categorical_regressor_mappings=categorical_regressor_mappings,
        output_dir=Path(output_dir),
        mask_path=Path(mask_path) if mask_path is not None else None,
        two_sided_tests=two_sided_tests,
        reindex_dataset=reindex_dataset,
        n_jobs=n_jobs,
        overwrite_existing=overwrite_existing,
        load_sidecars=load_sidecars,
        n_permutations=n_permutations,
        tfce=tfce,
        skip_contrast=skip_contrast,
    )

    cnf.output_dir.mkdir(exist_ok=True, parents=True)

    index_path = cnf.dataset_root / "index.csv"

    if reindex_dataset or not index_path.exists():
        layout = index_dataset(cnf)
        subject_metadata = pd.read_csv(cnf.dataset_root / "participants.tsv", sep="\t")
        layout.join_subject_metadata(subject_metadata, on="participant_id")
        layout.save_index(index_path, ignore_encoding_errors=True)
        logger.info("Saved index.")
    else:
        logger.info("Loading index...")
        layout = Layout(cnf.dataset_root)
        layout.load_index(index_path)

    # Get inputs
    list_images, df_regressors = get_input(cnf, layout)

    # Create design matrix
    design_matrix = create_design_matrix(cnf, df_regressors)
    save_design_matrix(cnf, design_matrix, list_images)

    # Load mask if specified
    if mask_path is not None:
        mask = nib.load(mask_path)
    else:
        mask = None

    # Compute contrasts
    if not cnf.skip_contrast:
        compute_contrast(cnf, list_images, design_matrix, mask)

    # Perform non parametric inference
    if cnf.n_permutations:
        perform_non_parametric_inference(cnf, list_images, design_matrix, mask)


if __name__ == "__main__":
    setup_logging(filename="glm_analysis.log")
    run_pipeline()
