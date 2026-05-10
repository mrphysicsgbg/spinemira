import logging
from pathlib import Path

import yaml

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config

from common.common import ImageBundle, index_dataset
from common.registration import (
    RegistrationJob,
    RegistrationPipelineConfig,
    collect_registration_metrics,
    describe_dice,
    evaluate_registration,
    get_moving_images_jobs,
    get_moving_input_image_paths,
    get_or_create_fixed,
    load_image_bundle,
    preprocess_image_bundle,
    run_registration,
    save_displacement_field,
    save_image_bundle,
    save_registration_metrics_to_derivative,
)


logger = logging.getLogger(__name__)


def process_moving(
    cnf: RegistrationPipelineConfig, jobs: list[RegistrationJob], fixed: ImageBundle
):
    for job in jobs:
        moving = load_image_bundle(job)
        moving = preprocess_image_bundle(moving, reference_bundle=fixed)
        df = run_registration(cnf, fixed, moving)
        registered = moving.apply_deformation_field(df)
        evaluation = evaluate_registration(fixed, registered, df)
        save_image_bundle(job, registered, sidecar=evaluation)
        save_displacement_field(job, df)


@with_cli_config(log_fn=logger.info)
def run_pipeline(
    dataset_root: Path,
    moving_images_query: str,
    segmentation_query: str,
    levels_query: str,
    output_derivative_name: str,
    image_weight: str,
    load_sidecars: bool = False,
    overwrite_existing: bool = False,
    pydeform: Path | str | dict | None = None,
    use_gpu: bool = False,
    fixed_image_query: str | None = None,
    fixed_image_bundle_path: Path | str | None = None,
):
    loaded_pydeform_config: dict | None

    if isinstance(pydeform, (Path, str)):
        logger.info(f"Loading pydeform configuration from: {pydeform}")
        loaded_pydeform_config = yaml.safe_load(Path(pydeform).read_text())
    else:
        loaded_pydeform_config = pydeform

    if fixed_image_bundle_path is not None:
        fixed_image_bundle_path = Path(fixed_image_bundle_path)

    if fixed_image_bundle_path is None and fixed_image_query is None:
        logger.error(
            "At least one of 'fixed_image_bundle_path' or 'fixed_image_query' needs to be specified."
        )
        return

    cnf = RegistrationPipelineConfig(
        dataset_root=dataset_root,
        fixed_image_query=fixed_image_query,
        fixed_image_bundle_path=fixed_image_bundle_path,
        moving_images_query=moving_images_query,
        segmentation_query=segmentation_query,
        levels_query=levels_query,
        output_derivative_name=output_derivative_name,
        image_weight=image_weight,
        load_sidecars=load_sidecars,
        overwrite_existing=overwrite_existing,
        pydeform=loaded_pydeform_config or {},
        use_gpu=use_gpu,
    )

    layout = index_dataset(cnf)

    fixed = get_or_create_fixed(cnf, layout)

    # Get moving image jobs
    moving_image_paths = get_moving_input_image_paths(cnf, layout, fixed.image_path)
    moving_image_jobs = get_moving_images_jobs(cnf, layout, moving_image_paths)

    # Process moving images
    process_moving(cnf, moving_image_jobs, fixed.bundle)

    # Reindex dataset to find new derivatives
    if moving_image_jobs:
        layout = index_dataset(cnf)

    # Collect and write registration metrics (dice)
    df_registration_metrics = collect_registration_metrics(layout=layout, cnf=cnf)
    df_dice_described = describe_dice(df_registration_metrics)
    save_registration_metrics_to_derivative(cnf, df_dice_described)


if __name__ == "__main__":
    setup_logging(filename="register_with_straightening_to_single_image_pipeline.log")
    run_pipeline()
