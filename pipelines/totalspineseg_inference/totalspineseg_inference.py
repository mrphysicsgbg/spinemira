"""
Spine segmentation pipeline using TotalSpineSeg.

This module implements a pipeline for segmenting images in a BIDS-like dataset.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
import tempfile
import multiprocessing as mp

from totalspineseg.init_inference import init_inference
from totalspineseg.utils.utils import ZIP_URLS as TOTALSPINSEG_URLS
from totalspineseg.inference import inference

from spinemira.core.logging import setup_logging
from spinemira.io.bids import Layout, resolve_derivative
from spinemira.pipelines.config import with_cli_config


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    dataset_root: Path
    image_query: str
    output_derivative_name: str
    image_weight: str
    totalspineseg_data_dir: Path
    load_sidecars: bool
    overwrite_existing: bool
    device: str
    totalspineseg_release: str | None
    totalspineseg_quiet: bool


@dataclass
class Job:
    input_image: Path

    output_label_map: Path
    output_levels: Path


def build_image_query(base: str, image_weight: str) -> str:
    extra = [
        "~is_sidecar",
        f"`suffix` == '{image_weight}'",
    ]
    return " and ".join([base, *extra])


def index_dataset(cnf: PipelineConfig) -> Layout:
    logging.info("Loading dataset...")
    layout = Layout(root=cnf.dataset_root, include_derivatives=True)
    layout.index(load_sidecars=cnf.load_sidecars)
    logging.info("Dataset indexed.")

    return layout


def get_jobs(cnf: PipelineConfig, layout: Layout) -> list[Job]:
    logging.info("Resolving jobs...")

    df_images = layout.query(build_image_query(cnf.image_query, cnf.image_weight))
    image_paths = [Path(p) for p in df_images["path"].to_list()]

    jobs = [
        Job(
            input_image=image_path,
            output_label_map=resolve_derivative(
                image_path, cnf.output_derivative_name, suffix="dseg"
            ),
            output_levels=resolve_derivative(
                image_path, cnf.output_derivative_name, suffix="levels"
            ),
        )
        for image_path in image_paths
    ]

    if not cnf.overwrite_existing:
        logging.info("Filtering already completed jobs...")
        jobs = [
            job
            for job in jobs
            if not job.output_label_map.exists() or not job.output_levels.exists()
        ]

    logging.info(f"Compiled {len(jobs)} jobs to perform.")

    return jobs


def process_job(cnf: PipelineConfig, job: Job):
    logging.info(f"Starting processing of {job.input_image}.")

    with tempfile.TemporaryDirectory() as tmpdir:
        inference(
            job.input_image,
            output_path=tmpdir,
            data_path=cnf.totalspineseg_data_dir,
            default_release=cnf.totalspineseg_release,
            device=cnf.device,
            quiet=cnf.totalspineseg_quiet,
        )

        tmpdir_path = Path(tmpdir)

        tmp_levels = tmpdir_path / "step1_levels" / Path(job.input_image.name)
        tmp_label_map = tmpdir_path / "step2_output" / Path(job.input_image.name)

        logging.info(f"Copying {tmp_label_map} --> {job.output_label_map}.")
        job.output_label_map.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_label_map, job.output_label_map)

        logging.info(f"Copying {tmp_levels} --> {job.output_levels}.")
        job.output_levels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_levels, job.output_levels)

    logging.info(f"Processing of {job.input_image} done.")


def process_jobs(cnf: PipelineConfig, jobs: list[Job]):
    logging.info("Starting processing of jobs...")

    for i, job in enumerate(jobs):
        logging.info(f"Starting processing of job ({i + 1} / {len(jobs)})")
        process_job(cnf, job)

    logging.info("Processing of jobs finished.")


def init_totalspineseg(cnf: PipelineConfig):
    logging.info("Initializing totalspineseg...")

    cnf.totalspineseg_data_dir.mkdir(exist_ok=True, parents=True)

    init_inference(
        data_path=cnf.totalspineseg_data_dir,
        dict_urls=TOTALSPINSEG_URLS,
        quiet=cnf.totalspineseg_quiet,
    )

    logging.info("Initialized totalspineseg.")


def configure_no_stalling(no_stalling: bool):
    if no_stalling and "forkserver" in mp.get_all_start_methods():
        mp.set_start_method("forkserver", force=True)
        logger.info("Start method for multiprocessing was set to `forkserver`.")


@with_cli_config(log_fn=logger.info)
def run_pipeline(
    dataset_root: Path,
    image_query: str,
    output_derivative_name: str,
    image_weight: str,
    totalspineseg_data_dir: Path,
    load_sidecars: bool = False,
    overwrite_existing: bool = False,
    device: str = "cpu",
    totalspineseg_release: str | None = None,
    totalspineseg_quiet: bool = True,
    no_stalling: bool = True,
):
    cnf = PipelineConfig(
        dataset_root=dataset_root,
        image_query=image_query,
        output_derivative_name=output_derivative_name,
        image_weight=image_weight,
        load_sidecars=load_sidecars,
        overwrite_existing=overwrite_existing,
        device=device,
        totalspineseg_release=totalspineseg_release,
        totalspineseg_data_dir=totalspineseg_data_dir,
        totalspineseg_quiet=totalspineseg_quiet,
    )

    configure_no_stalling(no_stalling)

    # Initialize and index the dataset
    layout = index_dataset(cnf)

    # Initialize totalspineseg
    init_totalspineseg(cnf)

    # Get all jobs (files to segment and output files)
    jobs = get_jobs(cnf, layout)

    # Process all jobs
    process_jobs(cnf, jobs)


if __name__ == "__main__":
    setup_logging(filename="segmentation_pipeline.log")
    run_pipeline()
