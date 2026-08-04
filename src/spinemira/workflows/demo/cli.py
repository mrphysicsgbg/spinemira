from pathlib import Path
from time import time

from spinemira.core.logging import setup_logging
from spinemira.pipelines.config import with_cli_config
from spinemira.workflows.demo.workflow import demo_workflow


@with_cli_config(log_platform_and_packages=False)
def run_demo_workflow(
    dataset_root: Path,
    query: str,
    output_derivative_name: str,
    overwrite: bool = False,
    rerun: bool = False,
    worker: str = "debug",
):
    dataset_root = Path(dataset_root).resolve()

    wf = demo_workflow(
        dataset_root=dataset_root,
        query=query,
        output_derivative_name=output_derivative_name,
        overwrite=overwrite,
    )

    start = time()

    wf(worker=worker, rerun=rerun)

    end = time()

    print(f"Total time: {end - start} s")


def main():
    setup_logging(filename="demo.log")
    run_demo_workflow()


if __name__ == "__main__":
    main()
