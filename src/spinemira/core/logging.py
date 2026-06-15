from datetime import datetime
from importlib.metadata import distributions
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Callable


def setup_logging(
    level: int | str = logging.INFO,
    filename: str | Path | None = None,
    suffix_with_datetime: bool = True,
):
    """Configure logging

    Parameters
    ----------
    level : int | str, optional
        Log level, by default logging.INFO
    filename : str | Path | None, optional
        File name for optional logfile, by default None
    suffix_with_datetime : bool, optional
        Control of a suffix with date and time should be appended to the name of the logfile, by default True
    """

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(stream_handler)

    if filename:
        logfile = Path(filename)

        if suffix_with_datetime:
            now_str = datetime.today().strftime("%Y-%m-%d_%H%M%S")
            logfile = logfile.with_name(f"{logfile.stem}_{now_str}{logfile.suffix}")

        file_handler = logging.FileHandler(filename=logfile)
        file_handler.setFormatter(formatter)

        root.addHandler(file_handler)


def log_environment(log_fn: Callable | None) -> None:
    """Log Python environment and installed packages

    log_fn : Callable | None, optional
        Method to call for logging, will print to standard output if unspecified, by default None
    """

    packages = sorted(
        (
            dist.metadata["Name"] if "Name" in dist.metadata else "UNKNOWN",
            dist.version,
        )
        for dist in distributions()
    )

    log_message = "\n".join(
        [
            "",
            "-" * 80,
            "Environment",
            "-" * 80,
            f"Python version       : {sys.version.replace(chr(10), ' ')}",
            f"Python executable    : {sys.executable}",
            f"Python implementation: {platform.python_implementation()}",
            f"Python build         : {platform.python_build()}",
            f"Python compiler      : {platform.python_compiler()}",
            f"Platform             : {platform.platform()}",
            f"System               : {platform.system()}",
            f"Release              : {platform.release()}",
            f"Machine              : {platform.machine()}",
            f"Processor            : {platform.processor()}",
            f"Hostname             : {platform.node()}",
            f"Current directory    : {os.getcwd()}",
            f"Virtual environment  : {sys.prefix != sys.base_prefix}",
            "-" * 80,
            f"Installed packages ({len(packages)})",
            "-" * 80,
            *(f"    {name}=={version}" for name, version in packages),
        ]
    )

    if log_fn:
        log_fn(log_message)
    else:
        print(log_message)
