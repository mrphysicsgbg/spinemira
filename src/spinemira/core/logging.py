from datetime import datetime
import logging
from pathlib import Path


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
