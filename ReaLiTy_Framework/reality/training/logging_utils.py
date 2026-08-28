"""Run logging: console plus a file in the run directory.

An unattended cluster run is only diagnosable from what it wrote down, so the
file log carries the same lines the console does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

LOGGER_NAME = "reality"


def setup_logging(run_dir: Union[str, Path], name: str = LOGGER_NAME,
                  level: int = logging.INFO, filename: str = "train.log") -> logging.Logger:
    """Attach a console and a file handler to the run's logger."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
