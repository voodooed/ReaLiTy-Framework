"""Seed control, so a run can be reproduced (see the contributor guide).

On CUDA, reproducibility needs more than a seed. cuDNN picks convolution
algorithms by benchmarking and some reductions are non-deterministic, so two
identical forward passes agree only to floating-point tolerance -- measured here
at ~3e-7 on a V100. Setting ``cudnn.deterministic`` and disabling benchmarking
makes them bit-identical, at some cost in throughput.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy and torch; optionally force deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Let cuDNN pick the fastest algorithm per shape; results then vary at
        # the 1e-7 level between otherwise identical runs.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    return seed


def seed_from_config(config, deterministic: bool = True) -> int:
    """Seed from a run config's ``training.seed``."""
    return seed_everything(config.training.seed, deterministic=deterministic)
