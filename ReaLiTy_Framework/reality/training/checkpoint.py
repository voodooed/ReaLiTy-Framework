"""Checkpoints: a full resumable state, and a slim inference file.

Native PyTorch ``state_dict`` is the primary format. Two kinds are written:

* **full** -- everything needed to resume: all four networks, both optimizers,
  the AMP scaler, the epoch, plus the run's normalization statistics and a
  snapshot of the resolved config.
* **slim** -- ``gen_R`` and the normalization statistics only. This is the file
  that travels back from a cluster for local inference.

The normalization statistics live *inside* both files deliberately. Weights
trained against statistics measured over a full dataset must be denormalised with
those same statistics, whatever data happens to be on the machine doing
inference; a checkpoint that carried only weights would silently denormalise with
whatever local statistics existed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from reality.core.config import Config
from reality.core.version import CHECKPOINT_FORMAT_VERSION, __version__
from reality.preprocessing.statistics import NormalizationStats

FULL_SUFFIX = "full.pt"
SLIM_SUFFIX = "gen_r.pt"
METADATA_NAME = "metadata.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be written or read."""


@dataclass
class LoadedCheckpoint:
    """A checkpoint read back from disk."""

    state: Dict[str, Any]
    stats: NormalizationStats
    metadata: Dict[str, Any]
    epoch: int = 0

    @property
    def in_channels_s(self) -> int:
        return int(self.metadata.get("in_channels_s", 3))

    @property
    def is_slim(self) -> bool:
        return "gen_S" not in self.state


def build_metadata(config: Optional[Config], model, epoch: int = 0,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Provenance recorded beside the weights (README -> *Checkpoints*)."""
    metadata: Dict[str, Any] = {
        "framework_version": __version__,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "model": getattr(model, "name", type(model).__name__),
        "in_channels_s": getattr(model, "in_channels_s", None),
        "in_channels_r": getattr(model, "in_channels_r", None),
        "normalization_mode": getattr(getattr(model, "stats", None), "mode", None),
    }
    if config is not None:
        metadata.update({
            "source_dataset": config.source.dataset,
            "target_dataset": config.target.dataset,
            "task": config.task.type,
            "weather": config.geometric_degradation.weather,
            "degradation": (config.geometric_degradation.type
                            if config.geometric_degradation.enabled else None),
            "run_name": config.run_name,
            "config": config.to_dict(),
        })
    if extra:
        metadata.update(extra)
    return metadata


def save_full(model, path: Union[str, Path], config: Optional[Config] = None,
              epoch: int = 0, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a resumable checkpoint: networks, optimizers, scaler, stats, config."""
    if not model.is_built:
        raise CheckpointError("build the model before saving a checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "full",
        "epoch": epoch,
        "gen_R": model.gen_R.state_dict(),
        "gen_S": model.gen_S.state_dict(),
        "disc_R": model.disc_R.state_dict(),
        "disc_S": model.disc_S.state_dict(),
        "opt_gen": model.opt_gen.state_dict(),
        "opt_disc": model.opt_disc.state_dict(),
        "g_scaler": model.g_scaler.state_dict(),
        "d_scaler": model.d_scaler.state_dict(),
        "normalization_stats": model.stats.to_dict(),
        "metadata": build_metadata(config, model, epoch, extra),
    }
    torch.save(payload, path)
    return path


def save_slim(model, path: Union[str, Path], config: Optional[Config] = None,
              epoch: int = 0, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the inference checkpoint: gen_R plus the normalization statistics."""
    if not model.is_built:
        raise CheckpointError("build the model before saving a checkpoint")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "slim",
        "epoch": epoch,
        "gen_R": model.gen_R.state_dict(),
        "normalization_stats": model.stats.to_dict(),
        "metadata": build_metadata(config, model, epoch, extra),
    }, path)
    return path


def save(model, checkpoint_dir: Union[str, Path], config: Optional[Config] = None,
         epoch: int = 0, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Write both checkpoints plus a readable metadata sidecar."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "full": save_full(model, directory / FULL_SUFFIX, config, epoch, extra),
        "slim": save_slim(model, directory / SLIM_SUFFIX, config, epoch, extra),
    }
    metadata = build_metadata(config, model, epoch, extra)
    (directory / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
    paths["metadata"] = directory / METADATA_NAME
    return paths


def load(path: Union[str, Path], map_location: str = "cpu") -> LoadedCheckpoint:
    """Read a checkpoint of either kind."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if "gen_R" not in payload:
        raise CheckpointError(f"{path}: not a ReaLiTy checkpoint (no gen_R weights)")
    stats_data = payload.get("normalization_stats")
    if stats_data is None:
        raise CheckpointError(
            f"{path}: carries no normalization statistics, so its outputs cannot be "
            f"denormalised correctly"
        )
    return LoadedCheckpoint(
        state=payload, stats=NormalizationStats.from_dict(stats_data),
        metadata=payload.get("metadata", {}), epoch=int(payload.get("epoch", 0)),
    )


def restore(model, path: Union[str, Path], map_location: Optional[str] = None) -> int:
    """Restore a model from a checkpoint, rebuilding it if needed. Returns the epoch."""
    checkpoint = load(path, map_location or str(getattr(model, "device", "cpu")))
    if not model.is_built:
        model.build_model(checkpoint.in_channels_s,
                          int(checkpoint.metadata.get("in_channels_r", 1) or 1))
    model.use_statistics(checkpoint.stats)

    state = checkpoint.state
    model.gen_R.load_state_dict(state["gen_R"])
    if not checkpoint.is_slim:
        model.gen_S.load_state_dict(state["gen_S"])
        model.disc_R.load_state_dict(state["disc_R"])
        model.disc_S.load_state_dict(state["disc_S"])
        model.opt_gen.load_state_dict(state["opt_gen"])
        model.opt_disc.load_state_dict(state["opt_disc"])
        model.g_scaler.load_state_dict(state["g_scaler"])
        model.d_scaler.load_state_dict(state["d_scaler"])
    return checkpoint.epoch
