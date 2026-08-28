"""Prepared range-image cache: degrade and project once, train many epochs off it.

Projection runs at ~31 frames/s on CPU while a training step runs at ~23 frames/s
on the GPU, so projecting inside the training loop would leave the GPU waiting on
the CPU for the whole run. The preparation pass therefore happens once and its
output is reused by every epoch.

Layout, keyed by a hash of the settings that actually change the tensors::

    <cache_root>/<run_name>-<key>/
        manifest.json
        source/<frame>.npy      (4, H, W) range, incidence, reflectance, phy
                                (3, H, W) range, incidence, phy   -- no labels
        target/<frame>.npy      (2, H, W) range, intensity

Those are exactly the stacks PICGAN's own ``dataset.py`` reads, so the cache is
the model's native on-disk format rather than a ReaLiTy-specific one.

The pass is **resumable and idempotent**: each stack is written to a temporary
file and renamed, so a killed job leaves no half-written frame, and a re-run skips
what is already present. The manifest is only marked complete once every frame
exists, and statistics refuse to run against an incomplete cache.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.version import __version__

#: Bump when the cached tensor layout changes, so old caches are not reused.
CACHE_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
SOURCE_DIR = "source"
TARGET_DIR = "target"

#: Channel order written for each role. Source mirrors PICGAN's sim stack.
SOURCE_CHANNELS_LABELLED = ("range", "incidence", "reflectance", "phy")
SOURCE_CHANNELS_UNLABELLED = ("range", "incidence", "phy")
TARGET_CHANNELS = ("range", "intensity")


class CacheError(RuntimeError):
    """Raised when a cache cannot be built, read or trusted."""


def config_key(config: Config) -> str:
    """Hash the settings that change the prepared tensors, and nothing else.

    Normalization is deliberately excluded: statistics are measured *from* the
    cache, so they cannot be an input to it. Training hyperparameters are excluded
    for the same reason -- changing the learning rate must not invalidate frames.
    """
    def data(spec):
        return {
            "dataset": spec.dataset, "path": spec.path, "format": spec.format,
            "columns": list(spec.columns), "intensity_scale": spec.intensity_scale,
            "sequences": list(spec.sequences), "split": spec.split,
            "labels": None if spec.labels is None else {
                "path": spec.labels.path, "format": spec.labels.format},
        }

    degradation = config.geometric_degradation
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "framework_version": __version__,
        "task": config.task.type,
        "source": data(config.source),
        "target": data(config.target),
        "sensor": None if config.sensor is None else {
            "proj_H": config.sensor.proj_H, "proj_W": config.sensor.proj_W,
            "fov_up": config.sensor.fov_up, "fov_down": config.sensor.fov_down},
        "degradation": {
            "enabled": degradation.enabled, "type": degradation.type,
            "weather": degradation.weather,
            "precipitation_rate": degradation.precipitation_rate,
            "mode": degradation.mode, "rmax": degradation.rmax,
            "rmin": degradation.rmin, "bdiv": degradation.bdiv,
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class PrepareReport:
    """What a preparation pass did."""

    source_written: int = 0
    source_skipped: int = 0
    target_written: int = 0
    target_skipped: int = 0
    complete: bool = False

    @property
    def total_written(self) -> int:
        return self.source_written + self.target_written


class PrepareCache:
    """A config-keyed cache of prepared range-image stacks."""

    def __init__(self, config: Config, cache_root: Union[str, Path, None] = None) -> None:
        self.config = config
        self.key = config_key(config)
        root = Path(cache_root) if cache_root else Path(config.output.checkpoint_dir) / "cache"
        self.directory = root / f"{config.run_name}-{self.key}"
        self.source_dir = self.directory / SOURCE_DIR
        self.target_dir = self.directory / TARGET_DIR

    # -- manifest ------------------------------------------------------------- #

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def read_manifest(self) -> Optional[Dict]:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return None

    def write_manifest(self, **fields) -> Dict:
        manifest = self.read_manifest() or {}
        manifest.update(fields)
        manifest.setdefault("key", self.key)
        manifest.setdefault("format_version", CACHE_FORMAT_VERSION)
        manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def is_valid(self) -> bool:
        """True when a complete cache for exactly this configuration exists."""
        manifest = self.read_manifest()
        if not manifest or not manifest.get("complete"):
            return False
        if manifest.get("key") != self.key:
            return False
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            return False
        return (len(self.source_files()) == manifest.get("n_source")
                and len(self.target_files()) == manifest.get("n_target"))

    # -- contents -------------------------------------------------------------- #

    def source_files(self) -> List[Path]:
        return sorted(self.source_dir.glob("*.npy")) if self.source_dir.is_dir() else []

    def target_files(self) -> List[Path]:
        return sorted(self.target_dir.glob("*.npy")) if self.target_dir.is_dir() else []

    @property
    def source_channels(self) -> Tuple[str, ...]:
        manifest = self.read_manifest() or {}
        return tuple(manifest.get("source_channels", SOURCE_CHANNELS_LABELLED))

    @property
    def has_reflectance(self) -> bool:
        return "reflectance" in self.source_channels

    @property
    def image_shape(self) -> Optional[Tuple[int, int]]:
        manifest = self.read_manifest() or {}
        shape = manifest.get("image_shape")
        return tuple(shape) if shape else None

    def clear(self) -> None:
        """Delete the cache directory. Used when a rebuild is explicitly asked for."""
        if self.directory.exists():
            shutil.rmtree(self.directory)

    # -- building ---------------------------------------------------------------- #

    @staticmethod
    def _stack_for_source(sample: Sample) -> np.ndarray:
        """PICGAN's sim layout: range, incidence, [reflectance], phy."""
        if sample.phy is None:
            raise CacheError(
                f"{sample.meta.dataset}: no physics intensity to cache. On the weather "
                f"path the degradation stage produces it; on the sensor path it comes "
                f"from the source simulator."
            )
        names = (SOURCE_CHANNELS_LABELLED if sample.meta.has_reflectance
                 else SOURCE_CHANNELS_UNLABELLED)
        planes = [sample.phy[0] if name == "phy" else sample.channel(name)
                  for name in names]
        return np.stack(planes).astype(np.float32)

    @staticmethod
    def _stack_for_target(sample: Sample) -> np.ndarray:
        """PICGAN's real layout: range, intensity."""
        return np.stack([sample.channel(name) for name in TARGET_CHANNELS]).astype(np.float32)

    @staticmethod
    def _write(path: Path, array: np.ndarray) -> None:
        """Write atomically, so an interrupted pass never leaves a partial frame."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        # Write through a handle: np.save appends '.npy' to a path argument whose
        # name does not already end in it, which would defeat the rename.
        with open(temporary, "wb") as handle:
            np.save(handle, array)
        temporary.replace(path)

    def prepare(self, source_adapter, target_adapter, prepare_source: Callable[[Sample], Sample],
                prepare_target: Callable[[Sample], Sample], force: bool = False,
                limit: Optional[int] = None, log: Optional[Callable[[str], None]] = None
                ) -> PrepareReport:
        """Build the cache, skipping frames already present.

        ``prepare_source`` runs degradation then projection; ``prepare_target``
        projects. Both are supplied by the caller so this module stays unaware of
        which stages a task involves.
        """
        if force:
            self.clear()
        report = PrepareReport()
        say = log or (lambda message: None)

        source_frames = source_adapter.list_frames()[:limit]
        target_frames = target_adapter.list_frames()[:limit]
        if not source_frames or not target_frames:
            raise CacheError(
                f"nothing to prepare: {len(source_frames)} source and "
                f"{len(target_frames)} target frames found. Check the train/ folders "
                f"under {self.config.source.path} and {self.config.target.path}."
            )

        self.write_manifest(complete=False, n_source=len(source_frames),
                            n_target=len(target_frames), run=self.config.run_name)

        channels, shape = None, None
        for index, frame in enumerate(source_frames):
            destination = self.source_dir / f"{index:06d}.npy"
            if destination.is_file() and not force:
                report.source_skipped += 1
                continue
            prepared = prepare_source(source_adapter.load_sample(frame))
            stack = self._stack_for_source(prepared)
            channels = (SOURCE_CHANNELS_LABELLED if prepared.meta.has_reflectance
                        else SOURCE_CHANNELS_UNLABELLED)
            shape = stack.shape[1:]
            self._write(destination, stack)
            report.source_written += 1
            if report.source_written % 50 == 0:
                say(f"  prepared {report.source_written} source frames")

        for index, frame in enumerate(target_frames):
            destination = self.target_dir / f"{index:06d}.npy"
            if destination.is_file() and not force:
                report.target_skipped += 1
                continue
            stack = self._stack_for_target(prepare_target(target_adapter.load_sample(frame)))
            shape = shape or stack.shape[1:]
            self._write(destination, stack)
            report.target_written += 1
            if report.target_written % 50 == 0:
                say(f"  prepared {report.target_written} target frames")

        if channels is None:  # everything was already cached
            channels = self.source_channels
        if shape is None:
            shape = self.image_shape

        complete = (len(self.source_files()) == len(source_frames)
                    and len(self.target_files()) == len(target_frames))
        self.write_manifest(
            complete=complete, n_source=len(source_frames), n_target=len(target_frames),
            source_channels=list(channels), image_shape=list(shape) if shape else None,
            has_reflectance="reflectance" in channels,
        )
        report.complete = complete
        return report

    def require_complete(self) -> None:
        """Raise unless the cache is complete, so nothing measures a partial pass."""
        if not self.is_valid():
            manifest = self.read_manifest() or {}
            raise CacheError(
                f"cache at {self.directory} is not complete "
                f"({len(self.source_files())}/{manifest.get('n_source', '?')} source, "
                f"{len(self.target_files())}/{manifest.get('n_target', '?')} target). "
                f"Run preparation to completion before measuring statistics or training."
            )

    def __repr__(self) -> str:
        return (f"PrepareCache({self.directory.name}, "
                f"{len(self.source_files())} source, {len(self.target_files())} target, "
                f"complete={bool((self.read_manifest() or {}).get('complete'))})")
