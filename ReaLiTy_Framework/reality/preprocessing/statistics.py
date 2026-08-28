"""Per-channel normalization statistics computed from the configured datasets.

PICGAN's shipped constants were fitted on VoxelScape and do not describe other
source data: measured on real KITTI, the incidence channel has mean 0.346 where
the shipped transform assumes 0.7156. Rather than normalise real data with
another dataset's statistics, these are measured from a sample of the actual
configured source and target, cached next to the run's config snapshot, and fed
into the transforms.

Two roles, two datasets, deliberately:

* **source** — range, incidence, [reflectance] and phy, measured on the source.
* **target** — intensity, measured on the *target*, which is also what output
  denormaweather_modeltion uses. Denormalising a generated CADC intensity with statistics
  from anywhere else is the asymmetry this removes.

Statistics are taken over **occupied pixels only**. Empty pixels of a range image
carry no return, and their share depends on ``proj_W`` (41.5% of a KITTI frame is
projected at 1024, 92.4% at 4096), so including them would make the constants a
function of projection resolution rather than of the data. The all-pixel figures
are recorded alongside for reference.

The estimator is plain per-channel mean/std (z-scoring), matching what PICGAN's
transforms already do -- only the numbers change, not the method. A different
estimator (robust/percentile statistics) would change what the physics loss sees
in ``phy`` and is out of scope without approval (see docs/modifying_picgan.md).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample

#: Channels the source stack needs statistics for, in PICGAN's order.
SOURCE_CHANNELS = ("range", "incidence", "reflectance", "phy")
#: The target domain contributes one channel, which also drives denormaweather_modeltion.
TARGET_CHANNELS = ("intensity",)

#: PICGAN's original VoxelScape-fitted constants, kept so published runs
#: reproduce exactly. Sourced from the model's own transform_utils.py.
PICGAN_DEFAULT_STATS: Dict[str, Tuple[float, float]] = {
    "range": (0.0965, 0.1068),
    "incidence": (0.7156, 0.6352),
    "reflectance": (0.2979, 0.2743),
    "phy": (0.1745, 0.1515),
    "intensity": (0.0158, 0.0462),
}

MIN_STD = 1e-6


class StatisticsError(ValueError):
    """Raised when statistics cannot be computed from the configured data."""


@dataclass
class ChannelStats:
    """Mean and standard deviation of one channel over occupied pixels."""

    mean: float
    std: float
    count: int = 0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    #: Same statistics over every pixel including empties, for reference only.
    all_pixel_mean: Optional[float] = None
    all_pixel_std: Optional[float] = None

    def as_pair(self) -> Tuple[float, float]:
        return self.mean, max(self.std, MIN_STD)


class _Accumulator:
    """Streaming sums for one channel, so frames are never all held in memory."""

    def __init__(self) -> None:
        self.total = self.total_sq = 0.0
        self.count = 0
        self.all_total = self.all_total_sq = 0.0
        self.all_count = 0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None

    def update(self, plane: np.ndarray, occupied: np.ndarray) -> None:
        values = np.asarray(plane, dtype=np.float64)
        self.all_total += float(values.sum())
        self.all_total_sq += float((values ** 2).sum())
        self.all_count += values.size

        selected = values[occupied]
        if selected.size == 0:
            return
        self.total += float(selected.sum())
        self.total_sq += float((selected ** 2).sum())
        self.count += int(selected.size)
        low, high = float(selected.min()), float(selected.max())
        self.minimum = low if self.minimum is None else min(self.minimum, low)
        self.maximum = high if self.maximum is None else max(self.maximum, high)

    def result(self) -> Optional[ChannelStats]:
        if self.count == 0:
            return None
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean ** 2, 0.0)
        all_mean = self.all_total / self.all_count if self.all_count else 0.0
        all_variance = max(self.all_total_sq / self.all_count - all_mean ** 2, 0.0)
        return ChannelStats(
            mean=mean, std=float(np.sqrt(variance)), count=self.count,
            minimum=self.minimum, maximum=self.maximum,
            all_pixel_mean=all_mean, all_pixel_std=float(np.sqrt(all_variance)),
        )


@dataclass
class NormalizationStats:
    """Per-channel statistics plus the provenance needed to audit them."""

    channels: Dict[str, ChannelStats] = field(default_factory=dict)
    #: "computed" or "picgan_default".
    mode: str = "computed"
    source_dataset: Optional[str] = None
    target_dataset: Optional[str] = None
    n_source_frames: int = 0
    n_target_frames: int = 0
    seed: Optional[int] = None
    #: Channels that fell back to PICGAN's defaults because data was unavailable.
    fallbacks: List[str] = field(default_factory=list)

    # -- lookup -------------------------------------------------------------- #

    def pair(self, channel: str) -> Tuple[float, float]:
        """(mean, std) for a channel, falling back to PICGAN's default."""
        if channel in self.channels:
            return self.channels[channel].as_pair()
        if channel in PICGAN_DEFAULT_STATS:
            return PICGAN_DEFAULT_STATS[channel]
        raise StatisticsError(f"no statistics and no default for channel '{channel}'")

    def as_pairs(self) -> Dict[str, Tuple[float, float]]:
        names = set(self.channels) | set(PICGAN_DEFAULT_STATS)
        return {name: self.pair(name) for name in sorted(names)}

    # -- construction --------------------------------------------------------- #

    @classmethod
    def picgan_default(cls) -> "NormalizationStats":
        """PICGAN's original VoxelScape constants, exactly as published."""
        return cls(
            channels={name: ChannelStats(mean=mean, std=std)
                      for name, (mean, std) in PICGAN_DEFAULT_STATS.items()},
            mode="picgan_default",
        )

    # -- persistence ----------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "n_source_frames": self.n_source_frames,
            "n_target_frames": self.n_target_frames,
            "seed": self.seed,
            "fallbacks": list(self.fallbacks),
            "statistics_over": "occupied pixels only",
            "channels": {name: asdict(stats) for name, stats in self.channels.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NormalizationStats":
        return cls(
            channels={name: ChannelStats(**values)
                      for name, values in (data.get("channels") or {}).items()},
            mode=data.get("mode", "computed"),
            source_dataset=data.get("source_dataset"),
            target_dataset=data.get("target_dataset"),
            n_source_frames=int(data.get("n_source_frames", 0)),
            n_target_frames=int(data.get("n_target_frames", 0)),
            seed=data.get("seed"),
            fallbacks=list(data.get("fallbacks") or []),
        )

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{n}={s.mean:.4f}/{s.std:.4f}" for n, s in self.channels.items())
        return f"NormalizationStats({self.mode}: {pairs})"


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #


def accumulate(samples: Iterable[Sample], channels: Sequence[str]
               ) -> Dict[str, ChannelStats]:
    """Accumulate per-channel statistics over already-projected samples."""
    accumulators: Dict[str, _Accumulator] = {}
    for sample in samples:
        if sample.range_image is None:
            raise StatisticsError(
                f"{sample.meta.dataset}: statistics need projected samples; project first"
            )
        occupied = (sample.mapping >= 0) if sample.mapping is not None else np.ones(
            sample.range_image.shape[1:], dtype=bool)
        for channel in channels:
            if channel == "phy":
                plane = sample.phy[0] if sample.phy is not None else None
            elif channel in sample.channels:
                plane = sample.channel(channel)
            else:
                plane = None
            if plane is None:
                continue
            accumulators.setdefault(channel, _Accumulator()).update(plane, occupied)
    return {name: stats for name, acc in accumulators.items()
            if (stats := acc.result()) is not None}


def _sample_frames(adapter, n_frames: int, seed: Optional[int]):
    """Pick an evenly spread, reproducible subset of a dataset's frames."""
    frames = adapter.list_frames()
    if not frames:
        raise StatisticsError(f"{adapter.name}: no frames found to measure")
    if n_frames >= len(frames):
        return frames
    if seed is None:
        # Evenly spaced: cheap and covers the whole sequence.
        return [frames[i] for i in np.linspace(0, len(frames) - 1, n_frames).astype(int)]
    rng = np.random.default_rng(seed)
    return [frames[i] for i in sorted(rng.choice(len(frames), n_frames, replace=False))]


def compute_from_config(config: Config, n_frames: Optional[int] = None,
                        seed: Optional[int] = None,
                        degradation=None) -> NormalizationStats:
    """Measure statistics from the configured source and target datasets.

    On the weather path the physics intensity only exists once the degradation
    stage has run, so source frames are degraded before projection when a
    degradation is available. Measuring ``phy`` on anything else would leave the
    physics loss comparing against another dataset's constants -- the very
    mismatch this module exists to remove.
    """
    from reality.core.registry import DATASETS, DEGRADATIONS
    from reality.plugins import register_all
    from reality.preprocessing.projection import project

    register_all()
    spec = config.normalization
    n_frames = n_frames if n_frames is not None else spec.frames
    seed = seed if seed is not None else spec.seed

    if degradation is None and config.geometric_degradation.enabled:
        try:
            candidate = DEGRADATIONS.get(config.geometric_degradation.type)(config)
            # Availability is checked here rather than discovered mid-measurement:
            # the weather model resolves lazily, so an absent model would otherwise surface as a
            # failure part-way through the pass.
            degradation = candidate if candidate.available else None
        except Exception:
            degradation = None

    def measure(data_spec, channels, role):
        adapter = DATASETS.get(data_spec.dataset)(
            data_spec, sensor=config.sensor, task=config.task.type)
        frames = _sample_frames(adapter, n_frames, seed)
        sensor = config.sensor or adapter.sensor

        def prepared():
            for frame in frames:
                sample = adapter.load_sample(frame)
                if role == "source" and degradation is not None:
                    sample = degradation.apply(sample)
                yield project(sample, sensor)

        return accumulate(prepared(), channels), len(frames)

    source_stats, n_source = measure(config.source, SOURCE_CHANNELS, "source")
    target_stats, n_target = measure(config.target, TARGET_CHANNELS, "target")

    channels = dict(source_stats)
    channels.update(target_stats)  # the target owns 'intensity'
    fallbacks = [name for name in SOURCE_CHANNELS + TARGET_CHANNELS
                 if name not in channels]
    if "phy" in fallbacks and config.geometric_degradation.enabled:
        warnings.warn(
            "phy statistics could not be measured because the degradation stage was "
            "unavailable, so the physics loss will normalise phy with PICGAN's "
            "VoxelScape constant. Provide the weather model (geometric_degradation.weather_model_path) and "
            "recompute before training.",
            RuntimeWarning, stacklevel=2,
        )
    return NormalizationStats(
        channels=channels, mode="computed",
        source_dataset=config.source.dataset, target_dataset=config.target.dataset,
        n_source_frames=n_source, n_target_frames=n_target, seed=seed,
        fallbacks=fallbacks,
    )


def compute_from_cache(cache, config: Optional[Config] = None) -> NormalizationStats:
    """Measure statistics over **every** prepared stack in a complete cache.

    This is the correctness path for training: the constants describe the whole
    training set in the exact range-image form the model consumes, not a sample of
    it and not the raw clouds. ``phy`` is measured from the degraded stacks,
    because that is what the physics loss compares against.

    Occupancy is taken from the range plane: projection zero-fills pixels no point
    landed on, so ``range > 0`` marks the real returns. Empty pixels are excluded
    for the reason given at the top of this module -- their share is a function of
    ``proj_W``, not of the data.
    """
    cache.require_complete()  # never measure a partial preparation

    channels = list(cache.source_channels)
    accumulators: Dict[str, _Accumulator] = {name: _Accumulator() for name in channels}
    accumulators["intensity"] = _Accumulator()

    range_index = channels.index("range")
    for path in cache.source_files():
        stack = np.load(path)
        occupied = stack[range_index] > 0.0
        for position, name in enumerate(channels):
            accumulators[name].update(stack[position], occupied)

    for path in cache.target_files():
        stack = np.load(path)              # (2, H, W) range, intensity
        occupied = stack[0] > 0.0
        accumulators["intensity"].update(stack[1], occupied)

    measured = {name: result for name, accumulator in accumulators.items()
                if (result := accumulator.result()) is not None}
    fallbacks = [name for name in SOURCE_CHANNELS + TARGET_CHANNELS
                 if name not in measured]
    return NormalizationStats(
        channels=measured, mode="computed",
        source_dataset=config.source.dataset if config else None,
        target_dataset=config.target.dataset if config else None,
        n_source_frames=len(cache.source_files()),
        n_target_frames=len(cache.target_files()),
        seed=None,  # every frame is used, so there is nothing to seed
        fallbacks=fallbacks,
    )


def stats_path(config: Config) -> Path:
    """Where a run's statistics live: beside its config snapshot."""
    return Path(config.output.checkpoint_dir) / "normalization_stats.json"


def resolve(config: Config, cache: bool = True, recompute: bool = False,
            prepared=None) -> NormalizationStats:
    """Return the statistics this run should use, honouring the config switch.

    ``picgan_default`` reproduces the original constants exactly, and exists only
    so previously published behaviour stays reproducible. ``computed`` measures
    the data: over the whole prepared cache when one is given (the training path),
    otherwise over a sample of the raw datasets (used for quick inspection).
    """
    if config.normalization.source == "picgan_default":
        return NormalizationStats.picgan_default()

    path = stats_path(config)
    if path.is_file() and not recompute:
        return NormalizationStats.load(path)
    stats = (compute_from_cache(prepared, config) if prepared is not None
             else compute_from_config(config))
    if cache:
        stats.save(path)
    return stats
