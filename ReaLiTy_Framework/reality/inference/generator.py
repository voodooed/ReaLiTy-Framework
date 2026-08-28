"""End-to-end generation: Sample -> projection -> model -> back-projection -> file.

Source and target are parameters resolved from config through the registries, so
there is no per-dataset code here. The degradation stage is deliberately absent:
this is the sensor path, where the physics intensity comes from the source
simulator.  inserts the weather model between projection and the model for the weather
path, and reuses this same projection and back-projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.registry import DATASETS, DEGRADATIONS, MODELS
from reality.io.writers import OutputWriter
from reality.plugins import register_all
from reality.postprocessing.backprojection import BackProjection, backproject
from reality.preprocessing.projection import project
from reality.preprocessing import statistics


class GenerationError(RuntimeError):
    """Raised when a run cannot be assembled from its config."""


@dataclass
class GeneratedFrame:
    """One frame's result: the transformed cloud and what happened to it."""

    sample: Sample
    points: np.ndarray
    written: np.ndarray
    path: Optional[Path] = None
    stats: dict = field(default_factory=dict)


class IntensityGenerator:
    """Runs the sensor pipeline for a configured source/target pair."""

    def __init__(self, config: Config, model=None, writer: Optional[OutputWriter] = None,
                 fill: Union[str, float] = "keep",
                 stats: Optional[statistics.NormalizationStats] = None,
                 degradation=None, clamp: Optional[tuple] = None) -> None:
        register_all()
        self.config = config
        self.model = model
        self.writer = writer
        self.fill = fill
        self._stats = stats
        self._degradation = degradation
        #: Optional (low, high) clamp for generated intensity. gen_R ends in tanh,
        #: so denormalised output can undershoot 0 slightly; native formats expect
        #: non-negative intensity. Off by default so the raw output is preserved.
        self.clamp = clamp

    @property
    def stats(self) -> statistics.NormalizationStats:
        """Normalization statistics for this run, per ``normalization.source``.

        Resolved lazily: measuring reads the datasets, which a caller supplying its
        own model has no reason to pay for.
        """
        if self._stats is None:
            self._stats = statistics.resolve(self.config)
        return self._stats

    # -- assembly ------------------------------------------------------------ #

    def build_model(self, sample: Sample):
        """Resolve the model from config and build it for this sample's channels."""
        if self.model is None:
            model_cls = MODELS.get(self.config.model.type)
            self.model = model_cls(self.config, stats=self.stats)
        if not self.model.is_built:
            self.model.build_model(3 if sample.meta.has_reflectance else 2)
        return self.model

    @staticmethod
    def adapter_for(spec, task: str, sensor=None):
        """Instantiate a dataset adapter by name from the registry."""
        return DATASETS.get(spec.dataset)(spec, sensor=sensor, task=task)

    def source_adapter(self):
        return self.adapter_for(self.config.source, self.config.task.type, self.config.sensor)

    def target_adapter(self):
        return self.adapter_for(self.config.target, self.config.task.type, self.config.sensor)

    # -- running -------------------------------------------------------------- #

    @property
    def degradation(self):
        """The weather stage, when the run has one. None for sensor transfer."""
        spec = self.config.geometric_degradation
        if not spec.enabled:
            return None
        if self._degradation is None:
            self._degradation = DEGRADATIONS.get(spec.type)(self.config)
        return self._degradation

    def project(self, sample: Sample) -> Sample:
        """Degrade (weather runs only) and project a loaded sample.

        The same stage order as training: the weather model consumes the 3D cloud, so it runs
        before projection and its ``ref_new`` becomes ``phy``.
        """
        degradation = self.degradation
        if degradation is not None:
            sample = degradation.apply(sample)
        return project(sample, self.config.sensor or sample.meta.fov)

    def generate_frame(self, sample: Sample, projected: bool = False) -> GeneratedFrame:
        """Run one frame end to end and return the transformed cloud."""
        projected_sample = sample if projected else self.project(sample)
        model = self.build_model(projected_sample)

        # gen_R emits normalised intensity (it ends in tanh); return it to data
        # units with PICGAN's own constants so the written cloud is native.
        output = model.generate(projected_sample)
        if hasattr(model, "denormalize_real_intensity"):
            output = model.denormalize_real_intensity(output)
        intensity = output.detach().cpu().numpy()[0]  # (1, H, W)
        if self.clamp is not None:
            intensity = np.clip(intensity, *self.clamp)

        result: BackProjection = backproject(projected_sample, intensity, fill=self.fill)
        projection_stats = projected_sample.meta.extra.get("projection", {})
        stats = {
            "frame_id": projected_sample.meta.extra.get("frame_id"),
            "n_points": int(projected_sample.points.shape[0]),
            "n_projected": projection_stats.get("n_projected"),
            "n_written": result.n_written,
            "n_dropped": result.n_dropped,
            "image_shape": projection_stats.get("shape"),
            "normalization": getattr(getattr(model, "stats", None), "mode", None),
        }
        path = self.writer.write(projected_sample, result.points) if self.writer else None
        return GeneratedFrame(sample=projected_sample, points=result.points,
                              written=result.written, path=path, stats=stats)

    def stream(self, limit: Optional[int] = None, frames: Optional[Sequence] = None,
               log_every: int = 0, log=None) -> Iterator[GeneratedFrame]:
        """Yield one transformed frame at a time.

        Nothing about the run is proportional to the dataset size: frames are
        loaded, transformed, written and released one by one, so converting two
        thousand frames and converting the entire odometry set cost the same
        memory. Callers that do not retain the yielded frames stay flat.
        """
        adapter = self.source_adapter()
        selected = list(frames) if frames is not None else adapter.list_frames()
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            raise GenerationError(
                f"{self.config.source.dataset}: no frames found at "
                f"{self.config.source.path} (split '{self.config.source.split}')"
            )
        say = log or (lambda message: None)
        for index, frame in enumerate(selected, start=1):
            yield self.generate_frame(adapter.load_sample(frame))
            if log_every and index % log_every == 0:
                say(f"  {index}/{len(selected)} frames")

    def run(self, limit: Optional[int] = None,
            frames: Optional[Sequence] = None) -> List[GeneratedFrame]:
        """Convert a dataset and return every frame.

        Convenient for small jobs; for large ones use :meth:`stream`, which does
        not hold the whole run in memory.
        """
        return list(self.stream(limit=limit, frames=frames))
