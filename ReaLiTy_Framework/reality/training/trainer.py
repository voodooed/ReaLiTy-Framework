"""Weather-pipeline orchestration for training steps.

Source frames run cloud -> degradation -> projection; target frames run
projection only. The paired ``(sim, real, phy)`` batch is then handed to the
model adapter, which delegates the optimisation step to PICGAN's own ``train_fn``
so the loss formulation stays where it is.

This is the assembly layer, not a training loop:  owns epochs, scheduling
and checkpoint cadence. What exists here is enough to prove the pipeline runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.registry import DATASETS, DEGRADATIONS, MODELS
from reality.plugins import register_all
from reality.preprocessing.cache import PrepareCache, PrepareReport
from reality.preprocessing import statistics
from reality.preprocessing.projection import project


class TrainingError(RuntimeError):
    """Raised when a training run cannot be assembled."""


@dataclass
class StepResult:
    """What one training step did, for logging and assertions."""

    step: int
    source_channels: int
    image_shape: tuple
    weights_changed: bool
    loss_scale: float
    degradation: Dict = field(default_factory=dict)


class Trainer:
    """One command: prepare the cache, measure statistics, then train.

    The order matters and is enforced. Preparation runs to completion first;
    statistics are measured over the whole prepared set (never a partial cache,
    never a sample); only then does training start, with those statistics baked
    into every checkpoint so they travel with the weights.
    """

    def __init__(self, config: Config, model=None, degradation=None,
                 cache_root=None, run_dir=None) -> None:
        register_all()
        self.config = config
        self.pipeline = WeatherPipeline(config, model=model, degradation=degradation) \
            if config.task.is_weather else SensorPipeline(config, model=model)
        self.run_dir = Path(run_dir or config.output.checkpoint_dir)
        self.cache = PrepareCache(config, cache_root)
        self.logger = None
        self._stats = None

    # -- stages ------------------------------------------------------------------ #

    def prepare(self, force: bool = False, limit: Optional[int] = None) -> "PrepareReport":
        """Build the prepared cache, or reuse a valid one."""
        log = self.logger.info if self.logger else print
        if self.cache.is_valid() and not force:
            log(f"cache hit: {self.cache.directory} "
                f"({len(self.cache.source_files())} source, "
                f"{len(self.cache.target_files())} target) -- skipping prepare")
            from reality.preprocessing.cache import PrepareReport
            return PrepareReport(source_skipped=len(self.cache.source_files()),
                                 target_skipped=len(self.cache.target_files()),
                                 complete=True)
        log(f"preparing cache at {self.cache.directory}")
        report = self.cache.prepare(
            self.pipeline.adapter(self.config.source),
            self.pipeline.adapter(self.config.target),
            self.pipeline.prepare_source, self.pipeline.prepare_target,
            force=force, limit=limit, log=log,
        )
        log(f"prepared {report.source_written} source and {report.target_written} target "
            f"frames ({report.source_skipped + report.target_skipped} already cached)")
        return report

    @property
    def stats(self) -> statistics.NormalizationStats:
        """Statistics over the complete prepared set, measured once and cached."""
        if self._stats is None:
            self._stats = statistics.resolve(self.config, prepared=self.cache)
        return self._stats

    def model(self):
        model = self.pipeline.model
        model.use_statistics(self.stats)
        if not model.is_built:
            model.build_model(3 if self.cache.has_reflectance else 2)
        return model

    # -- the loop ------------------------------------------------------------------ #

    def train(self, epochs: Optional[int] = None, resume: bool = True,
              checkpoint_every: int = 1, batch_size: Optional[int] = None,
              num_workers: Optional[int] = None, limit: Optional[int] = None,
              force_prepare: bool = False) -> Dict[str, Any]:
        """Run the whole thing: prepare, measure, train, checkpoint."""
        import torch
        from torch.utils.data import DataLoader

        from reality.core.determinism import seed_everything
        from reality.training import checkpoint as ckpt
        from reality.training.dataset import PreparedDataset
        from reality.training.logging_utils import setup_logging

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.run_dir)
        log = self.logger.info

        seed_everything(self.config.training.seed, deterministic=True)
        log(f"run {self.config.run_name} | seed {self.config.training.seed}")

        self.prepare(force=force_prepare, limit=limit)
        self.cache.require_complete()

        stats = self.stats
        log(f"normalization ({stats.mode}) measured over {stats.n_source_frames} source "
            f"and {stats.n_target_frames} target frames, all pixels occupied-only")
        for name in ("range", "incidence", "reflectance", "phy", "intensity"):
            mean, std = stats.pair(name)
            marker = " (fallback)" if name in stats.fallbacks else ""
            log(f"  {name:12s} mean={mean:9.4f} std={std:8.4f}{marker}")
        stats.save(self.run_dir / "normalization_stats.json")
        self.config.snapshot(self.run_dir / "config.snapshot.yaml")

        model = self.model()
        log(f"model {self.config.model.type} | in_channels_s={model.in_channels_s} "
            f"| device {model.device}")

        start_epoch = 0
        resume_path = self.run_dir / ckpt.FULL_SUFFIX
        if resume and resume_path.is_file():
            start_epoch = ckpt.restore(model, resume_path) + 1
            log(f"resumed from {resume_path} at epoch {start_epoch}")

        total_epochs = epochs if epochs is not None else self.config.training.epochs
        if start_epoch >= total_epochs:
            log(f"nothing to do: already trained {start_epoch} of {total_epochs} epochs")
            return {"epochs_run": 0, "start_epoch": start_epoch, "stats": stats}

        dataset = PreparedDataset(self.cache, model)
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.config.training.batch_size,
            shuffle=True, drop_last=False,
            num_workers=(self.config.training.num_workers if num_workers is None
                         else num_workers),
            pin_memory=model.device.type == "cuda",
        )
        log(f"training on {len(dataset)} prepared frames, batch "
            f"{batch_size or self.config.training.batch_size}, "
            f"epochs {start_epoch}..{total_epochs - 1}")

        history = []
        for epoch in range(start_epoch, total_epochs):
            started = time.time()
            before = model.gen_R.last.weight.detach().clone()
            batches = 0
            term_totals: Dict[str, float] = {}
            for batch in loader:
                terms = model.train_step(tuple(t.to(model.device) for t in batch))
                for name, value in terms.items():
                    if name in ("batches", "source_channels"):
                        continue
                    term_totals[name] = term_totals.get(name, 0.0) + float(value)
                batches += 1
            term_means = {name: total / max(batches, 1)
                          for name, total in term_totals.items()}
            changed = not torch.equal(before, model.gen_R.last.weight)
            gradient = model.gen_R.last.weight.grad
            elapsed = time.time() - started
            log(f"epoch {epoch:4d} | {batches:5d} batches | {elapsed:7.1f}s "
                f"| loss scale {model.g_scaler.get_scale():8.0f} "
                f"| weights {'updated' if changed else 'unchanged (AMP warm-up)'} "
                f"| grad {float(gradient.abs().mean()) if gradient is not None else float('nan'):.3e}")
            if term_means:
                log("           losses | " + " ".join(
                    f"{name}={value:.4f}" for name, value in sorted(term_means.items())))
            history.append({"epoch": epoch, "batches": batches, "seconds": elapsed,
                            "updated": changed,
                            "loss_scale": float(model.g_scaler.get_scale()),
                            **term_means})
            (self.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

            finite = all(torch.isfinite(p).all() for p in model.gen_R.parameters())
            if not finite:
                raise TrainingError(f"epoch {epoch}: gen_R went non-finite; stopping")

            if (epoch + 1) % checkpoint_every == 0 or epoch == total_epochs - 1:
                ckpt.save(model, self.run_dir, self.config, epoch=epoch)
                log(f"  checkpointed epoch {epoch} -> {self.run_dir}")

        ckpt.save(model, self.run_dir, self.config, epoch=total_epochs - 1)
        log(f"finished: {len(history)} epochs, checkpoints in {self.run_dir}")
        return {"epochs_run": len(history), "start_epoch": start_epoch,
                "history": history, "stats": stats, "run_dir": self.run_dir}


class SensorPipeline:
    """Sensor path: no degradation stage, phy comes from the source simulator."""

    def __init__(self, config: Config, model=None, stats=None) -> None:
        register_all()
        self.config = config
        self._model = model
        self._stats = stats

    @property
    def stats(self):
        if self._stats is None:
            self._stats = statistics.resolve(self.config)
        return self._stats

    @property
    def model(self):
        if self._model is None:
            self._model = MODELS.get(self.config.model.type)(self.config, stats=self.stats)
        return self._model

    def adapter(self, spec):
        return DATASETS.get(spec.dataset)(spec, sensor=self.config.sensor,
                                          task=self.config.task.type)

    def prepare_source(self, sample: Sample) -> Sample:
        return project(sample, self.config.sensor or sample.meta.fov)

    def prepare_target(self, sample: Sample) -> Sample:
        return project(sample, self.config.sensor or sample.meta.fov)


class WeatherPipeline:
    """Assembles the KITTI -> the weather model -> PICGAN -> CADC path from config."""

    def __init__(self, config: Config, model=None, degradation=None,
                 stats: Optional[statistics.NormalizationStats] = None) -> None:
        register_all()
        if not config.task.is_weather:
            raise TrainingError(
                f"WeatherPipeline needs task.type='weather', got {config.task.type!r}"
            )
        self.config = config
        self._stats = stats
        self._model = model
        self._degradation = degradation

    # -- components ------------------------------------------------------------- #

    @property
    def stats(self) -> statistics.NormalizationStats:
        if self._stats is None:
            self._stats = statistics.resolve(self.config)
        return self._stats

    @property
    def degradation(self):
        if self._degradation is None:
            spec = self.config.geometric_degradation
            self._degradation = DEGRADATIONS.get(spec.type)(self.config)
        return self._degradation

    @property
    def model(self):
        if self._model is None:
            self._model = MODELS.get(self.config.model.type)(self.config, stats=self.stats)
        return self._model

    def adapter(self, spec):
        return DATASETS.get(spec.dataset)(spec, sensor=self.config.sensor,
                                          task=self.config.task.type)

    # -- per-frame preparation ---------------------------------------------------- #

    def prepare_source(self, sample: Sample) -> Sample:
        """Degrade the cloud, then project it. phy comes from the weather model's ref_new."""
        degraded = self.degradation.apply(sample)
        projected = project(degraded, self.config.sensor or sample.meta.fov)
        if projected.phy is None:
            raise TrainingError(
                f"{sample.meta.dataset}: the degradation stage produced no physics "
                f"intensity; PICGAN never computes it itself"
            )
        return projected

    def prepare_target(self, sample: Sample) -> Sample:
        """Project a target frame; only its intensity channel is consumed."""
        return project(sample, self.config.sensor or sample.meta.fov)

    # -- steps --------------------------------------------------------------------- #

    def build_batch(self, sources: Sequence[Sample], targets: Sequence[Sample]):
        model = self.model
        if not model.is_built:
            model.build_model(3 if sources[0].meta.has_reflectance else 2)
        return model.build_batch(sources, targets)

    def train_step(self, sources: Sequence[Sample], targets: Sequence[Sample],
                   step: int = 0) -> StepResult:
        """Run one optimisation step on prepared source/target samples."""
        import torch

        batch = self.build_batch(sources, targets)
        model = self.model
        before = model.gen_R.last.weight.detach().clone()
        model.train_step(batch)
        return StepResult(
            step=step,
            source_channels=int(model.in_channels_s),
            image_shape=tuple(batch[0].shape[-2:]),
            weights_changed=not torch.equal(before, model.gen_R.last.weight),
            loss_scale=float(model.g_scaler.get_scale()),
            degradation=dict(sources[0].meta.extra.get("degradation", {})),
        )

    def run_steps(self, n_steps: int, n_source: int = 4, n_target: int = 4,
                  batch_size: int = 1) -> List[StepResult]:
        """Prepare a tiny subset and run ``n_steps`` optimisation steps over it."""
        source_adapter = self.adapter(self.config.source)
        target_adapter = self.adapter(self.config.target)

        sources = [self.prepare_source(source_adapter.load_sample(frame))
                   for frame in source_adapter.list_frames()[:n_source]]
        targets = [self.prepare_target(target_adapter.load_sample(frame))
                   for frame in target_adapter.list_frames()[:n_target]]
        if not sources or not targets:
            raise TrainingError("no frames available for source or target")

        results = []
        for step in range(n_steps):
            batch_sources = [sources[(step * batch_size + i) % len(sources)]
                             for i in range(batch_size)]
            batch_targets = [targets[(step * batch_size + i) % len(targets)]
                             for i in range(batch_size)]
            results.append(self.train_step(batch_sources, batch_targets, step=step + 1))
        return results
