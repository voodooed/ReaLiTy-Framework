#!/usr/bin/env python3
"""Score one experiment run on the held-out test set. Identical protocol for every run.

    python scripts/evaluate_run.py --label that variant --checkpoint runs/that variant/gen_r.pt

Output goes to <out>/<label>/: metrics.json, panels, histograms. All runs are
compared on **clamped** output, since that is what ships.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve.parents[1]))

import numpy as np
import torch

from reality.core.config import Config
from reality.evaluation import IntensityDistributionEvaluator, histogram, occupied_values
from reality.models import PicganAdapter
from reality.plugins import register_all
from reality.preprocessing.cache import PrepareCache
from reality.training import checkpoint as ckpt
from reality.training.trainer import WeatherPipeline

#: SemanticKITTI traffic-sign / plate. The retroreflector class.
RETRO_LABEL = 81
#: Thresholds for the bright-tail metrics.
TAIL_THRESHOLDS = (0.2, 0.5, 0.9)


def build_test_cache(config, cache_root: Path) -> PrepareCache:
    """Prepare the held-out split once; reused by every run."""
    pipeline = WeatherPipeline(config)
    cache = PrepareCache(config, cache_root)
    if not cache.is_valid:
        print("preparing held-out test split...", flush=True)
        cache.prepare(pipeline.adapter(config.source), pipeline.adapter(config.target),
                      pipeline.prepare_source, pipeline.prepare_target,
                      log=lambda m: print(m, flush=True))
    cache.require_complete
    return cache


def retro_masks(config, cache: PrepareCache, path: Path) -> np.ndarray:
    """Per-frame boolean mask of retroreflector pixels, aligned to the cache.

    the weather model is stochastic, so a mask must come from the *same* degradation pass that
    produced the cached stacks. The masks are therefore built once alongside the
    cache and reused, never recomputed per run.
    """
    if path.is_file:
        return np.load(path)
    print("building retro masks (label 81)...", flush=True)
    from reality.preprocessing.projection import project

    pipeline = WeatherPipeline(config)
    adapter = pipeline.adapter(config.source)
    frames = adapter.list_frames[:len(cache.source_files)]
    shape = cache.image_shape
    masks = np.zeros((len(frames),) + tuple(shape), dtype=bool)
    for i, frame in enumerate(frames):
        sample = adapter.load_sample(frame)
        degraded = pipeline.degradation.apply(sample)
        projected = project(degraded, config.sensor)
        occupied = projected.mapping >= 0
        labels = degraded.point_column("label")[projected.mapping[occupied]]
        plane = np.zeros(shape, dtype=bool)
        plane[occupied] = labels == RETRO_LABEL
        masks[i] = plane
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)
    np.save(path, masks)
    return masks


def tail_metrics(values: np.ndarray, prefix: str) -> dict:
    out = {f"{prefix}_p99": float(np.percentile(values, 99)),
           f"{prefix}_max": float(values.max)}
    for threshold in TAIL_THRESHOLDS:
        out[f"{prefix}_frac_above_{threshold}"] = float((values > threshold).mean)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="run name, e.g. the released model or that variant")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("reality/configs/weather/kitti_to_cadc.yaml"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("cache/test"))
    parser.add_argument("--out", type=Path,
                        default=Path("runs/evaluation"))
    parser.add_argument("--no-clamp", dest="clamp", action="store_false")
    parser.set_defaults(clamp=True)
    args = parser.parse_args(argv)

    register_all
    config = Config.load(args.config)
    config.source.split = config.target.split = "test"
    out_dir = args.out / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = build_test_cache(config, args.cache_root)
    masks_path = args.cache_root / "retro_masks.npy"
    retro = retro_masks(config, cache, masks_path)

    loaded = ckpt.load(args.checkpoint)
    activation = loaded.metadata.get("output_activation", "tanh")
    config.model.output_activation = activation
    model = PicganAdapter(config, workspace=out_dir, stats=loaded.stats)
    ckpt.restore(model, args.checkpoint)
    assert model.stats.to_dict == loaded.stats.to_dict
    model.gen_R.eval
    print(f"{args.label}: epoch {loaded.epoch}, head={activation}, "
          f"channels={loaded.in_channels_s}, clamp={args.clamp}")

    channels = list(cache.source_channels)
    phy_index = channels.index("phy")
    generated, physics, occupancy = [], [], []
    started = time.perf_counter
    for path in cache.source_files:
        stack = np.load(path)
        sim, _, _ = model.to_tensors(stack, np.zeros((2,) + stack.shape[1:], np.float32))
        with torch.no_grad:
            out = model.denormalize_real_intensity(
                model.gen_R(sim.unsqueeze(0).to(model.device)))
        image = out.cpu.numpy[0, 0]
        if args.clamp:
            image = np.clip(image, 0.0, 1.0)
        generated.append(image)
        physics.append(stack[phy_index])
        occupancy.append(stack[0] > 0)
    elapsed = time.perf_counter - started

    real_images, real_masks = [], []
    for path in cache.target_files:
        stack = np.load(path)
        real_images.append(stack[1])
        real_masks.append(stack[0] > 0)

    evaluator = IntensityDistributionEvaluator
    results = {
        "label": args.label, "epoch": loaded.epoch, "output_activation": activation,
        "in_channels_s": loaded.in_channels_s, "clamped": args.clamp,
        "frames": len(generated),
        "gen_frames_per_s": len(generated) / elapsed,
        "generated_vs_real": evaluator.evaluate(
            generated, real_images, generated_masks=occupancy, reference_masks=real_masks),
        "physics_vs_real": evaluator.evaluate(
            physics, real_images, generated_masks=occupancy, reference_masks=real_masks),
        "generated_vs_physics_paired": evaluator.evaluate_paired(
            generated, physics, occupancy),
    }

    gen_values = np.concatenate([g[m] for g, m in zip(generated, occupancy)])
    real_values = np.concatenate([r[m] for r, m in zip(real_images, real_masks)])
    phy_values = np.concatenate([p[m] for p, m in zip(physics, occupancy)])
    results["tail"] = {**tail_metrics(gen_values, "generated"),
                       **tail_metrics(real_values, "real"),
                       **tail_metrics(phy_values, "physics")}
    results["negative_fraction"] = float((gen_values < 0).mean)
    results["output_range"] = [float(gen_values.min), float(gen_values.max)]

    # Retro-pixel contrast: brightness on label-81 pixels versus everywhere else.
    retro_pixels = np.concatenate([g[m & r] for g, m, r in zip(generated, occupancy, retro)])
    other_pixels = np.concatenate([g[m & ~r] for g, m, r in zip(generated, occupancy, retro)])
    phy_retro = np.concatenate([p[m & r] for p, m, r in zip(physics, occupancy, retro)])
    results["retro"] = {
        "n_retro_pixels": int(retro_pixels.size),
        "generated_retro_mean": float(retro_pixels.mean),
        "generated_other_mean": float(other_pixels.mean),
        "generated_contrast": float(retro_pixels.mean / max(other_pixels.mean, 1e-9)),
        "generated_retro_frac_above_0.5": float((retro_pixels > 0.5).mean),
        "physics_retro_mean": float(phy_retro.mean),
        "real_overall_mean": float(real_values.mean),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    def show(block, name):
        print(f"\n{name}")
        for k, v in block.items:
            print(f"   {k:34s} {v:.6f}" if isinstance(v, float) else f"   {k:34s} {v}")
    for key in ("generated_vs_real", "physics_vs_real", "generated_vs_physics_paired",
                "tail", "retro"):
        show(results[key], key)
    print(f"\nnegative fraction {results['negative_fraction']:.6f} | "
          f"output range {results['output_range']}")

    # -- figures ---------------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, 101)
    centres = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for values, lab, style in ((phy_values, "physics (the weather model)", "--"),
                               (gen_values, f"{args.label} generated", "-"),
                               (real_values, "real CADC", "-")):
        ax.plot(centres, histogram(values, 100, (0, 1)), style, label=lab, linewidth=1.8)
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend
    ax.set_xlabel("intensity"); ax.set_ylabel("normalised frequency")
    ax.set_title(f"{args.label}: intensity distributions, 2,000 held-out frames")
    fig.savefig(out_dir / "histograms.png", dpi=130); plt.close(fig)

    # Frames with the most retro pixels, so sign behaviour is visible.
    ranked = np.argsort([-(r & m).sum for r, m in zip(retro, occupancy)])[:2]
    for idx in ranked:
        rows = np.any(retro[idx], axis=1); cols = np.any(retro[idx], axis=0)
        r0, r1 = max(np.argmax(rows) - 6, 0), min(len(rows) - np.argmax(rows[::-1]) + 6, retro[idx].shape[0])
        c0, c1 = max(np.argmax(cols) - 60, 0), min(len(cols) - np.argmax(cols[::-1]) + 60, retro[idx].shape[1])
        fig, axes = plt.subplots(3, 1, figsize=(13, 6), constrained_layout=True)
        for ax, image, title in zip(axes,
                [physics[idx][r0:r1, c0:c1], generated[idx][r0:r1, c0:c1],
                 real_images[idx % len(real_images)][r0:r1, c0:c1]],
                ["physics (the weather model)", f"{args.label} generated", "real CADC (unpaired)"]):
            im = ax.imshow(image, cmap="viridis", aspect="auto", vmin=0, vmax=1.0)
            ax.set_title(f"{title} — retro close-up, frame {idx}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.03)
        fig.savefig(out_dir / f"retro_closeup_{idx:04d}.png", dpi=120); plt.close(fig)

    for idx in (0, 1000):
        fig, axes = plt.subplots(3, 1, figsize=(16, 7), constrained_layout=True)
        for ax, image, title in zip(axes,
                [physics[idx], generated[idx], real_images[idx % len(real_images)]],
                ["physics (the weather model)", f"{args.label} generated", "real CADC (unpaired)"]):
            im = ax.imshow(image, cmap="viridis", aspect="auto", vmin=0, vmax=0.5)
            ax.set_title(f"{title} — frame {idx}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.02)
        fig.savefig(out_dir / f"panel_{idx:04d}.png", dpi=110); plt.close(fig)

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main)
