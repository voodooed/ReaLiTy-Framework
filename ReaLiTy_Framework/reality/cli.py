"""Command-line entry point: ``python -m reality <command> --config ...``.

Commands: ``prepare-data``, ``train``, ``generate``, ``evaluate``.

Each command loads and validates its config, prints the resolved run, and
delegates to the corresponding stage of the framework.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from reality.core.config import Config, ConfigError
from reality.core.pipeline import plan_stages
from reality.plugins import register_all
from reality.core.version import __version__

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="reality",
        description="ReaLiTy — LiDAR intensity adaptation across sensors and weather.",
    )
    parser.add_argument("--version", action="version", version=f"reality {__version__}")
    subs = parser.add_subparsers(dest="command", metavar="{train,generate,evaluate,prepare-data}")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subs.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("--config", required=True, help="path to the run config YAML")
        sub.add_argument("--data-root", default=None,
                         help="root holding one folder per dataset (overrides the "
                              "config's data_root)")
        return sub

    prepare = add("prepare-data",
                  "Build the prepared range-image cache (optional: train does this).")
    prepare.add_argument("--force", action="store_true",
                         help="rebuild even if a valid cache exists")
    prepare.add_argument("--limit", type=int, default=None,
                         help="prepare only the first N frames")
    prepare.add_argument("--cache-root", default=None, help="where to write the cache")

    train = add("train", "Train the intensity model described by the config.")
    train.add_argument("--epochs", type=int, default=None,
                       help="override training.epochs")
    train.add_argument("--no-resume", action="store_true",
                       help="start from scratch instead of continuing a run")
    train.add_argument("--checkpoint-every", type=int, default=1,
                       help="write a full checkpoint every N epochs")
    train.add_argument("--cache-root", default=None,
                       help="where the prepared cache lives (fast local scratch on a cluster)")
    train.add_argument("--limit", type=int, default=None,
                       help="use only the first N frames (smoke runs)")
    train.add_argument("--force-prepare", action="store_true",
                       help="rebuild the prepared cache first")
    generate = add("generate", "Apply a trained checkpoint to the source dataset.")
    generate.add_argument("--checkpoint", required=True, help="path to the model checkpoint")
    generate.add_argument("--split", default=None, choices=["train", "test"],
                          help="which split to convert (default: the config's)")
    generate.add_argument("--output", default=None,
                          help="where to write clouds (default: <checkpoint_dir>/generated)")
    generate.add_argument("--format", default="bin", choices=["bin", "npy"],
                          help="output cloud format")
    generate.add_argument("--limit", type=int, default=None,
                          help="convert only the first N frames")
    generate.add_argument("--log-every", type=int, default=100,
                          help="progress line every N frames")
    generate.add_argument("--no-clamp", dest="clamp", action="store_false",
                          help="write the raw generator output instead of clamping it "
                               "to [0, 1]. Intensity is physically non-negative and "
                               "clamped output is what any downstream detector consumes, "
                               "so clamping is the default; use this only to analyse the "
                               "raw distribution.")
    generate.set_defaults(clamp=True)
    evaluate = add("evaluate",
                   "Score a checkpoint against the target intensity distribution.")
    evaluate.add_argument("--checkpoint", required=True, help="checkpoint to score")
    evaluate.add_argument("--label", default="run", help="name for the results folder")
    evaluate.add_argument("--output", default=None,
                          help="where to write metrics and figures")
    evaluate.add_argument("--no-clamp", dest="clamp", action="store_false",
                          help="score the raw output instead of clamped [0, 1]")
    evaluate.set_defaults(clamp=True)
    return parser


def _describe(command: str, config: Config) -> None:
    """Print the resolved run the command would execute."""
    plan = " -> ".join(str(s) for s in plan_stages(config))
    print(f"[reality {__version__}] {command}: {config.run_name} (task={config.task.type})")
    print(f"  model:       {config.model.type}")
    print(f"  stages:      {plan}")
    print(f"  checkpoints: {config.output.checkpoint_dir}")


def _apply_data_root(config: Config, data_root) -> Config:
    """Re-point a config at a different dataset root, keeping explicit paths."""
    if not data_root:
        return config
    return Config.from_dict({**config.to_dict(), "data_root": str(data_root),
                             **_without_resolved_paths(config)})


def _without_resolved_paths(config: Config) -> dict:
    """Drop paths that came from the old data_root so they resolve afresh."""
    data = config.to_dict()
    previous = config.data_root
    out = {}
    for role in ("source", "target"):
        spec = dict(data[role])
        if previous and spec.get("path", "").startswith(str(previous)):
            spec.pop("path", None)
        out[role] = spec
    return out


def _cmd_train(args: argparse.Namespace) -> int:
    """Prepare (if needed), measure statistics, then train."""
    from reality.training import Trainer

    config = _apply_data_root(Config.load(args.config), args.data_root)
    register_all()
    _describe("train", config)
    trainer = Trainer(config, cache_root=args.cache_root)
    trainer.train(epochs=args.epochs, resume=not args.no_resume,
                  checkpoint_every=args.checkpoint_every, limit=args.limit,
                  force_prepare=args.force_prepare)
    return EXIT_OK


def _cmd_prepare(args: argparse.Namespace) -> int:
    """Build the prepared cache without training -- for cluster pre-staging."""
    from reality.training import Trainer

    config = _apply_data_root(Config.load(args.config), args.data_root)
    register_all()
    _describe("prepare-data", config)
    trainer = Trainer(config, cache_root=args.cache_root)
    report = trainer.prepare(force=args.force, limit=args.limit)
    print(f"[reality] cache {'complete' if report.complete else 'INCOMPLETE'} at "
          f"{trainer.cache.directory}")
    return EXIT_OK if report.complete else EXIT_CONFIG_ERROR


def _cmd_generate(args: argparse.Namespace) -> int:
    """Convert a dataset split with a trained checkpoint."""
    import time

    from reality.inference import IntensityGenerator
    from reality.io import OutputWriter
    from reality.models import PicganAdapter
    from reality.training import checkpoint as ckpt
    from reality.training.logging_utils import setup_logging

    config = _apply_data_root(Config.load(args.config), args.data_root)
    if args.split:
        config.source.split = args.split
    register_all()

    output_dir = Path(args.output or Path(config.output.checkpoint_dir) / "generated")
    logger = setup_logging(output_dir, filename="generate.log")
    log = logger.info
    _describe("generate", config)

    # The statistics that denormalise the output must be the ones the weights were
    # trained with. They travel inside the checkpoint precisely so that inference
    # never depends on what data happens to be on this machine.
    loaded = ckpt.load(args.checkpoint)
    stats = loaded.stats
    log(f"checkpoint {args.checkpoint} (epoch {loaded.epoch}, "
        f"in_channels_s={loaded.in_channels_s})")
    log(f"normalization source: CHECKPOINT (mode={stats.mode}, measured over "
        f"{stats.n_source_frames} source / {stats.n_target_frames} target frames)")
    for name in ("range", "incidence", "reflectance", "phy", "intensity"):
        mean, std = stats.pair(name)
        log(f"  {name:12s} mean={mean:9.4f} std={std:8.4f}")

    model = PicganAdapter(config, workspace=output_dir, stats=stats)
    ckpt.restore(model, args.checkpoint)
    if model.stats.to_dict() != stats.to_dict():
        raise RuntimeError(
            "the model is not using the checkpoint's statistics; refusing to write "
            "output that would be denormalised with the wrong constants"
        )
    log(f"model on {model.device}, statistics confirmed identical to the checkpoint")

    writer = OutputWriter(output_dir / "clouds", fmt=args.format,
                          columns=("x", "y", "z", "intensity"))
    generator = IntensityGenerator(config, model=model, writer=writer, stats=stats,
                                   clamp=(0.0, 1.0) if args.clamp else None)
    log("intensity clamped to [0, 1] (released clouds are clamped)" if args.clamp
        else "NOT clamping: writing raw generator output for distribution analysis")

    started = time.perf_counter()
    written = points = 0
    for frame in generator.stream(limit=args.limit, log_every=args.log_every, log=log):
        written += 1
        points += frame.stats["n_points"]
    elapsed = time.perf_counter() - started

    log(f"converted {written} frames ({points:,} points) in {elapsed:.1f}s "
        f"= {written / elapsed:.2f} frames/s")
    log(f"output: {output_dir / 'clouds'}")
    return EXIT_OK


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Score a checkpoint on the held-out split against the target distribution."""
    from reality.training.logging_utils import setup_logging

    config = _apply_data_root(Config.load(args.config), args.data_root)
    register_all()
    output_dir = Path(args.output or Path(config.output.checkpoint_dir) / "evaluation")
    logger = setup_logging(output_dir, filename="evaluate.log")
    _describe("evaluate", config)

    # The heavy lifting lives in the standalone harness so a run can also be
    # scored outside the CLI; this wires it to the same config and checkpoint.
    from scripts.evaluate_run import main as evaluate_main

    argv = ["--label", args.label, "--checkpoint", str(args.checkpoint),
            "--config", str(args.config), "--out", str(output_dir.parent)]
    if not args.clamp:
        argv.append("--no-clamp")
    logger.info(f"scoring {args.checkpoint} as '{args.label}'")
    return evaluate_main(argv)


#: Command name -> handler.
HANDLERS: Dict[str, Handler] = {
    "prepare-data": _cmd_prepare,
    "train": _cmd_train,
    "generate": _cmd_generate,
    "evaluate": _cmd_evaluate,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse ``argv``, dispatch to the command handler and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.command:
        parser.print_help()
        return EXIT_OK

    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"[reality] config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
