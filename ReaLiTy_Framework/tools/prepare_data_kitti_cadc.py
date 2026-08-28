#!/usr/bin/env python3
"""Build the Data/ layout from raw KITTI and CADC.

Run once, before training. Datasets are organised **by identity, not by role**:

    Data/
    ├── KITTI/
    │   ├── train/                *.bin
    │   ├── test/                 *.bin
    │   └── labels/{train,test}/  *.label  (SemanticKITTI, drives the reflectance channel)
    └── CADC/
        ├── train/                *.bin
        └── test/                 *.bin

Which dataset is the source and which is the target is decided by the run config,
so the same ``Data/KITTI`` is the source of ``kitti_to_cadc`` and could be the
target of another run without moving a byte. Weather type is likewise a config
attribute (``weather: snow``), not a folder level.

Usage:

    python prepare_data_kitti_cadc.py \\
        --kitti /path/to/raw/KITTI \\
        --cadc  /path/to/raw/CADC \\
        --out   Data

Split policy differs by role, deliberately:

* **Source** (KITTI) contributes both splits: ``train`` trains, ``test`` is held
  back for evaluation and inference.
* **Target** (CADC) is only ever an unpaired distribution for the discriminator.
  Its ``test`` split is not consumed by training or inference, so the target's
  ``train`` is filled to parity with the source's ``train`` **first** and whatever
  remains becomes ``test``. Target ``train`` shrinks only if the dataset is
  smaller than the requested size, and that is warned about loudly.

This script is deliberately outside the ``reality`` package: it is a one-off data
organisation step, not part of the pipeline. Once Data/ exists, zip it together
with the framework and ship both to the cluster.

Sampling is deterministic for a given ``--seed``, and ``manifest.json`` records
which raw frame every output file came from, so the selection is reproducible and
auditable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# KITTI odometry sequences 00-10 carry SemanticKITTI labels; 11-21 do not, and
# mixing them would mix 3-channel and 2-channel sources in one training set.
LABELLED_SEQUENCES = [f"{i:02d}" for i in range(11)]

KITTI_VELODYNE = "data_odometry_velodyne/dataset/sequences"
KITTI_LABELS = "data_odometry_labels/dataset/sequences"
CADC_LIDAR = "labeled/lidar_points/data"


class PrepareError(RuntimeError):
    """Raised when the raw data is not where it is expected."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def find_kitti_frames(root: Path) -> List[Tuple[Path, Path]]:
    """Every labelled KITTI frame, as (cloud, label) pairs."""
    velodyne, labels = root / KITTI_VELODYNE, root / KITTI_LABELS
    if not velodyne.is_dir:
        raise PrepareError(f"no KITTI velodyne sequences at {velodyne}")

    frames: List[Tuple[Path, Path]] = []
    for sequence in LABELLED_SEQUENCES:
        cloud_dir = velodyne / sequence / "velodyne"
        label_dir = labels / sequence / "labels"
        if not cloud_dir.is_dir:
            continue
        if not label_dir.is_dir:
            print(f"  ! sequence {sequence} has no labels; skipping "
                  f"(the source needs them for the reflectance channel)")
            continue
        for cloud in sorted(cloud_dir.glob("*.bin")):
            label = label_dir / f"{cloud.stem}.label"
            if label.is_file:
                frames.append((cloud, label))
    return frames


def find_cadc_frames(root: Path) -> List[Path]:
    """Every CADC lidar frame, skipping drives that ship none."""
    base = root / "cadcd" if (root / "cadcd").is_dir else root
    if not base.is_dir:
        raise PrepareError(f"no CADC data at {base}")

    frames: List[Path] = []
    skipped: List[str] = []
    for date in sorted(p for p in base.iterdir if p.is_dir):
        for drive in sorted(p for p in date.iterdir if p.is_dir):
            if drive.name == "calib":
                continue
            cloud_dir = drive / CADC_LIDAR
            if not cloud_dir.is_dir:
                skipped.append(f"{date.name}/{drive.name}")
                continue
            frames.extend(sorted(cloud_dir.glob("*.bin")))
    if skipped:
        print(f"  ! {len(skipped)} drive(s) without lidar, skipped: {', '.join(skipped)}")
    return frames


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def source_counts(available: int, n_train: int, n_test: int) -> Tuple[int, int]:
    """Source splits: both are used, so a shortfall is shared between them."""
    requested = n_train + n_test
    if requested <= available:
        return n_train, n_test
    scale = available / requested
    train = int(n_train * scale)
    test = available - train
    print(f"  ! source: {requested:,} frames requested but only {available:,} exist; "
          f"keeping the {n_train}:{n_test} ratio -> {train:,} train / {test:,} test")
    return train, test


def target_counts(available: int, n_train: int) -> Tuple[int, int]:
    """Target splits: train reaches parity with the source first, test is leftover.

    The weather target is consumed only as an unpaired distribution for the
    discriminator, so its test split is never read by training or inference.
    Filling train first is what matters; anything left over goes to test rather
    than being taken away from train.
    """
    if available < n_train:
        print(f"  ! target: only {available:,} frames available, fewer than the "
              f"{n_train:,} needed for parity with the source train split. Training "
              f"will see a smaller target distribution than source; consider "
              f"lowering --source-train.")
        return available, 0
    return n_train, available - n_train


def sample(items: List, n_train: int, n_test: int, seed: int) -> Tuple[List, List]:
    """Deterministically draw disjoint train and test subsets."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    train_idx = sorted(order[:n_train].tolist)
    test_idx = sorted(order[n_train:n_train + n_test].tolist)
    return [items[i] for i in train_idx], [items[i] for i in test_idx]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def place(source: Path, destination: Path, symlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists or destination.is_symlink:
        destination.unlink
    if symlink:
        destination.symlink_to(source.resolve)
    else:
        shutil.copy2(source, destination)


def write_kitti(frames, out_root: Path, split: str, symlink: bool) -> List[Dict]:
    records = []
    for index, (cloud, label) in enumerate(frames):
        name = f"{index:06d}"
        place(cloud, out_root / split / f"{name}.bin", symlink)
        place(label, out_root / "labels" / split / f"{name}.label", symlink)
        records.append({"output": f"{split}/{name}.bin", "source": str(cloud),
                        "label_source": str(label)})
    return records


def write_cadc(frames, out_root: Path, split: str, symlink: bool) -> List[Dict]:
    records = []
    for index, cloud in enumerate(frames):
        name = f"{index:06d}"
        place(cloud, out_root / split / f"{name}.bin", symlink)
        records.append({"output": f"{split}/{name}.bin", "source": str(cloud)})
    return records


def directory_size(path: Path) -> float:
    """Size in GiB, following symlinks only for real copies."""
    total = sum(p.stat.st_size for p in path.rglob("*")
                if p.is_file and not p.is_symlink)
    return total / 2 ** 30


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti", required=True, type=Path,
                        help="raw KITTI odometry root (holds data_odometry_velodyne/)")
    parser.add_argument("--cadc", required=True, type=Path,
                        help="raw CADC root (holds cadcd/)")
    parser.add_argument("--out", default=Path("Data"), type=Path,
                        help="where to build the dataset folders (default: Data)")
    parser.add_argument("--source-train", type=int, default=5000,
                        help="source training frames (default: 5000)")
    parser.add_argument("--source-test", type=int, default=2000,
                        help="source test frames, held back for evaluation (default: 2000)")
    parser.add_argument("--target-train", type=int, default=None,
                        help="target training frames (default: match --source-train). "
                             "The target's test split is whatever is left over and is "
                             "not consumed by training or inference.")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed; the same seed reproduces the same split")
    parser.add_argument("--symlink", action="store_true",
                        help="link instead of copy (fast and small, but not portable: "
                             "use real copies for anything you intend to ship)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written and stop")
    args = parser.parse_args(argv)

    print("Scanning raw datasets")
    kitti_frames = find_kitti_frames(args.kitti)
    cadc_frames = find_cadc_frames(args.cadc)
    print(f"  KITTI: {len(kitti_frames):,} labelled frames "
          f"(sequences {LABELLED_SEQUENCES[0]}-{LABELLED_SEQUENCES[-1]})")
    print(f"  CADC : {len(cadc_frames):,} frames")
    if not kitti_frames or not cadc_frames:
        raise PrepareError("nothing to sample; check --kitti and --cadc")

    kitti_train_n, kitti_test_n = source_counts(len(kitti_frames), args.source_train,
                                                args.source_test)
    target_train_request = args.target_train or kitti_train_n
    cadc_train_n, cadc_test_n = target_counts(len(cadc_frames), target_train_request)

    kitti_train, kitti_test = sample(kitti_frames, kitti_train_n, kitti_test_n, args.seed)
    cadc_train, cadc_test = sample(cadc_frames, cadc_train_n, cadc_test_n, args.seed + 1)

    source_root = args.out / "KITTI"
    target_root = args.out / "CADC"
    print(f"\nPlanned layout under {args.out}/")
    print(f"  KITTI/train {len(kitti_train):,}   KITTI/test {len(kitti_test):,}"
          f"   (source: both splits used)")
    print(f"  CADC/train  {len(cadc_train):,}   CADC/test  {len(cadc_test):,}"
          f"   (target: train at parity with source, test is unused leftover)")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    action = "linking" if args.symlink else "copying"
    print(f"\n{action.capitalize} frames (this is the slow part)")
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "symlink": args.symlink,
        "kitti_root": str(args.kitti), "cadc_root": str(args.cadc),
        "labelled_sequences": LABELLED_SEQUENCES,
        "layout": "dataset-centric: Data/<Dataset>/{train,test}; role is set in the run config",
        "split_policy": {
            "source": "train and test both used (train trains, test evaluates)",
            "target": ("train filled to parity with source train; test is the "
                       "leftover and is not consumed by training or inference"),
        },
        "counts": {
            "KITTI_train": len(kitti_train), "KITTI_test": len(kitti_test),
            "CADC_train": len(cadc_train), "CADC_test": len(cadc_test),
        },
        "frames": {},
    }
    manifest["frames"]["KITTI_train"] = write_kitti(kitti_train, source_root, "train",
                                                    args.symlink)
    print(f"  KITTI/train done ({len(kitti_train):,})")
    manifest["frames"]["KITTI_test"] = write_kitti(kitti_test, source_root, "test",
                                                   args.symlink)
    print(f"  KITTI/test  done ({len(kitti_test):,})")
    manifest["frames"]["CADC_train"] = write_cadc(cadc_train, target_root, "train",
                                                  args.symlink)
    print(f"  CADC/train  done ({len(cadc_train):,})")
    manifest["frames"]["CADC_test"] = write_cadc(cadc_test, target_root, "test",
                                                 args.symlink)
    print(f"  CADC/test   done ({len(cadc_test):,})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nBuilt {args.out}/")
    for label, path in (("KITTI", source_root), ("CADC", target_root)):
        for split in ("train", "test"):
            files = list((path / split).glob("*.bin"))
            print(f"  {label}/{split:5s} {len(files):6,} frames")
    print(f"  KITTI/labels {sum(1 for _ in (source_root / 'labels').rglob('*.label')):,} files")
    if not args.symlink:
        print(f"  total size {directory_size(args.out):.1f} GiB")
    print(f"  manifest   {args.out / 'manifest.json'}")
    print("\nPoint a config at it (role is decided here, not on disk):")
    print(f"  data_root: {args.out}")
    print(f"  source: {{dataset: kitti}}")
    print(f"  target: {{dataset: cadc}}")
    return 0


if __name__ == "__main__":
    sys.exit(main)
