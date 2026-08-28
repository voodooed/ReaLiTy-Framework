# Datasets

Datasets are **not** committed — they are tens of gigabytes. Build this directory
locally; the layout is the only thing the framework requires.

## Layout

One directory per dataset, named for the dataset rather than the role it plays:

```
data/
├── KITTI/
│   ├── train/                 point clouds used for training
│   ├── test/                  held back for evaluation and inference
│   └── labels/{train,test}/   optional, enables the reflectance channel
├── CADC/
│   ├── train/
│   └── test/
├── nuScenes/                  same shape
└── Boreas/                    same shape
```

Whether a dataset is the *source* or the *target* is decided in the run config,
not by the filesystem, so the same `data/KITTI` can serve several experiments:

```yaml
data_root: data
source: {dataset: kitti}     # -> data/KITTI
target: {dataset: cadc}      # -> data/CADC
```

## Formats

| format | notes |
|---|---|
| `.bin` | headerless `float32`, one row per point |
| `.npy` | `(N, C)` array of points |

Columns are **declared in the config**, never inferred:

```yaml
source: {dataset: generic, path: data/MySensor, format: bin,
         columns: [x, y, z, intensity], intensity_scale: 1.0}
```

`x`, `y`, `z` and `intensity` are required; extra columns (`ring`, `timestamp`,
…) are carried through untouched. Both formats may be mixed in one folder, and
sub-directories inside `train/` are walked.

## Building KITTI → CADC

```bash
python tools/prepare_data_kitti_cadc.py \
    --kitti /path/to/raw/KITTI \
    --cadc  /path/to/raw/CADC \
    --out   data
```

Deterministic for a given `--seed`; writes `data/manifest.json` recording the raw
source of every file. `--dry-run` previews the counts.

## Splits

| role | `train/` | `test/` |
|---|---|---|
| source | trains the model | held back for evaluation and inference |
| weather target | unpaired distribution for the discriminator | not consumed |

Because a weather target's `test/` split is never read during training or
inference, the preparation tool fills the target's `train/` to parity with the
source's first and puts the remainder in `test/`.
