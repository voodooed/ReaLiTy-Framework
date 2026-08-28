# ReaLiTy — quick start

Turn clear-weather LiDAR into snow-degraded LiDAR: **KITTI → physics weather degradation → PICGAN → CADC**.
Two commands. This page is the short version; `README.md` has the full architecture.

---

## 1. Install

```bash
conda create -n reality python=3.11 && conda activate reality
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install -r requirements.txt
```

The CUDA 12.6 wheels matter: the cu130 build has no `sm_70` kernels and every CUDA
op fails on a V100. Check the GPU really works, not just that it is visible:

```bash
python -c "import torch; torch.mm(torch.randn(8,8,device='cuda'), torch.randn(8,8,device='cuda')); print('ok')"
```

---

## 2. Organise the data

One command builds the layout the framework reads:

```bash
python tools/prepare_data_kitti_cadc.py \
    --kitti /path/to/KITTI \
    --cadc  /path/to/CADC \
    --out   data
```

```
data/
├── KITTI/
│   ├── train/                 000000.bin ...
│   ├── test/                  000000.bin ...
│   └── labels/{train,test}/   000000.label ...   (found automatically)
└── CADC/
    ├── train/
    └── test/
```

**Datasets are named on disk; roles are chosen in config.** Nothing in
`data/KITTI/` says "source" — that is decided by the run:

```yaml
data_root: Data
source: {dataset: kitti}          # -> data/KITTI
target: {dataset: cadc}           # -> data/CADC
task: {type: weather}
geometric_degradation: {weather: snow, ...}
```

So the same `data/KITTI` is the source of `kitti_to_cadc` and, with a different
config, the source of a future `kitti_to_nuscenes` — no data is copied or moved.
Weather type is a config attribute (`weather: snow`), never a folder level: CADC
*is* the snow dataset, so identity stays one level deep.

**Adding a dataset is a folder plus a config** — drop `data/<Name>/{train,test}`
in place and name it in a config. No code change.

### How the splits are filled

The two roles are treated differently, on purpose:

| | train | test |
|---|---|---|
| **Source** (KITTI) | trains the model | held back for evaluation and inference |
| **Target** (CADC) | the unpaired distribution the discriminator sees | **unused** |

The weather target is only ever an unpaired distribution, so its `test` split is
never read by training or inference. The script therefore fills target `train` to
**parity with the source train split first**, and whatever is left becomes `test`.
CADC's 6,323 frames give **5,000 train / 1,323 test** — the deficit lands in test,
not taken out of train. Target `train` shrinks only if the dataset is smaller than
the requested size, and the script warns loudly if that happens.

Control it with `--source-train` (default 5,000), `--source-test` (default 2,000)
and `--target-train` (defaults to matching `--source-train`). Add `--dry-run` to
preview counts, `--symlink` to link instead of copy.

**Formats:** `.bin` (headerless float32) and `.npy` both work; columns are declared
in config, never guessed. **Scale:** ~7,000 frames is the *training* convention,
not a limit — at inference you can point a run at a `test/` folder holding the
entire odometry set. Bringing your own data? Same shape, `dataset: generic`, with
your `columns` declared. See README → *Dataset Organization*.

---

## 3. Train

```bash
python -m reality train --config reality/configs/weather/kitti_to_cadc.yaml
```

That single command does everything, in order:

1. **Prepare** — runs physics weather degradation on each source cloud, projects both domains to
   range images, and caches the stacks. Keyed by a hash of the settings that
   affect the tensors, so it happens **once**; later runs print `cache hit` and
   skip it. Roughly 13 min for 5,000 frames.
2. **Measure** — computes normalization constants over the *whole* prepared set
   (PICGAN's shipped constants were fitted on different data and are wrong here).
3. **Train** — epochs over the cache, checkpointing as it goes.

Useful flags:

| Flag | Why |
|---|---|
| `--cache-root /fast/scratch` | put the cache on fast local disk (do this on a cluster) |
| `--epochs 10` | override the config |
| `--checkpoint-every 5` | checkpoint less often |
| `--limit 50` | tiny smoke run |
| `--no-resume` | start over instead of continuing |

Re-running the same command **resumes** from the last checkpoint. A killed job just
needs resubmitting.

On a cluster, use `scripts/train_cadc.sh` as a starting point (~12 h for 5,000
frames × 200 epochs on one V100; the script requests 16 h for margin).

### What you get

```
checkpoints/kitti_to_cadc/
├── full.pt                     resumable: all networks, optimizers, AMP scaler, epoch
├── gen_r.pt                    inference only: gen_R + normalization stats (~44 MB)
├── metadata.json               datasets, task, weather, versions, config snapshot
├── normalization_stats.json    the measured constants
└── train.log
```

Watch `train.log`. Note that AMP skips the first few optimizer steps while the
loss scale calibrates — `weights unchanged (AMP warm-up)` on early epochs is
normal, not a stalled run.

---

## 4. Bring the weights home

Copy back **`gen_r.pt`** — it is small (~44 MB) and carries the normalization
statistics *inside* it, so generated intensities denormalize with the constants
the weights were trained against, no matter what data is local.

## 5. Convert data

Apply a trained checkpoint to a dataset split:

```bash
python -m reality generate \
    --config reality/configs/weather/kitti_to_cadc.yaml \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt \
    --split test \
    --output runs/kitti_to_cadc/generated
```

Each frame is degraded (the weather model), projected, passed through `gen_R`, and
back-projected onto its own points, so **geometry is unchanged** and the output is
a drop-in `.bin` in the source's native column layout. The run logs which
statistics it is using and asserts they came from the checkpoint — inference never
recomputes them from whatever data happens to be present.

**Released clouds are clamped to `[0, 1]`.** Intensity is physically non-negative
and clamped output is what any downstream detector consumes, so clamping is the
default. Pass `--no-clamp` to write the raw generator output instead — useful only
when analysing the distribution itself, since the current `tanh` head undershoots
zero on a fraction of points.

Other flags: `--limit N` (smoke run), `--format npy`, `--log-every N`.

### Converting your own data, at any scale

Nothing in the conversion path scales with dataset size. Frames are loaded,
transformed, written and released one at a time, so converting 2,000 frames and
converting the entire KITTI odometry set (43,552 frames) use the same memory —
only wall time differs. Labels are not required at inference: an unlabelled source
simply produces a 2-channel stack, provided the checkpoint was trained that way.

To convert a whole dataset, put it in a `test/` folder and point a run at it:

```
data/
└── KITTI_full/
    └── test/          all 43,552 frames
```

```bash
python -m reality generate \
    --config reality/configs/weather/kitti_to_cadc.yaml \
    --data-root Data \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt \
    --split test \
    --output converted/kitti_full_snow
```

with `source: {dataset: kitti}` resolving to `data/KITTI_full`, or an explicit
`--data-root`/`path:` for data kept elsewhere. The same applies to your own
dataset under the `generic` adapter: declare its columns, arrange a `test/`
folder, and convert.

Budget roughly from the measured single-frame cost on your hardware (the weather model +
projection + `gen_R` + back-projection + write); the run prints frames/s as it
goes. Splitting a large conversion across several jobs is just several runs with
different source folders — there is no shared state between frames.

## 6. Score a model

```bash
python -m reality evaluate \
    --config reality/configs/weather/kitti_to_cadc.yaml \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt \
    --label my_run
```

Compares generated intensity against the target domain on the held-out split —
Wasserstein, KL both directions, histogram MSE, tail fractions, retroreflector
contrast — and writes metrics plus histogram and range-image panels. The physics
baseline is scored alongside, so you can see what the model adds over the weather model alone.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no kernel image is available` | wrong torch build — reinstall from the cu126 index |
| `weather model not found` | set `$REALITY_WEATHER_MODEL_PATH` or `geometric_degradation.weather_model_path` (see docs/weather_model.md) |
| `cannot import name 'trapz'` | SciPy ≥ 1.14; the wrapper shims it, so this means the weather model was imported directly |
| `a batch mixes source stacks of [2, 3] channels` | labelled and unlabelled frames in one `train/` folder |
| `cache is not complete` | a prepare pass was interrupted — just re-run, it resumes |
| phy falls back to PICGAN's constant | the weather model was unavailable when statistics were measured; fix the path and re-run with `--force-prepare` |
