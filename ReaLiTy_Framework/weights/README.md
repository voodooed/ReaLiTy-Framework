# Weights

Organised by task, mirroring `reality/configs/`:

```
weights/
├── sensor/     sensor-to-sensor intensity transfer
└── weather/    clear-to-adverse-weather transfer
```

## Released checkpoint

`weather/kitti_to_cadc_gen_r.pt` — KITTI → CADC (snow), 200 epochs on 5,000
source / 5,000 target frames.

Each checkpoint carries its **normalization statistics inside the file**, so
generated intensity is denormalised with the constants the weights were trained
against regardless of what data is present locally. `*_metadata.json` records the
datasets, task, framework version and a snapshot of the run config.

```bash
python -m reality generate \
    --config reality/configs/weather/kitti_to_cadc.yaml \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt \
    --split test
```

Training writes two files per run: `full.pt` (resumable — networks, optimizers,
AMP scaler, epoch) and `gen_r.pt` (inference only, ~44 MB). Only the second is
needed to convert data.

The released inference checkpoint is **committed to the repository** (44 MB, well
within GitHub's limits) so a fresh clone can convert data immediately with no
extra download. Large per-run training checkpoints (`full.pt`) are gitignored.
