# ReaLiTy: Realistic LiDAR Transformation Framework 
## Sim2Real Adaptation for Realistic LiDAR Sensor and Weather Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Tested-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**ReaLiTy** transforms clear-weather simulated or real LiDAR point clouds so their distribution matches a different target condition — either another sensor's response characteristics, or the same scene as it would appear under adverse weather. 

Scene Layout is preserved exactly, and outputs are written in the source's native format, so a converted cloud is a drop-in replacement for the original in downstream training and evaluation.

```
Input cloud → [physics weather degradation] → spherical projection
            → intensity transfer (PICGAN) → back-projection → output cloud
```

Built around our Physics-Informed Cycle-Consistent generative adversarial architecture (PICGAN) and physics-informed atmospheric modeling, ReaLiTy bridges the sim-to-real gap for autonomous vehicle perception systems. It enables robust adaptation across heterogeneous sensors and adverse weather conditions (such as rain and snow) without requiring full data recollection.

Using this framework, we introduce **LADS** (LiDAR Adaptation Dataset Suite), a large-scale benchmark that injects physically accurate adverse weather (snow and rain) into standard clear-weather autonomous driving datasets like KITTI and nuScenes.

---

## Core Capabilities

* **Sensor-to-Sensor Adaptation:** Transform synthetic or source-domain point clouds to emulate the intensity, beam divergence, and noise profiles of target LiDAR senso.
* **Physics-Informed Weather Adaptation:** Seamlessly integrate an atmospheric model to simulate the geometric and intensity effects of adverse weather (fog, snow, rain) on clean point clouds.
* **Sim-to-Real Intensity Bridging:** Reduce the domain gap by learning target-domain intensity distributions conditioned on local geometry, incidence angles, and acquisition context.
* **Fast Vectorized Backprojection:** Efficiently map 2D predicted intensity tensors back to 3D spherical point clouds using an optimized, loop-free projection module.
---

## Installation

```bash
conda create -n reality python=3.11 && conda activate reality
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install -r requirements.txt
```
The CUDA build matters. Check that the GPU can actually run a kernel, not merely
that it is visible:

```bash
python -c "import torch; torch.mm(torch.randn(8,8,device='cuda'), torch.randn(8,8,device='cuda')); print('ok')"
```

The physics weather-degradation model ships with the repository, so the weather
path works out of the box. To substitute your own, see
[docs/weather_model.md](docs/weather_model.md).

---

## Data

One directory per dataset; the role each plays is chosen in the run config, not by
the filesystem:

```
data/
├── KITTI/
│   ├── train/                 point clouds
│   ├── test/                  held back for evaluation and inference
│   └── labels/{train,test}/   optional, enables the reflectance channel
└── CADC/
    ├── train/
    └── test/
```

```bash
python tools/prepare_data_kitti_cadc.py \
    --kitti /path/to/raw/KITTI --cadc /path/to/raw/CADC --out data
```

`.bin` (headerless float32) and `.npy` are both accepted, and columns are declared
in config rather than inferred. Full details, including the split policy and
bring-your-own-data, are in [data/README.md](data/README.md).

---

## Usage

Four commands cover the workflow. Configurations live in
`reality/configs/{sensor,weather}/`.

```bash
# 1. Build the cached range-image stacks (optional — train does this automatically)
python -m reality prepare-data --config reality/configs/weather/kitti_to_cadc.yaml

# 2. Train: prepares the cache, measures normalization over the full training set,
#    then trains. Re-running resumes from the last checkpoint.
python -m reality train --config reality/configs/weather/kitti_to_cadc.yaml

# 3. Convert a dataset split with trained weights
python -m reality generate --config reality/configs/weather/kitti_to_cadc.yaml \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt --split test

# 4. Score against the target distribution
python -m reality evaluate --config reality/configs/weather/kitti_to_cadc.yaml \
    --checkpoint weights/weather/kitti_to_cadc_gen_r.pt --label my_run
```

Preparation is cached and keyed by the settings that affect the tensors, so it
runs once and every later epoch and run reuses it. Training writes a run directory
containing the checkpoints, the measured statistics, a snapshot of the resolved
config and a log.

A SLURM template for cluster training is in `scripts/train_weather.sh`.

### Pretrained weights

`weights/weather/kitti_to_cadc_gen_r.pt` — KITTI → CADC (snow). The normalization
statistics are stored **inside** the checkpoint, so intensities denormalise
correctly wherever it is used. See [weights/README.md](weights/README.md).

### Bring your own data

Arrange any dataset in the same `train/` + `test/` shape and declare its layout:

```yaml
data_root: data
source:
  dataset: generic
  path: data/MySimulator
  format: npy
  columns: [x, y, z, intensity, physics_intensity]
target:
  dataset: generic
  path: data/MySensor
  format: bin
  columns: [x, y, z, intensity, ring]
  intensity_scale: 255.0
sensor: {proj_H: 64, proj_W: 1024, fov_up: 3.0, fov_down: -25.0}
```

A source that already carries a physics-based intensity should declare a
`physics_intensity` column; the weather stage is then unnecessary. Start from
`reality/configs/sensor/template_sensor.yaml`.

---

## Extending

Every stage is selected by name from a registry, so new components are additions
rather than edits:

| to add | implement | select with |
|---|---|---|
| a dataset | `DatasetAdapter` | `source: {dataset: ...}` |
| a weather model | `GeometricDegradation` | `geometric_degradation: {type: ...}` |
| an intensity model | `IntensityModel` | `model: {type: ...}` |
| a metric | `Evaluator` | — |

See [docs/weather_model.md](docs/weather_model.md) and
[docs/modifying_picgan.md](docs/modifying_picgan.md).

---

## Repository structure

```
reality/
├── cli.py                 entry point for prepare-data / train / generate / evaluate
├── configs/
│   ├── sensor/             sensor-transfer configs (e.g. voxelscape_to_kitti.yaml)
│   └── weather/             weather-transfer configs (e.g. kitti_to_cadc.yaml)
├── core/                   config loading, pipeline orchestration, determinism, registry
├── datasets/               per-dataset adapters (KITTI, nuScenes, CADC, Boreas, VoxelScape, generic)
├── degradation/            geometric degradation plugins (physics weather, learned)
├── preprocessing/          spherical projection, statistics, disk cache
├── models/                 intensity models, incl. the vendored PICGAN/ network and its adapter
├── postprocessing/         back-projection from range image to point cloud
├── inference/               checkpoint-driven generation and ONNX export
├── training/               trainer, dataset wrapper, checkpointing, logging
├── evaluation/             distributional metrics against the target sensor/weather
├── io/                     output writers (native per-source format)
├── physics/                reflectance LUT and ECOSTRESS-derived reflectance lookup
├── structure/weather_model/ vendored LISA scattering model (GPL-3.0, see THIRD_PARTY.md)
└── tests/                  pytest suite mirroring the package layout

data/          per-dataset train/test point clouds (see data/README.md)
weights/       pretrained checkpoints (see weights/README.md)
docs/          extending the weather model and PICGAN
scripts/       SLURM training template, run evaluation helper
tools/         raw-dataset preparation (prepare_data_kitti_cadc.py)
```

---


## The LADS Dataset

LADS provides physically accurate, adverse-weather augmented versions of standard autonomous driving benchmarks. It allows researchers to evaluate 3D object detection and semantic segmentation models under harsh conditions without recording new real-world data.

### Supported Modalities

- **KITTI-Snow & KITTI-Rain:** Generated from the KITTI Odometry dataset (Sequence 00–10). Includes updated `.label` files mapping scattered snow points to SemanticKITTI Class 1 (Noise/Ignore).

- **nuScenes-Snow & nuScenes-Rain:** Generated from the nuScenes Trainval `samples/LIDAR_TOP` directory. Retains the original 5-column structure `(x, y, z, intensity, ring_index)`.

> The LADS dataset is publicly available for download:

**[🔗 Download LADS Dataset](https://www.lidaverse.com/category/?dataset=synthetic)**

---

## Scope and known limitations

- **Intensity adaptation targets the bulk distribution.** Against raw physics
  degradation the model reduces the Wasserstein distance to the real target
  distribution by roughly an order of magnitude, and reproduces its mean closely.
- **Unpaired evaluation.** No paired ground truth exists for a clear-weather scene
  observed under adverse weather, so target comparisons are distributional across
  scenes rather than per-frame.
- **Projection coverage.** At `proj_W = 1024` a substantial share of points share
  pixels and only the nearest is written; raising the projection width increases
  coverage at proportional cost.

---

## 📝 Citation
If you use the ReaLiTy framework or the LADS dataset in your research, please cite our primary paper:

```bibtex

  @article{anand2026sim2real,
    title   = {Toward Closing the Sim-to-Real Gap: A Physics-Guided Learning Approach for LiDAR Intensity Simulation},
    author  = {Anand, Vivek and Lohani, Bharat and Kumar, Vaibhav and Mishra, Rakesh and Pandey, Gaurav},
    journal = {IEEE Transactions on Intelligent Transportation Systems},
    year    = {2026},
    note    = {Early access},
    doi     = {10.1109/TITS.2026.3681982}
  }
  
  @misc{anand2026weather,
    title         = {Simulating Realistic LiDAR Data Under Adverse Weather for Autonomous Vehicles: A Physics-Informed Learning Approach},
    author        = {Anand, Vivek and Lohani, Bharat and Mishra, Rakesh and Pandey, Gaurav},
    year          = {2026},
    eprint        = {2604.01254},
    archivePrefix = {arXiv},
    primaryClass  = {cs.RO},
    note          = {arXiv preprint},
    url           = {https://arxiv.org/abs/2604.01254}
  }
  
  @article{anand2025lblis,
    title   = {Advancing LiDAR Intensity Simulation Through Learning With Novel Physics-Based Modalities},
    author  = {Anand, Vivek and Lohani, Bharat and Pandey, Gaurav and Mishra, Rakesh},
    journal = {IEEE Transactions on Intelligent Transportation Systems},
    year    = {2025},
    volume  = {26},
    number  = {5},
    pages   = {6493--6502},
    doi     = {10.1109/TITS.2025.3532687}
  }
  
  @inproceedings{anand2025snow,
    title     = {Towards Realistic LiDAR Intensity Simulation in Snowy Weather Using Physics-Informed Learning},
    author    = {Anand, Vivek and Lohani, Bharat and Mishra, Rakesh and Pandey, Gaurav},
    booktitle = {IEEE Intelligent Vehicles Symposium (IV)},
    year      = {2025},
    pages     = {2552--2557},
    doi       = {10.1109/IV64158.2025.11097501}
  }
  
  @misc{anand2026reality_lads,
    title         = {ReaLiTy and LADS: A Unified Framework and Dataset Suite for LiDAR Adaptation Across Sensors and Adverse Weather Conditions},
    author        = {Anand, Vivek and others},
    year          = {2026},
    eprint        = {XXXX.XXXXX},
    archivePrefix = {arXiv},
    primaryClass  = {cs.RO},
    note          = {arXiv preprint}
  }

```
---

## 📄 License

This project is released under the **CC BY-NC-SA 4.0** license. It is strictly for academic and non-commercial use. The underlying KITTI and nuScenes data remain subject to their original respective licenses.
