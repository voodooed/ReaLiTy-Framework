# ReaLiTy: A Realistic LiDAR Transformation Framework & LADS Dataset
## Sim2Real Adaptation for Realistic LiDAR Sensor and Weather Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Tested-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**ReaLiTy** (Realistic LiDAR Transformation) is a unified, domain-adaptive framework designed for the physically and statistically consistent transformation of LiDAR point clouds. 

Built around a conditional generative adversarial architecture (PICGAN) and physics-informed atmospheric modeling, ReaLiTy bridges the sim-to-real gap for autonomous vehicle perception systems. It enables robust adaptation across heterogeneous sensors and adverse weather conditions (such as rain and snow) without requiring full data recollection.

Using this framework, we introduce **LADS** (LiDAR Adaptation Dataset Suite), a large-scale benchmark that injects physically accurate adverse weather (snow and rain) into standard clear-weather autonomous driving datasets like KITTI and nuScenes.

---

## 🔹 Core Capabilities

* **Sensor-to-Sensor Adaptation:** Transform synthetic or source-domain point clouds to emulate the intensity, beam divergence, and noise profiles of target LiDAR hardware.
* **Physics-Informed Weather Adaptation:** Seamlessly integrate the LISA atmospheric model to simulate the geometric and intensity effects of adverse weather (fog, snow, rain) on clean point clouds.
* **Sim-to-Real Intensity Bridging:** Reduce the domain gap by learning target-domain intensity distributions conditioned on local geometry, incidence angles, and acquisition context.
* **Fast Vectorized Backprojection:** Efficiently map 2D predicted intensity tensors back to 3D spherical point clouds using an optimized, loop-free projection module.

## 📂 Framework Structure

```plaintext
ReaLiTy/
│
├── ReaLiTy.py
│
├── models/
│     ├── PICGAN/
│     
│
├── prepare_training_data.py
│
├── structure/
│     ├── projection.py
│     ├── weather.py
│     ├── backprojection.py
│
├── data/
│     └── prepare_training_data.py
│
├── training/
│     └── train_picgan.py
│
├── transform/
│     └── transform.py
│
├── weights/
│     ├── sensor/
│     └── weather/
│
├── configs/
│     ├── sensor.yaml
│     └── weather.yaml
│
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/voodooed/ReaLiTy-Framework.git
cd ReaLiTy_Framework
```

### 2. Create a Virtual Environment

We recommend using Conda to manage dependencies.

```bash
conda create -n reality python=3.9
conda activate reality
```

### 3. Install Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
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

## 💻 Usage Guide

ReaLiTy uses a single entry point (reality.py) driven by a master config.yaml.

### 0. Configuration (config.yaml)
Define your LiDAR sensor parameters, semantic reflectance mappings, and weather conditions:

YAML
mode: "weather"             # "sensor" or "weather"
experiment_name: "T1"
fov_up: 2.0
fov_down: -24.9
width: 1024
height: 64
intensity_mean: 0.5
intensity_std: 0.2
atm_model: "snow"
precipitation_rate: 10.0

### 1. Run Inference / Transformation

Process a directory of raw KITTI .bin files into realism-consistent, weather-adapted point clouds:

Bash
python reality.py \
  --mode transform \
  --config config.yaml \
  --picgan_root /path/to/PICGAN \
  --weights weights/weather/kitti_clear2snow.pth.tar \
  --input /path/to/raw/dataset \
  --output /path/to/transformed/dataset
  
### 2. Train on a New Sensor/Weather Target
   
⚠️ **Note:** The training pipeline is currently being finalized. While the script outlines the intended workflow, some components (e.g., data preprocessing and configuration handling) are subject to updates. A stable and fully reproducible version will be released soon.

```bash
python training/train_picgan.py \
  --mode train \
  --data_dir /path/to/training/tensors \
  --config config/sensor.yaml \
  --epochs 100 \
  --batch_size 4
```

### 3. Visualizing Results

We highly recommend using **Open3D** to visualize the 3D point clouds. When visualizing nuScenes, remember to isolate the first 4 columns, as the 5th column contains the beam ring index.

---

## ⚙️ Configuration (`config/`)

The YAML configuration files control the strict geometric parameters of the LiDAR sensors and the normalization statistics for the neural network.

**Example `weather.yaml` parameters:**

```yaml
# Sensor Geometry (KITTI HDL-64E)
proj_H: 64
proj_W: 2048
proj_fov_up: 3.0
proj_fov_down: -25.0

# Neural Normalization (Required for accurate Intensity prediction)
range_mean: 0.0965
range_std: 0.1068
incidence_mean: 0.7156
incidence_std: 0.6352

# Output Denormalization (Target Domain stats)
intensity_mean: 0.0158
intensity_std: 0.0462
```

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
