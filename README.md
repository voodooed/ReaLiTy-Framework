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
├── ReaLiTy.py                  # Main orchestrator for the framework
├── build_lads_kitti.py         # Automated LADS generator for KITTI (64-beam, 3-channel)
│
├── models/
│   └── PICGAN/                 # Physics-Informed Conditional GAN architecture
│
├── structure/                  # Core Geometry & Physics Engine
│   ├── projection.py           # 3D Point Cloud -> 2D Spherical Range Image
│   ├── weather.py              # LISA Atmospheric scattering engine integration
│   └── backprojection.py       # 2D Prediction -> 3D Point Cloud mapping
│
├── data/
│   └── prepare_training_data.py # Data normalization and tensor preparation
│
├── training/
│   └── train_picgan.py         # Training loop for the generative models
│
├── transform/
│   └── transform.py            # End-to-end inference pipeline (Load -> Project -> Infer -> Backproject)
│
├── weights/                    # Pretrained Model Checkpoints
│   ├── sensor/                 # Weights for Sim-to-Real sensor translation
│   └── weather/                # Weights for Clear-to-Weather translation
│
├── configs/                    # Hyperparameters and Normalization Statistics
│   ├── sensor.yaml
│   └── weather.yaml
│
└── README.md
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/voodooed/ReaLiTy-Framework.git
cd ReaLiTy
```

### 2. Create a Virtual Environment

We recommend using Conda to manage dependencies.

```bash
conda create -n reality python=3.9
conda activate reality
```

### 3. Install Dependencies

> **Note:** Due to PyTorch compatibility, ensure you are using NumPy 1.x.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open3d matplotlib tqdm pyyaml
pip install "numpy<2"
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

Run Inference / Transformation

Process a directory of raw KITTI .bin files into realism-consistent, weather-adapted point clouds:

Bash
python reality.py \
  --mode transform \
  --config config.yaml \
  --picgan_root /path/to/PICGAN \
  --weights weights/weather/kitti_clear2snow.pth.tar \
  --input /path/to/raw/dataset \
  --output /path/to/transformed/dataset
3. Train on a New Sensor Target
Wrap your existing PiCGAN training logic to automatically manage checkpoints and output paths based on your configuration:

Bash
python reality.py \
  --mode train \
  --config config.yaml \
  --picgan_root /path/to/PICGAN \
  --exp_name Custom_Sensor_V1

### 1. Using the trained models (Inference)

To generate your own weather-augmented data from raw clear-weather KITTI files, use the provided dataset builder scripts. Ensure your `configs/config.yaml` contains the correct normalization statistics from your training run.

**For KITTI:**

```bash
python build_lads_kitti.py \
  --kitti_root /path/to/kitti/dataset/sequences \
  --output_dir /path/to/output/LADS/KITTI \
  --config config/config.yaml \
  --picgan_root models/PICGAN \
  --weights weights/weather/kitti_clear2snow.pth.tar_T1 \
  --weather_mode snow
```

### 2. Training the PiCGAN Model

To train the model on a new sensor configuration or specific weather intensity, prepare your normalized 2D tensors and run the training script:

```bash
python training/train_picgan.py \
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
@misc{anand2026reality,
  title={ReaLiTy and LADS: A Unified Framework and Dataset Suite for LiDAR Adaptation Across Sensors and Adverse Weather Conditions},
  author={Vivek Anand and Bharat Lohani and Rakesh Mishra and Gaurav Pandey},
  year={2026},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
```bibtex
If your work builds upon the underlying physics-informed intensity simulation, please also consider citing our foundational works:
@ARTICLE{Vivek_Advancing,
  author={Anand, Vivek and Lohani, Bharat and Pandey, Gaurav and Mishra, Rakesh},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Advancing LiDAR Intensity Simulation Through Learning With Novel Physics-Based Modalities}, 
  year={2025},
  volume={26},
  number={5},
  pages={6493-6502},
  doi={10.1109/TITS.2025.3532687}
}

@INPROCEEDINGS{anand_snow_iv,
  author={Anand, Vivek and Lohani, Bharat and Mishra, Rakesh and Pandey, Gaurav},
  booktitle={2025 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Towards Realistic LiDAR Intensity Simulation in Snowy Weather Using Physics-Informed Learning}, 
  year={2025},
  pages={2552-2557},
  doi={10.1109/IV64158.2025.11097501}
}

@article{anand2024toward,
  title={Toward Physics-Aware Deep Learning Architectures for LiDAR Intensity Simulation},
  author={Anand, Vivek and Lohani, Bharat and Pandey, Gaurav and Mishra, Rakesh},
  journal={arXiv preprint arXiv:2404.15774},
  year={2024}
}

```
---

## 📄 License

This project is released under the **CC BY-NC-SA 4.0** license. It is strictly for academic and non-commercial use. The underlying KITTI and nuScenes data remain subject to their original respective licenses.
