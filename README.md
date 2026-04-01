# ReaLiTy Framework & LADS Dataset
## Sim2Real Adaptation for Realistic LiDAR Sensor and Weather Simulation

**ReaLiTy** (Realistic LiDAR Transformatiob) is a hybrid, physics-guided and learning-based framework that bridges the domain gap between simulated and real-world LiDAR point clouds.

Using this framework, we introduce **LADS** (LiDAR Adaptation Dataset Suite), a large-scale benchmark that injects physically accurate adverse weather (snow and rain) into standard clear-weather autonomous driving datasets like KITTI and nuScenes.

---

## 📂 Project Structure

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
  --config configs/sensor.yaml \
  --epochs 100 \
  --batch_size 4
```

### 3. Visualizing Results

We highly recommend using **Open3D** to visualize the 3D point clouds. When visualizing nuScenes, remember to isolate the first 4 columns, as the 5th column contains the beam ring index.

---

## ⚙️ Configuration (`configs/`)

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
