# ReaLITy Dataset and Framework
### Sim2Real Dataset- Sensor and Weather Specific Adaptation for Realistic LiDAR Simulation

ReaLITy-Framework/
│── configs/
│   ├── sensor_transfer.yaml
│   ├── weather_transfer.yaml
│
│── data/
│   ├── preprocessing/        # range image generation
│   ├── augmentations/        # 3D geometric ops for weather
│
│── models/
│   ├── cyclegan_base.py      # shared physics-informed core
│   ├── losses.py             # custom + physics-aware losses
│
│── pipelines/
│   ├── sensor_pipeline.py    # sensor mode wrapper
│   ├── weather_pipeline.py   # weather mode wrapper
│
│── scripts/
│   ├── train.py              # unified CLI for training
│   ├── inference.py          # unified CLI for inference
│
│── weights/
│   ├── sensor/
│   ├── weather/
│
│── examples/
│   ├── demo_sensor_transfer.ipynb
│   ├── demo_weather_transfer.ipynb
│
│── docs/
│   ├── README.md
│   ├── tutorials.md
│   ├── API.md
