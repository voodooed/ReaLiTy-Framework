import torch
import os

Trial_Num = "T1" #Change
Output_Trial_Num = "Output_1.0"

from pathlib import Path

BASE_DIR = Path("Trial/Output")

Trial_Path = BASE_DIR / f"{Output_Trial_Num}" / f"{Trial_Num}" / "Model"
OUTPUT_FOLDER = BASE_DIR / f"{Output_Trial_Num}" / f"{Trial_Num}" / "Output"

if not os.path.exists(Trial_Path):
    os.makedirs(Trial_Path)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS_R = 1 #Input channel for real Intensity - CADC
IN_CHANNELS_S = 3 #Input channel for simulated data - Range + Incidence + Reflectance

LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10
LAMBDA_Physics = 10
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

SAVE_MODEL = True

LOAD_MODEL = False #Change
START_EPOCH = 0
NUM_EPOCHS = 200 #Change



CHECKPOINT_GEN_S = f"{Trial_Path}/gen_s.pth.tar_{Trial_Num}"
CHECKPOINT_GEN_R = f"{Trial_Path}/gen_r.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_S = f"{Trial_Path}/disc_s.pth.tar_{Trial_Num}"
CHECKPOINT_DISC_R = f"{Trial_Path}/disc_r.pth.tar_{Trial_Num}"

#Input Directory

# Base paths (relative)
base_path_real_adverse = "Data/KITTI/Range_Image/Snow/Train/CADC/"  # Range+Intensity #CADC
base_path_sim_adverse  = "Data/KITTI/Range_Image/Snow/Train/"       # Range+Intensity

# Train directories
TRAIN_Lidar_Real_DIR = base_path_real_adverse + "train_range_image"
TRAIN_Lidar_Sim_Adverse_DIR = base_path_sim_adverse + "train_range_image"

# Validation directories
VAL_Lidar_Real_DIR = base_path_real_adverse + "train_range_image"
VAL_Lidar_Sim_Adverse_DIR = base_path_sim_adverse + "train_range_image"
