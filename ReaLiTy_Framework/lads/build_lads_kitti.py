import os
import sys
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from inference.transform import ReaLiTyInference

def load_generator(weights_path, device, picgan_root):
    if picgan_root not in sys.path:
        sys.path.insert(0, picgan_root)
    
    from model.PICGAN.generator import Generator 
    
    model = Generator(img_channels=3, out_channels=1, num_residuals=9)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def process_kitti_sequence(seq_dir, out_seq_dir, inference, device):
    velo_dir = seq_dir / "velodyne"
    label_dir = seq_dir / "labels"
    
    out_velo_dir = out_seq_dir / "velodyne"
    out_label_dir = out_seq_dir / "labels"
    
    out_velo_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    
    bin_files = sorted([f for f in velo_dir.iterdir() if f.suffix == '.bin'])
    
    for bin_path in tqdm(bin_files, desc=f"Processing Seq {seq_dir.name}"):
        out_bin_path = out_velo_dir / bin_path.name
        label_path = label_dir / (bin_path.stem + ".label")
        out_label_path = out_label_dir / label_path.name
        
        # SKIP LOGIC
        if out_bin_path.exists() and out_label_path.exists():
            continue

        pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        
        # 1. Load full 32-bit labels to preserve Instance IDs
        full_labels = None
        if label_path.exists():
            full_labels = np.fromfile(label_path, dtype=np.uint32)
            semantic_labels = full_labels & 0xFFFF
            pc = np.hstack((pc, semantic_labels[:, np.newaxis])).astype(np.float32)
        else:
            pc = np.hstack((pc, np.zeros((pc.shape[0], 1)))).astype(np.float32)

        temp_input_path = out_velo_dir / f"_temp_{bin_path.name}"
        pc.tofile(temp_input_path)

        try:
            final_pc = inference.process_single_file(str(temp_input_path))
            
            if final_pc is not None:
                # 2. Save Geometry (.bin)
                final_pc_4col = final_pc[:, :4].astype(np.float32)
                final_pc_4col[:, 3] = np.clip(final_pc_4col[:, 3], 0.0, 1.0)
                final_pc_4col.tofile(out_bin_path)
                
                # 3. Save Updated Labels (.label)
                if full_labels is not None:
                    # Get the new semantic labels outputted by weather.py
                    updated_semantics = final_pc[:, 4].astype(np.uint32)
                    
                    # Find points altered by weather (1 = snow/noise, 0 = lost)
                    altered_mask = (updated_semantics == 1) | (updated_semantics == 0)
                    
                    final_labels = full_labels.copy()
                    
                    # Overwrite ONLY the scattered points with the new semantic class
                    # This drops the instance ID for snow, but keeps it for original objects!
                    final_labels[altered_mask] = updated_semantics[altered_mask]
                    
                    final_labels.tofile(out_label_path)

        except Exception as e:
            print(f"Error processing {bin_path.name}: {e}")
        finally:
            if temp_input_path.exists():
                temp_input_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="LADS Dataset Builder - KITTI")
    parser.add_argument("--kitti_root", type=str, required=True, help="Path to original KITTI sequences")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the LADS dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--picgan_root", type=str, required=True, help="Path to PICGAN repo")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--weather_mode", type=str, choices=["rain", "snow"], required=True)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting LADS KITTI Pipeline | Mode: {args.weather_mode.upper()}")
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "weather"
    cfg["atm_model"] = args.weather_mode

    model = load_generator(args.weights, device, args.picgan_root)
    inference = ReaLiTyInference(model=model, device=device, config=cfg, mode="weather")

    dataset_path = Path(args.kitti_root)
    sequence_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()])

    if not sequence_dirs:
        print(f"Error: No sequence directories found in {dataset_path}")
        sys.exit(1)

    for seq_dir in sequence_dirs:
        print(f"\n{'='*40}\nProcessing Sequence {seq_dir.name}\n{'='*40}")
        out_seq_dir = Path(args.output_dir) / args.weather_mode / "sequences" / seq_dir.name
        process_kitti_sequence(seq_dir, out_seq_dir, inference, device)

    print(f"\nLADS {args.weather_mode.upper()} Pipeline complete. Ready for distribution.")

if __name__ == "__main__":
    main()