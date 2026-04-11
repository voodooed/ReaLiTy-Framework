import os
import sys
import yaml
import argparse
import torch

# Framework modules
from training.train_picgan import PICGANTrainer
from inference.transform import ReaLiTyInference  

# ==========================================
# Model Loading
# ==========================================

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

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="ReaLiTy Unified Framework")
    parser.add_argument("--mode", type=str, choices=["train", "transform"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--picgan_root", type=str, required=True)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--exp_name", type=str, default="T1")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        cfg["picgan_root"] = args.picgan_root
        cfg["experiment_name"] = args.exp_name
        trainer = PICGANTrainer(cfg)
        trainer.run()
        
    elif args.mode == "transform":
        #  Hand over control to transform.py
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_generator(args.weights, device, args.picgan_root)
        
        # Instantiate the robust class from transform.py
        inference = ReaLiTyInference(
            model=model, 
            device=device, 
            config=cfg, 
            mode=cfg.get("mode", "sensor")
        )
        
        # Run the robust folder processing
        inference.process_folder(args.input, args.output)

if __name__ == "__main__":
    main()