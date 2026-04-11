import os
import numpy as np
import torch
from tqdm import tqdm

from structure.projection import project_pointcloud
from structure.weather import apply_weather
from structure.backprojection import backproject_intensity

SUPPORTED_EXTENSIONS = [".bin", ".npy"]

class ReaLiTyInference:

    def __init__(self, model, device, config, mode="sensor"):
        assert mode in ["sensor", "weather"], "Mode must be 'sensor' or 'weather'"
        self.model = model
        self.device = device
        self.config = config
        self.mode = mode

    def process_folder(self, input_root, output_root):
        print(f"\n Running ReaLiTy Inference | Mode: {self.mode}")
        
        # 1. Gather all valid files first so tqdm knows the total count
        file_paths = []
        for root, _, files in os.walk(input_root):
            for file in files:
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_paths.append(os.path.join(root, file))

        # 2. Process with a clean progress bar
        for input_path in tqdm(file_paths, desc="Processing"):
            relative_path = os.path.relpath(input_path, input_root)
            output_path = os.path.join(output_root, relative_path)
            
            output_path = os.path.splitext(output_path)[0] + ".bin"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                final_pc = self.process_single_file(input_path)
                if final_pc is not None:
                    final_pc.astype(np.float32).tofile(output_path)
            except Exception as e:
                print(f"\nError processing {input_path}: {str(e)}")

        print("\n Folder processing complete.\n")

    def process_single_file(self, input_path):
        # 1️⃣ Load PC
        pc = self.load_pointcloud(input_path)

        # 2️⃣ Apply Weather (if required)
        if self.mode == "weather":
            pc = apply_weather(pc, self.config)

        # 3️⃣ Projection
        range_image, mapping = project_pointcloud(pc, self.config)
        
        # Guard against empty projections
        if range_image is None or mapping is None:
            return None

        # 4️⃣ Model Inference
        predicted_range = self.run_model(range_image)

        # 5️⃣ Backprojection 
        final_pc = backproject_intensity(pc, predicted_range[3], mapping)

        return final_pc

    def run_model(self, range_image):
        # 1. Extract raw arrays
        raw_range = range_image[0]
        raw_incidence = range_image[1]
        raw_reflectance = range_image[2]

        # 2. Apply Normalization from Config
        r_mean = self.config.get("range_mean", 0.0965)
        r_std = self.config.get("range_std", 0.1068)
        
        i_mean = self.config.get("incidence_mean", 0.7156)
        i_std = self.config.get("incidence_std", 0.6352)
        
        ref_mean = self.config.get("reflectance_mean", 0.2979)
        ref_std = self.config.get("reflectance_std", 0.2743)

        norm_range = (raw_range - r_mean) / r_std
        norm_incidence = (raw_incidence - i_mean) / i_std
        norm_reflectance = (raw_reflectance - ref_mean) / ref_std

        # 3. Convert to Tensors and add Batch/Channel dimensions (1, 1, H, W)
        range_ch = torch.tensor(norm_range).unsqueeze(0).unsqueeze(0)
        incidence_ch = torch.tensor(norm_incidence).unsqueeze(0).unsqueeze(0)
        reflectance_ch = torch.tensor(norm_reflectance).unsqueeze(0).unsqueeze(0)

        # 4. Concatenate along Channel dimension (dim=1) -> Shape: (1, 3, H, W)
        input_tensor = torch.cat(
            [range_ch, incidence_ch, reflectance_ch],
            dim=1
        ).to(self.device).float()

        # 5. Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # 6. Denormalize the Output
        out_mean = self.config.get("intensity_mean", 0.0158)
        out_std = self.config.get("intensity_std", 0.0462)

        output = output * out_std + out_mean
        output_np = output.squeeze().cpu().numpy()
        
        # 7. Clip the 2D output physical bounds
        output_np = np.clip(output_np, 0.0, 1.0)

        predicted_range = range_image.copy()
        predicted_range[3] = output_np

        return predicted_range

    @staticmethod
    def load_pointcloud(path):
        ext = os.path.splitext(path)[1]
        if ext == ".bin":
            pc = np.fromfile(path, dtype=np.float32)
            pc = pc.reshape(-1, 4) if pc.size % 5 != 0 else pc.reshape(-1, 5)
        elif ext == ".npy":
            pc = np.load(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return pc