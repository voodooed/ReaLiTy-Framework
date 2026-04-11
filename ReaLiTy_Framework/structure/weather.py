import numpy as np
from .LISA.atmos_models import LISA

# Global cache to hold the model in memory
_LISA_MODEL = None
_CURRENT_WEATHER = None

def apply_weather(pc, config):
    global _LISA_MODEL, _CURRENT_WEATHER
    
    atm_model = config["atm_model"]
    precipitation_rate = config["precipitation_rate"]

    # Only initialize LISA once.
    if _LISA_MODEL is None or _CURRENT_WEATHER != atm_model:
        print(f"\n[Physics] Initializing LISA engine for {atm_model.upper()}... (This only happens once!)")
        _LISA_MODEL = LISA(atm_model=atm_model)
        _CURRENT_WEATHER = atm_model

    # Separate geometry + intensity
    xyz_intensity = pc[:, :4]
    original_labels = pc[:, 4:5]

    # Apply LISA using the cached model
    # pc_aug shape is (N, 5) -> x, y, z, ref_new, lisa_flag
    pc_aug = _LISA_MODEL.augment(xyz_intensity, precipitation_rate)

    # Extract the physics flags from LISA
    lisa_flags = pc_aug[:, 4:5]

    # Create a fresh copy of the original labels to modify
    new_labels = np.copy(original_labels)

    # 1. Re-assign scattered weather points to SemanticKITTI Class 1 (Outlier/Noise)
    scatter_mask = (lisa_flags == 1.0)
    new_labels[scatter_mask] = 1 

    # 2. Re-assign completely lost points to Class 0 (Unlabeled)
    lost_mask = (lisa_flags == 0.0)
    new_labels[lost_mask] = 0 

    # Reattach the correctly mapped labels to the augmented geometry
    pc_final = np.concatenate([pc_aug[:, :4], new_labels], axis=1)

    return pc_final