import numpy as np
from .LISA.atmos_models import LISA


def apply_weather(pc, config):
    """
    pc: numpy array (N, 5) → x,y,z,intensity,label
    config: dict with:
        - atm_model: 'snow' or 'rain'
        - precipitation_rate: float (mm/hr)
    """

    atm_model = config["atm_model"]
    precipitation_rate = config["precipitation_rate"]

    lisa = LISA(atm_model=atm_model)

    # Separate geometry + intensity
    xyz_intensity = pc[:, :4]
    semantic_labels = pc[:, 4:5]

    # Apply LISA
    pc_aug = lisa.augment(xyz_intensity, precipitation_rate)

    # Drop LISA-generated label
    pc_aug = pc_aug[:, :4]

    # Reattach original labels
    pc_final = np.concatenate([pc_aug, semantic_labels], axis=1)

    return pc_final