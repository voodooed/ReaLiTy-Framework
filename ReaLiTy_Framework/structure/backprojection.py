import numpy as np


def backproject_intensity(pc, predicted_intensity, mapping):
    """
    pc: (N, 4 or 5)
    predicted_intensity: (1, H, W) or (H, W)
    mapping: dict with 'u', 'v'

    Returns:
        new_pc with updated intensity
    """

    if predicted_intensity.ndim == 3:
        predicted_intensity = predicted_intensity[0]

    u = mapping["u"]
    v = mapping["v"]

    H, W = predicted_intensity.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    new_pc = pc.copy()
    new_pc[:, 3] = predicted_intensity[v, u]
    new_pc[:, 3] = np.nan_to_num(new_pc[:, 3], nan=0.0)

    return new_pc