"""Intensity-distribution metrics: MSE, SSIM, KL and Wasserstein.

Two comparisons, because they answer different questions and only one of them
can be paired:

* **Unpaired, against real CADC** — there is no ground truth for "this KITTI
  frame in snow", so the honest comparison is distributional: does the generated
  intensity look like real snow-condition intensity in aggregate? KL, Wasserstein
  and a histogram MSE over pooled occupied-pixel intensities.
* **Paired, against the physics baseline** — the same frame's the weather model ``phy`` is
  available pixel for pixel, so MSE and SSIM there measure what PICGAN changed
  relative to raw physics degradation.

Only occupied pixels are compared. Empty pixels of a range image are absent
returns, and their share depends on projection width, so including them would
measure the projection rather than the intensities.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import wasserstein_distance

from reality.evaluation.base import Evaluator

#: Histogram resolution for the distributional metrics.
DEFAULT_BINS = 100
#: Guards a log of zero in the KL divergence.
EPS = 1e-10


def _pool(images: Iterable[np.ndarray],
          masks: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """Concatenate the occupied values of a set of images."""
    images = list(images)
    if masks is None:
        return np.concatenate([occupied_values(image) for image in images])
    if len(masks) != len(images):
        raise ValueError(
            f"got {len(masks)} masks for {len(images)} images; pass one per image"
        )
    return np.concatenate([occupied_values(image, mask)
                           for image, mask in zip(images, masks)])


def occupied_values(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Flatten the pixels that carry a return."""
    image = np.asarray(image, dtype=np.float64)
    if mask is None:
        return image[np.isfinite(image) & (image != 0.0)]
    return image[np.asarray(mask, dtype=bool) & np.isfinite(image)]


def histogram(values: np.ndarray, bins: int = DEFAULT_BINS,
              value_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Normalised histogram over a fixed range, so two sets are comparable."""
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) over histograms, with both smoothed away from zero."""
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None,
         window: int = 7, data_range: float = 1.0) -> float:
    """Mean structural similarity between two images (Wang et al. 2004).

    Local statistics come from a uniform filter, as in the reference
    implementation. When a mask is given, the score is averaged over the masked
    pixels only.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"SSIM needs matching shapes, got {a.shape} and {b.shape}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mean_a = uniform_filter(a, window)
    mean_b = uniform_filter(b, window)
    mean_aa = uniform_filter(a * a, window)
    mean_bb = uniform_filter(b * b, window)
    mean_ab = uniform_filter(a * b, window)

    var_a = mean_aa - mean_a * mean_a
    var_b = mean_bb - mean_b * mean_b
    covar = mean_ab - mean_a * mean_b

    numerator = (2 * mean_a * mean_b + c1) * (2 * covar + c2)
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2)
    similarity = numerator / np.maximum(denominator, EPS)

    if mask is None:
        return float(similarity.mean())
    selected = np.asarray(mask, dtype=bool)
    return float(similarity[selected].mean()) if selected.any() else float("nan")


class IntensityDistributionEvaluator(Evaluator):
    """MSE, SSIM, KL and Wasserstein over intensity images."""

    name = "intensity_distribution"

    def __init__(self, bins: int = DEFAULT_BINS,
                 value_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.bins = bins
        self.value_range = value_range

    # -- unpaired ------------------------------------------------------------- #

    def evaluate(self, generated: Iterable[np.ndarray],
                 reference: Iterable[np.ndarray],
                 generated_masks: Optional[Sequence[np.ndarray]] = None,
                 reference_masks: Optional[Sequence[np.ndarray]] = None
                 ) -> Dict[str, float]:
        """Distributional comparison of two unpaired sets of intensity images.

        **Pass the masks.** A generator emits a value on every pixel, including
        those no point landed on, while physics and real images are zero there.
        Without masks the non-zero heuristic then pools ~58% non-returns into the
        generated set only, and the two sides stop being comparable.  made
        exactly that mistake; ``n_generated_points`` in the result is the check
        that catches it -- it should match the occupied count, not H*W*frames.
        """
        generated_values = _pool(generated, generated_masks)
        reference_values = _pool(reference, reference_masks)
        if generated_values.size == 0 or reference_values.size == 0:
            raise ValueError("no occupied pixels to compare")

        p = histogram(generated_values, self.bins, self.value_range)
        q = histogram(reference_values, self.bins, self.value_range)
        return {
            "histogram_mse": float(np.mean((p - q) ** 2)),
            "kl_divergence": kl_divergence(p, q),
            "kl_divergence_reverse": kl_divergence(q, p),
            "wasserstein": float(wasserstein_distance(generated_values, reference_values)),
            "generated_mean": float(generated_values.mean()),
            "reference_mean": float(reference_values.mean()),
            "generated_std": float(generated_values.std()),
            "reference_std": float(reference_values.std()),
            "n_generated_points": int(generated_values.size),
            "n_reference_points": int(reference_values.size),
        }

    # -- paired ---------------------------------------------------------------- #

    def evaluate_paired(self, generated: Sequence[np.ndarray],
                        baseline: Sequence[np.ndarray],
                        masks: Optional[Sequence[np.ndarray]] = None) -> Dict[str, float]:
        """Per-frame comparison against a baseline of the same frames.

        Used against the physics ``phy`` image, which exists pixel for pixel, to
        show what the model changed relative to raw physics degradation.
        """
        if len(generated) != len(baseline):
            raise ValueError(
                f"paired comparison needs equal counts, got {len(generated)} and "
                f"{len(baseline)}"
            )
        errors: List[float] = []
        similarities: List[float] = []
        for index, (image, other) in enumerate(zip(generated, baseline)):
            mask = None if masks is None else np.asarray(masks[index], dtype=bool)
            selected = mask if mask is not None else np.isfinite(image) & (other != 0.0)
            if selected.any():
                errors.append(float(np.mean((image[selected] - other[selected]) ** 2)))
            similarities.append(ssim(image, other, mask=selected))
        return {
            "mse": float(np.mean(errors)) if errors else float("nan"),
            "ssim": float(np.nanmean(similarities)) if similarities else float("nan"),
            "n_frames": len(generated),
        }
