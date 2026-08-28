"""Intensity-distribution metrics."""

import numpy as np
import pytest

from reality.evaluation import (
    Evaluator,
    IntensityDistributionEvaluator,
    histogram,
    kl_divergence,
    occupied_values,
    ssim,
)


def image(mean=0.3, std=0.1, seed=0, shape=(32, 64), fill=0.6):
    """An intensity image with a realistic share of empty (zero) pixels."""
    rng = np.random.default_rng(seed)
    values = np.clip(rng.normal(mean, std, shape), 0, 1)
    occupied = rng.random(shape) < fill
    return (values * occupied).astype(np.float32)


def test_occupied_pixels_only():
    plane = np.array([[0.0, 0.5], [0.25, 0.0]])
    assert sorted(occupied_values(plane).tolist()) == [0.25, 0.5]
    mask = np.array([[True, True], [False, False]])
    assert sorted(occupied_values(plane, mask).tolist()) == [0.0, 0.5]


def test_histogram_is_normalised():
    values = np.random.default_rng(0).random(1000)
    counts = histogram(values)
    assert counts.shape == (100,)
    assert counts.sum() == pytest.approx(1.0)


def test_kl_is_zero_for_identical_distributions():
    values = np.random.default_rng(1).random(5000)
    p = histogram(values)
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_grows_with_separation():
    rng = np.random.default_rng(2)
    base = histogram(np.clip(rng.normal(0.3, 0.05, 20000), 0, 1))
    near = histogram(np.clip(rng.normal(0.35, 0.05, 20000), 0, 1))
    far = histogram(np.clip(rng.normal(0.7, 0.05, 20000), 0, 1))
    assert kl_divergence(base, near) < kl_divergence(base, far)


def test_kl_is_asymmetric():
    rng = np.random.default_rng(3)
    p = histogram(np.clip(rng.normal(0.3, 0.05, 10000), 0, 1))
    q = histogram(np.clip(rng.normal(0.4, 0.15, 10000), 0, 1))
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_ssim_is_one_for_identical_images():
    plane = image(seed=4)
    assert ssim(plane, plane) == pytest.approx(1.0, abs=1e-6)


def test_ssim_falls_with_corruption():
    plane = image(seed=5)
    rng = np.random.default_rng(6)
    slight = plane + rng.normal(0, 0.01, plane.shape)
    heavy = plane + rng.normal(0, 0.3, plane.shape)
    assert ssim(plane, slight) > ssim(plane, heavy)
    assert ssim(plane, heavy) < 1.0


def test_ssim_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        ssim(np.zeros((4, 4)), np.zeros((4, 5)))


def test_ssim_honours_a_mask():
    a, b = image(seed=7), image(seed=8)
    mask = np.zeros(a.shape, dtype=bool)
    mask[:8, :8] = True
    assert ssim(a, b, mask=mask) != ssim(a, b)


def test_unpaired_evaluation_reports_the_metric_set():
    evaluator = IntensityDistributionEvaluator()
    generated = [image(0.3, 0.1, seed=i) for i in range(4)]
    reference = [image(0.3, 0.1, seed=100 + i) for i in range(4)]
    result = evaluator.evaluate(generated, reference)
    for key in ("histogram_mse", "kl_divergence", "wasserstein",
                "generated_mean", "reference_mean"):
        assert key in result
    assert result["wasserstein"] < 0.05, "same distribution should be close"
    assert result["n_generated_points"] > 0


def test_unpaired_metrics_separate_different_distributions():
    evaluator = IntensityDistributionEvaluator()
    reference = [image(0.3, 0.08, seed=200 + i) for i in range(4)]
    close = evaluator.evaluate([image(0.3, 0.08, seed=i) for i in range(4)], reference)
    far = evaluator.evaluate([image(0.7, 0.08, seed=i) for i in range(4)], reference)
    assert far["wasserstein"] > close["wasserstein"]
    assert far["kl_divergence"] > close["kl_divergence"]
    assert far["histogram_mse"] > close["histogram_mse"]


def test_empty_input_is_reported():
    evaluator = IntensityDistributionEvaluator()
    with pytest.raises(ValueError, match="no occupied pixels"):
        evaluator.evaluate([np.zeros((8, 8))], [np.zeros((8, 8))])


def test_paired_evaluation_against_a_baseline():
    evaluator = IntensityDistributionEvaluator()
    baseline = [image(seed=i) for i in range(3)]
    identical = evaluator.evaluate_paired(baseline, baseline)
    assert identical["mse"] == pytest.approx(0.0, abs=1e-12)
    assert identical["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert identical["n_frames"] == 3

    rng = np.random.default_rng(9)
    changed = [plane + rng.normal(0, 0.1, plane.shape) for plane in baseline]
    altered = evaluator.evaluate_paired(changed, baseline)
    assert altered["mse"] > 0
    assert altered["ssim"] < identical["ssim"]


def test_paired_evaluation_requires_equal_counts():
    evaluator = IntensityDistributionEvaluator()
    with pytest.raises(ValueError, match="equal counts"):
        evaluator.evaluate_paired([image()], [image(), image()])


def test_evaluator_interface():
    assert issubclass(IntensityDistributionEvaluator, Evaluator)
    assert IntensityDistributionEvaluator().name == "intensity_distribution"


# --------------------------------------------------------------------------- #
# Regression: the pooled-vs-occupied mistake
# --------------------------------------------------------------------------- #


def dense_and_sparse(shape=(32, 64), seed=0):
    """A generator-style dense image and a physics-style sparse one, same mask."""
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < 0.42          # 42% occupied, like a real projection
    dense = rng.normal(0.05, 0.03, shape)    # a value on EVERY pixel
    sparse = np.where(mask, rng.normal(0.05, 0.03, shape), 0.0)
    return dense, sparse, mask


def test_masks_are_honoured_and_counted():
    """With masks, both sides contribute exactly their occupied pixels."""
    dense, sparse, mask = dense_and_sparse()
    result = IntensityDistributionEvaluator().evaluate(
        [dense], [sparse], generated_masks=[mask], reference_masks=[mask])
    assert result["n_generated_points"] == int(mask.sum())
    assert result["n_reference_points"] == int(mask.sum())


def test_unmasked_dense_input_pools_every_pixel():
    """The failure mode itself: without masks a dense image contributes H*W.

    an earlier first evaluation pass did this, diluting the generated distribution
    with ~58% non-returns while physics and real contributed occupied pixels only.
    """
    dense, sparse, mask = dense_and_sparse()
    unmasked = IntensityDistributionEvaluator().evaluate([dense], [sparse])
    assert unmasked["n_generated_points"] == dense.size, "dense pools all pixels"
    assert unmasked["n_reference_points"] == int(mask.sum())
    assert unmasked["n_generated_points"] > 2 * unmasked["n_reference_points"], (
        "this imbalance is the bug signature: counts must match when masks are used"
    )


def test_masking_changes_the_metrics_materially():
    """Masking is not cosmetic: it moves the numbers, so it must not be forgotten."""
    dense, sparse, mask = dense_and_sparse(seed=3)
    evaluator = IntensityDistributionEvaluator()
    wrong = evaluator.evaluate([dense], [sparse])
    right = evaluator.evaluate([dense], [sparse],
                               generated_masks=[mask], reference_masks=[mask])
    assert wrong["wasserstein"] != pytest.approx(right["wasserstein"], abs=1e-6)
    assert right["wasserstein"] < wrong["wasserstein"], (
        "like-for-like comparison of the same distribution should score better"
    )


def test_mask_count_must_match_image_count():
    dense, sparse, mask = dense_and_sparse()
    with pytest.raises(ValueError, match="one per image"):
        IntensityDistributionEvaluator().evaluate(
            [dense, dense], [sparse], generated_masks=[mask])
