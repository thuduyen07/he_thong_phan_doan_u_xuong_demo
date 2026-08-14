from __future__ import annotations

import math
from typing import Any

import numpy as np


def softmax_probabilities(logits: np.ndarray, *, class_axis: int = 0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=class_axis, keepdims=True)
    exp_values = np.exp(shifted)
    denom = np.clip(exp_values.sum(axis=class_axis, keepdims=True), a_min=1e-12, a_max=None)
    return exp_values / denom


def binary_probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    positive = 1.0 / (1.0 + np.exp(-values))
    negative = 1.0 - positive
    return np.concatenate([negative, positive], axis=0)


def predictive_entropy(probabilities: np.ndarray, *, class_axis: int = 0, normalize: bool = True) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    num_classes = probs.shape[class_axis]
    entropy = -np.sum(probs * np.log(np.clip(probs, a_min=1e-12, a_max=1.0)), axis=class_axis)
    if normalize and num_classes > 1:
        entropy = entropy / np.log(float(num_classes))
    return entropy.astype(np.float64)


def max_softmax_confidence(probabilities: np.ndarray, *, class_axis: int = 0) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    return np.max(probs, axis=class_axis).astype(np.float64)


def probability_margin(probabilities: np.ndarray, *, class_axis: int = 0) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    moved = np.moveaxis(probs, class_axis, -1)
    if moved.shape[-1] == 1:
        return np.ones(moved.shape[:-1], dtype=np.float64)
    top2 = np.partition(moved, kth=moved.shape[-1] - 2, axis=-1)[..., -2:]
    top1 = np.max(top2, axis=-1)
    top2_second = np.min(top2, axis=-1)
    return (top1 - top2_second).astype(np.float64)


def one_minus_margin(probabilities: np.ndarray, *, class_axis: int = 0) -> np.ndarray:
    margin = probability_margin(probabilities, class_axis=class_axis)
    return np.clip(1.0 - margin, a_min=0.0, a_max=1.0)


def _neighbor_reduce(mask: np.ndarray, *, mode: str) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant", constant_values=False)
    neighborhoods = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighborhoods.append(
                padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            )
    stacked = np.stack(neighborhoods, axis=0)
    if mode == "any":
        return np.any(stacked, axis=0)
    if mode == "all":
        return np.all(stacked, axis=0)
    raise ValueError(f"Unsupported reduction mode: {mode}")


def binary_dilation(mask: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        result = _neighbor_reduce(result, mode="any")
    return result


def binary_erosion(mask: np.ndarray, *, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        result = _neighbor_reduce(result, mode="all")
    return result


def binary_boundary_band(mask: np.ndarray, *, width: int = 1) -> np.ndarray:
    region = np.asarray(mask, dtype=bool)
    if not region.any():
        return np.zeros_like(region, dtype=bool)
    width = max(1, int(width))
    inner = binary_erosion(region, iterations=1)
    edge = np.logical_xor(region, inner)
    return binary_dilation(edge, iterations=width - 1 if width > 1 else 0)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return None
    return float(selected.mean())


def masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return None
    return float(np.percentile(selected, percentile))


def masked_fraction_below(values: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return None
    return float((selected < float(threshold)).mean())


def conformal_prediction_sets(
    probabilities: np.ndarray,
    *,
    lambda_threshold: float,
    class_axis: int = 0,
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probability_floor = 1.0 - float(lambda_threshold)
    moved = np.moveaxis(probs, class_axis, 0)
    included = moved >= probability_floor
    return np.moveaxis(included, 0, class_axis)


def conformal_quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Conformal calibration scores are empty.")
    ordered = np.sort(values)
    safe_alpha = min(max(float(alpha), 1e-9), 1.0 - 1e-9)
    rank = int(math.ceil((ordered.size + 1) * (1.0 - safe_alpha))) - 1
    rank = min(max(rank, 0), ordered.size - 1)
    return float(ordered[rank])


def build_binary_conformal_region_maps(
    probabilities: np.ndarray,
    *,
    lambda_threshold: float,
    tumor_class_id: int = 1,
    class_axis: int = 0,
) -> dict[str, np.ndarray]:
    prediction_sets = conformal_prediction_sets(
        probabilities,
        lambda_threshold=lambda_threshold,
        class_axis=class_axis,
    )
    included = np.moveaxis(np.asarray(prediction_sets, dtype=bool), class_axis, 0)
    if included.shape[0] < 2:
        raise ValueError("Binary conformal regions require at least two classes.")

    background_included = included[0]
    tumor_included = included[int(tumor_class_id)]
    set_size = included.sum(axis=0).astype(np.int64)

    sure_tumor = np.logical_and(tumor_included, set_size == 1)
    sure_background = np.logical_and(background_included, set_size == 1)
    outer_tumor = tumor_included
    uncertain = np.logical_not(np.logical_or(sure_tumor, sure_background))

    return {
        "set_size": set_size,
        "sure_tumor": sure_tumor.astype(bool),
        "sure_background": sure_background.astype(bool),
        "uncertain": uncertain.astype(bool),
        "outer_tumor": outer_tumor.astype(bool),
    }


def sanitize_float(value: float | None) -> float | None:
    if value is None:
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return float(value)


def summarize_binary_conformal_regions(
    region_maps: dict[str, np.ndarray],
    *,
    alpha: float,
    lambda_threshold: float,
    calibration_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sure_tumor = np.asarray(region_maps["sure_tumor"], dtype=bool)
    sure_background = np.asarray(region_maps["sure_background"], dtype=bool)
    uncertain = np.asarray(region_maps["uncertain"], dtype=bool)
    outer_tumor = np.asarray(region_maps["outer_tumor"], dtype=bool)
    set_size = np.asarray(region_maps["set_size"], dtype=np.int64)
    total_pixels = int(set_size.size)
    safe_total = max(total_pixels, 1)

    return {
        "available": True,
        "alpha": float(alpha),
        "target_coverage": float(1.0 - float(alpha)),
        "lambda_threshold": float(lambda_threshold),
        "probability_floor": float(1.0 - float(lambda_threshold)),
        "calibration_size": int(calibration_size) if calibration_size is not None else None,
        "mean_set_size": sanitize_float(float(set_size.mean()) if total_pixels else None),
        "max_set_size": int(set_size.max()) if total_pixels else None,
        "sure_tumor_pixels": int(sure_tumor.sum()),
        "outer_tumor_pixels": int(outer_tumor.sum()),
        "sure_background_pixels": int(sure_background.sum()),
        "uncertain_pixels": int(uncertain.sum()),
        "uncertain_fraction": sanitize_float(float(uncertain.sum()) / safe_total),
        "sure_tumor_fraction": sanitize_float(float(sure_tumor.sum()) / safe_total),
        "sure_background_fraction": sanitize_float(float(sure_background.sum()) / safe_total),
        "metadata": dict(metadata or {}),
    }
