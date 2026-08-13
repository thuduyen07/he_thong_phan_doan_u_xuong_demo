from __future__ import annotations

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
