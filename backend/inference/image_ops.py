from __future__ import annotations

import numpy as np
from PIL import Image


def resize_grayscale(image: np.ndarray, size: int) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
    resized = pil_image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def blend_rgb(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    base_arr = np.asarray(base, dtype=np.float32)
    overlay_arr = np.asarray(overlay, dtype=np.float32)
    mixed = ((1.0 - float(alpha)) * base_arr) + (float(alpha) * overlay_arr)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def save_rgb_png(path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path, format="PNG")


def save_grayscale_png(path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(path, format="PNG")


def colorize_heatmap(values: np.ndarray) -> np.ndarray:
    """Use a sequential display palette; values remain unchanged in the saved artifact."""
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    low = np.array([23.0, 42.0, 79.0], dtype=np.float32)
    high = np.array([249.0, 203.0, 82.0], dtype=np.float32)
    rgb = low + arr[..., None] * (high - low)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
