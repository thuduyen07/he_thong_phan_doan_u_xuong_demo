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
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 * arr, 0.0, 1.0)
    green = np.clip(1.5 - np.abs((arr * 2.0) - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - arr), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
