from __future__ import annotations

import cv2
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
    """Render the fixed [0, 1] Turbo colormap used by the runtime reference."""
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    heat = np.rint(clipped * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def blend_heatmap_rgb(base: np.ndarray, heatmap_rgb: np.ndarray) -> np.ndarray:
    """Match the historic runtime heatmap blend: 45% X-ray, 55% Turbo map."""
    return cv2.addWeighted(
        np.asarray(base, dtype=np.uint8),
        0.45,
        np.asarray(heatmap_rgb, dtype=np.uint8),
        0.55,
        0.0,
    )
