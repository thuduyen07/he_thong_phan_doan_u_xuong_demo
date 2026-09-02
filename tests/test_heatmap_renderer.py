from __future__ import annotations

import unittest

import cv2
import numpy as np

from backend.inference.image_ops import blend_heatmap_rgb, colorize_heatmap


class HeatmapRendererTests(unittest.TestCase):
    def test_colorize_heatmap_matches_fixed_opencv_turbo_scale(self):
        values = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
        expected_bgr = cv2.applyColorMap(
            np.rint(values * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        expected_rgb = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(colorize_heatmap(values), expected_rgb)

    def test_heatmap_blend_matches_runtime_reference_weights(self):
        base = np.full((2, 2, 3), 40, dtype=np.uint8)
        heatmap = np.full((2, 2, 3), [122, 4, 3], dtype=np.uint8)
        expected = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0.0)
        np.testing.assert_array_equal(blend_heatmap_rgb(base, heatmap), expected)


if __name__ == "__main__":
    unittest.main()
