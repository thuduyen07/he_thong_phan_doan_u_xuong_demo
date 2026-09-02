from __future__ import annotations

import os
import unittest

import numpy as np
from PIL import Image

os.environ["DEMO_MODE"] = "static"

from backend.api import app
from backend.static_samples import SAMPLE_DIR, get_static_result, load_static_samples


class StaticHeatmapContractTests(unittest.TestCase):
    def setUp(self) -> None:
        load_static_samples.cache_clear()
        self.entries = load_static_samples()

    def test_all_curated_samples_declare_colored_heatmaps(self):
        self.assertGreaterEqual(len(self.entries), 10)
        self.assertEqual({entry.get("dataset") for entry in self.entries}, {"btxrd", "fracatlas"})
        for entry in self.entries:
            artifacts = entry.get("artifacts") or {}
            for key in ("probability_heatmap", "entropy_heatmap"):
                relative_path = artifacts.get(key)
                self.assertIsInstance(relative_path, str, msg=f"{entry['id']} missing {key}")
                image_path = SAMPLE_DIR / relative_path
                self.assertTrue(image_path.is_file(), msg=f"{entry['id']} missing {image_path}")
                with Image.open(image_path) as image:
                    self.assertIn(image.mode, {"RGB", "RGBA"}, msg=f"{entry['id']} {key} is not color-rendered")
                    self.assertGreater(len(image.getcolors(maxcolors=1 << 24) or []), 32, msg=f"{entry['id']} {key} lacks heatmap variation")

    def test_static_response_uses_only_explicit_heatmap_keys_and_preserves_scalars(self):
        for entry in self.entries:
            result = get_static_result(entry["id"])
            artifacts = result["artifacts"]
            self.assertIn("probability_heatmap", artifacts)
            self.assertIn("entropy_heatmap", artifacts)
            self.assertIn("prediction_mask", artifacts)
            self.assertIn("reference_mask", artifacts)
            self.assertNotIn("probability", artifacts)
            self.assertNotIn("entropy", artifacts)
            self.assertTrue(artifacts["probability_heatmap"])
            self.assertTrue(artifacts["entropy_heatmap"])
            self.assertTrue(artifacts["prediction_mask"])
            if (entry.get("foreground_presence") or {}).get("ground_truth") is not None:
                self.assertTrue(artifacts["reference_mask"])
            self.assertNotIn("/Users/", artifacts["probability_heatmap"])
            self.assertNotIn("/Users/", artifacts["entropy_heatmap"])
            self.assertEqual(
                result["uncertainty"]["summary"],
                (entry.get("uncertainty") or {}).get("summary") or {},
            )

    def test_representative_heatmaps_are_not_single_yellow_overlays(self):
        for kind in ("probability_heatmap", "entropy_heatmap"):
            image_path = SAMPLE_DIR / "artifacts/btxrd/IMG001462" / f"{kind}.png"
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
            red_dominant = (rgb[..., 0] > rgb[..., 1] + 20) & (rgb[..., 0] > rgb[..., 2] + 20)
            cool_dominant = rgb[..., 2] > rgb[..., 0] + 20
            cyan_like = (rgb[..., 1] > rgb[..., 0] + 15) & (rgb[..., 2] > rgb[..., 0] + 15)
            self.assertGreater(int(red_dominant.sum()), 100, msg=f"{kind} lacks warm Turbo colors")
            self.assertGreater(int(cool_dominant.sum()), 100, msg=f"{kind} lacks cool Turbo colors")
            self.assertGreater(int(cyan_like.sum()), 100, msg=f"{kind} lacks intermediate Turbo colors")

    def test_static_api_serves_every_explicit_heatmap_without_ml_runtime(self):
        with app.test_client() as client:
            for entry in self.entries:
                response = client.get(f"/static-samples/{entry['id']}")
                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                for key in ("probability_heatmap", "entropy_heatmap", "prediction_mask", "reference_mask"):
                    asset_response = client.get(payload["artifacts"][key])
                    try:
                        self.assertEqual(asset_response.status_code, 200, msg=f"{entry['id']} {key}")
                    finally:
                        asset_response.close()


if __name__ == "__main__":
    unittest.main()
