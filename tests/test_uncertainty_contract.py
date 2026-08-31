from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.inference.contracts import PredictionArtifacts
from backend.inference import service


def build_artifacts(*, boundary_enabled: bool, boundary_entropy: float | None, conformal_available: bool):
    summary = {
        "available": True,
        "global_entropy": 0.25,
        "boundary_entropy": boundary_entropy,
        "boundary_available": boundary_enabled,
        "boundary_pixel_count": 12 if boundary_enabled else None,
        "uncertain_pixel_ratio": 0.1,
        "predicted_tumor_ratio": 0.05,
        "mean_tumor_probability": 0.08,
    }
    return PredictionArtifacts(
        mask=np.zeros((4, 4), dtype=np.uint8),
        overlay=np.zeros((4, 4, 3), dtype=np.uint8),
        original=np.zeros((4, 4, 3), dtype=np.uint8),
        model_id="fixture",
        checkpoint_path=Path("fixture.pt"),
        note="fixture",
        extra_metadata={
            "uncertainty_summary": summary,
            "conformal_summary": {"available": conformal_available},
        },
        auxiliary_maps={
            "predictive_entropy": np.full((4, 4), 0.25, dtype=np.float32),
            "tumor_probability": np.full((4, 4), 0.5, dtype=np.float32),
        },
    )


class UncertaintyContractTests(unittest.TestCase):
    def build_payload(self, artifacts):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(service, "RESULT_DIR", Path(temp_dir)):
            return service._build_uncertainty_payload("fixture", artifacts)

    def test_entropy_only_checkpoint_marks_boundary_and_crc_unavailable(self):
        payload = self.build_payload(
            build_artifacts(boundary_enabled=False, boundary_entropy=None, conformal_available=False)
        )
        self.assertTrue(payload["pixel_entropy"]["available"])
        self.assertEqual(payload["pixel_entropy"]["normalization"], "normalized_binary_entropy")
        self.assertTrue(payload["global"]["available"])
        self.assertFalse(payload["boundary"]["available"])
        self.assertEqual(payload["boundary"]["reason"], "boundary_disabled_in_model_config")
        self.assertFalse(payload["risk_control"]["available"])
        self.assertEqual(payload["risk_control"]["reason"], "not_applicable")

    def test_boundary_enabled_checkpoint_exposes_hb(self):
        payload = self.build_payload(
            build_artifacts(boundary_enabled=True, boundary_entropy=0.4, conformal_available=False)
        )
        self.assertTrue(payload["boundary"]["available"])
        self.assertEqual(payload["boundary"]["value"], 0.4)
        self.assertIsNone(payload["boundary"]["reason"])

    def test_empty_predicted_boundary_is_not_presented_as_meaningful_hb(self):
        artifacts = build_artifacts(boundary_enabled=True, boundary_entropy=0.0, conformal_available=False)
        artifacts = replace(
            artifacts,
            extra_metadata={
                **artifacts.extra_metadata,
                "uncertainty_summary": {
                    **artifacts.extra_metadata["uncertainty_summary"],
                    "boundary_pixel_count": 0,
                },
            },
        )
        payload = self.build_payload(artifacts)
        self.assertFalse(payload["boundary"]["available"])
        self.assertIsNone(payload["boundary"]["value"])
        self.assertEqual(payload["boundary"]["reason"], "empty_predicted_boundary")

    def test_crc_fixture_is_only_available_when_backend_supplies_it(self):
        payload = self.build_payload(
            build_artifacts(boundary_enabled=True, boundary_entropy=0.4, conformal_available=True)
        )
        self.assertTrue(payload["risk_control"]["available"])
        self.assertIsNone(payload["risk_control"]["reason"])


if __name__ == "__main__":
    unittest.main()
