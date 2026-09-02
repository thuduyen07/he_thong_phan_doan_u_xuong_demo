from __future__ import annotations

import unittest

from backend.inference.registry import list_model_records, list_model_specs
from backend.inference.service import resolve_models_for_dataset


MODEL_ID = "btxrd_segformer_b0_boundary_adaptive_crc"
FRACATLAS_MODEL_ID = "fracatlas_segformer_b0_boundary_adaptive_crc"


class BoundaryAdaptiveLiveRegistryTests(unittest.TestCase):
    def test_boundary_adaptive_checkpoint_is_live_capable_without_calibration_state(self):
        records = {record["model_id"]: record for record in list_model_records()}
        record = records[MODEL_ID]
        self.assertTrue(record["available"])
        self.assertEqual(record["threshold"], 0.50)
        self.assertNotIn("calibration_available", record)

        specs = {spec.model_id: spec for spec in list_model_specs()}
        metadata = dict(specs[MODEL_ID].extra_metadata or {})
        self.assertEqual(metadata["method"], "boundary_adaptive_crc")
        self.assertTrue(metadata["trained_with_crc"])
        self.assertFalse(metadata["requires_calibration_at_inference"])
        self.assertTrue(metadata["inference_calibration_declared"])
        self.assertEqual(metadata["inference_rule"], "binary_threshold")

    def test_model_registry_routes_exactly_one_live_model_per_dataset(self):
        records = {record["model_id"]: record for record in list_model_records()}
        self.assertTrue(records[FRACATLAS_MODEL_ID]["available"])
        self.assertEqual(
            [record["model_id"] for record in resolve_models_for_dataset("btxrd")],
            [MODEL_ID],
        )
        self.assertEqual(
            [record["model_id"] for record in resolve_models_for_dataset("fracatlas")],
            [FRACATLAS_MODEL_ID],
        )


if __name__ == "__main__":
    unittest.main()
