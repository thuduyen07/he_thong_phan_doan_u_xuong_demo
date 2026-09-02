from __future__ import annotations

import unittest

from backend.dashboard import get_dashboard_payload


EXPECTED_TOTALS = {
    "btxrd": {"images": 3746, "positive": 1867, "negative": 1879},
    "fracatlas": {"images": 4024, "positive": 719, "negative": 3305},
}

EXPECTED_ROWS = {
    "btxrd": [
        ("Huấn luyện", 2247, 1120, 1127, "60%"),
        ("Xác thực", 375, 187, 188, "10%"),
        ("Hiệu chỉnh phát triển", 375, 187, 188, "10%"),
        ("Hiệu chỉnh cuối", 375, 187, 188, "10%"),
        ("Kiểm thử", 374, 186, 188, "10%"),
        ("Tổng", 3746, 1867, 1879, "100%"),
    ],
    "fracatlas": [
        ("Huấn luyện", 2414, 431, 1983, "60%"),
        ("Xác thực", 403, 72, 331, "10%"),
        ("Hiệu chỉnh phát triển", 403, 72, 331, "10%"),
        ("Hiệu chỉnh cuối", 402, 72, 330, "10%"),
        ("Kiểm thử", 402, 72, 330, "10%"),
        ("Tổng", 4024, 719, 3305, "100%"),
    ],
}


class DatasetSplitTableTests(unittest.TestCase):
    def test_verified_split_rows_sum_to_exact_dataset_totals(self):
        datasets = get_dashboard_payload()["overview"]["dataset_splits"]
        self.assertEqual([item["id"] for item in datasets], ["btxrd", "fracatlas"])
        for dataset in datasets:
            rows = dataset["rows"]
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                [(row["split"], row["images"], row["positive"], row["negative"], row["ratio"]) for row in rows],
                EXPECTED_ROWS[dataset["id"]],
            )
            self.assertTrue(rows[-1]["is_total"])
            self.assertEqual(rows[-1]["split"], "Tổng")
            for key, expected in EXPECTED_TOTALS[dataset["id"]].items():
                self.assertEqual(sum(row[key] for row in rows[:-1]), expected)
                self.assertEqual(rows[-1][key], expected)
            self.assertEqual(rows[-1]["ratio"], "100%")


if __name__ == "__main__":
    unittest.main()
