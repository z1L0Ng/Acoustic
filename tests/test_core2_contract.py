import unittest

import numpy as np

from baseline.multidataset_pipeline.core2_hf_positive_kauh_external import (
    CORE_NODES,
    hf_positive_report,
    select_core_shared_thresholds,
)


class Core2ContractTest(unittest.TestCase):
    def test_shared_threshold_uses_equal_dataset_f1_and_has_no_other(self):
        predictions = {
            "dataset_ids": np.asarray(["icbhi"] * 4 + ["sprsound"] * 4),
            "eligible": np.ones((8, 3), dtype=bool),
            "targets": np.asarray(
                [
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            ),
            "attribute_probabilities": np.asarray(
                [
                    [0.1, 0.2],
                    [0.8, 0.1],
                    [0.2, 0.7],
                    [0.9, 0.8],
                    [0.3, 0.2],
                    [0.6, 0.4],
                    [0.4, 0.6],
                    [0.7, 0.9],
                ],
                dtype=np.float32,
            ),
        }
        thresholds, details = select_core_shared_thresholds(predictions)
        self.assertEqual(CORE_NODES, ("level1", "crackle", "wheeze"))
        self.assertEqual(set(thresholds), {"crackle", "wheeze"})
        for node in thresholds:
            self.assertEqual(
                set(details[node]["f1_by_dataset"]), {"icbhi", "sprsound"}
            )

    def test_hf_report_is_positive_only_without_detector_metrics(self):
        predictions = {
            "dataset_ids": np.asarray(["hf_lung", "hf_lung", "icbhi"]),
            "eligible": np.asarray(
                [[False, True, False], [False, False, True], [True, True, True]]
            ),
            "attribute_probabilities": np.asarray(
                [[0.8, 0.1], [0.2, 0.9], [0.9, 0.9]], dtype=np.float32
            ),
        }
        report = hf_positive_report(
            predictions, {"crackle": 0.5, "wheeze": 0.5}
        )
        self.assertEqual(set(report["nodes"]), {"crackle", "wheeze"})
        self.assertEqual(report["nodes"]["crackle"]["positive_recall"], 1.0)
        self.assertNotIn("f1", report["nodes"]["crackle"])
        self.assertNotIn("specificity", report["nodes"]["crackle"])


if __name__ == "__main__":
    unittest.main()
