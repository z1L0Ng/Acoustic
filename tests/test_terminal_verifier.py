import unittest

from baseline.multidataset_pipeline.terminal_verifier import (
    _average_precision,
    _binary_auc,
    _multiclass_metrics,
)


class TerminalVerifierTest(unittest.TestCase):
    def test_independent_metric_recompute(self):
        metrics = _multiclass_metrics(
            [0, 0, 1, 1], [0, 1, 1, 1], ["normal", "abnormal"], "SPRSound_binary"
        )
        self.assertEqual(metrics["confusion"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["native_score"], 17 / 24)
        self.assertAlmostEqual(_binary_auc([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.7]), 1.0)
        self.assertAlmostEqual(
            _average_precision([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.7]), 1.0
        )


if __name__ == "__main__":
    unittest.main()
