from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from baseline.multidataset_pipeline.hf_thresholds import (
    HFValidationThresholdBatch,
    select_and_write_hf_threshold_receipt,
    select_hf_validation_thresholds,
)
from baseline.multidataset_pipeline.hf_validation_exporter import (
    _load_prediction_artifact,
    _pad_validation_rows,
    _write_prediction_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(*, outer_test_accessed: bool = False) -> HFValidationThresholdBatch:
    scores = torch.tensor([0.9, 0.8, 0.8, 0.8, 0.8, 0.0], dtype=torch.float64)
    labels = torch.tensor([1, 1, 0, 0, 0, 1], dtype=torch.float64)
    probabilities = scores.reshape(1, 6, 1).repeat(1, 1, 4)
    targets = labels.reshape(1, 6, 1).repeat(1, 1, 4)
    return HFValidationThresholdBatch(
        prediction_ids=("hf:validation:one",),
        probabilities=probabilities,
        targets=targets,
        window_mask=torch.ones((1, 6), dtype=torch.bool),
        annotation_mask=torch.ones((1, 6, 4), dtype=torch.bool),
        valid_mask=torch.ones((1, 6, 4), dtype=torch.bool),
        time_map=torch.tensor(
            [[[float(index), float(index + 1)] for index in range(6)]],
            dtype=torch.float64,
        ),
        outer_test_accessed=outer_test_accessed,
    )


class HFThresholdToolchainTest(unittest.TestCase):
    def test_validation_export_artifact_preserves_masks_time_map_and_isolation(self):
        rows = []
        for index, windows in enumerate((2, 1)):
            rows.append(
                (
                    f"hf:train:validation-{index}",
                    torch.full((windows, 4), 0.75, dtype=torch.float32),
                    torch.tensor(
                        [([1.0, 0.0, 1.0, 0.0] if index == 0 else [0.0, 1.0, 0.0, 1.0])]
                        * windows
                    ),
                    torch.ones(windows, dtype=torch.bool),
                    torch.ones(windows, 4, dtype=torch.bool),
                    torch.ones(windows, 4, dtype=torch.bool),
                    torch.tensor(
                        [[float(step), float(step + 1)] for step in range(windows)],
                        dtype=torch.float64,
                    ),
                )
            )
        batch = _pad_validation_rows(rows)
        self.assertEqual(batch.probabilities.shape, (2, 2, 4))
        self.assertFalse(batch.window_mask[1, 1])
        self.assertEqual(batch.time_map[1, 1].tolist(), [0.0, 0.0])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.pt"
            artifact = _write_prediction_artifact(path, pipeline_id="P2", batch=batch)
            loaded = _load_prediction_artifact(path, artifact["sha256"], "P2")
        self.assertEqual(loaded.prediction_ids, batch.prediction_ids)
        self.assertTrue(torch.equal(loaded.valid_mask, batch.valid_mask))
        self.assertFalse(loaded.outer_test_accessed)

    def test_validation_max_f1_uses_highest_threshold_tie_break(self):
        selection = select_hf_validation_thresholds(_batch())
        self.assertEqual(selection.thresholds, (0.9, 0.9, 0.9, 0.9))
        self.assertEqual([row["max_f1"] for row in selection.per_channel], [0.5] * 4)
        with self.assertRaisesRegex(PermissionError, "validation"):
            select_hf_validation_thresholds(_batch(outer_test_accessed=True))

    def test_generation_binds_full_selection_checkpoint_and_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "selected.pt"
            checkpoint.write_bytes(b"synthetic-selected-checkpoint")
            checkpoint_sha256 = _sha256(checkpoint)
            approval = {
                "status": "approved",
                "pipeline_id": "P1",
                "phase": "full",
                "config_sha256": "a" * 64,
                "data_identity_sha256": "b" * 64,
                "authorized_by": "synthetic-test",
                "outer_test_authorized": False,
            }
            approval_path = root / "approval.json"
            approval_path.write_text(json.dumps(approval), encoding="utf-8")
            approval_sha256 = _sha256(approval_path)
            batch = _batch()
            ordered_ids_sha256 = hashlib.sha256(
                json.dumps(
                    {"ordered_prediction_ids": list(batch.prediction_ids)},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            selected = {
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": checkpoint_sha256,
                "outer_test_accessed": False,
                "native_metrics_only": True,
            }
            selection = {
                "schema_version": "validation_selection_v2",
                "runner_schema_version": "shared_window_training_v5",
                "pipeline_id": "P1",
                "config_sha256": "a" * 64,
                "data_identity_sha256": "b" * 64,
                "full_approval_receipt_sha256": approval_sha256,
                "hf_validation_threshold_identity": {
                    "validation_data_identity_sha256": "c" * 64,
                    "hf_validation_manifest_identity_sha256": "d" * 64,
                    "hf_validation_ordered_prediction_ids_sha256": ordered_ids_sha256,
                },
                "candidates": [selected],
                "selected_checkpoint": selected,
                "outer_test_accessed": False,
                "reported_as_pooled_performance": False,
            }
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            result = select_and_write_hf_threshold_receipt(
                root / "hf_thresholds.json",
                batch,
                validation_data_identity_sha256="c" * 64,
                hf_validation_manifest_identity_sha256="d" * 64,
                expected_hf_validation_ordered_prediction_ids_sha256=(
                    ordered_ids_sha256
                ),
                full_approval_receipt_path=approval_path,
                expected_full_approval_receipt_sha256=approval_sha256,
                validation_selection_receipt_path=selection_path,
                expected_validation_selection_receipt_sha256=_sha256(selection_path),
                selected_checkpoint_path=checkpoint,
                expected_selected_checkpoint_sha256=checkpoint_sha256,
            )
            self.assertEqual(
                result["status"], "hf_threshold_artifact_generated_validation_only"
            )
            self.assertEqual(result["thresholds"], [0.9] * 4)
            self.assertFalse(result["outer_test_accessed"])
            self.assertEqual(len(result["threshold_bytes_hex"]), 64)


if __name__ == "__main__":
    unittest.main()
