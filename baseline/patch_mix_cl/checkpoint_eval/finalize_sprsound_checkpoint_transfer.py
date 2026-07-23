"""Finalize B0 receipts after a post-inference artifact-writing interruption."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .restricted_checkpoint import restricted_torch_load
from .run_icbhi_checkpoint_eval import LABELS, sha256_file
from .run_sprsound_checkpoint_transfer import (
    BINARY_LABELS,
    EXPECTED_EVENTS,
    EXPECTED_RAW_COUNTS,
    FOUR_TO_INDEX,
    METHOD_STATUS,
    classification_metrics,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_csv(args.result_root / "b0_predictions.csv")
    manifest = read_csv(args.result_root / "inference_manifest_label_free.csv")
    expected_ids = {row["event_id"] for row in manifest}
    observed_ids = {row["event_id"] for row in predictions}
    if (
        len(predictions) != EXPECTED_EVENTS
        or len(observed_ids) != EXPECTED_EVENTS
        or observed_ids != expected_ids
    ):
        raise RuntimeError("cannot finalize incomplete predictions")
    if any("label" in key for key in manifest[0]):
        raise RuntimeError("inference manifest is not label-free")

    y_four = np.asarray([FOUR_TO_INDEX[row["narrow_four_label"]] for row in predictions])
    p_four = np.asarray([int(row["pred_narrow_four_index"]) for row in predictions])
    y_binary = np.asarray([0 if row["binary_broad_label"] == "normal" else 1 for row in predictions])
    p_binary = np.asarray([int(row["pred_binary_broad_index"]) for row in predictions])
    four_metrics, four_matrix = classification_metrics(y_four, p_four, LABELS)
    binary_metrics, binary_matrix = classification_metrics(y_binary, p_binary, BINARY_LABELS)
    floor_binary, floor_binary_matrix = classification_metrics(
        y_binary, np.zeros(EXPECTED_EVENTS, dtype=int), BINARY_LABELS
    )
    floor_four, floor_four_matrix = classification_metrics(
        y_four, np.zeros(EXPECTED_EVENTS, dtype=int), LABELS
    )
    coverage = {
        "events_total": EXPECTED_EVENTS,
        "binary_broad_included": EXPECTED_EVENTS,
        "narrow_four_included": EXPECTED_EVENTS,
        "excluded_rhonchi": 0,
        "excluded_stridor": 0,
        "excluded_poor_quality_record_origin": 0,
        "coverage_binary_broad": 1.0,
        "coverage_narrow_four": 1.0,
        "raw_label_counts": EXPECTED_RAW_COUNTS,
        "narrow_four_counts": dict(
            sorted(Counter(row["narrow_four_label"] for row in predictions).items())
        ),
        "binary_broad_counts": dict(
            sorted(Counter(row["binary_broad_label"] for row in predictions).items())
        ),
    }
    checkpoint = restricted_torch_load(args.checkpoint)
    receipt = {
        "status": "completed_pending_independent_verification",
        "method_status": METHOD_STATUS,
        "claim_boundary": "exploratory transfer performance only; no C0 target reference and no degradation conclusion",
        "source_checkpoint": {
            "sha256": sha256_file(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
            "frozen": True,
            "selected_on_icbhi_official_test": True,
        },
        "source_alignment_receipt": "result/patch_mix_cl_author_checkpoint_20260722_154852/icbhi/metrics.json",
        "target": "SPRSound BioCAS2022 inter-subject event-level only",
        "target_use": "event audio and official boundaries for inference; labels used only for final scoring",
        "label_isolation_receipt": "inference_manifest_label_free.csv contains no label fields; scoring labels loaded only after all frozen-model logits were complete",
        "preprocessing": "8 kHz mono source resampled to 16 kHz; source full-recording fade; official event crop; source 8 s truncate/repeat+fade, 128-bin Kaldi fbank, AudioSet normalization, resize 798x128, augmentation off",
        "binary_decision_rule": "collapse source four-class argmax: normal iff source argmax is normal; otherwise abnormal; no target threshold",
        "target_training": False,
        "target_adaptation": False,
        "target_calibration": False,
        "target_threshold_search": False,
        "target_model_or_row_selection": False,
        "broad_cas_executed": False,
        "intra_executed": False,
        "coverage": coverage,
        "b0": {
            "binary_broad": {"metrics": binary_metrics, "confusion_matrix": binary_matrix.tolist()},
            "narrow_four": {"metrics": four_metrics, "confusion_matrix": four_matrix.tolist()},
        },
        "b0_floor_all_normal": {
            "binary_broad": {
                "metrics": floor_binary,
                "confusion_matrix": floor_binary_matrix.tolist(),
            },
            "narrow_four": {
                "metrics": floor_four,
                "confusion_matrix": floor_four_matrix.tolist(),
            },
        },
        "runtime_seconds": None,
        "runtime_note": "full inference completed; exact timer was not persisted because the first receipt-finalization attempt failed after prediction export",
        "resume_from_completed_predictions": True,
    }
    (args.result_root / "metrics.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    protocol = {
        "protocol": "B0_plus_B0_floor_v1",
        "method_status": METHOD_STATUS,
        "dataset_commit": "874eeb8736ddb78937c2fb5332fc7e7293d0f0ca",
        "target_partition": "BioCAS2022 inter-subject only",
        "prediction_unit": "official event segment",
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_checkpoint_test_selected": True,
        "weights_frozen": True,
        "tasks": ["binary_broad", "narrow_four"],
        "floor": "all-normal on identical target rows",
        "intra": "not executed",
        "c0": "not executed",
        "target_tuning": "none",
        "inference_manifest_label_free": True,
        "receipt_finalized_from_completed_predictions": True,
    }
    (args.result_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    print("sprsound_b0_finalize_ok events=1429 inference_reused=true")


if __name__ == "__main__":
    main()
