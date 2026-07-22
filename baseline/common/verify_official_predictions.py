"""Export and verify ICBHI cycle-level predictions from a logits archive.

The maintained method adapters write ``cycle_id``, integer ``label``, and
``logits`` arrays. This command converts them into a portable CSV, recomputes
the author Sp/Se/Score metric, and optionally enforces complete official-split
coverage against the audited manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ["normal", "crackle", "wheeze", "both"]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_ids(manifest: Path, split: str) -> set[str]:
    with manifest.open(newline="") as handle:
        return {
            row["cycle_id"]
            for row in csv.DictReader(handle)
            if row["official_split"] == split
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-split", choices=["train", "test"])
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as archive:
        cycle_ids = archive["cycle_id"].astype(str)
        labels = archive["label"].astype(np.int64)
        logits = archive["logits"].astype(np.float64)

    n = len(cycle_ids)
    if logits.shape != (n, 4) or labels.shape != (n,):
        raise ValueError(f"invalid shapes: ids={cycle_ids.shape}, labels={labels.shape}, logits={logits.shape}")
    if len(set(cycle_ids)) != n:
        raise ValueError("cycle_id values are not unique")
    if args.expected_rows is not None and n != args.expected_rows:
        raise ValueError(f"row count mismatch: expected={args.expected_rows} observed={n}")
    if not np.isfinite(logits).all() or not np.isin(labels, np.arange(4)).all():
        raise ValueError("non-finite logits or invalid labels")

    if bool(args.manifest) != bool(args.expected_split):
        raise ValueError("--manifest and --expected-split must be supplied together")
    if args.manifest:
        expected = expected_ids(args.manifest, args.expected_split)
        observed = set(cycle_ids)
        if observed != expected:
            raise ValueError(
                f"split coverage mismatch: expected={len(expected)} observed={len(observed)} "
                f"missing={len(expected - observed)} extra={len(observed - expected)}"
            )

    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for true, pred in zip(labels, predictions):
        confusion[true, pred] += 1
    normal_specificity = float(confusion[0, 0] / max(confusion[0].sum(), 1))
    abnormal_total = int(confusion[1:].sum())
    abnormal_sensitivity = float(np.trace(confusion[1:, 1:]) / max(abnormal_total, 1))
    score = (normal_specificity + abnormal_sensitivity) / 2.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", newline="") as handle:
        fields = ["cycle_id", "true_index", "true_label", "pred_index", "pred_label"] + [
            f"prob_{label}" for label in LABELS
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, cycle_id in enumerate(cycle_ids):
            row = {
                "cycle_id": cycle_id,
                "true_index": int(labels[index]),
                "true_label": LABELS[labels[index]],
                "pred_index": int(predictions[index]),
                "pred_label": LABELS[predictions[index]],
            }
            row.update({f"prob_{name}": float(probabilities[index, i]) for i, name in enumerate(LABELS)})
            writer.writerow(row)

    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    receipt = {
        "status": "verified",
        "input_npz": str(args.input_npz),
        "rows": n,
        "unique_cycle_ids": n,
        "confusion_total": int(confusion.sum()),
        "label_order": LABELS,
        "normal_specificity": normal_specificity,
        "abnormal_sensitivity": abnormal_sensitivity,
        "icbhi_score": score,
        "metric_scale": "fraction; multiply by 100 for author paper tables",
        "prediction_csv": str(prediction_path),
        "expected_split": args.expected_split,
        "expected_rows": args.expected_rows,
    }
    (args.output_dir / "metric_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
