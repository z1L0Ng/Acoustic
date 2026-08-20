"""Posthoc native-format readouts from saved unified-head predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ICBHI_LABELS = ("normal", "crackle", "wheeze", "both")
CONDITIONS = ("AST_HF_off", "BEATs_HF_off", "PANNs_Cnn14")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def native_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    labels: Sequence[str],
) -> dict[str, object]:
    indices = np.arange(len(labels))
    matrix = confusion_matrix(target, prediction, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=indices, zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum())
    abnormal_total = int(matrix[1:].sum())
    sensitivity = float(np.trace(matrix[1:, 1:]) / abnormal_total)
    average = (specificity + sensitivity) / 2
    harmonic = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    return {
        "scale": "fraction_0_to_1",
        "rows": int(len(target)),
        "confusion": matrix.astype(int).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "macro_f1": float(
            f1_score(target, prediction, labels=indices, average="macro", zero_division=0)
        ),
        "uar": float(recall.mean()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "average_score": average,
        "icbhi_score": average,
        "harmonic_score": harmonic,
    }


def decode_icbhi_flat4(
    level1_prediction: np.ndarray,
    attribute_probability: np.ndarray,
    crackle_threshold: float,
    wheeze_threshold: float,
) -> np.ndarray:
    output = np.zeros(len(level1_prediction), dtype=np.int64)
    for index in np.flatnonzero(level1_prediction == 1):
        crackle = attribute_probability[index, 0] >= crackle_threshold
        wheeze = attribute_probability[index, 1] >= wheeze_threshold
        if crackle and wheeze:
            output[index] = 3
        elif crackle:
            output[index] = 1
        elif wheeze:
            output[index] = 2
        else:
            crackle_margin = attribute_probability[index, 0] - crackle_threshold
            wheeze_margin = attribute_probability[index, 1] - wheeze_threshold
            output[index] = 1 if crackle_margin >= wheeze_margin else 2
    return output


def condition_readout(result_root: Path, condition: str) -> dict[str, object]:
    run_dir = result_root / condition / "seed_42"
    predictions = np.load(run_dir / "selected_test_predictions.npz", allow_pickle=False)
    threshold_payload = json.loads((run_dir / "validation_thresholds.json").read_text())
    thresholds = threshold_payload["thresholds"]

    icbhi_mask = predictions["dataset_ids"] == "icbhi"
    raw_icbhi = [
        json.loads(value) for value in predictions["raw_ground_truth"][icbhi_mask]
    ]
    icbhi_target = np.asarray([ICBHI_LABELS.index(value) for value in raw_icbhi])
    icbhi_prediction = decode_icbhi_flat4(
        predictions["level1_predictions"][icbhi_mask],
        predictions["attribute_probabilities"][icbhi_mask],
        float(thresholds["crackle"]),
        float(thresholds["wheeze"]),
    )
    icbhi = native_metrics(icbhi_target, icbhi_prediction, ICBHI_LABELS)
    icbhi.update(
        {
            "task": "ICBHI official-test flat4",
            "protocol": "official recording split 60/40; test n=2756; not strict patient-held-out",
            "official_train_test_patient_overlap": ["156", "218"],
            "prediction_unit": "annotated respiratory cycle",
            "readout": "posthoc native-format decoder from unified head; not a paper-faithful native head",
            "decoder": (
                "Level1 Normal->Normal; abnormal with both/one attribute over shared "
                "validation thresholds->Both/Crackle/Wheeze; neither over threshold->"
                "larger probability-minus-threshold margin; tie=Crackle"
            ),
            "thresholds": thresholds,
            "official_score": icbhi["average_score"],
        }
    )

    spr_mask = predictions["dataset_ids"] == "sprsound"
    spr_target = predictions["targets"][spr_mask, 0].astype(np.int64)
    spr_prediction = predictions["level1_predictions"][spr_mask].astype(np.int64)
    spr = native_metrics(spr_target, spr_prediction, ("normal", "abnormal"))
    spr.update(
        {
            "task": "SPRSound BioCAS2022 Task1-1 Normal/Adventitious",
            "prediction_unit": "official inter-subject respiratory event",
            "protocol": "official BioCAS2022 inter; event n=1429; not Task1-2 raw7",
            "readout": "posthoc native-format Level1 readout from unified head",
            "test_threshold_tuning": False,
            "official_score": (
                spr["average_score"] + spr["harmonic_score"]
            )
            / 2,
        }
    )
    payload = {
        "condition": condition,
        "source_predictions": str(run_dir / "selected_test_predictions.npz"),
        "new_test_inference": False,
        "threshold_source": "saved ICBHI+SPRSound validation shared thresholds",
        "icbhi_flat4": icbhi,
        "sprsound_inter_binary": spr,
    }
    write_json(run_dir / "native_comparable_metrics.json", payload)
    return payload


def build_comparison(result_root: Path) -> dict[str, object]:
    results = {condition: condition_readout(result_root, condition) for condition in CONDITIONS}
    metrics = (
        "macro_f1",
        "uar",
        "sensitivity",
        "specificity",
        "average_score",
        "harmonic_score",
    )
    delta = {
        task: {
            metric: (
                results["BEATs_HF_off"][task][metric]
                - results["AST_HF_off"][task][metric]
            )
            for metric in metrics
        }
        for task in ("icbhi_flat4", "sprsound_inter_binary")
    }
    payload = {
        "status": "posthoc_native_format_readout_complete",
        "conditions": results,
        "beats_minus_ast": delta,
        "paper_comparisons": {
            "icbhi_official_recording_split": [
                {
                    "work": "Bae23 AST+CE",
                    "icbhi_score": 0.5955,
                    "checkpoint_selection": "test-selected",
                    "macro_f1": None,
                    "uar": None
                },
                {
                    "work": "Bae23 Patch-Mix CL",
                    "icbhi_score": 0.6237,
                    "checkpoint_selection": "test-selected",
                    "macro_f1": None,
                    "uar": None
                },
                {
                    "work": "Jeong25 BEATs+CE",
                    "icbhi_score": 0.6349,
                    "checkpoint_selection": "test-selected",
                    "macro_f1": None,
                    "uar": None
                },
                {
                    "work": "Jeong25 PAFA",
                    "icbhi_score": 0.6484,
                    "checkpoint_selection": "test-selected",
                    "macro_f1": None,
                    "uar": None
                },
                {
                    "work": "Kim24 SG-SCL",
                    "icbhi_score": 0.6171,
                    "checkpoint_selection": "test-selected",
                    "macro_f1": None,
                    "uar": None
                }
            ],
            "sprsound_biocas2022_task1_1_inter": [
                {
                    "work": "Zhang22 SPR Task1-1",
                    "sensitivity": 0.7583,
                    "specificity": 0.7904,
                    "average_score": 0.7744,
                    "harmonic_score": 0.7740,
                    "official_score": 0.7742
                }
            ]
        },
        "excluded_comparison_scopes": [
            "strict patient-held-out",
            "LungMix full-test/COMB",
            "random split",
            "pooled rows"
        ],
        "claim_boundary": (
            "readout from frozen unified-head predictions; no new inference and no "
            "paper-faithful dataset-native head claim"
        ),
    }
    write_json(result_root / "native_comparable_ast_vs_beats.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("result/reproduce/core2_hf_positive_kauh_external"),
    )
    args = parser.parse_args()
    comparison = build_comparison(args.result_root.resolve())
    compact = {
        condition: {
            "icbhi": comparison["conditions"][condition]["icbhi_flat4"],
            "sprsound": comparison["conditions"][condition]["sprsound_inter_binary"],
        }
        for condition in CONDITIONS
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
