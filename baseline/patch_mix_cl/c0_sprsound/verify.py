"""Independent structural and metric verification for C0 smoke/full artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import (
    EXPECTED_INTER_EVENTS,
    EXPECTED_INTER_ID_SHA256,
    PROTOCOL_NAME,
    TASK_LABELS,
    classification_metrics,
    id_sha256,
    load_run_manifest,
    read_csv,
    task_label,
    validate_cache_root,
    validate_result_root,
)


def compare_metrics(observed: dict, expected: dict) -> float:
    keys = ["specificity", "sensitivity", "icbhi_score", "macro_f1", "weighted_f1", "uar"]
    if "both_recall" in expected:
        keys.append("both_recall")
    return max(abs(float(observed[key]) - float(expected[key])) for key in keys)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package() -> None:
    project_root = Path(__file__).resolve().parents[3]
    package = project_root / "baseline" / "patch_mix_cl" / "c0_sprsound"
    manifest_entries = {}
    for line in (package / "package_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        manifest_entries[relative] = digest
    expected_files = {
        str(path.relative_to(project_root))
        for path in package.iterdir()
        if path.is_file() and path.name != "package_manifest.sha256"
    }
    if set(manifest_entries) != expected_files:
        raise RuntimeError("C0 package manifest membership mismatch")
    mismatches = {
        relative: (digest, sha256_file(project_root / relative))
        for relative, digest in manifest_entries.items()
        if digest != sha256_file(project_root / relative)
    }
    if mismatches:
        raise RuntimeError(f"C0 package hash mismatch: {mismatches}")
    notebook = json.loads((package / "patch_mix_c0_sprsound.ipynb").read_text())
    if notebook.get("nbformat") != 4 or any(
        cell.get("execution_count") is not None or cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ):
        raise RuntimeError("C0 notebook is not clean nbformat 4")
    protocol = json.loads((package / "protocol.json").read_text())
    if protocol.get("experiment_id") != "sprsound_patchmix_target_training":
        raise RuntimeError("C0 protocol experiment ID mismatch")
    experiment = (project_root / "experiments" / "sprsound_patchmix_target_training.yaml").read_text()
    required = [
        "id: sprsound_patchmix_target_training",
        "result_root: result/sprsound_patchmix_target_training",
        "cache_root: .cache/sprsound_patchmix_target_training",
    ]
    if any(token not in experiment for token in required):
        raise RuntimeError("C0 experiment definition path contract mismatch")
    package_text = "\n".join(
        path.read_text(errors="ignore")
        for path in package.iterdir()
        if path.is_file() and path.name not in {"verify.py", "package_manifest.sha256"}
    )
    forbidden = ["/Users/", "/files1/", "results/", "result/c0_patch_mix_sprsound_"]
    found = [token for token in forbidden if token in package_text]
    if found:
        raise RuntimeError(f"C0 package contains retired/nonportable path tokens: {found}")
    print(f"c0_package_verification_ok files={len(expected_files)} notebook=clean paths=canonical")


def read_confusion(path: Path, labels: list[str]) -> np.ndarray:
    rows = read_csv(path)
    if [row["true/pred"] for row in rows] != labels:
        raise RuntimeError(f"confusion label order mismatch: {path}")
    return np.asarray([[int(row[label]) for label in labels] for row in rows], dtype=np.int64)


def verify_probabilities(rows: list[dict[str, str]], labels: list[str], task: str) -> float:
    max_error = 0.0
    for row in rows:
        logits = np.asarray([float(row[f"logit_{label}"]) for label in labels])
        probabilities = np.asarray([float(row[f"prob_{label}"]) for label in labels])
        if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise RuntimeError(f"non-finite prediction: {task}")
        shifted = logits - logits.max()
        expected = np.exp(shifted) / np.exp(shifted).sum()
        max_error = max(
            max_error,
            abs(float(probabilities.sum()) - 1.0),
            float(np.max(np.abs(probabilities - expected))),
        )
        if int(row["pred_index"]) != int(np.argmax(probabilities)):
            raise RuntimeError(f"argmax mismatch: {task}")
        if row["pred_label"] != labels[int(row["pred_index"])]:
            raise RuntimeError(f"prediction label mismatch: {task}")
    if max_error > 1e-5:
        raise RuntimeError(f"probability mismatch: {task} {max_error}")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["package", "smoke", "full"], required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-name", default="full")
    args = parser.parse_args()
    if args.mode == "package":
        verify_package()
        return
    if args.result_root is None or args.cache_root is None:
        parser.error("--result-root and --cache-root are required for smoke/full verification")
    root = validate_result_root(args.result_root)
    cache_root = validate_cache_root(args.cache_root)
    if args.mode != args.cache_name:
        raise ValueError("--mode and --cache-name must match")

    run_manifest = load_run_manifest(root)
    bootstrap = run_manifest["bootstrap"]
    data_receipt = run_manifest["data_protocol"]
    cache_receipt = json.loads((cache_root / args.cache_name / "cache_receipt.json").read_text())
    if cache_receipt != run_manifest[f"cache_{args.cache_name}"]:
        raise RuntimeError("cache receipt/run-manifest mismatch")
    if bootstrap["status"] != "bootstrap_verified":
        raise RuntimeError("bootstrap receipt failed")
    if data_receipt["validation"]["patient_overlap"] != 0:
        raise RuntimeError("inner split patient leakage")
    assignments = read_csv(root / "data" / "patient_validation_assignment.csv")
    if len(assignments) != 243 or len({row["patient_id"] for row in assignments}) != 243:
        raise RuntimeError("patient assignment mismatch")
    train_rows = read_csv(root / "data" / "train_events.csv")
    patient_splits: dict[str, set[str]] = {}
    for row in train_rows:
        patient_splits.setdefault(row["patient_id"], set()).add(row["inner_split"])
    if any(len(splits) != 1 for splits in patient_splits.values()):
        raise RuntimeError("independent patient-group leakage check failed")
    if not cache_receipt["train"]["finite"] or not cache_receipt["inter"]["finite"]:
        raise RuntimeError("cache finite gate failed")

    verified = {}
    inter_manifest = {
        row["event_id"]: row
        for row in read_csv(root / "data" / "inter_events_label_free.csv")
    }
    inter_ids = sorted(inter_manifest)
    if len(inter_ids) != EXPECTED_INTER_EVENTS or id_sha256(inter_ids) != EXPECTED_INTER_ID_SHA256:
        raise RuntimeError("C0 inter manifest differs from frozen B0 target")
    cache_index = read_csv(cache_root / args.cache_name / "inter" / "index.csv")
    expected_forward_ids = [row["event_id"] for row in cache_index]
    for task, labels in TASK_LABELS.items():
        task_dir = root / args.mode / task
        receipt = run_manifest[f"{args.mode}_{task}"]
        label_free = read_csv(task_dir / "inter_predictions_label_free.csv")
        ids = [row["event_id"] for row in label_free]
        if (
            ids != expected_forward_ids
            or len(ids) != len(set(ids))
            or any("true_" in key or "raw_label" in key for key in label_free[0])
        ):
            raise RuntimeError(f"label-free prediction contract failed: {task}")
        probability_error = verify_probabilities(label_free, labels, task)
        if not receipt["finite_loss_and_gradient"] or receipt["inter_metrics_used_for_selection"]:
            raise RuntimeError(f"training boundary failed: {task}")
        history = read_csv(task_dir / "training_history.csv")
        expected_epochs = 1 if args.mode == "smoke" else 50
        if len(history) != expected_epochs or [int(row["epoch"]) for row in history] != list(range(1, expected_epochs + 1)):
            raise RuntimeError(f"training history coverage mismatch: {task}")
        if list(task_dir.glob("validation_predictions_epoch_*.csv")) or list(
            task_dir.glob("validation_confusion_epoch_*.csv")
        ):
            raise RuntimeError(f"non-compact per-epoch validation artifacts found: {task}")
        eligible = [
            row
            for row in history
            if args.mode == "smoke" or row["selection_eligible"].lower() == "true"
        ]
        expected_best = max(eligible, key=lambda row: float(row["validation_score"]))
        if (
            int(expected_best["epoch"]) != int(receipt["best_epoch"])
            or abs(float(expected_best["validation_score"]) - float(receipt["best_inner_validation_score"])) > 1e-12
        ):
            raise RuntimeError(f"best checkpoint selection mismatch: {task}")
        best_checkpoint = None
        last_checkpoint = None
        for checkpoint_name in ("best_validation.pth", "last.pth"):
            checkpoint = torch.load(task_dir / "checkpoints" / checkpoint_name, map_location="cpu")
            required = {"protocol", "task", "seed", "mode", "cache_name", "batch_size", "labels", "epoch", "model", "classifier", "projector"}
            if not required <= set(checkpoint):
                raise RuntimeError(f"checkpoint completeness mismatch: {task} {checkpoint_name}")
            if checkpoint_name == "last.pth":
                resume_required = {
                    "optimizer",
                    "scaler",
                    "best_validation_score",
                    "best_epoch",
                    "python_random_state",
                    "numpy_random_state",
                    "torch_random_state",
                    "cuda_random_states",
                    "data_generator_state",
                }
                if not resume_required <= set(checkpoint):
                    raise RuntimeError(f"resume-state completeness mismatch: {task}")
            if (
                checkpoint["protocol"] != PROTOCOL_NAME
                or checkpoint["task"] != task
                or checkpoint["mode"] != args.mode
                or checkpoint["cache_name"] != args.cache_name
                or checkpoint["labels"] != labels
            ):
                raise RuntimeError(f"checkpoint protocol mismatch: {task} {checkpoint_name}")
            if checkpoint_name == "best_validation.pth":
                best_checkpoint = checkpoint
            else:
                last_checkpoint = checkpoint
        if best_checkpoint is None or int(best_checkpoint["epoch"]) != int(receipt["best_epoch"]):
            raise RuntimeError(f"best checkpoint epoch mismatch: {task}")
        if (
            last_checkpoint is None
            or int(last_checkpoint["epoch"]) != int(history[-1]["epoch"])
            or int(receipt["epochs_completed"]) != int(history[-1]["epoch"])
        ):
            raise RuntimeError(f"last checkpoint/history mismatch: {task}")
        validation_predictions = read_csv(task_dir / "validation_predictions_best.csv")
        if len(validation_predictions) != int(receipt["validation_rows"]):
            raise RuntimeError(f"best validation prediction coverage mismatch: {task}")
        validation_true = np.asarray(
            [int(row["true_index"]) for row in validation_predictions], dtype=np.int64
        )
        validation_pred = np.asarray(
            [int(row["pred_index"]) for row in validation_predictions], dtype=np.int64
        )
        validation_metrics, validation_matrix = classification_metrics(
            validation_true, validation_pred, labels
        )
        validation_difference = compare_metrics(
            best_checkpoint["validation_metrics"], validation_metrics
        )
        if validation_difference > 1e-12 or not np.array_equal(
            read_confusion(task_dir / "validation_confusion_best.csv", labels),
            validation_matrix,
        ):
            raise RuntimeError(f"best validation artifact mismatch: {task}")
        if args.mode == "smoke":
            if receipt["inter_label_access"] != "none" or (task_dir / "inter_metrics.json").exists():
                raise RuntimeError(f"smoke accessed inter labels: {task}")
            verified[task] = {
                "inter_forward_rows": len(ids),
                "metrics": "not_computed",
                "probability_max_abs_error": probability_error,
                "validation_metric_max_abs_difference": validation_difference,
            }
            continue

        if len(ids) != EXPECTED_INTER_EVENTS or id_sha256(ids) != EXPECTED_INTER_ID_SHA256:
            raise RuntimeError(f"full inter coverage mismatch: {task}")
        scored = read_csv(task_dir / "inter_predictions.csv")
        if len(scored) != EXPECTED_INTER_EVENTS or [row["event_id"] for row in scored] != ids:
            raise RuntimeError(f"scored prediction coverage mismatch: {task}")
        for row in scored:
            annotation = inter_manifest[row["event_id"]]
            raw_payload = json.loads(Path(annotation["annotation_path"]).read_text())
            raw = str(raw_payload["event_annotation"][int(annotation["event_index"])]["type"])
            expected = task_label(raw, task)
            if row["raw_label"] != raw or row["task_label"] != (expected or ""):
                raise RuntimeError(f"independent label rebuild mismatch: {row['event_id']}")
        included = [row for row in scored if row["mapping_status"] == "included"]
        y_true = np.asarray([int(row["true_index"]) for row in included], dtype=np.int64)
        y_pred = np.asarray([int(row["pred_index"]) for row in included], dtype=np.int64)
        expected_metrics, matrix = classification_metrics(y_true, y_pred, labels)
        observed_metrics = json.loads((task_dir / "inter_metrics.json").read_text())
        difference = compare_metrics(observed_metrics, expected_metrics)
        observed_confusion = read_confusion(task_dir / "inter_confusion.csv", labels)
        if difference > 1e-12 or not np.array_equal(observed_confusion, matrix):
            raise RuntimeError(f"independent metric mismatch: {task} {difference}")
        verified[task] = {
            "inter_forward_rows": len(ids),
            "inter_scored_rows": len(included),
            "confusion_total": int(matrix.sum()),
            "max_metric_difference": difference,
            "probability_max_abs_error": probability_error,
            "validation_metric_max_abs_difference": validation_difference,
        }

    print(
        f"c0_{args.mode}_verification_ok tasks=2 cache={args.cache_name} "
        f"details={json.dumps(verified, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
