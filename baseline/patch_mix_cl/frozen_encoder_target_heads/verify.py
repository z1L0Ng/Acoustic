"""Independent verifier for the Patch-Mix frozen-encoder target-head pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from acoustic.evaluation.sprsound_inter import EXPECTED_ID_SHA256, id_sha256

from .run import (
    CHECKPOINT_SIZE_BYTES,
    CHECKPOINT_SHA256,
    EXPECTED_INTER_RAW,
    EXPERIMENT_ID,
    TASK_LABELS,
    raw_to_task,
    score_metrics,
    sha256_file,
    validate_roots,
)


PACKAGE_RELATIVE_FILES = {
    "baseline/patch_mix_cl/frozen_encoder_target_heads/__init__.py",
    "baseline/patch_mix_cl/frozen_encoder_target_heads/README.md",
    "baseline/patch_mix_cl/frozen_encoder_target_heads/protocol.json",
    "baseline/patch_mix_cl/frozen_encoder_target_heads/run.py",
    "baseline/patch_mix_cl/frozen_encoder_target_heads/verify.py",
    "experiments/sprsound_patchmix_frozen_encoder_target_heads.yaml",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_package() -> None:
    package = Path(__file__).resolve().parent
    project_root = package.parents[2]
    entries = {}
    for line in (package / "package_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    expected = PACKAGE_RELATIVE_FILES
    if set(entries) != expected:
        raise RuntimeError("package manifest membership mismatch")
    for relative, expected_digest in entries.items():
        if sha256_file(project_root / relative) != expected_digest:
            raise RuntimeError(f"package digest mismatch: {relative}")
    protocol = json.loads((package / "protocol.json").read_text())
    if (
        protocol["experiment_id"] != EXPERIMENT_ID
        or protocol["checkpoint"]["sha256"] != CHECKPOINT_SHA256
        or protocol["checkpoint"]["size_bytes"] != CHECKPOINT_SIZE_BYTES
    ):
        raise RuntimeError("package protocol identity mismatch")
    experiment = (
        project_root / "experiments" / "sprsound_patchmix_frozen_encoder_target_heads.yaml"
    ).read_text()
    required = [
        f"id: {EXPERIMENT_ID}",
        f"result_root: result/{EXPERIMENT_ID}",
        f"cache_root: .cache/{EXPERIMENT_ID}",
    ]
    if any(token not in experiment for token in required):
        raise RuntimeError("experiment path contract mismatch")
    combined = "\n".join(
        (project_root / relative).read_text(errors="ignore")
        for relative in PACKAGE_RELATIVE_FILES
    )
    forbidden = ("/" + "Users/", "/" + "files1/", "result" + "s/")
    if any(token in combined for token in forbidden):
        raise RuntimeError("nonportable package path")
    print(f"frozen_head_package_verification_ok files={len(PACKAGE_RELATIVE_FILES)}")


def compare_metrics(observed: dict[str, object], expected: dict[str, object], task: str) -> None:
    keys = [
        "specificity_percent",
        "sensitivity_percent",
        "average_score_percent",
        "harmonic_score_percent",
        "macro_f1",
        "weighted_f1",
        "uar",
    ]
    keys.append(
        "official_sprsound_score_percent"
        if task != "narrow_four"
        else "narrow4_icbhi_as_shared_ontology_diagnostic_percent"
    )
    if max(abs(float(observed[key]) - float(expected[key])) for key in keys) > 1e-10:
        raise RuntimeError(f"{task} metric recomputation mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["package", "profile", "full"], required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.mode == "package":
        verify_package()
        return
    if args.result_root is None or args.cache_root is None:
        parser.error("--result-root and --cache-root are required for profile/full")
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
    manifest = [json.loads(line) for line in (result_root / "event_manifest.jsonl").read_text().splitlines()]
    train = [row for row in manifest if row["partition"] == "train"]
    inter = [row for row in manifest if row["partition"] == "inter"]
    if len(train) != 6_656 or len(inter) != 1_429:
        raise RuntimeError("event manifest coverage mismatch")
    if any("raw_label" in row for row in inter):
        raise RuntimeError("inter manifest is not label-free")
    subtrain_patients = {row["patient_id"] for row in train if row["inner_split"] == "subtrain"}
    validation_patients = {row["patient_id"] for row in train if row["inner_split"] == "validation"}
    train_patients = {row["patient_id"] for row in train}
    inter_patients = {row["patient_id"] for row in inter}
    if (
        sum(row["inner_split"] == "subtrain" for row in train),
        sum(row["inner_split"] == "validation" for row in train),
        len(subtrain_patients),
        len(validation_patients),
        len(subtrain_patients & validation_patients),
        len(train_patients),
        len(inter_patients),
        len(train_patients & inter_patients),
    ) != (5_219, 1_437, 194, 49, 0, 243, 41, 0):
        raise RuntimeError("patient-grouped split verification failed")
    inter_ids = [row["event_id"] for row in inter]
    if len(set(inter_ids)) != 1_429 or id_sha256(inter_ids) != EXPECTED_ID_SHA256:
        raise RuntimeError("inter event ID verification failed")
    run_manifest = json.loads((result_root / "run_manifest.json").read_text())
    if run_manifest["feature_receipt"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint receipt mismatch")
    cache_path = cache_root / f"{args.mode}_embeddings.npz"
    if args.mode == "full":
        if Path(run_manifest["feature_receipt"]["cache_path"]) != cache_path:
            raise RuntimeError("feature cache path mismatch")
        if sha256_file(cache_path) != run_manifest["feature_receipt"]["cache_sha256"]:
            raise RuntimeError("feature cache digest mismatch")
    with np.load(cache_path) as payload:
        embeddings = payload["embeddings"]
        ids = payload["event_ids"].astype(str).tolist()
    expected_ids = (
        [row["event_id"] for row in manifest]
        if args.mode == "full"
        else ids
    )
    if ids != expected_ids or embeddings.shape != (len(ids), 768) or not np.isfinite(embeddings).all():
        raise RuntimeError("embedding cache structural verification failed")
    if args.mode == "profile":
        gate = run_manifest["runtime_gate"]
        if (
            gate["profile_events"] != 100
            or len(ids) != 100
            or len(set(ids)) != 100
            or not all(
            np.isfinite(float(gate[key]))
            for key in (
                "profile_extraction_seconds",
                "projected_total_seconds",
                "peak_rss_gib",
            )
            )
        ):
            raise RuntimeError("profile gate receipt mismatch")
        print(
            f"frozen_head_profile_verification_ok passed={gate['passed']} "
            f"projected_seconds={gate['projected_total_seconds']:.2f} rss_gib={gate['peak_rss_gib']:.3f}"
        )
        return

    label_free = read_csv(result_root / "inter_predictions_label_free.csv")
    scored = read_csv(result_root / "inter_predictions_scored.csv")
    if (
        len(label_free) != 1_429
        or [row["event_id"] for row in label_free] != inter_ids
        or any("raw_label" in row for row in label_free)
        or [row["event_id"] for row in scored] != inter_ids
    ):
        raise RuntimeError("prediction ID/label isolation verification failed")
    if dict(sorted(Counter(row["raw_label"] for row in scored).items())) != EXPECTED_INTER_RAW:
        raise RuntimeError("inter scoring support mismatch")
    stored_metrics = json.loads((result_root / "metrics.json").read_text())
    for task, labels in TASK_LABELS.items():
        included = [row for row in scored if raw_to_task(row["raw_label"], task) is not None]
        probability_columns = [
            f"{task}_prob_{label.lower().replace(' ', '_').replace('+', '_and_')}"
            for label in labels
        ]
        for row in label_free:
            probabilities = np.asarray([float(row[column]) for column in probability_columns])
            if not np.isfinite(probabilities).all() or abs(float(probabilities.sum()) - 1) > 1e-5:
                raise RuntimeError(f"{task} probability verification failed")
            if int(row[f"{task}_pred_index"]) != int(probabilities.argmax()):
                raise RuntimeError(f"{task} argmax verification failed")
        y_true = np.asarray(
            [labels.index(str(raw_to_task(row["raw_label"], task))) for row in included]
        )
        y_pred = np.asarray([int(row[f"{task}_pred_index"]) for row in included])
        recomputed, matrix = score_metrics(y_true, y_pred, labels, task)
        compare_metrics(stored_metrics[task], recomputed, task)
        confusion = read_csv(result_root / "tasks" / task / "confusion.csv")
        observed_matrix = np.asarray(
            [[int(row[label]) for label in labels] for row in confusion], dtype=np.int64
        )
        if not np.array_equal(matrix, observed_matrix) or int(matrix.sum()) != len(included):
            raise RuntimeError(f"{task} confusion verification failed")
    print(
        "frozen_head_full_verification_ok tasks=3 inter_ids=1429 "
        "inner_patient_overlap=0 train_inter_patient_overlap=0"
    )


if __name__ == "__main__":
    main()
