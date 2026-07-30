"""Shared orchestration for matched frozen-encoder SPRSound target-head pilots."""

from __future__ import annotations

import json
import os
import time
import csv
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from acoustic.evaluation.sprsound_inter import EXPECTED_ID_SHA256, id_sha256

from baseline.patch_mix_cl.frozen_encoder_target_heads.run import (
    FULL_EVENTS,
    MAX_PROJECTED_SECONDS,
    MAX_RSS_GIB,
    PROFILE_EVENTS,
    TASK_LABELS,
    build_manifest,
    load_inter_labels,
    peak_rss_gib,
    profile_rows,
    raw_to_task,
    save_embedding_cache,
    score_metrics,
    sha256_file,
    train_head,
    write_csv,
    write_json,
    write_manifest,
)


BuildEncoder = Callable[[Path, torch.device], tuple[torch.nn.Module, dict[str, object]]]
ExtractEmbeddings = Callable[
    [list[dict[str, object]], Path, torch.nn.Module, torch.device, int],
    tuple[np.ndarray, float],
]


def validate_roots(
    result_root: Path, cache_root: Path, experiment_id: str
) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != experiment_id or result.parent.name != "result":
        raise ValueError(f"result root must be result/{experiment_id}")
    if cache.name != experiment_id or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{experiment_id}")
    return result, cache


def verify_source_repo(source_repo: Path, expected_commit: str) -> Path:
    import subprocess

    repo = source_repo.resolve()
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"author source is not a Git checkout: {repo}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != expected_commit or status:
        raise RuntimeError("author source identity/status gate failed")
    return repo


def run_pilot(
    *,
    phase: str,
    dataset_root: Path,
    source_repo: Path,
    result_root: Path,
    cache_root: Path,
    device_name: str,
    threads: int,
    batch_size: int,
    experiment_id: str,
    protocol_name: str,
    method_id: str,
    author_repo_commit: str,
    source_preprocessing: str,
    direct_transfer_binary_score: float,
    build_encoder: BuildEncoder,
    extract_embeddings: ExtractEmbeddings,
) -> None:
    if phase not in {"profile", "full"}:
        raise ValueError(phase)
    if device_name != "cpu":
        raise ValueError("these bounded local pilots are preregistered for CPU")
    result_root, cache_root = validate_roots(result_root, cache_root, experiment_id)
    result_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    for variable, relative in (
        ("NUMBA_CACHE_DIR", "runtime/numba"),
        ("MPLCONFIGDIR", "runtime/matplotlib"),
        ("XDG_CACHE_HOME", "runtime/xdg"),
    ):
        runtime_cache = cache_root / relative
        runtime_cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(runtime_cache)

    source_repo = verify_source_repo(source_repo, author_repo_commit)
    torch.set_num_threads(threads)
    device = torch.device(device_name)
    rows = build_manifest(dataset_root)
    write_manifest(result_root / "event_manifest.jsonl", rows)
    train_rows = [row for row in rows if row["partition"] == "train"]
    inter_rows = [row for row in rows if row["partition"] == "inter"]
    profile_manifest = None
    if phase == "full":
        profile_manifest = json.loads((result_root / "run_manifest.json").read_text())
        if not profile_manifest["runtime_gate"]["passed"]:
            raise RuntimeError("full run forbidden because the local profile gate did not pass")

    model_started = time.perf_counter()
    encoder, identity_receipt = build_encoder(source_repo, device)
    model_load_seconds = time.perf_counter() - model_started
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("encoder freeze gate failed")
    selected_rows = profile_rows(rows) if phase == "profile" else rows
    embeddings, extraction_seconds = extract_embeddings(
        selected_rows, source_repo, encoder, device, batch_size
    )
    del encoder
    cache_path = cache_root / f"{phase}_embeddings.npz"
    save_embedding_cache(cache_path, selected_rows, embeddings)
    feature_receipt = {
        **identity_receipt,
        "author_repo_commit": author_repo_commit,
        "source_preprocessing": source_preprocessing,
        "encoder_frozen": True,
        "source_classifier_loaded_or_reused": False,
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(np.isfinite(embeddings).all()),
        "model_load_seconds": model_load_seconds,
        "extraction_seconds": extraction_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
    }

    if phase == "profile":
        profile_training_seconds = 0.0
        for task in TASK_LABELS:
            _, _, task_receipt = train_head(
                task,
                selected_rows,
                embeddings,
                device,
                cache_root / "profile_heads" / task,
            )
            profile_training_seconds += float(task_receipt["runtime_seconds"])
        projected_extraction = extraction_seconds / PROFILE_EVENTS * FULL_EVENTS
        projected_training = profile_training_seconds / PROFILE_EVENTS * len(train_rows)
        projected_total = model_load_seconds + projected_extraction + projected_training
        gate = {
            "profile_events": PROFILE_EVENTS,
            "profile_model_load_seconds": model_load_seconds,
            "profile_extraction_seconds": extraction_seconds,
            "profile_head_training_seconds": profile_training_seconds,
            "projected_full_extraction_seconds": projected_extraction,
            "projected_three_head_training_seconds": projected_training,
            "projected_total_seconds": projected_total,
            "peak_rss_gib": peak_rss_gib(),
            "runtime_limit_seconds": MAX_PROJECTED_SECONDS,
            "rss_limit_gib": MAX_RSS_GIB,
            "passed": projected_total <= MAX_PROJECTED_SECONDS
            and peak_rss_gib() <= MAX_RSS_GIB,
        }
        write_json(
            result_root / "run_manifest.json",
            {
                "experiment_id": experiment_id,
                "method_id": method_id,
                "protocol": protocol_name,
                "phase": "profile",
                "data": {
                    "train_events": len(train_rows),
                    "inter_events": len(inter_rows),
                    "subtrain_events": sum(
                        row["inner_split"] == "subtrain" for row in train_rows
                    ),
                    "validation_events": sum(
                        row["inner_split"] == "validation" for row in train_rows
                    ),
                },
                "feature_receipt": feature_receipt,
                "runtime_gate": gate,
            },
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return

    if profile_manifest is None:
        raise RuntimeError("missing accepted profile manifest")
    train_embeddings = embeddings[: len(train_rows)]
    inter_embeddings = embeddings[len(train_rows) :]
    heads: dict[str, torch.nn.Module] = {}
    task_receipts: dict[str, dict[str, object]] = {}
    for task in TASK_LABELS:
        head, _, receipt = train_head(
            task,
            train_rows,
            train_embeddings,
            device,
            result_root / "tasks" / task,
        )
        heads[task] = head
        task_receipts[task] = receipt

    task_outputs = {}
    inter_tensor = torch.from_numpy(inter_embeddings).to(device)
    with torch.inference_mode():
        for task, head in heads.items():
            logits = head(inter_tensor).cpu().numpy()
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            task_outputs[task] = (logits, probabilities)
    label_free_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(inter_rows):
        output: dict[str, object] = {
            key: row[key]
            for key in (
                "event_id",
                "partition",
                "recording_id",
                "patient_id",
                "event_index",
                "start_ms",
                "end_ms",
            )
        }
        for task, labels in TASK_LABELS.items():
            logits, probabilities = task_outputs[task]
            prediction = int(probabilities[row_index].argmax())
            output[f"{task}_pred_index"] = prediction
            output[f"{task}_pred_label"] = labels[prediction]
            for label_index, label in enumerate(labels):
                safe_label = label.lower().replace(" ", "_").replace("+", "_and_")
                output[f"{task}_logit_{safe_label}"] = float(
                    logits[row_index, label_index]
                )
                output[f"{task}_prob_{safe_label}"] = float(
                    probabilities[row_index, label_index]
                )
        label_free_rows.append(output)
    write_csv(result_root / "inter_predictions_label_free.csv", label_free_rows)

    scoring_labels = load_inter_labels(inter_rows)
    scored_rows = [
        {**row, "raw_label": scoring_labels[str(row["event_id"])]}
        for row in label_free_rows
    ]
    metrics_payload: dict[str, dict[str, object]] = {}
    for task, labels in TASK_LABELS.items():
        included = [
            row
            for row in scored_rows
            if raw_to_task(str(row["raw_label"]), task) is not None
        ]
        y_true = np.asarray(
            [labels.index(str(raw_to_task(str(row["raw_label"]), task))) for row in included]
        )
        y_pred = np.asarray([int(row[f"{task}_pred_index"]) for row in included])
        task_metrics, matrix = score_metrics(y_true, y_pred, labels, task)
        task_metrics["included_inter_events"] = len(included)
        task_metrics["excluded_inter_events"] = len(scored_rows) - len(included)
        metrics_payload[task] = task_metrics
        confusion_rows = [
            {
                "true/pred": label,
                **{pred: int(value) for pred, value in zip(labels, values)},
            }
            for label, values in zip(labels, matrix)
        ]
        write_csv(result_root / "tasks" / task / "confusion.csv", confusion_rows)
    write_csv(result_root / "inter_predictions_scored.csv", scored_rows)
    binary_score = float(metrics_payload["binary"]["official_sprsound_score_percent"])
    metrics_payload["binary"]["comparators"] = {
        f"{method_id}_direct_transfer_score_percent": direct_transfer_binary_score,
        "patch_mix_target_head_score_percent": 87.56146910627386,
        "pilot_minus_own_direct_transfer_percentage_points": binary_score
        - direct_transfer_binary_score,
        "pilot_minus_patch_mix_target_head_percentage_points": binary_score
        - 87.56146910627386,
        "claim_limit": "matched descriptive pilot comparison; not SOTA or significance",
    }
    total_runtime = (
        model_load_seconds
        + extraction_seconds
        + sum(float(receipt["runtime_seconds"]) for receipt in task_receipts.values())
    )
    final_manifest = {
        **profile_manifest,
        "phase": "full_complete",
        "feature_receipt": feature_receipt,
        "training": task_receipts,
        "metrics": metrics_payload,
        "total_runtime_seconds": total_runtime,
        "peak_rss_gib": peak_rss_gib(),
        "inter_evaluated_once_after_validation_selection": True,
        "inter_labels_loaded_after_label_free_predictions_written": True,
        "intra_used": False,
    }
    write_json(result_root / "metrics.json", metrics_payload)
    write_json(result_root / "run_manifest.json", final_manifest)
    print(
        json.dumps(
            {"total_runtime_seconds": total_runtime, "peak_rss_gib": peak_rss_gib()},
            indent=2,
        )
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_package(
    *,
    package_dir: Path,
    project_root: Path,
    relative_files: set[str],
    experiment_id: str,
) -> None:
    entries = {}
    for line in (package_dir / "package_manifest.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    if set(entries) != relative_files:
        raise RuntimeError("package manifest membership mismatch")
    for relative, expected_digest in entries.items():
        if sha256_file(project_root / relative) != expected_digest:
            raise RuntimeError(f"package digest mismatch: {relative}")
    experiment_path = project_root / "experiments" / f"{experiment_id}.yaml"
    experiment = experiment_path.read_text()
    required = [
        f"id: {experiment_id}",
        f"result_root: result/{experiment_id}",
        f"cache_root: .cache/{experiment_id}",
    ]
    if any(token not in experiment for token in required):
        raise RuntimeError("experiment path contract mismatch")
    combined = "\n".join(
        (project_root / relative).read_text(errors="ignore")
        for relative in relative_files
    )
    forbidden = ("/" + "Users/", "/" + "files1/", "result" + "s/")
    if any(token in combined for token in forbidden):
        raise RuntimeError("nonportable package path")
    print(f"{experiment_id}_package_verification_ok files={len(relative_files)}")


def verify_pilot(
    *,
    mode: str,
    result_root: Path,
    cache_root: Path,
    experiment_id: str,
    protocol_name: str,
    method_id: str,
    task_checkpoint_sha256: str,
    backbone_checkpoint_sha256: str | None = None,
) -> None:
    result_root, cache_root = validate_roots(result_root, cache_root, experiment_id)
    manifest = [
        json.loads(line)
        for line in (result_root / "event_manifest.jsonl").read_text().splitlines()
    ]
    train = [row for row in manifest if row["partition"] == "train"]
    inter = [row for row in manifest if row["partition"] == "inter"]
    if len(train) != 6_656 or len(inter) != 1_429:
        raise RuntimeError("event manifest coverage mismatch")
    if any("raw_label" in row for row in inter):
        raise RuntimeError("inter manifest is not label-free")
    subtrain_patients = {
        row["patient_id"] for row in train if row["inner_split"] == "subtrain"
    }
    validation_patients = {
        row["patient_id"] for row in train if row["inner_split"] == "validation"
    }
    train_patients = {row["patient_id"] for row in train}
    inter_patients = {row["patient_id"] for row in inter}
    split_receipt = (
        sum(row["inner_split"] == "subtrain" for row in train),
        sum(row["inner_split"] == "validation" for row in train),
        len(subtrain_patients),
        len(validation_patients),
        len(subtrain_patients & validation_patients),
        len(train_patients),
        len(inter_patients),
        len(train_patients & inter_patients),
    )
    if split_receipt != (5_219, 1_437, 194, 49, 0, 243, 41, 0):
        raise RuntimeError("patient-grouped split verification failed")
    inter_ids = [row["event_id"] for row in inter]
    if len(set(inter_ids)) != 1_429 or id_sha256(inter_ids) != EXPECTED_ID_SHA256:
        raise RuntimeError("inter event ID verification failed")

    run_manifest = json.loads((result_root / "run_manifest.json").read_text())
    if (
        run_manifest["experiment_id"] != experiment_id
        or run_manifest["protocol"] != protocol_name
        or run_manifest["method_id"] != method_id
    ):
        raise RuntimeError("run identity mismatch")
    feature = run_manifest["feature_receipt"]
    if (
        feature["task_checkpoint_sha256"] != task_checkpoint_sha256
        or not feature["encoder_frozen"]
        or feature["source_classifier_loaded_or_reused"]
    ):
        raise RuntimeError("encoder/checkpoint contract mismatch")
    if (
        backbone_checkpoint_sha256 is not None
        and feature["backbone_checkpoint_sha256"] != backbone_checkpoint_sha256
    ):
        raise RuntimeError("backbone checkpoint contract mismatch")
    cache_path = cache_root / f"{mode}_embeddings.npz"
    if mode == "full":
        if Path(feature["cache_path"]) != cache_path:
            raise RuntimeError("feature cache path mismatch")
        if sha256_file(cache_path) != feature["cache_sha256"]:
            raise RuntimeError("feature cache digest mismatch")
    with np.load(cache_path) as payload:
        embeddings = payload["embeddings"]
        ids = payload["event_ids"].astype(str).tolist()
    expected_ids = [row["event_id"] for row in manifest] if mode == "full" else ids
    if (
        ids != expected_ids
        or embeddings.shape != (len(ids), 768)
        or not np.isfinite(embeddings).all()
    ):
        raise RuntimeError("embedding cache structural verification failed")
    if mode == "profile":
        gate = run_manifest["runtime_gate"]
        if (
            gate["profile_events"] != PROFILE_EVENTS
            or len(ids) != PROFILE_EVENTS
            or len(set(ids)) != PROFILE_EVENTS
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
            f"{method_id}_frozen_head_profile_verification_ok "
            f"passed={gate['passed']} projected_seconds={gate['projected_total_seconds']:.2f} "
            f"rss_gib={gate['peak_rss_gib']:.3f}"
        )
        return

    label_free = _read_csv(result_root / "inter_predictions_label_free.csv")
    scored = _read_csv(result_root / "inter_predictions_scored.csv")
    if (
        len(label_free) != 1_429
        or [row["event_id"] for row in label_free] != inter_ids
        or any("raw_label" in row for row in label_free)
        or [row["event_id"] for row in scored] != inter_ids
    ):
        raise RuntimeError("prediction ID/label isolation verification failed")
    from baseline.patch_mix_cl.frozen_encoder_target_heads.run import EXPECTED_INTER_RAW

    if dict(sorted(Counter(row["raw_label"] for row in scored).items())) != EXPECTED_INTER_RAW:
        raise RuntimeError("inter scoring support mismatch")
    stored_metrics = json.loads((result_root / "metrics.json").read_text())
    for task, labels in TASK_LABELS.items():
        included = [
            row for row in scored if raw_to_task(row["raw_label"], task) is not None
        ]
        probability_columns = [
            f"{task}_prob_{label.lower().replace(' ', '_').replace('+', '_and_')}"
            for label in labels
        ]
        for row in label_free:
            probabilities = np.asarray(
                [float(row[column]) for column in probability_columns]
            )
            if (
                not np.isfinite(probabilities).all()
                or abs(float(probabilities.sum()) - 1) > 1e-5
            ):
                raise RuntimeError(f"{task} probability verification failed")
            if int(row[f"{task}_pred_index"]) != int(probabilities.argmax()):
                raise RuntimeError(f"{task} argmax verification failed")
        y_true = np.asarray(
            [labels.index(str(raw_to_task(row["raw_label"], task))) for row in included]
        )
        y_pred = np.asarray([int(row[f"{task}_pred_index"]) for row in included])
        recomputed, matrix = score_metrics(y_true, y_pred, labels, task)
        keys = [
            "specificity_percent",
            "sensitivity_percent",
            "average_score_percent",
            "harmonic_score_percent",
            "macro_f1",
            "weighted_f1",
            "uar",
            (
                "official_sprsound_score_percent"
                if task != "narrow_four"
                else "narrow4_icbhi_as_shared_ontology_diagnostic_percent"
            ),
        ]
        if max(
            abs(float(stored_metrics[task][key]) - float(recomputed[key])) for key in keys
        ) > 1e-10:
            raise RuntimeError(f"{task} metric recomputation mismatch")
        confusion = _read_csv(result_root / "tasks" / task / "confusion.csv")
        observed_matrix = np.asarray(
            [[int(row[label]) for label in labels] for row in confusion],
            dtype=np.int64,
        )
        if not np.array_equal(matrix, observed_matrix) or int(matrix.sum()) != len(
            included
        ):
            raise RuntimeError(f"{task} confusion verification failed")
    finetune_profile_path = result_root / "full_finetune_profile.json"
    if finetune_profile_path.is_file():
        finetune_profile = json.loads(finetune_profile_path.read_text())
        if (
            finetune_profile["local_90_minute_gate_passed"]
            or not np.isfinite(float(finetune_profile["step_seconds"]))
            or not np.isfinite(float(finetune_profile["peak_rss_gib"]))
            or not np.isfinite(
                float(finetune_profile["projected_five_epoch_train_seconds_lower_bound"])
            )
        ):
            raise RuntimeError("full-finetune profile receipt mismatch")
    print(
        f"{method_id}_frozen_head_full_verification_ok tasks=3 inter_ids=1429 "
        "inner_patient_overlap=0 train_inter_patient_overlap=0"
    )
