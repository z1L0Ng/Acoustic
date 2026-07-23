"""Frozen Patch-Mix author-checkpoint B0 transfer to SPRSound inter events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from .run_icbhi_checkpoint_eval import LABELS, build_model, sha256_file


METHOD_STATUS = (
    "published-model / author-checkpoint / ICBHI-test-selected / "
    "zero-target-tuning exploratory transfer"
)
RAW_TO_FOUR = {
    "Normal": "normal",
    "Fine Crackle": "crackle",
    "Coarse Crackle": "crackle",
    "Wheeze": "wheeze",
    "Wheeze+Crackle": "both",
}
FOUR_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
BINARY_LABELS = ["normal", "abnormal"]
EXPECTED_EVENTS = 1429
EXPECTED_RAW_COUNTS = {
    "Coarse Crackle": 3,
    "Fine Crackle": 80,
    "Normal": 1040,
    "Wheeze": 305,
    "Wheeze+Crackle": 1,
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_inter_manifest(dataset_root: Path) -> list[dict[str, object]]:
    """Build the label-free inference manifest from official event boundaries."""
    json_dir = dataset_root / "test2022_json" / "inter_test_json"
    wav_dir = dataset_root / "test2022_wav"
    if not json_dir.is_dir() or not wav_dir.is_dir():
        raise FileNotFoundError(f"invalid BioCAS2022 root: {dataset_root}")
    rows: list[dict[str, object]] = []
    for json_path in sorted(json_dir.glob("*.json")):
        recording_id = json_path.stem
        audio_path = wav_dir / f"{recording_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        payload = json.loads(json_path.read_text())
        for event_index, event in enumerate(payload.get("event_annotation", [])):
            start_ms = int(event["start"])
            end_ms = int(event["end"])
            if start_ms < 0 or end_ms <= start_ms:
                raise RuntimeError(f"invalid event boundary in {json_path}: {event}")
            rows.append(
                {
                    "event_id": f"inter:{recording_id}:event_{event_index:03d}",
                    "source_partition": "BioCAS2022_inter_subject",
                    "recording_id": recording_id,
                    "patient_id": recording_id.split("_", 1)[0],
                    "event_index": event_index,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "audio_path": str(audio_path.resolve()),
                    "annotation_path": str(json_path.resolve()),
                }
            )
    if len(rows) != EXPECTED_EVENTS or len({str(row["event_id"]) for row in rows}) != EXPECTED_EVENTS:
        raise RuntimeError(f"inter must contain {EXPECTED_EVENTS} unique events")
    return rows


def load_final_scoring_labels(dataset_root: Path) -> dict[str, dict[str, str]]:
    """Load target labels only after every frozen-model logit has been produced."""
    json_dir = dataset_root / "test2022_json" / "inter_test_json"
    labels: dict[str, dict[str, str]] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        for event_index, event in enumerate(payload.get("event_annotation", [])):
            raw_label = str(event["type"])
            if raw_label not in RAW_TO_FOUR:
                raise RuntimeError(
                    "inter contains an unsupported label; broad-CAS is not approved: "
                    f"{raw_label} in {json_path}"
                )
            labels[f"inter:{json_path.stem}:event_{event_index:03d}"] = {
                "raw_event_label": raw_label,
                "binary_broad_label": "normal" if raw_label == "Normal" else "abnormal",
                "narrow_four_label": RAW_TO_FOUR[raw_label],
                "narrow_four_mapping_status": "included",
            }
    counts = dict(sorted(Counter(row["raw_event_label"] for row in labels.values()).items()))
    if len(labels) != EXPECTED_EVENTS or counts != EXPECTED_RAW_COUNTS:
        raise RuntimeError(f"unexpected final-scoring label receipt: n={len(labels)} counts={counts}")
    return labels


def preprocess_recording(
    rows: list[dict[str, object]], generate_fbank, cut_pad, protocol_args
) -> list[torch.Tensor]:
    waveform, sample_rate = torchaudio.load(str(rows[0]["audio_path"]))
    waveform = waveform.mean(dim=0, keepdim=True)
    source_duration_ms = waveform.shape[1] / sample_rate * 1000
    if sample_rate != protocol_args.sample_rate:
        waveform = audio_transforms.Resample(sample_rate, protocol_args.sample_rate)(waveform)
    fade_samples = int(protocol_args.sample_rate / 16)
    waveform = audio_transforms.Fade(
        fade_in_len=fade_samples,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )(waveform)
    resize = transforms.Resize((798, 128))
    images = []
    for row in rows:
        if float(row["end_ms"]) > source_duration_ms + 1.0:
            raise RuntimeError(f"event exceeds recording: {row['event_id']}")
        start = int(float(row["start_ms"]) / 1000 * protocol_args.sample_rate)
        end = int(float(row["end_ms"]) / 1000 * protocol_args.sample_rate)
        event_audio = cut_pad(waveform[:, start:end], protocol_args)
        fbank = generate_fbank(event_audio, protocol_args.sample_rate, n_mels=128)
        image = resize(transforms.ToTensor()(fbank))
        if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
            raise RuntimeError(f"invalid fbank for {row['event_id']}: {tuple(image.shape)}")
        images.append(image)
    return images


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> tuple[dict[str, object], np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    predicted_counts = np.bincount(y_pred, minlength=len(labels))
    metrics: dict[str, object] = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": (specificity + sensitivity) / 2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
                "predicted": int(predicted_counts[index]),
            }
            for index, label in enumerate(labels)
        },
        "collapse_diagnostics": {
            "predicted_class_counts": {
                label: int(predicted_counts[index]) for index, label in enumerate(labels)
            },
            "predicted_class_fractions": {
                label: float(predicted_counts[index] / len(y_pred))
                for index, label in enumerate(labels)
            },
            "zero_prediction_classes": [
                label for index, label in enumerate(labels) if predicted_counts[index] == 0
            ],
            "zero_recall_classes": [
                label for index, label in enumerate(labels) if recall[index] == 0
            ],
            "majority_prediction_fraction": float(predicted_counts.max() / len(y_pred)),
            "single_class_collapse": bool(np.count_nonzero(predicted_counts) == 1),
        },
    }
    if labels == LABELS:
        metrics["both_recall"] = float(recall[3])
        metrics["both_support_warning"] = (
            "SPRSound inter has Both support=1; do not draw minority conclusions."
        )
    return metrics, matrix


def write_confusion(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for label, values in zip(labels, matrix):
            writer.writerow([label, *values.tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dataset_root = args.dataset_root.resolve()
    author_repo = args.author_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=False)

    rows = build_inter_manifest(dataset_root)
    write_csv(result_root / "inference_manifest_label_free.csv", rows)
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    protocol_args = argparse.Namespace(sample_rate=16000, desired_length=8, pad_types="repeat", n_mels=128)
    model, classifier, checkpoint_epoch = build_model(author_repo, checkpoint, device)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)

    predictions: list[dict[str, object]] = []
    pending_images: list[torch.Tensor] = []
    pending_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_images:
            return
        batch = torch.stack(pending_images).to(device)
        with torch.inference_mode():
            logits = classifier(model(batch))
            probabilities = torch.softmax(logits, dim=1)
        for row, logit, probability in zip(
            pending_rows, logits.cpu().numpy(), probabilities.cpu().numpy()
        ):
            four_index = int(probability.argmax())
            binary_index = 0 if four_index == 0 else 1
            record = dict(row)
            record.update(
                {
                    "pred_narrow_four_label": LABELS[four_index],
                    "pred_narrow_four_index": four_index,
                    "pred_binary_broad_label": BINARY_LABELS[binary_index],
                    "pred_binary_broad_index": binary_index,
                    "prob_binary_normal": float(probability[0]),
                    "prob_binary_abnormal": float(probability[1:].sum()),
                    "binary_decision_rule": "collapse_source_four_class_argmax",
                }
            )
            for index, label in enumerate(LABELS):
                record[f"logit_{label}"] = float(logit[index])
                record[f"prob_{label}"] = float(probability[index])
            predictions.append(record)
        pending_images.clear()
        pending_rows.clear()

    for recording_index, recording_id in enumerate(sorted(grouped), start=1):
        recording_rows = grouped[recording_id]
        images = preprocess_recording(
            recording_rows, generate_fbank, cut_pad_sample_torchaudio, protocol_args
        )
        for row, image in zip(recording_rows, images):
            pending_rows.append(row)
            pending_images.append(image)
            if len(pending_images) == args.batch_size:
                flush()
        if recording_index % 50 == 0:
            print(
                f"inter_recordings={recording_index}/{len(grouped)} events={len(predictions)}",
                flush=True,
            )
    flush()
    runtime_seconds = time.perf_counter() - started
    predictions.sort(key=lambda row: str(row["event_id"]))
    if len(predictions) != EXPECTED_EVENTS or len({str(row["event_id"]) for row in predictions}) != EXPECTED_EVENTS:
        raise RuntimeError("prediction coverage mismatch")
    numeric_keys = [f"logit_{label}" for label in LABELS] + [f"prob_{label}" for label in LABELS]
    numeric_keys += ["prob_binary_normal", "prob_binary_abnormal"]
    if not all(math.isfinite(float(row[key])) for row in predictions for key in numeric_keys):
        raise RuntimeError("non-finite prediction output")

    scoring_labels = load_final_scoring_labels(dataset_root)
    if set(scoring_labels) != {str(row["event_id"]) for row in predictions}:
        raise RuntimeError("final-scoring labels do not cover the frozen predictions")
    for row in predictions:
        row.update(scoring_labels[str(row["event_id"])])
    write_csv(result_root / "b0_predictions.csv", predictions)

    y_four = np.asarray([FOUR_TO_INDEX[str(row["narrow_four_label"])] for row in predictions])
    p_four = np.asarray([int(row["pred_narrow_four_index"]) for row in predictions])
    y_binary = np.asarray([0 if row["binary_broad_label"] == "normal" else 1 for row in predictions])
    p_binary = np.asarray([int(row["pred_binary_broad_index"]) for row in predictions])
    four_metrics, four_matrix = classification_metrics(y_four, p_four, LABELS)
    binary_metrics, binary_matrix = classification_metrics(y_binary, p_binary, BINARY_LABELS)
    write_confusion(result_root / "b0_narrow_four_confusion.csv", LABELS, four_matrix)
    write_confusion(result_root / "b0_binary_broad_confusion.csv", BINARY_LABELS, binary_matrix)

    floor_rows = []
    for row in predictions:
        floor_rows.append(
            {
                "event_id": row["event_id"],
                "raw_event_label": row["raw_event_label"],
                "binary_broad_label": row["binary_broad_label"],
                "narrow_four_label": row["narrow_four_label"],
                "floor_pred_binary_broad_label": "normal",
                "floor_pred_binary_broad_index": 0,
                "floor_pred_narrow_four_label": "normal",
                "floor_pred_narrow_four_index": 0,
            }
        )
    write_csv(result_root / "b0_floor_predictions.csv", floor_rows)
    floor_binary, floor_binary_matrix = classification_metrics(
        y_binary, np.zeros(EXPECTED_EVENTS, dtype=int), BINARY_LABELS
    )
    floor_four, floor_four_matrix = classification_metrics(
        y_four, np.zeros(EXPECTED_EVENTS, dtype=int), LABELS
    )
    write_confusion(result_root / "b0_floor_binary_broad_confusion.csv", BINARY_LABELS, floor_binary_matrix)
    write_confusion(result_root / "b0_floor_narrow_four_confusion.csv", LABELS, floor_four_matrix)

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
            sorted(Counter(str(row["narrow_four_label"]) for row in predictions).items())
        ),
        "binary_broad_counts": dict(
            sorted(Counter(str(row["binary_broad_label"]) for row in predictions).items())
        ),
    }
    receipt = {
        "status": "completed_pending_independent_verification",
        "method_status": METHOD_STATUS,
        "claim_boundary": "exploratory transfer performance only; no C0 target reference and no degradation conclusion",
        "source_checkpoint": {
            "sha256": sha256_file(checkpoint),
            "epoch": checkpoint_epoch,
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
            "binary_broad": {"metrics": floor_binary, "confusion_matrix": floor_binary_matrix.tolist()},
            "narrow_four": {"metrics": floor_four, "confusion_matrix": floor_four_matrix.tolist()},
        },
        "runtime_seconds": runtime_seconds,
        "device": str(device),
        "batch_size": args.batch_size,
        "threads": args.threads,
    }
    (result_root / "metrics.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    protocol = {
        "protocol": "B0_plus_B0_floor_v1",
        "method_status": METHOD_STATUS,
        "dataset_commit": "874eeb8736ddb78937c2fb5332fc7e7293d0f0ca",
        "target_partition": "BioCAS2022 inter-subject only",
        "prediction_unit": "official event segment",
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_test_selected": True,
        "weights_frozen": True,
        "tasks": ["binary_broad", "narrow_four"],
        "floor": "all-normal on identical target rows",
        "intra": "not executed",
        "c0": "not executed",
        "target_tuning": "none",
    }
    (result_root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
