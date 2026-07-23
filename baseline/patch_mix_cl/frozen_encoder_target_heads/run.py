"""Profile and run frozen Patch-Mix embeddings with SPRSound target heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from acoustic.evaluation.sprsound_inter import (
    EXPECTED_ID_SHA256,
    id_sha256,
    resolve_biocas_root,
)
from baseline.patch_mix_cl.checkpoint_eval.restricted_checkpoint import restricted_torch_load


EXPERIMENT_ID = "sprsound_patchmix_frozen_encoder_target_heads"
PROTOCOL = "patch_mix_frozen_encoder_target_heads_v1"
AUTHOR_REPO_URL = "https://github.com/raymin0223/patch-mix_contrastive_learning"
AUTHOR_REPO_COMMIT = "836b09fea1b70eb29fe0b25afa481286b56f5104"
CHECKPOINT_SHA256 = "0cdeeec3b834ad18a5a945ea935ef8656922b0e444ed80e050206304de597949"
CHECKPOINT_SIZE_BYTES = 1_047_665_181
SEED = 20260722
PROFILE_EVENTS = 100
FULL_EVENTS = 6_656 + 1_429
MAX_PROJECTED_SECONDS = 90 * 60
MAX_RSS_GIB = 24.0
RAW_LABELS = [
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
]
TASK_LABELS = {
    "binary": ["normal", "adventitious"],
    "seven_class": RAW_LABELS,
    "narrow_four": ["normal", "crackle", "wheeze", "both"],
}
EXPECTED_TRAIN_RAW = {
    "Coarse Crackle": 49,
    "Fine Crackle": 912,
    "Normal": 5159,
    "Rhonchi": 39,
    "Stridor": 15,
    "Wheeze": 452,
    "Wheeze+Crackle": 30,
}
EXPECTED_INTER_RAW = {
    "Coarse Crackle": 3,
    "Fine Crackle": 80,
    "Normal": 1040,
    "Wheeze": 305,
    "Wheeze+Crackle": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_roots(result_root: Path, cache_root: Path) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != EXPERIMENT_ID or result.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}")
    if cache.name != EXPERIMENT_ID or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}")
    return result, cache


def clone_or_verify_author_repo(cache_root: Path) -> Path:
    repo = cache_root / "source" / "repo"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", AUTHOR_REPO_URL, str(repo)], check=True)
    subprocess.run(["git", "checkout", "--detach", AUTHOR_REPO_COMMIT], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if commit != AUTHOR_REPO_COMMIT or status:
        raise RuntimeError("author source identity/status gate failed")
    return repo


def checkpoint_gate(path: Path) -> None:
    if path.stat().st_size != CHECKPOINT_SIZE_BYTES or sha256_file(path) != CHECKPOINT_SHA256:
        raise RuntimeError("Patch-Mix author task checkpoint identity mismatch")


def event_rows(
    json_dir: Path, wav_dir: Path, partition: str, include_labels: bool
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for annotation_path in sorted(json_dir.glob("*.json")):
        recording_id = annotation_path.stem
        audio_path = wav_dir / f"{recording_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        payload = json.loads(annotation_path.read_text())
        for index, event in enumerate(payload.get("event_annotation", [])):
            start_ms, end_ms = int(event["start"]), int(event["end"])
            if start_ms < 0 or end_ms <= start_ms:
                raise RuntimeError(f"invalid event boundary: {annotation_path} {event}")
            row: dict[str, object] = {
                "event_id": f"{partition}:{recording_id}:event_{index:03d}",
                "partition": partition,
                "recording_id": recording_id,
                "patient_id": recording_id.split("_", 1)[0],
                "event_index": index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "audio_path": str(audio_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
            }
            if include_labels:
                raw = str(event["type"])
                if raw not in RAW_LABELS:
                    raise RuntimeError(f"unknown train event label: {raw}")
                row["raw_label"] = raw
            rows.append(row)
    rows.sort(key=lambda row: str(row["event_id"]))
    if len(rows) != len({str(row["event_id"]) for row in rows}):
        raise RuntimeError(f"duplicate {partition} event IDs")
    return rows


def build_manifest(dataset_root: Path) -> list[dict[str, object]]:
    root = resolve_biocas_root(dataset_root)
    train = event_rows(root / "train2022_json", root / "train2022_wav", "train", True)
    inter = event_rows(
        root / "test2022_json" / "inter_test_json", root / "test2022_wav", "inter", False
    )
    if len(train) != 6_656 or dict(sorted(Counter(row["raw_label"] for row in train).items())) != EXPECTED_TRAIN_RAW:
        raise RuntimeError("SPRSound train count/support gate failed")
    if len(inter) != 1_429 or id_sha256([str(row["event_id"]) for row in inter]) != EXPECTED_ID_SHA256:
        raise RuntimeError("SPRSound inter ID contract gate failed")
    train_patients = {str(row["patient_id"]) for row in train}
    inter_patients = {str(row["patient_id"]) for row in inter}
    if len(train_patients) != 243 or len(inter_patients) != 41 or train_patients & inter_patients:
        raise RuntimeError("official train/inter patient disjointness gate failed")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = np.asarray([str(row["raw_label"]) for row in train])
    groups = np.asarray([str(row["patient_id"]) for row in train])
    train_indices, validation_indices = next(splitter.split(np.arange(len(train)), y, groups))
    validation_set = set(validation_indices.tolist())
    for index, row in enumerate(train):
        row["inner_split"] = "validation" if index in validation_set else "subtrain"
    subtrain_patients = {str(train[index]["patient_id"]) for index in train_indices}
    validation_patients = {str(train[index]["patient_id"]) for index in validation_indices}
    if (
        len(train_indices),
        len(validation_indices),
        len(subtrain_patients),
        len(validation_patients),
        len(subtrain_patients & validation_patients),
    ) != (5_219, 1_437, 194, 49, 0):
        raise RuntimeError("fixed C0-compatible patient split drift")
    return train + inter


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines()]
        if existing != rows:
            raise RuntimeError("existing event manifest differs from current data contract")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def profile_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    train = [row for row in rows if row["partition"] == "train"]
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train:
        buckets[f"{row['inner_split']}::{row['raw_label']}"].append(row)
    selected: list[dict[str, object]] = []
    while len(selected) < PROFILE_EVENTS:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < PROFILE_EVENTS:
                selected.append(buckets[key].pop(0))
    return sorted(selected, key=lambda row: str(row["event_id"]))


def preprocess_recording(rows: list[dict[str, object]], author_repo: Path) -> list[torch.Tensor]:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    waveform, sample_rate = torchaudio.load(str(rows[0]["audio_path"]))
    waveform = waveform.mean(dim=0, keepdim=True)
    source_duration_ms = waveform.shape[1] / sample_rate * 1000
    if sample_rate != 16_000:
        waveform = audio_transforms.Resample(sample_rate, 16_000)(waveform)
    waveform = audio_transforms.Fade(
        fade_in_len=1_000, fade_out_len=1_000, fade_shape="linear"
    )(waveform)
    args = argparse.Namespace(sample_rate=16_000, desired_length=8, pad_types="repeat")
    resize = transforms.Resize((798, 128))
    images = []
    for row in rows:
        if float(row["end_ms"]) > source_duration_ms + 1:
            raise RuntimeError(f"event exceeds recording: {row['event_id']}")
        start = int(float(row["start_ms"]) / 1000 * 16_000)
        end = int(float(row["end_ms"]) / 1000 * 16_000)
        event = cut_pad_sample_torchaudio(waveform[:, start:end], args)
        fbank = generate_fbank(event, 16_000, n_mels=128)
        image = resize(transforms.ToTensor()(fbank)).to(torch.float32)
        if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
            raise RuntimeError(f"invalid fbank: {row['event_id']}")
        images.append(image)
    return images


def build_encoder(author_repo: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(author_repo))
    from models.ast import ASTModel

    checkpoint = restricted_torch_load(checkpoint_path)
    model = ASTModel(
        label_dim=4,
        fstride=10,
        tstride=10,
        input_fdim=798,
        input_tdim=128,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        model_size="base384",
        verbose=False,
        mix_beta=1.0,
    )
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"encoder state mismatch: {result}")
    model.mlp_head = torch.nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("encoder freeze gate failed")
    return model.to(device).eval()


def extract_embeddings(
    rows: list[dict[str, object]],
    author_repo: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording_id"])].append(row)
    embeddings: dict[str, np.ndarray] = {}
    pending_images: list[torch.Tensor] = []
    pending_ids: list[str] = []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_images:
            return
        with torch.inference_mode():
            values = encoder(torch.stack(pending_images).to(device))
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise RuntimeError(f"invalid Patch-Mix embedding batch: {tuple(values.shape)}")
        for event_id, value in zip(pending_ids, values.cpu().numpy()):
            embeddings[event_id] = value.astype(np.float32, copy=False)
        pending_images.clear()
        pending_ids.clear()

    for recording_id in sorted(grouped):
        recording_rows = grouped[recording_id]
        for row, image in zip(recording_rows, preprocess_recording(recording_rows, author_repo)):
            pending_ids.append(str(row["event_id"]))
            pending_images.append(image)
            if len(pending_images) == batch_size:
                flush()
    flush()
    runtime = time.perf_counter() - started
    ordered = np.stack([embeddings[str(row["event_id"])] for row in rows])
    if ordered.shape != (len(rows), 768) or not np.isfinite(ordered).all():
        raise RuntimeError("embedding coverage/finite gate failed")
    return ordered, runtime


def raw_to_task(raw: str, task: str) -> str | None:
    if task == "binary":
        return "normal" if raw == "Normal" else "adventitious"
    if task == "seven_class":
        return raw
    if task == "narrow_four":
        return {
            "Normal": "normal",
            "Fine Crackle": "crackle",
            "Coarse Crackle": "crackle",
            "Wheeze": "wheeze",
            "Wheeze+Crackle": "both",
        }.get(raw)
    raise ValueError(task)


class LinearTargetHead(torch.nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.normalization = torch.nn.LayerNorm(768)
        self.classifier = torch.nn.Linear(768, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.normalization(inputs))


def score_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], task: str
) -> tuple[dict[str, object], np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    specificity = float(recall[0] * 100)
    abnormal_total = int(matrix[1:].sum())
    sensitivity = float(np.trace(matrix[1:, 1:]) / abnormal_total * 100)
    average_score = (specificity + sensitivity) / 2
    harmonic_score = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    composite = (average_score + harmonic_score) / 2
    payload: dict[str, object] = {
        "specificity_percent": specificity,
        "sensitivity_percent": sensitivity,
        "average_score_percent": average_score,
        "harmonic_score_percent": harmonic_score,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    if task in {"binary", "seven_class"}:
        payload["official_sprsound_score_percent"] = composite
        payload["metric_scope"] = f"official SPRSound {'Task 1-1' if task == 'binary' else 'Task 1-2'}"
    else:
        payload["narrow4_icbhi_as_shared_ontology_diagnostic_percent"] = average_score
        payload["narrow4_mean_as_hs_descriptive_percent"] = composite
        payload["metric_scope"] = "shared-ontology diagnostic; not official SPRSound Task 1-2 Score"
        payload["both_support_warning"] = "inter Both support=1; no minority conclusion"
    return payload, matrix


def labels_for_rows(
    rows: list[dict[str, object]], task: str
) -> tuple[list[dict[str, object]], np.ndarray]:
    labels = TASK_LABELS[task]
    included = []
    indices = []
    for row in rows:
        mapped = raw_to_task(str(row["raw_label"]), task)
        if mapped is not None:
            included.append(row)
            indices.append(labels.index(mapped))
    return included, np.asarray(indices, dtype=np.int64)


def train_head(
    task: str,
    rows: list[dict[str, object]],
    embeddings: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> tuple[LinearTargetHead, list[dict[str, object]], dict[str, object]]:
    labels = TASK_LABELS[task]
    row_index = {str(row["event_id"]): index for index, row in enumerate(rows)}
    task_rows, targets = labels_for_rows(rows, task)
    sub_indices = np.asarray(
        [row_index[str(row["event_id"])] for row in task_rows if row["inner_split"] == "subtrain"]
    )
    val_indices = np.asarray(
        [row_index[str(row["event_id"])] for row in task_rows if row["inner_split"] == "validation"]
    )
    sub_targets = targets[
        np.asarray([row["inner_split"] == "subtrain" for row in task_rows], dtype=bool)
    ]
    val_targets = targets[
        np.asarray([row["inner_split"] == "validation" for row in task_rows], dtype=bool)
    ]
    torch.manual_seed(SEED)
    head = LinearTargetHead(len(labels)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(embeddings[sub_indices])),
        torch.from_numpy(sub_targets),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, generator=generator, num_workers=0
    )
    validation_x = torch.from_numpy(np.asarray(embeddings[val_indices])).to(device)
    history: list[dict[str, object]] = []
    best_value = -math.inf
    best_state = None
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, 6):
        head.train()
        losses = []
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(inputs)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {task} loss")
            loss.backward()
            if not all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ):
                raise RuntimeError(f"non-finite {task} head gradient")
            optimizer.step()
            losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
        validation_pred = validation_logits.argmax(dim=1).cpu().numpy()
        metrics, _ = score_metrics(val_targets, validation_pred, labels, task)
        selection_value = float(
            metrics["narrow4_icbhi_as_shared_ontology_diagnostic_percent"]
            if task == "narrow_four"
            else metrics["official_sprsound_score_percent"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_selection_metric": selection_value,
            }
        )
        if selection_value > best_value:
            best_value = selection_value
            best_state = deepcopy(head.state_dict())
            best_metrics = metrics
    runtime = time.perf_counter() - started
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"no best head selected for {task}")
    head.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "history.csv", history)
    np.savez_compressed(
        output_dir / "best_head.npz",
        normalization_weight=best_state["normalization.weight"].cpu().numpy(),
        normalization_bias=best_state["normalization.bias"].cpu().numpy(),
        classifier_weight=best_state["classifier.weight"].cpu().numpy(),
        classifier_bias=best_state["classifier.bias"].cpu().numpy(),
    )
    receipt = {
        "best_epoch": int(max(history, key=lambda row: row["validation_selection_metric"])["epoch"]),
        "best_validation_metrics": best_metrics,
        "runtime_seconds": runtime,
        "head": "LayerNorm(768)+Linear; random target initialization; no source classifier reuse",
        "optimizer": "Adam(lr=1e-3), unweighted CrossEntropyLoss",
        "epochs_completed": 5,
        "subtrain_rows": len(sub_indices),
        "validation_rows": len(val_indices),
    }
    return head.eval(), history, receipt


def peak_rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_value = value if sys.platform == "darwin" else value * 1024
    return float(bytes_value / 1024**3)


def save_embedding_cache(path: Path, rows: list[dict[str, object]], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            event_ids=np.asarray([str(row["event_id"]) for row in rows]),
            embeddings=embeddings,
        )
    temporary.replace(path)


def load_inter_labels(rows: list[dict[str, object]]) -> dict[str, str]:
    labels = {}
    for row in rows:
        payload = json.loads(Path(str(row["annotation_path"])).read_text())
        raw = str(payload["event_annotation"][int(row["event_index"])]["type"])
        if raw not in RAW_LABELS:
            raise RuntimeError(f"unknown inter event label: {raw}")
        labels[str(row["event_id"])] = raw
    if dict(sorted(Counter(labels.values()).items())) != EXPECTED_INTER_RAW:
        raise RuntimeError("inter scoring-label support drift")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["profile", "full"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.device != "cpu":
        raise ValueError("this bounded local pilot is preregistered for CPU")
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
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
    checkpoint = args.checkpoint.resolve()
    checkpoint_gate(checkpoint)
    author_repo = clone_or_verify_author_repo(cache_root)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rows = build_manifest(args.dataset_root)
    write_manifest(result_root / "event_manifest.jsonl", rows)
    train_rows = [row for row in rows if row["partition"] == "train"]
    inter_rows = [row for row in rows if row["partition"] == "inter"]
    profile_manifest = None
    if args.phase == "full":
        profile_manifest = json.loads((result_root / "run_manifest.json").read_text())
        if not profile_manifest["runtime_gate"]["passed"]:
            raise RuntimeError("full run forbidden because the local profile gate did not pass")

    model_started = time.perf_counter()
    encoder = build_encoder(author_repo, checkpoint, device)
    model_load_seconds = time.perf_counter() - model_started
    selected_rows = profile_rows(rows) if args.phase == "profile" else rows
    embeddings, extraction_seconds = extract_embeddings(
        selected_rows, author_repo, encoder, device, args.batch_size
    )
    del encoder
    save_embedding_cache(
        cache_root / f"{args.phase}_embeddings.npz", selected_rows, embeddings
    )
    feature_receipt = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
        "author_repo_commit": AUTHOR_REPO_COMMIT,
        "encoder_frozen": True,
        "source_classifier_loaded_or_reused": False,
        "embedding_shape": list(embeddings.shape),
        "embedding_finite": bool(np.isfinite(embeddings).all()),
        "model_load_seconds": model_load_seconds,
        "extraction_seconds": extraction_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "cache_path": str(cache_root / f"{args.phase}_embeddings.npz"),
        "cache_sha256": sha256_file(cache_root / f"{args.phase}_embeddings.npz"),
    }
    if args.phase == "profile":
        profile_training_seconds = 0.0
        profile_train = selected_rows
        for task in TASK_LABELS:
            _, _, task_receipt = train_head(
                task,
                profile_train,
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
            "passed": projected_total <= MAX_PROJECTED_SECONDS and peak_rss_gib() <= MAX_RSS_GIB,
        }
        write_json(
            result_root / "run_manifest.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "protocol": PROTOCOL,
                "phase": "profile",
                "data": {
                    "train_events": len(train_rows),
                    "inter_events": len(inter_rows),
                    "subtrain_events": sum(row["inner_split"] == "subtrain" for row in train_rows),
                    "validation_events": sum(row["inner_split"] == "validation" for row in train_rows),
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
    heads = {}
    task_receipts = {}
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

    label_free_rows: list[dict[str, object]] = []
    inter_tensor = torch.from_numpy(inter_embeddings).to(device)
    task_outputs = {}
    with torch.inference_mode():
        for task, head in heads.items():
            logits = head(inter_tensor).cpu().numpy()
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            task_outputs[task] = (logits, probabilities)
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
                output[f"{task}_logit_{safe_label}"] = float(logits[row_index, label_index])
                output[f"{task}_prob_{safe_label}"] = float(probabilities[row_index, label_index])
        label_free_rows.append(output)
    write_csv(result_root / "inter_predictions_label_free.csv", label_free_rows)

    scoring_labels = load_inter_labels(inter_rows)
    scored_rows = []
    metrics_payload = {}
    for row in label_free_rows:
        raw = scoring_labels[str(row["event_id"])]
        scored_rows.append({**row, "raw_label": raw})
    for task, labels in TASK_LABELS.items():
        included = [
            row for row in scored_rows if raw_to_task(str(row["raw_label"]), task) is not None
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
    metrics_payload["binary"]["comparators"] = {
        "frozen_patch_mix_b0_score_percent": 59.38,
        "exact_inter_native_sprsound_baseline_score_percent": 77.42,
        "pilot_minus_b0_percentage_points": float(
            metrics_payload["binary"]["official_sprsound_score_percent"] - 59.38
        ),
        "pilot_minus_native_baseline_percentage_points": float(
            metrics_payload["binary"]["official_sprsound_score_percent"] - 77.42
        ),
        "claim_limit": "descriptive pilot comparison; not SOTA or degradation significance",
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
    print(json.dumps({"total_runtime_seconds": total_runtime, "peak_rss_gib": peak_rss_gib()}, indent=2))


if __name__ == "__main__":
    main()
