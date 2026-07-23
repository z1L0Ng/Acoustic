"""Shared C0 data, model, metric, and artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from acoustic.evaluation.sprsound_inter import (
    EXPECTED_EVENTS as EXPECTED_INTER_EVENTS,
    EXPECTED_ID_SHA256 as EXPECTED_INTER_ID_SHA256,
    id_sha256,
    resolve_biocas_root,
)


RAW_LABELS = [
    "Normal",
    "Fine Crackle",
    "Coarse Crackle",
    "Wheeze",
    "Wheeze+Crackle",
    "Rhonchi",
    "Stridor",
]
TASK_LABELS = {
    "binary_broad": ["normal", "abnormal"],
    "narrow_four": ["normal", "crackle", "wheeze", "both"],
}
NARROW_MAP = {
    "Normal": "normal",
    "Fine Crackle": "crackle",
    "Coarse Crackle": "crackle",
    "Wheeze": "wheeze",
    "Wheeze+Crackle": "both",
}
AUTHOR_REPO_URL = "https://github.com/raymin0223/patch-mix_contrastive_learning"
AUTHOR_REPO_COMMIT = "836b09fea1b70eb29fe0b25afa481286b56f5104"
AST_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
AST_SHA256 = "dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f"
AST_SIZE = 352_587_836
EXPERIMENT_ID = "sprsound_patchmix_target_training"
PROTOCOL_NAME = "c0_patch_mix_sprsound_target_native_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_result_root(path: Path) -> Path:
    root = path.resolve()
    if root.name != EXPERIMENT_ID or root.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}, got {root}")
    return root


def validate_cache_root(path: Path) -> Path:
    root = path.resolve()
    if root.name != EXPERIMENT_ID or root.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}, got {root}")
    return root


def load_run_manifest(result_root: Path) -> dict[str, object]:
    path = validate_result_root(result_root) / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def update_run_manifest(result_root: Path, section: str, payload: object) -> None:
    root = validate_result_root(result_root)
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {
        "experiment_id": EXPERIMENT_ID,
        "protocol": PROTOCOL_NAME,
    }
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("run manifest identity mismatch")
    manifest[section] = payload
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, manifest)
    temporary.replace(path)


def task_label(raw_label: str, task: str) -> str | None:
    if raw_label not in RAW_LABELS:
        return None
    if task == "binary_broad":
        return "normal" if raw_label == "Normal" else "abnormal"
    if task == "narrow_four":
        return NARROW_MAP.get(raw_label)
    raise ValueError(task)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def protocol_args() -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=16000,
        desired_length=8,
        pad_types="repeat",
        n_mels=128,
        specaug_policy="icbhi_ast_sup",
        specaug_mask="mean",
        negative_pair="all",
    )


def preprocess_event_rows(
    rows: list[dict[str, str]], author_repo: Path
) -> dict[str, np.ndarray]:
    """Apply the source waveform/fbank contract, grouped by recording."""
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    args = protocol_args()
    resize = transforms.Resize((798, 128))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)
    output: dict[str, np.ndarray] = {}
    for recording_id in sorted(grouped):
        recording_rows = grouped[recording_id]
        waveform, sample_rate = torchaudio.load(recording_rows[0]["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != args.sample_rate:
            waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
        fade_samples = args.sample_rate // 16
        waveform = audio_transforms.Fade(
            fade_in_len=fade_samples,
            fade_out_len=fade_samples,
            fade_shape="linear",
        )(waveform)
        duration_ms = waveform.shape[1] / args.sample_rate * 1000
        for row in recording_rows:
            if float(row["end_ms"]) > duration_ms + 1:
                raise RuntimeError(f"event exceeds recording: {row['event_id']}")
            start = int(float(row["start_ms"]) / 1000 * args.sample_rate)
            end = int(float(row["end_ms"]) / 1000 * args.sample_rate)
            event = cut_pad_sample_torchaudio(waveform[:, start:end], args)
            fbank = generate_fbank(event, args.sample_rate, n_mels=args.n_mels)
            image = resize(transforms.ToTensor()(fbank)).to(torch.float32)
            if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
                raise RuntimeError(f"invalid fbank: {row['event_id']} {tuple(image.shape)}")
            output[row["event_id"]] = image.numpy()
    return output


def load_cache(cache_dir: Path) -> tuple[np.ndarray, list[dict[str, str]], dict[str, int]]:
    array = np.load(cache_dir / "features.npy", mmap_mode="r")
    rows = read_csv(cache_dir / "index.csv")
    if len(rows) != len(array):
        raise RuntimeError(f"cache row mismatch: {cache_dir}")
    index = {row["event_id"]: int(row["feature_index"]) for row in rows}
    if len(index) != len(rows) or array.shape[1:] != (1, 798, 128) or not np.isfinite(array).all():
        raise RuntimeError(f"invalid cache: {cache_dir}")
    return array, rows, index


class CachedEventDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        cache: np.ndarray,
        index: dict[str, int],
        labels: list[str] | None,
        specaugment=None,
    ) -> None:
        self.rows = rows
        self.cache = cache
        self.index = index
        self.labels = labels
        self.specaugment = specaugment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        row = self.rows[item]
        image = torch.from_numpy(np.array(self.cache[self.index[row["event_id"]]], copy=True))
        if self.specaugment is not None:
            image = self.specaugment(image)
        if self.labels is None:
            return image, row["event_id"]
        return image, self.labels.index(row["task_label"]), row["event_id"]


def build_model(author_repo: Path, init_checkpoint: Path, n_classes: int, device: torch.device):
    """Build the author AST/Projector while retaining the author init loader."""
    sys.path.insert(0, str(author_repo))
    from models import Projector
    from models.ast import ASTModel

    if init_checkpoint.stat().st_size != AST_SIZE or sha256_file(init_checkpoint) != AST_SHA256:
        raise RuntimeError("AST initialization checkpoint identity mismatch")
    model = ASTModel(
        label_dim=n_classes,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=798,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
        verbose=False,
        mix_beta=1.0,
    )
    classifier = deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, 768)
    return model.to(device), classifier.to(device), projector.to(device)


def contrastive_loss(
    projection1: torch.Tensor,
    projection2: torch.Tensor,
    lam: float,
    index: torch.Tensor,
    temperature: float = 0.06,
) -> torch.Tensor:
    projection1 = torch.nn.functional.normalize(projection1)
    projection2 = torch.nn.functional.normalize(projection2)
    logits = projection2 @ projection1.T / temperature
    identity = torch.eye(projection1.shape[0], device=projection1.device)
    paired = torch.zeros_like(identity)
    paired[torch.arange(projection1.shape[0], device=projection1.device), index] = 1
    mask = lam * identity + (1 - lam) * paired
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    return -((mask * log_prob).sum(dim=1) / mask.sum(dim=1)).mean()


def ema_update(module: torch.nn.Module, previous: dict[str, torch.Tensor], beta: float) -> None:
    with torch.no_grad():
        current = module.state_dict()
        updated = {key: previous[key] * beta + current[key] * (1 - beta) for key in current}
        module.load_state_dict(updated)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> tuple[dict[str, object], np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum() * 100)
    sensitivity = float(np.trace(matrix[1:, 1:]) / matrix[1:, :].sum() * 100)
    predicted = np.bincount(y_pred, minlength=len(labels))
    metrics: dict[str, object] = {
        "specificity": specificity,
        "sensitivity": sensitivity,
        "icbhi_score": (specificity + sensitivity) / 2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "uar": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(class_f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, label in enumerate(labels)
        },
    }
    if labels == TASK_LABELS["narrow_four"]:
        metrics["both_recall"] = float(recall[3])
        metrics["both_support_warning"] = "inter Both support is one; no minority claim"
    if not all(math.isfinite(float(value)) for value in [specificity, sensitivity, *recall]):
        raise RuntimeError("non-finite metric")
    return metrics, matrix


def write_confusion(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    rows = [{"true/pred": label, **{pred: int(v) for pred, v in zip(labels, values)}} for label, values in zip(labels, matrix)]
    write_csv(path, rows)


def class_counts(rows: Iterable[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row[key] for row in rows).items()))
