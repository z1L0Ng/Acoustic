"""Protocol, data, preprocessing, model, and artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torchaudio import transforms as audio_transforms
from torchvision import transforms

from acoustic.evaluation.sprsound_inter import id_sha256, resolve_biocas_root


EXPERIMENT_ID = "icbhi_sprsound_shared_encoder_native_heads"
PROTOCOL_VERSION = "shared_encoder_native_heads_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ICBHI_MANIFEST = REPO_ROOT / "dataset/processed/manifests/icbhi_2017_cycles.csv"
AUTHOR_REPO_URL = "https://github.com/raymin0223/patch-mix_contrastive_learning"
AUTHOR_REPO_COMMIT = "836b09fea1b70eb29fe0b25afa481286b56f5104"
AST_URL = "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
AST_SHA256 = "dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f"
AST_SIZE_BYTES = 352_587_836
HF_AST_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"
HF_AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
HF_AST_FILENAME = "model.safetensors"
HF_AST_SHA256 = "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d"
HF_AST_SIZE_BYTES = 346_404_948
HF_AST_LEGACY_CANONICAL_SHA256 = (
    "7fdbccf6986cdc372fc0ed90b2d39584cdabb21ed69efe550d495fe2d3b0cf85"
)
SPRSOUND_COMMIT = "874eeb8736ddb78937c2fb5332fc7e7293d0f0ca"
ICBHI_VALIDATION_SEED = 20260712
SPR_VALIDATION_SEED = 20260722
TRAINING_SEED = 20260728
ICBHI_LABELS = ["normal", "crackle", "wheeze", "both"]
SPR_LABELS = [
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
]
TASK_LABELS = {
    "icbhi_flat4": ICBHI_LABELS,
    "spr_binary": ["normal", "adventitious"],
    "spr_seven": SPR_LABELS,
}
EXPECTED_ICBHI_LABELS = {"normal": 3642, "crackle": 1864, "wheeze": 886, "both": 506}
EXPECTED_SPR_SUPPORT = {
    "train": {
        "Coarse Crackle": 49,
        "Fine Crackle": 912,
        "Normal": 5159,
        "Rhonchi": 39,
        "Stridor": 15,
        "Wheeze": 452,
        "Wheeze+Crackle": 30,
    },
    "inter": {
        "Coarse Crackle": 3,
        "Fine Crackle": 80,
        "Normal": 1040,
        "Wheeze": 305,
        "Wheeze+Crackle": 1,
    },
    "intra": {
        "Coarse Crackle": 14,
        "Fine Crackle": 175,
        "Normal": 688,
        "Rhonchi": 14,
        "Stridor": 2,
        "Wheeze": 108,
        "Wheeze+Crackle": 3,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_digest(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_roots(result_root: Path, cache_root: Path) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != EXPERIMENT_ID or result.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}")
    if cache.name != EXPERIMENT_ID or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}")
    return result, cache


def discover_icbhi_audio_dir(dataset_root: Path) -> Path:
    candidates: dict[Path, int] = {}
    for wav in dataset_root.rglob("*.wav"):
        candidates[wav.parent] = candidates.get(wav.parent, 0) + 1
    valid = []
    for parent, wav_count in candidates.items():
        annotation_count = sum(
            1 for path in parent.glob("*.txt") if len(path.stem.split("_")) >= 5
        )
        if wav_count == 920 and annotation_count == 920:
            valid.append(parent.resolve())
    if len(valid) != 1:
        raise RuntimeError(f"expected one ICBHI 920-WAV directory, found {valid}")
    return valid[0]


def load_icbhi_rows(dataset_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = pd.read_csv(ICBHI_MANIFEST, dtype={"patient_id": str, "cycle_id": str})
    if (
        len(frame) != 6898
        or not frame["cycle_id"].is_unique
        or frame["official_split"].value_counts().to_dict() != {"train": 4142, "test": 2756}
        or frame["native_four_class_label"].value_counts().to_dict() != EXPECTED_ICBHI_LABELS
    ):
        raise RuntimeError("ICBHI manifest count/ID/label gate failed")
    audio_dir = discover_icbhi_audio_dir(dataset_root)
    frame["audio_path"] = frame["recording_id"].map(
        lambda value: str((audio_dir / f"{value}.wav").resolve())
    )
    if not all(Path(value).is_file() for value in frame["audio_path"]):
        raise RuntimeError("ICBHI manifest cannot be joined to raw WAVs")
    train = frame[frame["official_split"].eq("train")].reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=ICBHI_VALIDATION_SEED
    )
    subtrain_index, validation_index = next(
        splitter.split(train, train["native_four_class_label"], train["patient_id"])
    )
    validation_ids = set(train.iloc[validation_index]["cycle_id"])
    frame["partition"] = np.where(
        frame["official_split"].eq("test"),
        "test",
        np.where(frame["cycle_id"].isin(validation_ids), "validation", "subtrain"),
    )
    subtrain_patients = set(frame.loc[frame["partition"].eq("subtrain"), "patient_id"])
    validation_patients = set(frame.loc[frame["partition"].eq("validation"), "patient_id"])
    official_train_patients = set(frame.loc[frame["official_split"].eq("train"), "patient_id"])
    official_test_patients = set(frame.loc[frame["official_split"].eq("test"), "patient_id"])
    if (
        frame["partition"].value_counts().to_dict()
        != {"subtrain": 3055, "test": 2756, "validation": 1087}
        or subtrain_patients & validation_patients
        or official_train_patients & official_test_patients != {"156", "218"}
    ):
        raise RuntimeError("ICBHI grouped-validation or official-overlap gate failed")
    rows = frame.to_dict("records")
    receipt = {
        "prediction_unit": "annotated respiratory cycle",
        "manifest": str(ICBHI_MANIFEST),
        "manifest_sha256": sha256_file(ICBHI_MANIFEST),
        "rows": len(rows),
        "unique_cycle_ids": int(frame["cycle_id"].nunique()),
        "official_split": dict(sorted(frame["official_split"].value_counts().items())),
        "partition": dict(sorted(frame["partition"].value_counts().items())),
        "label_support": dict(sorted(frame["native_four_class_label"].value_counts().items())),
        "subtrain_patients": len(subtrain_patients),
        "validation_patients": len(validation_patients),
        "subtrain_validation_patient_overlap": 0,
        "official_train_test_patient_overlap": ["156", "218"],
        "split_caveat": "official recording split is literature-comparable but not patient-independent",
        "validation": "StratifiedGroupKFold fold 0 inside official train; patient_id; seed 20260712",
    }
    return rows, receipt


def spr_event_rows(
    json_dir: Path,
    wav_dir: Path,
    partition: str,
    include_labels: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for annotation_path in sorted(json_dir.glob("*.json")):
        recording_id = annotation_path.stem
        audio_path = wav_dir / f"{recording_id}.wav"
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        payload = json.loads(annotation_path.read_text())
        for event_index, event in enumerate(payload.get("event_annotation", [])):
            start_ms, end_ms = int(event["start"]), int(event["end"])
            if start_ms < 0 or end_ms <= start_ms:
                raise RuntimeError(f"invalid SPRSound event boundary: {annotation_path}")
            row: dict[str, object] = {
                "event_id": f"{partition}:{recording_id}:event_{event_index:03d}",
                "recording_id": recording_id,
                "patient_id": recording_id.split("_", 1)[0],
                "event_index": event_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "audio_path": str(audio_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
                "partition": partition,
            }
            if include_labels:
                raw = str(event["type"])
                if raw not in SPR_LABELS:
                    raise RuntimeError(f"unknown SPRSound event label: {raw}")
                row["raw_label"] = raw
            rows.append(row)
    rows.sort(key=lambda row: str(row["event_id"]))
    if len(rows) != len({str(row["event_id"]) for row in rows}):
        raise RuntimeError(f"duplicate SPRSound IDs in {partition}")
    return rows


def load_spr_rows(dataset_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = resolve_biocas_root(dataset_root)
    if SPRSOUND_COMMIT not in str(root):
        raise RuntimeError(f"SPRSound root is not the pinned {SPRSOUND_COMMIT} source snapshot")
    train = spr_event_rows(root / "train2022_json", root / "train2022_wav", "train", True)
    inter = spr_event_rows(
        root / "test2022_json/inter_test_json", root / "test2022_wav", "inter", False
    )
    intra = spr_event_rows(
        root / "test2022_json/intra_test_json", root / "test2022_wav", "intra", False
    )
    scoring_support: dict[str, dict[str, int]] = {}
    for partition, rows in (("inter", inter), ("intra", intra)):
        raw = []
        for row in rows:
            payload = json.loads(Path(str(row["annotation_path"])).read_text())
            raw.append(str(payload["event_annotation"][int(row["event_index"])]["type"]))
        scoring_support[partition] = dict(sorted(Counter(raw).items()))
    if (
        len(train) != 6656
        or len(inter) != 1429
        or len(intra) != 1004
        or dict(sorted(Counter(str(row["raw_label"]) for row in train).items()))
        != EXPECTED_SPR_SUPPORT["train"]
        or scoring_support != {
            "inter": EXPECTED_SPR_SUPPORT["inter"],
            "intra": EXPECTED_SPR_SUPPORT["intra"],
        }
    ):
        raise RuntimeError("SPRSound row/support gate failed")
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SPR_VALIDATION_SEED
    )
    y = np.asarray([str(row["raw_label"]) for row in train])
    groups = np.asarray([str(row["patient_id"]) for row in train])
    subtrain_index, validation_index = next(splitter.split(np.arange(len(train)), y, groups))
    validation_set = set(validation_index.tolist())
    for index, row in enumerate(train):
        row["partition"] = "validation" if index in validation_set else "subtrain"
    subtrain_patients = {str(train[index]["patient_id"]) for index in subtrain_index}
    validation_patients = {str(train[index]["patient_id"]) for index in validation_index}
    train_patients = subtrain_patients | validation_patients
    inter_patients = {str(row["patient_id"]) for row in inter}
    intra_patients = {str(row["patient_id"]) for row in intra}
    train_archive_patients = {
        path.stem.split("_", 1)[0] for path in (root / "train2022_json").glob("*.json")
    }
    intra_archive_patients = {
        path.stem.split("_", 1)[0]
        for path in (root / "test2022_json/intra_test_json").glob("*.json")
    }
    if (
        (len(subtrain_index), len(validation_index)) != (5219, 1437)
        or (len(subtrain_patients), len(validation_patients)) != (194, 49)
        or subtrain_patients & validation_patients
        or train_patients & inter_patients
        or len(train_patients & intra_patients) != 156
        or len(train_archive_patients & intra_archive_patients) != 162
    ):
        raise RuntimeError("SPRSound grouped split/overlap gate failed")
    receipt = {
        "source_commit": SPRSOUND_COMMIT,
        "prediction_unit": "official BioCAS2022 event",
        "train_events": len(train),
        "subtrain_events": len(subtrain_index),
        "validation_events": len(validation_index),
        "inter_events": len(inter),
        "intra_events": len(intra),
        "subtrain_patients": len(subtrain_patients),
        "validation_patients": len(validation_patients),
        "subtrain_validation_patient_overlap": 0,
        "train_inter_patient_overlap": 0,
        "train_intra_event_bearing_patient_overlap": 156,
        "train_intra_archive_patient_overlap": 162,
        "train_support": EXPECTED_SPR_SUPPORT["train"],
        "inter_support_audit_only": EXPECTED_SPR_SUPPORT["inter"],
        "intra_support_audit_only": EXPECTED_SPR_SUPPORT["intra"],
        "inter_ordered_id_sha256": id_sha256([str(row["event_id"]) for row in inter]),
        "intra_ordered_id_sha256": id_digest([str(row["event_id"]) for row in intra]),
        "validation": "StratifiedGroupKFold fold 0 inside official train; patient_id; seed 20260722",
        "test_policy": "inter primary; intra separate repeated-subject diagnostic; never pooled",
        "test_manifests_are_label_free": True,
    }
    return train + inter + intra, receipt


def bootstrap_assets(cache_root: Path) -> dict[str, object]:
    source = cache_root / "source" / "repo"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", AUTHOR_REPO_URL, str(source)], check=True)
    subprocess.run(["git", "checkout", "--detach", AUTHOR_REPO_COMMIT], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, check=True, capture_output=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    if commit != AUTHOR_REPO_COMMIT or status:
        raise RuntimeError("Patch-Mix source pin/cleanliness gate failed")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf_hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "hf_xet"))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from baseline.patch_mix_cl.convert_hf_ast_checkpoint import convert_state_dict

    hf_source = Path(
        hf_hub_download(
            HF_AST_REPO,
            HF_AST_FILENAME,
            revision=HF_AST_REVISION,
            cache_dir=cache_root / "hf_hub",
        )
    )
    if hf_source.stat().st_size != HF_AST_SIZE_BYTES or sha256_file(hf_source) != HF_AST_SHA256:
        raise RuntimeError("pinned HF AudioSet source identity mismatch")
    checkpoint = cache_root / "checkpoints/hf_ast_legacy_compat.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        converted = convert_state_dict(load_file(hf_source))
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(converted, temporary)
        temporary.replace(checkpoint)
    canonical = legacy_checkpoint_canonical_sha256(checkpoint)
    if canonical != HF_AST_LEGACY_CANONICAL_SHA256:
        raise RuntimeError("HF-to-Patch-Mix compatibility conversion tensor gate failed")
    return {
        "author_repo": str(source.resolve()),
        "author_repo_url": AUTHOR_REPO_URL,
        "author_repo_commit": commit,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_source": HF_AST_REPO,
        "checkpoint_revision": HF_AST_REVISION,
        "checkpoint_source_path": str(hf_source.resolve()),
        "checkpoint_source_size_bytes": hf_source.stat().st_size,
        "checkpoint_source_sha256": sha256_file(hf_source),
        "checkpoint_url": f"https://huggingface.co/{HF_AST_REPO}/tree/{HF_AST_REVISION}",
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_canonical_tensor_sha256": canonical,
        "identity_boundary": (
            "pinned MIT/Hugging Face AudioSet checkpoint converted to the Patch-Mix legacy "
            "key layout; the 155 model-used tensors were previously verified equivalent to "
            "the then-served author artifacts; not original serialized-byte identity; "
            "no ICBHI task checkpoint used"
        ),
    }


def legacy_checkpoint_canonical_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or len(state) != 155:
        raise RuntimeError("legacy compatibility checkpoint must contain 155 tensors")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"non-tensor legacy checkpoint value: {key}")
        tensor_sha = hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        line = json.dumps(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": tensor_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def preprocessing_args() -> SimpleNamespace:
    return SimpleNamespace(sample_rate=16000, desired_length=8, pad_types="repeat")


def preprocess_rows(rows: list[dict[str, object]], author_repo: Path) -> torch.Tensor:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["audio_path"])].append(row)
    output: dict[str, torch.Tensor] = {}
    resize = transforms.Resize((798, 128), antialias=False)
    args = preprocessing_args()
    for audio_path in sorted(grouped):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = audio_transforms.Resample(sample_rate, 16000)(waveform)
        waveform = audio_transforms.Fade(
            fade_in_len=1000, fade_out_len=1000, fade_shape="linear"
        )(waveform)
        for row in grouped[audio_path]:
            if "cycle_start_s" in row:
                start = int(float(row["cycle_start_s"]) * 16000)
                end = int(float(row["cycle_end_s"]) * 16000)
                row_id = str(row["cycle_id"])
            else:
                start = int(float(row["start_ms"]) / 1000 * 16000)
                end = int(float(row["end_ms"]) / 1000 * 16000)
                row_id = str(row["event_id"])
            segment = waveform[:, min(start, waveform.shape[1]) : min(end, waveform.shape[1])]
            if segment.shape[-1] == 0:
                raise RuntimeError(f"empty segment: {row_id}")
            segment = cut_pad_sample_torchaudio(segment, args)
            fbank = generate_fbank(segment, 16000, n_mels=128)
            image = resize(transforms.ToTensor()(fbank)).to(torch.float32)
            if image.shape != (1, 798, 128) or not torch.isfinite(image).all():
                raise RuntimeError(f"invalid fbank: {row_id}")
            output[row_id] = image
    key = "cycle_id" if "cycle_id" in rows[0] else "event_id"
    return torch.stack([output[str(row[key])] for row in rows])


class SharedEncoderNativeHeads(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, len(labels)))
                for name, labels in TASK_LABELS.items()
            }
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(images)
        if embeddings.ndim != 2 or embeddings.shape[1] != 768:
            raise RuntimeError(f"unexpected shared embedding shape: {tuple(embeddings.shape)}")
        return embeddings

    def routed_logits(
        self, images: torch.Tensor, dataset_ids: list[str]
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        embeddings = self.encode(images)
        dataset_array = np.asarray(dataset_ids)
        masks = {
            "icbhi_flat4": torch.as_tensor(dataset_array == "icbhi", device=images.device),
            "spr_binary": torch.as_tensor(dataset_array == "sprsound", device=images.device),
            "spr_seven": torch.as_tensor(dataset_array == "sprsound", device=images.device),
        }
        return embeddings, {
            task: (mask, self.heads[task](embeddings[mask]))
            for task, mask in masks.items()
            if bool(mask.any())
        }


def build_model(author_repo: Path, checkpoint: Path, work_dir: Path, device: torch.device):
    if legacy_checkpoint_canonical_sha256(checkpoint) != HF_AST_LEGACY_CANONICAL_SHA256:
        raise RuntimeError("neutral AudioSet compatibility initialization mismatch")
    sys.path.insert(0, str(author_repo))
    from models.ast import ASTModel

    pretrained = work_dir / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    expected = pretrained / "audioset_10_10_0.4593.pth"
    if expected.exists() or expected.is_symlink():
        expected.unlink()
    expected.symlink_to(checkpoint.resolve())
    previous = Path.cwd()
    os.chdir(work_dir)
    try:
        encoder = ASTModel(
            label_dim=4,
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
    finally:
        os.chdir(previous)
    encoder.mlp_head = torch.nn.Identity()
    model = SharedEncoderNativeHeads(encoder).to(device)
    return model


def labels_for_rows(rows: list[dict[str, object]], dataset: str) -> dict[str, torch.Tensor]:
    if dataset == "icbhi":
        return {
            "icbhi_flat4": torch.tensor(
                [ICBHI_LABELS.index(str(row["native_four_class_label"])) for row in rows],
                dtype=torch.long,
            )
        }
    raw = [str(row["raw_label"]) for row in rows]
    return {
        "spr_binary": torch.tensor([0 if value == "Normal" else 1 for value in raw], dtype=torch.long),
        "spr_seven": torch.tensor([SPR_LABELS.index(value) for value in raw], dtype=torch.long),
    }


def routed_loss(
    model: SharedEncoderNativeHeads,
    images: torch.Tensor,
    dataset_ids: list[str],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, object]]:
    _, outputs = model.routed_logits(images, dataset_ids)
    expected_tasks = {
        "icbhi_flat4" if "icbhi" in dataset_ids else None,
        "spr_binary" if "sprsound" in dataset_ids else None,
        "spr_seven" if "sprsound" in dataset_ids else None,
    } - {None}
    if set(outputs) != expected_tasks:
        raise RuntimeError(f"head routing mismatch: {set(outputs)} != {expected_tasks}")
    task_losses = {}
    task_rows = {}
    for task, (mask, logits) in outputs.items():
        target = targets[task].to(images.device)
        if len(target) != int(mask.sum()):
            raise RuntimeError(f"target/mask mismatch for {task}")
        value = torch.nn.functional.cross_entropy(logits, target)
        if not torch.isfinite(value):
            raise RuntimeError(f"non-finite loss for {task}")
        task_losses[task] = value
        task_rows[task] = int(mask.sum())
    dataset_losses = []
    dataset_weights = []
    if "icbhi_flat4" in task_losses:
        dataset_losses.append(task_losses["icbhi_flat4"])
        dataset_weights.append(task_rows["icbhi_flat4"])
    if "spr_binary" in task_losses:
        dataset_losses.append((task_losses["spr_binary"] + task_losses["spr_seven"]) / 2)
        dataset_weights.append(task_rows["spr_binary"])
    total = sum(value * weight for value, weight in zip(dataset_losses, dataset_weights)) / sum(
        dataset_weights
    )
    return total, {
        "task_losses": {key: float(value.detach()) for key, value in task_losses.items()},
        "task_rows": task_rows,
        "dataset_loss_policy": "ICBHI CE; SPR mean(binary CE, seven-class CE); sample-count weighted",
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> tuple[dict[str, object], np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(labels)), zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(labels)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(len(labels)),
                average="weighted",
                zero_division=0,
            )
        ),
        "uar": float(recall.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }, matrix


def peak_rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float((value if sys.platform == "darwin" else value * 1024) / 1024**3)


def parameter_receipt(model: SharedEncoderNativeHeads) -> dict[str, object]:
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    heads = {
        name: sum(parameter.numel() for parameter in head.parameters())
        for name, head in model.heads.items()
    }
    return {
        "encoder_parameters": encoder,
        "head_parameters": heads,
        "total_parameters": encoder + sum(heads.values()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def protocol_receipt() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "scope": "minimal local controlled multi-dataset reference; not a paper reproduction",
        "encoder": "Patch-Mix-compatible AST base384 initialized from pinned MIT/HF AudioSet weights",
        "encoder_initialization_source_sha256": HF_AST_SHA256,
        "encoder_initialization_legacy_canonical_sha256": HF_AST_LEGACY_CANONICAL_SHA256,
        "forbidden_initialization": "any ICBHI task-selected checkpoint",
        "heads": {
            "icbhi_flat4": ICBHI_LABELS,
            "spr_binary": TASK_LABELS["spr_binary"],
            "spr_seven": SPR_LABELS,
        },
        "preprocessing": (
            "mono; resample 16 kHz; full-recording fade; annotated cycle/event crop; "
            "8 s truncate/repeat+fade; 128-bin Kaldi fbank; resize 798x128; no augmentation"
        ),
        "loss": "unweighted cross entropy; missing labels are never synthesized",
        "spr_head_loss": "mean of independent binary and seven-class CE on each SPR event batch",
        "sampling": {
            "primary": "naive source-proportional homogeneous batches",
            "epoch_definition": "one shuffled pass through each dataset subtrain batch list",
            "dataset_balanced": "interface reserved but not executed in this run",
        },
        "optimizer": "Adam(lr=5e-5, weight_decay=1e-6)",
        "batch_size": 8,
        "max_epochs": 50,
        "schedule": "cosine",
        "selection": (
            "minimum source-proportional validation CE, with SPR binary/seven CE averaged; "
            "official tests never used for selection"
        ),
        "reporting": "dataset-native metrics only; no cross-dataset raw Score aggregation",
        "test_policy": "ICBHI official test; SPR inter primary and intra separate diagnostic",
        "seed": TRAINING_SEED,
    }
