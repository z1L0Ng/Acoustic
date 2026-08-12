"""Run Shared Respiratory Attributes Phase 1A on local cached features."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from baseline.common.data import load_manifest
from baseline.common.metrics import evaluate_predictions
from baseline.common.strict_patient_v3 import build_fold_assignments
from baseline.shared_encoder_native_heads.protocol import (
    ICBHI_LABELS,
    SPR_LABELS,
    load_icbhi_rows,
    load_spr_rows,
)
from model.shared_respiratory_attributes_phase1a.models import (
    ATTRIBUTES,
    CONDITIONS,
    Phase1AModel,
    condition_parameter_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "shared_respiratory_attributes_phase1a"
RESULT_ROOT = ROOT / "result/model/2026-07-30" / EXPERIMENT_ID
DATASET_ROOT = ROOT / "dataset/raw"
CACHE_PATH = (
    ROOT
    / ".cache/four_dataset_representation_attribution/r1_beats_as2m_audioset_only/embeddings.npz"
)
CACHE_SHA256 = "3b3798cc9d01dbdfa8168a1cd641d658eb2fd4553799e59b84b7aae7ad0f5a69"
DRAW_SEEDS = [20260730, 20260731, 20260732, 20260733, 20260734]
OPTIMIZATION_SEEDS = [20260712, 20260713, 20260714]
SHOT_LEVELS: list[int | str] = [4, 8, 16, 32, 64, "full"]
SOURCE_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 64
MAX_STEPS = 400
EVAL_EVERY = 20
PATIENCE_EVALS = 8
SMOKE_STEPS = 40
PROFILE_BOOTSTRAP_REPEATS = 0
FULL_BOOTSTRAP_REPEATS = 200
VALIDATION_SUPPORTED_SPR_LABELS = ["Normal", "Fine Crackle", "Wheeze"]
SPR_COMPATIBLE_LABELS = {
    "Normal",
    "Fine Crackle",
    "Coarse Crackle",
    "Wheeze",
    "Wheeze+Crackle",
}
SPR_UNKNOWN_LABELS = {"Rhonchi", "Stridor"}
SOFTPLUS_INIT = math.log(math.e - 1.0)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    fieldnames.extend(
        key
        for row in rows
        for key in row
        if key not in fieldnames
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty gzip CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_sha(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cache(path: Path) -> tuple[dict[str, int], np.ndarray, dict[str, object]]:
    if sha256_file(path) != CACHE_SHA256:
        raise RuntimeError("AudioSet-only cache SHA256 mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"sample_ids", "embeddings", "receipt_json"}:
            raise RuntimeError(f"unexpected cache keys: {payload.files}")
        sample_ids = payload["sample_ids"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        receipt = json.loads(str(payload["receipt_json"]))
    if embeddings.shape != (25084, 768) or len(sample_ids) != 25084:
        raise RuntimeError(f"unexpected AudioSet cache shape: {embeddings.shape}")
    if len(sample_ids) != len(set(sample_ids)) or not np.isfinite(embeddings).all():
        raise RuntimeError("cache uniqueness/finite gate failed")
    return {sample_id: index for index, sample_id in enumerate(sample_ids)}, embeddings, receipt


def map_icbhi_attributes(label: str) -> tuple[int, int, int, int, int]:
    if label == "normal":
        return 0, 0, 1, 1, 1
    if label == "crackle":
        return 1, 0, 1, 1, 0
    if label == "wheeze":
        return 0, 1, 1, 1, 0
    if label == "both":
        return 1, 1, 1, 1, 0
    raise ValueError(label)


def map_spr_attributes(label: str) -> tuple[float | None, float | None, int, int, int]:
    if label == "Normal":
        return 0.0, 0.0, 1, 1, 1
    if label in {"Fine Crackle", "Coarse Crackle"}:
        return 1.0, 0.0, 1, 1, 0
    if label == "Wheeze":
        return 0.0, 1.0, 1, 1, 0
    if label == "Wheeze+Crackle":
        return 1.0, 1.0, 1, 1, 0
    if label in SPR_UNKNOWN_LABELS:
        return None, None, 0, 0, 0
    raise ValueError(label)


def build_phase1_frames(cache_index: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    icbhi_rows, icbhi_receipt = load_icbhi_rows(DATASET_ROOT / "icbhi_2017")
    spr_rows, spr_receipt = load_spr_rows(DATASET_ROOT / "sprsound")
    icbhi_records = []
    for row in icbhi_rows:
        label = str(row["native_four_class_label"])
        crackle, wheeze, mask_c, mask_w, normal = map_icbhi_attributes(label)
        sample_id = f"icbhi:{row['cycle_id']}"
        icbhi_records.append(
            {
                "sample_id": sample_id,
                "dataset": "icbhi",
                "group_id": str(row["patient_id"]),
                "patient_id": str(row["patient_id"]),
                "recording_id": str(row["recording_id"]),
                "cycle_id": str(row["cycle_id"]),
                "source_partition": str(row["partition"]),
                "official_split": str(row["official_split"]),
                "native_label_name": label,
                "native_label_index": ICBHI_LABELS.index(label),
                "attr_crackle": float(crackle),
                "attr_wheeze": float(wheeze),
                "mask_crackle": int(mask_c),
                "mask_wheeze": int(mask_w),
                "normal_certified": int(normal),
                "both_positive": int(label == "both"),
                "annotation_path": "",
                "event_index": -1,
                "embed_index": int(cache_index[sample_id]),
            }
        )
    spr_records = []
    for row in spr_rows:
        if str(row["partition"]) == "intra":
            continue
        sample_id = f"spr:{row['event_id']}"
        is_inter = str(row["partition"]) == "inter"
        label = None if is_inter else str(row["raw_label"])
        if label is None:
            crackle = wheeze = None
            mask_c = mask_w = normal = 0
            native_index = -1
        else:
            crackle, wheeze, mask_c, mask_w, normal = map_spr_attributes(label)
            native_index = SPR_LABELS.index(label)
        spr_records.append(
            {
                "sample_id": sample_id,
                "dataset": "sprsound",
                "group_id": str(row["patient_id"]),
                "patient_id": str(row["patient_id"]),
                "recording_id": str(row["recording_id"]),
                "cycle_id": "",
                "source_partition": "test" if is_inter else str(row["partition"]),
                "official_split": "test" if is_inter else "train",
                "native_label_name": label if label is not None else "",
                "native_label_index": int(native_index),
                "attr_crackle": None if crackle is None else float(crackle),
                "attr_wheeze": None if wheeze is None else float(wheeze),
                "mask_crackle": int(mask_c),
                "mask_wheeze": int(mask_w),
                "normal_certified": int(normal),
                "both_positive": int(label == "Wheeze+Crackle") if label is not None else 0,
                "annotation_path": str(row["annotation_path"]),
                "event_index": int(row["event_index"]),
                "embed_index": int(cache_index[sample_id]),
            }
        )
    icbhi = pd.DataFrame(icbhi_records).sort_values("sample_id").reset_index(drop=True)
    spr = pd.DataFrame(spr_records).sort_values("sample_id").reset_index(drop=True)
    if len(icbhi) != 6898 or len(spr) != 8085:
        raise RuntimeError("unexpected Phase 1A dataset row counts")
    if icbhi["sample_id"].duplicated().any() or spr["sample_id"].duplicated().any():
        raise RuntimeError("duplicate sample IDs in Phase 1A frames")
    if set(icbhi["sample_id"]) - set(cache_index) or set(spr["sample_id"]) - set(cache_index):
        raise RuntimeError("cache alignment failed for Phase 1A frames")
    receipt = {
        "status": "phase1a_data_contract_passed",
        "cache_sha256": CACHE_SHA256,
        "icbhi_rows": len(icbhi),
        "spr_rows_without_intra": len(spr),
        "spr_intra_excluded": int(sum(str(row["partition"]) == "intra" for row in spr_rows)),
        "icbhi_receipt": icbhi_receipt,
        "spr_receipt": spr_receipt,
        "icbhi_official_train_rows": int(icbhi["official_split"].eq("train").sum()),
        "spr_official_train_rows": int(spr["official_split"].eq("train").sum()),
        "spr_inter_rows": int(spr["official_split"].eq("test").sum()),
        "spr_unknown_mask_rows_in_train": int(
            (
                spr["official_split"].eq("train").to_numpy()
                & spr["mask_crackle"].eq(0).to_numpy()
            ).sum()
        ),
    }
    return icbhi, spr, receipt


def build_strict_icbhi_receipt() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = load_manifest()
    outer_assignment, nested_assignment, counts = build_fold_assignments(manifest)
    if len(outer_assignment) != 6898 or len(nested_assignment) != 6898 * 5:
        raise RuntimeError("strict-patient assignment row count failed")
    receipt = {
        "protocol_version": "formal-strict-patient-v3",
        "outer_fold_rows": len(outer_assignment),
        "nested_rows": len(nested_assignment),
        "outer_fold_counts": (
            outer_assignment["outer_fold"].value_counts().sort_index().astype(int).to_dict()
        ),
        "nested_unique_cycles_per_fold": {
            str(fold): int(
                nested_assignment[nested_assignment["evaluation_outer_fold"].eq(fold)]["cycle_id"].nunique()
            )
            for fold in range(5)
        },
        "patient_disjoint": True,
        "fold_class_patient_counts_rows": int(len(counts)),
    }
    return outer_assignment, nested_assignment, counts, receipt


def support_signature(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for patient_id, subset in frame.groupby("patient_id", sort=True):
        crackle_present = bool((subset["mask_crackle"].eq(1) & subset["attr_crackle"].fillna(0).eq(1)).any())
        wheeze_present = bool((subset["mask_wheeze"].eq(1) & subset["attr_wheeze"].fillna(0).eq(1)).any())
        groups.append(
            {
                "patient_id": patient_id,
                "normal_certified_present": bool(subset["normal_certified"].eq(1).any()),
                "crackle_present": crackle_present,
                "wheeze_present": wheeze_present,
                "both_present": bool(subset["both_positive"].eq(1).any()),
                "rows": int(len(subset)),
            }
        )
    return pd.DataFrame(groups).sort_values("patient_id").reset_index(drop=True)


def _shuffled(rng: np.random.Generator, values: list[str]) -> list[str]:
    copied = list(values)
    rng.shuffle(copied)
    return copied


def build_group_order(signature: pd.DataFrame, draw_seed: int) -> list[str]:
    rng = np.random.default_rng(draw_seed)
    normal_groups = signature.loc[signature["normal_certified_present"], "patient_id"].tolist()
    crackle_groups = signature.loc[signature["crackle_present"], "patient_id"].tolist()
    wheeze_groups = signature.loc[signature["wheeze_present"], "patient_id"].tolist()
    both_groups = signature.loc[
        signature["crackle_present"] & signature["wheeze_present"],
        "patient_id",
    ].tolist()
    all_groups = signature["patient_id"].tolist()
    used: list[str] = []

    def add_from(pool: list[str], count: int = 1) -> None:
        for patient_id in _shuffled(rng, pool):
            if patient_id not in used:
                used.append(patient_id)
                if count == 1:
                    return
                count -= 1
                if count == 0:
                    return

    add_from(normal_groups)
    add_from(both_groups or crackle_groups)
    add_from(
        [value for value in crackle_groups if value not in both_groups] or crackle_groups
    )
    add_from(
        [value for value in wheeze_groups if value not in both_groups] or wheeze_groups
    )
    while len(used) < 4:
        add_from(all_groups)

    def quota(prefix: list[str], column: str) -> int:
        selected = signature[signature["patient_id"].isin(prefix)]
        return int(selected[column].sum())

    while quota(used, "normal_certified_present") < 2:
        add_from(normal_groups)
    while quota(used, "crackle_present") < 2:
        add_from(crackle_groups)
    while quota(used, "wheeze_present") < 2:
        add_from(wheeze_groups)
    while len(used) < 8:
        add_from(all_groups)

    remaining = [value for value in _shuffled(rng, all_groups) if value not in used]
    order = used + remaining
    prefix4 = signature[signature["patient_id"].isin(order[:4])]
    prefix8 = signature[signature["patient_id"].isin(order[:8])]
    if int(prefix4["normal_certified_present"].sum()) < 1:
        raise RuntimeError("prefix-4 normal support failed")
    if int(prefix4["crackle_present"].sum()) < 1:
        raise RuntimeError("prefix-4 crackle support failed")
    if int(prefix4["wheeze_present"].sum()) < 1:
        raise RuntimeError("prefix-4 wheeze support failed")
    if int(prefix8["normal_certified_present"].sum()) < 2:
        raise RuntimeError("prefix-8 normal support failed")
    if int(prefix8["crackle_present"].sum()) < 2:
        raise RuntimeError("prefix-8 crackle support failed")
    if int(prefix8["wheeze_present"].sum()) < 2:
        raise RuntimeError("prefix-8 wheeze support failed")
    return order


def build_draw_receipts(
    icbhi: pd.DataFrame,
    spr: pd.DataFrame,
    nested_assignment: pd.DataFrame,
) -> tuple[dict[str, Any], list[int | str]]:
    receipts: dict[str, Any] = {"r1_icbhi_to_spr": {}, "r2_spr_to_icbhi": {}}
    target_spr_support = spr[spr["source_partition"].eq("subtrain")].copy()
    spr_signature = support_signature(target_spr_support)
    retained = set(SHOT_LEVELS)
    for draw_index, draw_seed in enumerate(DRAW_SEEDS):
        order = build_group_order(spr_signature, draw_seed)
        draw_rows = {}
        for shot in SHOT_LEVELS:
            groups = order if shot == "full" else order[: int(shot)]
            subset = target_spr_support[target_spr_support["patient_id"].isin(groups)]
            draw_rows[str(shot)] = {
                "groups": len(set(groups)),
                "group_sha256": ordered_sha(list(groups)),
                "rows": int(len(subset)),
                "crackle_positive_groups": int(
                    support_signature(subset)["crackle_present"].sum()
                ),
                "wheeze_positive_groups": int(
                    support_signature(subset)["wheeze_present"].sum()
                ),
                "normal_certified_groups": int(
                    support_signature(subset)["normal_certified_present"].sum()
                ),
            }
        receipts["r1_icbhi_to_spr"][f"draw_{draw_index}"] = {
            "seed": draw_seed,
            "full_groups": int(len(order)),
            "order_sha256": ordered_sha(order),
            "shot_support": draw_rows,
        }
    for outer_fold in range(5):
        fold_roles = nested_assignment[
            nested_assignment["evaluation_outer_fold"].eq(outer_fold)
        ][["cycle_id", "role"]]
        role_map = dict(zip(fold_roles["cycle_id"], fold_roles["role"]))
        fold_support = icbhi[icbhi["cycle_id"].map(role_map).eq("inner_train")].copy()
        signature = support_signature(fold_support)
        fold_receipt: dict[str, Any] = {
            "inner_train_groups": int(signature["patient_id"].nunique())
        }
        for draw_index, draw_seed in enumerate(DRAW_SEEDS):
            order = build_group_order(signature, draw_seed + outer_fold * 1000)
            shot_rows = {}
            for shot in SHOT_LEVELS:
                groups = order if shot == "full" else order[: int(shot)]
                subset = fold_support[fold_support["patient_id"].isin(groups)]
                shot_rows[str(shot)] = {
                    "groups": len(set(groups)),
                    "group_sha256": ordered_sha(list(groups)),
                    "rows": int(len(subset)),
                    "crackle_positive_groups": int(
                        support_signature(subset)["crackle_present"].sum()
                    ),
                    "wheeze_positive_groups": int(
                        support_signature(subset)["wheeze_present"].sum()
                    ),
                    "normal_certified_groups": int(
                        support_signature(subset)["normal_certified_present"].sum()
                    ),
                }
            fold_receipt[f"draw_{draw_index}"] = {
                "seed": draw_seed + outer_fold * 1000,
                "full_groups": int(len(order)),
                "order_sha256": ordered_sha(order),
                "shot_support": shot_rows,
            }
        receipts["r2_spr_to_icbhi"][f"outer_fold_{outer_fold}"] = fold_receipt
    for shot in SHOT_LEVELS:
        key = str(shot)
        if shot == "full":
            continue
        if any(
            receipts["r1_icbhi_to_spr"][draw][ "shot_support"][key]["groups"] < int(shot)
            for draw in receipts["r1_icbhi_to_spr"]
        ):
            retained.discard(shot)
        for fold_key in receipts["r2_spr_to_icbhi"]:
            if any(
                receipts["r2_spr_to_icbhi"][fold_key][draw]["shot_support"][key]["groups"] < int(shot)
                for draw in receipts["r2_spr_to_icbhi"][fold_key]
                if draw.startswith("draw_")
            ):
                retained.discard(shot)
    retained_levels = [shot for shot in SHOT_LEVELS if shot in retained]
    return receipts, retained_levels


def protocol_receipt(retained_shots: list[int | str]) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": "Phase 1A only",
        "encoder": {
            "representation": "r1_beats_as2m_audioset_only",
            "cache_path": str(CACHE_PATH.resolve()),
            "cache_sha256": CACHE_SHA256,
            "claim": "neutral AudioSet-only frozen encoder",
        },
        "conditions": list(CONDITIONS),
        "shots_requested": [str(value) for value in SHOT_LEVELS],
        "shots_retained_after_support_gate": [str(value) for value in retained_shots],
        "draw_seeds": DRAW_SEEDS,
        "optimization_seeds": OPTIMIZATION_SEEDS,
        "source_batch_size": SOURCE_BATCH_SIZE,
        "target_batch_size": TARGET_BATCH_SIZE,
        "max_steps": MAX_STEPS,
        "eval_every": EVAL_EVERY,
        "patience_evals": PATIENCE_EVALS,
        "selection": {
            "uses_outer_test": False,
            "scaler_fit": "source_train + target_support only",
            "icbhi_target_validation_metric": "native flat4 one-vs-rest macro-AUPRC",
            "spr_target_validation_metric": "native supported-class one-vs-rest macro-AUPRC",
            "tie_break": "earlier checkpoint",
        },
        "claim_boundary": [
            "attribute-support-constrained patient-group low-shot adaptation",
            "target-supervised frozen-representation downstream training",
            "Phase 1B joint/consistency remains HOLD",
            "PAFA attribution remains HOLD",
            "no calibration fitting",
            "no MoE/router/general-specific/new pooling/full encoder/server path",
        ],
        "parameter_contract": condition_parameter_receipt(),
    }


def to_numpy(frame: pd.DataFrame, embeddings: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    values = embeddings[frame["embed_index"].to_numpy(dtype=np.int64)]
    if scaler is None:
        return values.astype(np.float32, copy=False)
    return scaler.transform(values.astype(np.float64, copy=False)).astype(np.float32)


def fit_scaler(source_frame: pd.DataFrame, target_frame: pd.DataFrame, embeddings: np.ndarray) -> StandardScaler:
    values = np.concatenate(
        [
            embeddings[source_frame["embed_index"].to_numpy(dtype=np.int64)],
            embeddings[target_frame["embed_index"].to_numpy(dtype=np.int64)],
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler


def sample_group_batch(
    frame: pd.DataFrame,
    batch_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    groups = sorted(frame["patient_id"].unique().tolist())
    positions = {
        patient_id: frame.index[frame["patient_id"].eq(patient_id)].to_numpy(dtype=np.int64)
        for patient_id in groups
    }
    picked_groups = rng.choice(groups, size=batch_size, replace=True)
    rows = []
    for patient_id in picked_groups:
        rows.append(int(rng.choice(positions[str(patient_id)])))
    return frame.loc[rows].reset_index(drop=True)


def masked_attr_loss(logits: torch.Tensor, frame: pd.DataFrame) -> torch.Tensor | None:
    mask = torch.tensor(
        frame[["mask_crackle", "mask_wheeze"]].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=logits.device,
    )
    if float(mask.sum()) <= 0.0:
        return None
    target = torch.tensor(
        frame[["attr_crackle", "attr_wheeze"]].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=logits.device,
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    terms = []
    for index in range(len(ATTRIBUTES)):
        denominator = float(mask[:, index].sum().item())
        if denominator > 0:
            terms.append((raw[:, index] * mask[:, index]).sum() / mask[:, index].sum())
    if not terms:
        return None
    return torch.stack(terms).mean()


def native_loss(logits: torch.Tensor, frame: pd.DataFrame) -> torch.Tensor:
    target = torch.tensor(
        frame["native_label_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=logits.device,
    )
    return torch.nn.functional.cross_entropy(logits, target)


def batch_loss(
    model: Phase1AModel,
    frame: pd.DataFrame,
    values: np.ndarray,
    dataset: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    hidden = model.encode(torch.from_numpy(values))
    native = native_loss(model.native_logits(hidden, dataset), frame)
    total = native
    stats = {"native_loss": float(native.detach().cpu())}
    if model.condition != "N":
        attr = masked_attr_loss(model.attr_logits(hidden, dataset), frame)
        if attr is not None:
            total = total + attr
            stats["attr_loss"] = float(attr.detach().cpu())
    return total, stats


def multiclass_macro_auprc(probabilities: np.ndarray, truth: np.ndarray, labels: list[str]) -> float:
    values = []
    for class_index, _ in enumerate(labels):
        binary = (truth == class_index).astype(np.int64)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        values.append(float(average_precision_score(binary, probabilities[:, class_index])))
    if not values:
        return float("nan")
    return float(np.mean(values))


def multiclass_subset_macro_auprc(
    probabilities: np.ndarray,
    truth: np.ndarray,
    labels: list[str],
    subset: list[str],
) -> float:
    values = []
    for label in subset:
        class_index = labels.index(label)
        binary = (truth == class_index).astype(np.int64)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        values.append(float(average_precision_score(binary, probabilities[:, class_index])))
    if not values:
        return float("nan")
    return float(np.mean(values))


def predict_native_and_attr(
    model: Phase1AModel,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = to_numpy(frame, embeddings, scaler)
    dataset = str(frame["dataset"].iloc[0])
    model.eval()
    with torch.inference_mode():
        hidden = model.encode(torch.from_numpy(values))
        native_logits_tensor = model.native_logits(hidden, dataset)
        native_prob = torch.softmax(native_logits_tensor, dim=1).cpu().numpy()
        if model.condition == "N":
            if dataset == "icbhi":
                attr_prob = np.stack(
                    [
                        native_prob[:, 1] + native_prob[:, 3],
                        native_prob[:, 2] + native_prob[:, 3],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
            else:
                attr_prob = np.stack(
                    [
                        native_prob[:, 4] + native_prob[:, 5] + native_prob[:, 6],
                        native_prob[:, 2] + native_prob[:, 6],
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
        else:
            attr_prob = torch.sigmoid(model.attr_logits(hidden, dataset)).cpu().numpy()
    native_pred = native_prob.argmax(axis=1)
    return native_prob, native_pred, attr_prob


def target_validation_metric(
    model: Phase1AModel,
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    scaler: StandardScaler,
    direction: str,
) -> dict[str, float]:
    native_prob, _, attr_prob = predict_native_and_attr(model, frame, embeddings, scaler)
    truth = frame["native_label_index"].to_numpy(dtype=np.int64)
    if direction == "r1_icbhi_to_spr":
        selection = multiclass_subset_macro_auprc(
            native_prob,
            truth,
            SPR_LABELS,
            VALIDATION_SUPPORTED_SPR_LABELS,
        )
    else:
        selection = multiclass_macro_auprc(native_prob, truth, ICBHI_LABELS)
    output = {"selection_metric": float(selection)}
    if model.condition != "N":
        attr_truth = frame[["attr_crackle", "attr_wheeze"]].to_numpy(dtype=np.float32)
        attr_mask = frame[["mask_crackle", "mask_wheeze"]].to_numpy(dtype=np.int64)
        aps = []
        for index, name in enumerate(ATTRIBUTES):
            valid = attr_mask[:, index].astype(bool)
            if valid.any() and attr_truth[valid, index].sum() not in {0, valid.sum()}:
                aps.append(float(average_precision_score(attr_truth[valid, index], attr_prob[valid, index])))
                output[f"{name}_ap"] = aps[-1]
        if aps:
            output["attr_macro_auprc"] = float(np.mean(aps))
    return output


def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not mask.any():
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        total += abs(confidence - accuracy) * float(mask.mean())
    return total


def attr_metrics_from_rows(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, object]:
    truth = frame[["attr_crackle", "attr_wheeze"]].to_numpy(dtype=np.float32)
    mask = frame[["mask_crackle", "mask_wheeze"]].to_numpy(dtype=np.int64)
    scored = {}
    aps = []
    f1_values = []
    recall_values = []
    precision_values = []
    ece_values = []
    brier_values = []
    for index, name in enumerate(ATTRIBUTES):
        valid = mask[:, index].astype(bool)
        y_true = truth[valid, index]
        y_prob = probability[valid, index]
        y_pred = (y_prob >= 0.5).astype(np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        ap = (
            float(average_precision_score(y_true, y_prob))
            if y_true.sum() not in {0, len(y_true)}
            else float("nan")
        )
        ece = ece_binary(y_true, y_prob)
        brier = float(np.mean((y_true - y_prob) ** 2))
        scored[name] = {
            "rows": int(valid.sum()),
            "average_precision": ap,
            "precision_at_0_5": float(precision),
            "recall_at_0_5": float(recall),
            "f1_at_0_5": float(f1),
            "support_positive": int(y_true.sum()),
            "ece": ece,
            "brier": brier,
        }
        if not math.isnan(ap):
            aps.append(ap)
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        ece_values.append(ece)
        brier_values.append(brier)
    return {
        "supported_attr_macro_auprc": float(np.mean(aps)) if aps else float("nan"),
        "macro_precision_at_0_5": float(np.mean(precision_values)),
        "macro_recall_at_0_5": float(np.mean(recall_values)),
        "macro_f1_at_0_5": float(np.mean(f1_values)),
        "raw_ece": float(np.mean(ece_values)),
        "raw_brier": float(np.mean(brier_values)),
        "per_attribute": scored,
        "coverage_rows": int(mask[:, 0].sum()),
        "coverage_patients": int(frame["patient_id"].nunique()),
    }


def native_spr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    indices = list(range(len(SPR_LABELS)))
    matrix = confusion_matrix(y_true, y_pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    specificity = float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0
    abnormal_total = matrix[1:].sum()
    sensitivity = float(np.trace(matrix[1:, 1:]) / abnormal_total) if abnormal_total else 0.0
    average = (specificity + sensitivity) / 2.0
    harmonic = (
        2 * specificity * sensitivity / (specificity + sensitivity)
        if specificity + sensitivity
        else 0.0
    )
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=indices, average="weighted", zero_division=0)
        ),
        "uar": float(np.mean(recall)),
        "normal_specificity": specificity,
        "abnormal_sensitivity": sensitivity,
        "native_score": float((average + harmonic) / 2.0),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(SPR_LABELS)
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_labels": SPR_LABELS,
    }


def load_terminal_spr_truth(frame: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in frame.iterrows():
        payload = json.loads(Path(str(row["annotation_path"])).read_text())
        raw = str(payload["event_annotation"][int(row["event_index"])]["type"])
        labels.append(raw)
    output = frame.copy()
    output["native_label_name"] = labels
    output["native_label_index"] = [SPR_LABELS.index(label) for label in labels]
    attr_values = [map_spr_attributes(label) for label in labels]
    output["attr_crackle"] = [value[0] for value in attr_values]
    output["attr_wheeze"] = [value[1] for value in attr_values]
    output["mask_crackle"] = [value[2] for value in attr_values]
    output["mask_wheeze"] = [value[3] for value in attr_values]
    output["normal_certified"] = [value[4] for value in attr_values]
    output["both_positive"] = [int(label == "Wheeze+Crackle") for label in labels]
    if set(labels) - SPR_COMPATIBLE_LABELS:
        raise RuntimeError("unexpected incompatible label on SPR inter")
    return output


def score_target_predictions(
    frame: pd.DataFrame,
    native_prob: np.ndarray,
    native_pred: np.ndarray,
    attr_prob: np.ndarray,
) -> dict[str, object]:
    attr_metrics = attr_metrics_from_rows(frame, attr_prob)
    if str(frame["dataset"].iloc[0]) == "icbhi":
        y_true = frame["native_label_name"].to_numpy(dtype=str)
        y_pred = np.asarray([ICBHI_LABELS[index] for index in native_pred], dtype=str)
        native_metrics = evaluate_predictions(y_true, y_pred, ICBHI_LABELS)
    else:
        y_true = frame["native_label_index"].to_numpy(dtype=np.int64)
        native_metrics = native_spr_metrics(y_true, native_pred)
    return {"attr": attr_metrics, "native": native_metrics}


def prediction_rows(
    frame: pd.DataFrame,
    direction: str,
    condition: str,
    shot: int | str,
    draw_index: int,
    seed: int,
    outer_fold: int | None,
    native_prob: np.ndarray,
    native_pred: np.ndarray,
    attr_prob: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    label_list = ICBHI_LABELS if str(frame["dataset"].iloc[0]) == "icbhi" else SPR_LABELS
    for row_index, (_, row) in enumerate(frame.iterrows()):
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "direction": direction,
                "condition": condition,
                "shot_level": str(shot),
                "draw_index": draw_index,
                "seed": seed,
                "outer_fold": "" if outer_fold is None else outer_fold,
                "dataset": str(row["dataset"]),
                "patient_id": str(row["patient_id"]),
                "recording_id": str(row["recording_id"]),
                "cycle_id": str(row["cycle_id"]),
                "native_pred_index": int(native_pred[row_index]),
                "native_pred_label": label_list[int(native_pred[row_index])],
                "native_probabilities_json": json.dumps(native_prob[row_index].tolist()),
                "attr_prob_crackle": None if np.isnan(attr_prob[row_index, 0]) else float(attr_prob[row_index, 0]),
                "attr_prob_wheeze": None if np.isnan(attr_prob[row_index, 1]) else float(attr_prob[row_index, 1]),
                "attr_pred_crackle": None if np.isnan(attr_prob[row_index, 0]) else int(attr_prob[row_index, 0] >= 0.5),
                "attr_pred_wheeze": None if np.isnan(attr_prob[row_index, 1]) else int(attr_prob[row_index, 1] >= 0.5),
            }
        )
    return rows


def run_single(
    direction: str,
    condition: str,
    shot: int | str,
    draw_index: int,
    seed: int,
    icbhi: pd.DataFrame,
    spr: pd.DataFrame,
    nested_assignment: pd.DataFrame,
    draw_receipts: dict[str, Any],
    embeddings: np.ndarray,
    output_dir: Path,
    max_steps: int,
    outer_fold: int | None = None,
    score_outer_test: bool = True,
) -> dict[str, object]:
    set_seed(seed)
    if direction == "r1_icbhi_to_spr":
        source_frame = icbhi[icbhi["official_split"].eq("train")].copy()
        target_validation = spr[spr["source_partition"].eq("validation")].copy()
        target_test = spr[spr["source_partition"].eq("test")].copy()
        order_key = draw_receipts["r1_icbhi_to_spr"][f"draw_{draw_index}"]
        support_pool = spr[spr["source_partition"].eq("subtrain")].copy()
        groups = (
            support_pool["patient_id"].drop_duplicates().sort_values().tolist()
            if shot == "full"
            else None
        )
        if groups is None:
            order = build_group_order(support_signature(support_pool), DRAW_SEEDS[draw_index])
            groups = order[: int(shot)]
    else:
        if outer_fold is None:
            raise RuntimeError("r2 requires outer_fold")
        fold_roles = nested_assignment[
            nested_assignment["evaluation_outer_fold"].eq(outer_fold)
        ][["cycle_id", "role"]]
        role_map = dict(zip(fold_roles["cycle_id"], fold_roles["role"]))
        target_base = icbhi.copy()
        target_base["strict_role"] = target_base["cycle_id"].map(role_map)
        source_frame = spr[spr["official_split"].eq("train")].copy()
        target_validation = target_base[target_base["strict_role"].eq("inner_validation")].copy()
        target_test = target_base[target_base["strict_role"].eq("outer_test")].copy()
        support_pool = target_base[target_base["strict_role"].eq("inner_train")].copy()
        order_key = draw_receipts["r2_spr_to_icbhi"][f"outer_fold_{outer_fold}"][f"draw_{draw_index}"]
        groups = (
            support_pool["patient_id"].drop_duplicates().sort_values().tolist()
            if shot == "full"
            else None
        )
        if groups is None:
            order = build_group_order(
                support_signature(support_pool),
                DRAW_SEEDS[draw_index] + outer_fold * 1000,
            )
            groups = order[: int(shot)]
    target_support = support_pool[support_pool["patient_id"].isin(groups)].copy()
    scaler = fit_scaler(source_frame, target_support, embeddings)
    model = Phase1AModel(condition)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    source_rng = np.random.default_rng(seed + 100)
    target_rng = np.random.default_rng(seed + 200)
    history = []
    best_score = -math.inf
    best_state = None
    best_step = 0
    no_improve = 0
    step_digest = hashlib.sha256()
    started = time.perf_counter()
    gradient_finite = True
    loss_finite = True
    for step in range(1, max_steps + 1):
        model.train()
        source_batch = sample_group_batch(source_frame, SOURCE_BATCH_SIZE, source_rng)
        target_batch = sample_group_batch(target_support, TARGET_BATCH_SIZE, target_rng)
        step_digest.update(",".join(source_batch["sample_id"].astype(str).tolist()).encode())
        step_digest.update(b"\n")
        step_digest.update(",".join(target_batch["sample_id"].astype(str).tolist()).encode())
        step_digest.update(b"\n")
        optimizer.zero_grad(set_to_none=True)
        source_values = to_numpy(source_batch, embeddings, scaler)
        target_values = to_numpy(target_batch, embeddings, scaler)
        source_loss, source_stats = batch_loss(
            model,
            source_batch,
            source_values,
            str(source_batch["dataset"].iloc[0]),
        )
        target_loss, target_stats = batch_loss(
            model,
            target_batch,
            target_values,
            str(target_batch["dataset"].iloc[0]),
        )
        loss = 0.5 * (source_loss + target_loss)
        if not torch.isfinite(loss):
            loss_finite = False
            raise FloatingPointError("non-finite loss")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            gradient_finite = False
            raise FloatingPointError("non-finite gradient")
        optimizer.step()
        if step % EVAL_EVERY != 0 and step != max_steps:
            continue
        validation = target_validation_metric(
            model, target_validation, embeddings, scaler, direction
        )
        history.append(
            {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_selection_metric": float(validation["selection_metric"]),
                "source_native_loss": source_stats["native_loss"],
                "target_native_loss": target_stats["native_loss"],
                "source_attr_loss": source_stats.get("attr_loss"),
                "target_attr_loss": target_stats.get("attr_loss"),
            }
        )
        if validation["selection_metric"] > best_score:
            best_score = float(validation["selection_metric"])
            best_state = copy.deepcopy(model.state_dict())
            best_step = step
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE_EVALS:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    native_prob, native_pred, attr_prob = predict_native_and_attr(
        model, target_test, embeddings, scaler
    )
    label_free_rows = prediction_rows(
        target_test,
        direction,
        condition,
        shot,
        draw_index,
        seed,
        outer_fold,
        native_prob,
        native_pred,
        attr_prob,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label_free_path = output_dir / "predictions_label_free.csv.gz"
    write_gzip_csv(label_free_path, label_free_rows)
    scored_target = target_test if str(target_test["dataset"].iloc[0]) == "icbhi" else load_terminal_spr_truth(target_test)
    metrics = {}
    scored_rows = []
    if score_outer_test:
        metrics = score_target_predictions(scored_target, native_prob, native_pred, attr_prob)
        for row, (_, truth_row) in zip(label_free_rows, scored_target.iterrows()):
            scored_rows.append(
                {
                    **row,
                    "native_true_label": str(truth_row["native_label_name"]),
                    "native_true_index": int(truth_row["native_label_index"]),
                    "attr_true_crackle": None if pd.isna(truth_row["attr_crackle"]) else float(truth_row["attr_crackle"]),
                    "attr_true_wheeze": None if pd.isna(truth_row["attr_wheeze"]) else float(truth_row["attr_wheeze"]),
                    "mask_crackle": int(truth_row["mask_crackle"]),
                    "mask_wheeze": int(truth_row["mask_wheeze"]),
                }
            )
        write_gzip_csv(output_dir / "predictions_scored.csv.gz", scored_rows)
    else:
        metrics = {"outer_test_scored": False}
    torch.save(
        {
            "model": model.state_dict(),
            "condition": condition,
            "direction": direction,
            "seed": seed,
            "outer_fold": outer_fold,
            "best_step": best_step,
            "step_digest_sha256": step_digest.hexdigest(),
        },
        output_dir / "checkpoint.pt",
    )
    write_csv(output_dir / "training_curve.csv", history or [{"step": 0, "train_loss": 0.0, "validation_selection_metric": -1.0}])
    payload = {
        "condition": condition,
        "direction": direction,
        "shot_level": str(shot),
        "draw_index": draw_index,
        "seed": seed,
        "outer_fold": outer_fold,
        "runtime_seconds": float(time.perf_counter() - started),
        "best_step": best_step,
        "best_validation_selection_metric": best_score,
        "history_rows": len(history),
        "loss_finite": loss_finite,
        "gradient_finite": gradient_finite,
        "cache_sha256": CACHE_SHA256,
        "step_digest_sha256": step_digest.hexdigest(),
        "source_rows": int(len(source_frame)),
        "target_support_rows": int(len(target_support)),
        "target_validation_rows": int(len(target_validation)),
        "target_test_rows": int(len(target_test)),
        "target_support_groups": int(target_support["patient_id"].nunique()),
        "target_test_groups": int(target_test["patient_id"].nunique()),
        "parameter_receipt": condition_parameter_receipt()[condition],
        "shot_receipt": order_key["shot_support"][str(shot)],
        "metrics": metrics,
        "warnings": [],
    }
    write_json(output_dir / "metrics.json", payload)
    return payload


def summarize_run_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        native = row["metrics"]["native"]
        attr = row["metrics"]["attr"]
        output.append(
            {
                "direction": row["direction"],
                "condition": row["condition"],
                "shot_level": row["shot_level"],
                "draw_index": row["draw_index"],
                "seed": row["seed"],
                "outer_fold": row["outer_fold"] if row["outer_fold"] is not None else "",
                "supported_attr_macro_auprc": attr["supported_attr_macro_auprc"],
                "raw_brier": attr["raw_brier"],
                "raw_ece": attr["raw_ece"],
                "native_macro_f1": native["macro_f1"],
                "native_weighted_f1": native["weighted_f1"],
                "native_uar": native["uar"],
                "normal_specificity": native.get("normal_specificity"),
                "abnormal_sensitivity": native.get("abnormal_sensitivity"),
                "both_recall": native.get("both_recall"),
                "native_score": native.get("icbhi_score", native.get("native_score")),
            }
        )
    return output


def shot_axis_value(direction: str, shot: str, draw_receipts: dict[str, Any]) -> float:
    if shot != "full":
        return float(int(shot))
    if direction == "r1_icbhi_to_spr":
        return float(
            np.mean(
                [
                    draw_receipts["r1_icbhi_to_spr"][f"draw_{index}"]["full_groups"]
                    for index in range(len(DRAW_SEEDS))
                ]
            )
        )
    return float(
        np.mean(
            [
                draw_receipts["r2_spr_to_icbhi"][f"outer_fold_{fold}"]["inner_train_groups"]
                for fold in range(5)
            ]
        )
    )


def normalized_log_aulc(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    log_x = np.log2(np.asarray(xs, dtype=np.float64))
    y = np.asarray(ys, dtype=np.float64)
    area = np.trapz(y, log_x)
    return float(area / (log_x[-1] - log_x[0]))


def score_from_scored_predictions(rows: list[dict[str, str]]) -> dict[str, object]:
    frame = pd.DataFrame(rows)
    probability = np.vstack(frame["native_probabilities_json"].map(json.loads).to_list()).astype(np.float32)
    if frame["dataset"].iloc[0] == "icbhi":
        y_true = frame["native_true_label"].to_numpy(dtype=str)
        y_pred = frame["native_pred_label"].to_numpy(dtype=str)
        native = evaluate_predictions(y_true, y_pred, ICBHI_LABELS)
    else:
        y_true = frame["native_true_index"].to_numpy(dtype=np.int64)
        y_pred = frame["native_pred_index"].to_numpy(dtype=np.int64)
        native = native_spr_metrics(y_true, y_pred)
    attr_prob = frame[["attr_prob_crackle", "attr_prob_wheeze"]].astype(float).to_numpy(dtype=np.float32)
    attr_frame = pd.DataFrame(
        {
            "patient_id": frame["patient_id"],
            "attr_crackle": frame["attr_true_crackle"].astype(float),
            "attr_wheeze": frame["attr_true_wheeze"].astype(float),
            "mask_crackle": frame["mask_crackle"].astype(int),
            "mask_wheeze": frame["mask_wheeze"].astype(int),
        }
    )
    return {"native": native, "attr": attr_metrics_from_rows(attr_frame, attr_prob)}


def bootstrap_aulc_delta(
    paired_predictions: dict[tuple[str, str, int, int], pd.DataFrame],
    direction: str,
    retained_shots: list[str],
    repeats: int,
) -> dict[str, float]:
    if repeats <= 0:
        return {"mean_delta": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    deltas = []
    tuple_keys = sorted(
        {
            (draw_index, seed)
            for (_, _, draw_index, seed) in paired_predictions
        }
    )
    all_patients = sorted(
        set(
            pd.concat(list(paired_predictions.values()), ignore_index=True)["patient_id"].astype(str)
        )
    )
    for repeat in range(repeats):
        rng = np.random.default_rng(20260730 + repeat)
        sampled_patients = rng.choice(all_patients, size=len(all_patients), replace=True)
        per_tuple = []
        for draw_index, seed in tuple_keys:
            shot_metrics = {}
            for condition in ("I16", "S16"):
                xs = []
                ys = []
                for shot in retained_shots:
                    frame = paired_predictions[(condition, shot, draw_index, seed)]
                    selected = pd.concat(
                        [frame[frame["patient_id"].eq(patient_id)] for patient_id in sampled_patients],
                        ignore_index=True,
                    )
                    metrics = score_from_scored_predictions(selected.to_dict("records"))
                    xs.append(shot_axis_value(direction, shot, DRAW_RECEIPTS_GLOBAL))
                    ys.append(metrics["attr"]["supported_attr_macro_auprc"])
                shot_metrics[condition] = normalized_log_aulc(xs, ys)
            per_tuple.append(shot_metrics["S16"] - shot_metrics["I16"])
        deltas.append(float(np.mean(per_tuple)))
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }


DRAW_RECEIPTS_GLOBAL: dict[str, Any] = {}


def analyze(result_root: Path, retained_shots: list[int | str]) -> dict[str, object]:
    run_rows = []
    for path in sorted(result_root.glob("runs/**/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("metrics", {}).get("outer_test_scored") is False:
            continue
        run_rows.append(payload)
    if not run_rows:
        raise RuntimeError("no full run metrics found")
    write_csv(result_root / "run_level_results.csv", summarize_run_rows(run_rows))
    merged_metrics_rows = []
    paired_predictions: dict[str, dict[tuple[str, int, int], pd.DataFrame]] = defaultdict(dict)
    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        for condition in CONDITIONS:
            for shot in retained_shots:
                for draw_index in range(len(DRAW_SEEDS)):
                    for seed in OPTIMIZATION_SEEDS:
                        scored_paths = sorted(
                            result_root.glob(
                                (
                                    f"runs/{direction}_shot{shot}_draw{draw_index}_seed{seed}"
                                    f"{'_outer*_cond' + condition if direction == 'r2_spr_to_icbhi' else '_cond' + condition}"
                                    "*/predictions_scored.csv.gz"
                                )
                            )
                        )
                        if direction == "r1_icbhi_to_spr":
                            if len(scored_paths) != 1:
                                raise RuntimeError(f"unexpected R1 run count for {condition} shot={shot} draw={draw_index} seed={seed}")
                        else:
                            if len(scored_paths) != 5:
                                raise RuntimeError(f"unexpected R2 fold count for {condition} shot={shot} draw={draw_index} seed={seed}")
                        rows = []
                        for path in scored_paths:
                            rows.extend(read_gzip_csv(path))
                        metrics = score_from_scored_predictions(rows)
                        merged_metrics_rows.append(
                            {
                                "direction": direction,
                                "condition": condition,
                                "shot_level": str(shot),
                                "draw_index": draw_index,
                                "seed": seed,
                                "supported_attr_macro_auprc": metrics["attr"]["supported_attr_macro_auprc"],
                                "raw_brier": metrics["attr"]["raw_brier"],
                                "raw_ece": metrics["attr"]["raw_ece"],
                                "native_macro_f1": metrics["native"]["macro_f1"],
                                "native_weighted_f1": metrics["native"]["weighted_f1"],
                                "native_uar": metrics["native"]["uar"],
                                "normal_specificity": metrics["native"].get("normal_specificity"),
                                "abnormal_sensitivity": metrics["native"].get("abnormal_sensitivity"),
                                "both_recall": metrics["native"].get("both_recall"),
                                "native_score": metrics["native"].get("icbhi_score", metrics["native"].get("native_score")),
                            }
                        )
                        paired_predictions[direction][(condition, str(shot), draw_index, seed)] = pd.DataFrame(rows)
                        merged_path = result_root / "merged_predictions" / f"{direction}_shot{shot}_draw{draw_index}_seed{seed}_cond{condition}.csv.gz"
                        write_gzip_csv(merged_path, rows)
    write_csv(result_root / "merged_oof_metrics.csv", merged_metrics_rows)

    aulc_rows = []
    summary_rows = []
    retained_shot_strings = [str(value) for value in retained_shots]
    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        for condition in CONDITIONS:
            deltas = []
            for draw_index in range(len(DRAW_SEEDS)):
                for seed in OPTIMIZATION_SEEDS:
                    subset = [
                        row
                        for row in merged_metrics_rows
                        if row["direction"] == direction
                        and row["condition"] == condition
                        and row["draw_index"] == draw_index
                        and row["seed"] == seed
                    ]
                    subset = sorted(subset, key=lambda row: shot_axis_value(direction, row["shot_level"], DRAW_RECEIPTS_GLOBAL))
                    xs = [shot_axis_value(direction, row["shot_level"], DRAW_RECEIPTS_GLOBAL) for row in subset]
                    ys = [float(row["supported_attr_macro_auprc"]) for row in subset]
                    aulc = normalized_log_aulc(xs, ys)
                    aulc_rows.append(
                        {
                            "direction": direction,
                            "condition": condition,
                            "draw_index": draw_index,
                            "seed": seed,
                            "aulc": aulc,
                        }
                    )
            cond_rows = [row for row in aulc_rows if row["direction"] == direction and row["condition"] == condition]
            values = np.asarray([row["aulc"] for row in cond_rows], dtype=np.float64)
            summary_rows.append(
                {
                    "direction": direction,
                    "condition": condition,
                    "metric": "aulc",
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "runs": int(len(values)),
                }
            )
    write_csv(result_root / "aulc_by_draw_seed.csv", aulc_rows)

    delta_rows = []
    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        for baseline, candidate in (("I16", "S16"), ("I16", "S32"), ("N", "I16"), ("N", "S16")):
            for draw_index in range(len(DRAW_SEEDS)):
                for seed in OPTIMIZATION_SEEDS:
                    base = next(
                        row["aulc"]
                        for row in aulc_rows
                        if row["direction"] == direction and row["condition"] == baseline and row["draw_index"] == draw_index and row["seed"] == seed
                    )
                    comp = next(
                        row["aulc"]
                        for row in aulc_rows
                        if row["direction"] == direction and row["condition"] == candidate and row["draw_index"] == draw_index and row["seed"] == seed
                    )
                    delta_rows.append(
                        {
                            "direction": direction,
                            "baseline": baseline,
                            "candidate": candidate,
                            "draw_index": draw_index,
                            "seed": seed,
                            "delta_aulc": comp - base,
                        }
                    )
    write_csv(result_root / "paired_aulc_deltas.csv", delta_rows)

    bootstrap_rows = []
    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        subset = {
            key: value
            for key, value in paired_predictions[direction].items()
            if key[0] in {"I16", "S16"}
        }
        bootstrap = bootstrap_aulc_delta(subset, direction, retained_shot_strings, FULL_BOOTSTRAP_REPEATS)
        bootstrap_rows.append({"direction": direction, **bootstrap})
    write_csv(result_root / "patient_cluster_interval.csv", bootstrap_rows)

    s16_deltas = pd.DataFrame(
        [row for row in delta_rows if row["baseline"] == "I16" and row["candidate"] == "S16"]
    )
    delta_summary = (
        s16_deltas.groupby("direction")["delta_aulc"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"std": "sample_std"})
    )
    for _, row in delta_summary.iterrows():
        summary_rows.append(
            {
                "direction": row["direction"],
                "condition": "S16_minus_I16",
                "metric": "delta_aulc",
                "mean": float(row["mean"]),
                "sample_std": float(row["sample_std"]) if not math.isnan(float(row["sample_std"])) else 0.0,
                "runs": int(len(s16_deltas[s16_deltas["direction"].eq(row["direction"])])),
            }
        )
    write_csv(result_root / "summary.csv", summary_rows)

    guardrail_rows = []
    for direction in ("r1_icbhi_to_spr", "r2_spr_to_icbhi"):
        for shot in retained_shot_strings:
            base_rows = [
                row for row in merged_metrics_rows
                if row["direction"] == direction and row["condition"] == "I16" and row["shot_level"] == shot
            ]
            cand_rows = [
                row for row in merged_metrics_rows
                if row["direction"] == direction and row["condition"] == "S16" and row["shot_level"] == shot
            ]
            for metric in ("native_macro_f1", "native_uar", "normal_specificity", "both_recall"):
                base_mean = float(np.mean([float(row[metric]) for row in base_rows if row[metric] is not None]))
                cand_mean = float(np.mean([float(row[metric]) for row in cand_rows if row[metric] is not None]))
                guardrail_rows.append(
                    {
                        "direction": direction,
                        "shot_level": shot,
                        "metric": metric,
                        "I16_mean": base_mean,
                        "S16_mean": cand_mean,
                        "delta": cand_mean - base_mean,
                    }
                )
    write_csv(result_root / "guardrail_deltas.csv", guardrail_rows)

    bootstrap_map = {row["direction"]: row for row in bootstrap_rows}
    delta_map = {
        row["direction"]: float(row["mean"])
        for _, row in delta_summary.iterrows()
    }
    native_guard = all(
        row["delta"] >= -0.02
        for row in guardrail_rows
        if row["metric"] in {"native_macro_f1", "native_uar"}
    )
    spec_guard = all(
        row["delta"] >= -0.02
        for row in guardrail_rows
        if row["metric"] == "normal_specificity" and row["direction"] == "r2_spr_to_icbhi"
    )
    both_guard = all(
        row["delta"] >= -0.05
        for row in guardrail_rows
        if row["metric"] == "both_recall" and row["direction"] == "r2_spr_to_icbhi"
    )
    go = (
        delta_map.get("r1_icbhi_to_spr", -1.0) >= 0.02
        and delta_map.get("r2_spr_to_icbhi", -1.0) >= 0.02
        and bootstrap_map["r1_icbhi_to_spr"]["ci_lower"] > 0.0
        and bootstrap_map["r2_spr_to_icbhi"]["ci_lower"] > 0.0
        and native_guard
        and spec_guard
        and both_guard
    )
    partial = (
        (delta_map.get("r1_icbhi_to_spr", -1.0) >= 0.02 and bootstrap_map["r1_icbhi_to_spr"]["ci_lower"] > 0.0)
        or (delta_map.get("r2_spr_to_icbhi", -1.0) >= 0.02 and bootstrap_map["r2_spr_to_icbhi"]["ci_lower"] > 0.0)
    ) and not go
    decision = "go" if go else "partial" if partial else "no_go"
    receipt = {
        "status": "phase1a_analysis_complete",
        "decision": decision,
        "delta_aulc_mean": delta_map,
        "patient_cluster_interval": bootstrap_map,
        "native_guardrail_pass": native_guard,
        "icbhi_specificity_guardrail_pass": spec_guard,
        "icbhi_both_guardrail_pass": both_guard,
        "claim_boundary": [
            "local cached-feature target-supervised low-shot evidence",
            "Phase 1A only",
            "no joint/consistency",
            "no PAFA attribution",
            "no universal sharing claim",
        ],
    }
    write_json(result_root / "decision_receipt.json", receipt)
    return receipt


def run_audit(result_root: Path) -> dict[str, object]:
    cache_index, embeddings, cache_receipt = load_cache(CACHE_PATH)
    icbhi, spr, data_receipt = build_phase1_frames(cache_index)
    outer_assignment, nested_assignment, counts, strict_receipt = build_strict_icbhi_receipt()
    draw_receipts, retained_shots = build_draw_receipts(icbhi, spr, nested_assignment)
    global DRAW_RECEIPTS_GLOBAL
    DRAW_RECEIPTS_GLOBAL = draw_receipts
    write_json(result_root / "protocol.json", protocol_receipt(retained_shots))
    write_json(
        result_root / "data_receipt.json",
        {
            **data_receipt,
            "cache_embedded_receipt": cache_receipt,
            "embedding_shape": list(embeddings.shape),
        },
    )
    write_json(result_root / "strict_icbhi_split_receipt.json", strict_receipt)
    write_csv(
        result_root / "strict_icbhi_outer_fold_assignment.csv",
        outer_assignment.astype(str).to_dict("records"),
    )
    write_csv(
        result_root / "strict_icbhi_nested_assignment.csv",
        nested_assignment.astype(str).to_dict("records"),
    )
    write_csv(
        result_root / "strict_icbhi_fold_class_patient_counts.csv",
        counts.astype(str).to_dict("records"),
    )
    write_json(result_root / "shot_support_receipt.json", draw_receipts)
    write_json(result_root / "parameter_receipt.json", condition_parameter_receipt())
    receipt = {
        "status": "phase1a_protocol_data_split_audit_complete",
        "cache_sha256": CACHE_SHA256,
        "retained_shots": [str(value) for value in retained_shots],
        "conditions": list(CONDITIONS),
        "draws": len(DRAW_SEEDS),
        "seeds": OPTIMIZATION_SEEDS,
        "warnings": 0,
    }
    write_json(result_root / "audit_receipt.json", receipt)
    return receipt


def run_smoke(result_root: Path) -> dict[str, object]:
    cache_index, embeddings, _ = load_cache(CACHE_PATH)
    icbhi, spr, _ = build_phase1_frames(cache_index)
    _, nested_assignment, _, _ = build_strict_icbhi_receipt()
    draw_receipts, retained_shots = build_draw_receipts(icbhi, spr, nested_assignment)
    global DRAW_RECEIPTS_GLOBAL
    DRAW_RECEIPTS_GLOBAL = draw_receipts
    conditions = {}
    for condition in CONDITIONS:
        r1_dir = result_root / "smoke" / f"r1_cond{condition}"
        r2_dir = result_root / "smoke" / f"r2_cond{condition}"
        r1 = run_single(
            "r1_icbhi_to_spr",
            condition,
            retained_shots[0],
            0,
            OPTIMIZATION_SEEDS[0],
            icbhi,
            spr,
            nested_assignment,
            draw_receipts,
            embeddings,
            r1_dir,
            max_steps=SMOKE_STEPS,
            outer_fold=None,
            score_outer_test=False,
        )
        r2 = run_single(
            "r2_spr_to_icbhi",
            condition,
            retained_shots[0],
            0,
            OPTIMIZATION_SEEDS[0],
            icbhi,
            spr,
            nested_assignment,
            draw_receipts,
            embeddings,
            r2_dir,
            max_steps=SMOKE_STEPS,
            outer_fold=2,
            score_outer_test=False,
        )
        conditions[condition] = {
            "r1_best_step": r1["best_step"],
            "r2_best_step": r2["best_step"],
            "parameters": r1["parameter_receipt"],
            "gradient_finite": r1["gradient_finite"] and r2["gradient_finite"],
            "loss_finite": r1["loss_finite"] and r2["loss_finite"],
        }
    receipt = {
        "status": "phase1a_real_data_smoke_passed",
        "conditions": conditions,
        "warnings": 0,
        "outer_test_metrics_evaluated": False,
    }
    write_json(result_root / "smoke_receipt.json", receipt)
    return receipt


def run_profile(result_root: Path) -> dict[str, object]:
    cache_index, embeddings, _ = load_cache(CACHE_PATH)
    icbhi, spr, _ = build_phase1_frames(cache_index)
    _, nested_assignment, _, _ = build_strict_icbhi_receipt()
    draw_receipts, retained_shots = build_draw_receipts(icbhi, spr, nested_assignment)
    global DRAW_RECEIPTS_GLOBAL
    DRAW_RECEIPTS_GLOBAL = draw_receipts
    started = time.perf_counter()
    r1 = run_single(
        "r1_icbhi_to_spr",
        "I16",
        16 if 16 in retained_shots else retained_shots[0],
        0,
        OPTIMIZATION_SEEDS[0],
        icbhi,
        spr,
        nested_assignment,
        draw_receipts,
        embeddings,
        result_root / "profile" / "r1_i16",
        max_steps=MAX_STEPS,
        outer_fold=None,
        score_outer_test=True,
    )
    r2 = run_single(
        "r2_spr_to_icbhi",
        "I16",
        16 if 16 in retained_shots else retained_shots[0],
        0,
        OPTIMIZATION_SEEDS[0],
        icbhi,
        spr,
        nested_assignment,
        draw_receipts,
        embeddings,
        result_root / "profile" / "r2_i16_fold2",
        max_steps=MAX_STEPS,
        outer_fold=2,
        score_outer_test=True,
    )
    receipt = {
        "status": "phase1a_profile_complete",
        "total_runtime_seconds": float(time.perf_counter() - started),
        "r1_runtime_seconds": r1["runtime_seconds"],
        "r2_runtime_seconds": r2["runtime_seconds"],
        "max_steps": MAX_STEPS,
        "warnings": 0,
    }
    write_json(result_root / "profile_receipt.json", receipt)
    return receipt


def run_full(result_root: Path, resume: bool) -> dict[str, object]:
    cache_index, embeddings, _ = load_cache(CACHE_PATH)
    icbhi, spr, _ = build_phase1_frames(cache_index)
    _, nested_assignment, _, _ = build_strict_icbhi_receipt()
    draw_receipts, retained_shots = build_draw_receipts(icbhi, spr, nested_assignment)
    global DRAW_RECEIPTS_GLOBAL
    DRAW_RECEIPTS_GLOBAL = draw_receipts
    run_count = 0
    for condition in CONDITIONS:
        for shot in retained_shots:
            for draw_index in range(len(DRAW_SEEDS)):
                for seed in OPTIMIZATION_SEEDS:
                    r1_dir = result_root / "runs" / f"r1_icbhi_to_spr_shot{shot}_draw{draw_index}_seed{seed}_cond{condition}"
                    metrics_path = r1_dir / "metrics.json"
                    if not (resume and metrics_path.exists()):
                        run_single(
                            "r1_icbhi_to_spr",
                            condition,
                            shot,
                            draw_index,
                            seed,
                            icbhi,
                            spr,
                            nested_assignment,
                            draw_receipts,
                            embeddings,
                            r1_dir,
                            max_steps=MAX_STEPS,
                            outer_fold=None,
                            score_outer_test=True,
                        )
                    run_count += 1
                    for outer_fold in range(5):
                        r2_dir = result_root / "runs" / f"r2_spr_to_icbhi_shot{shot}_draw{draw_index}_seed{seed}_outer{outer_fold}_cond{condition}"
                        metrics_path = r2_dir / "metrics.json"
                        if not (resume and metrics_path.exists()):
                            run_single(
                                "r2_spr_to_icbhi",
                                condition,
                                shot,
                                draw_index,
                                seed,
                                icbhi,
                                spr,
                                nested_assignment,
                                draw_receipts,
                                embeddings,
                                r2_dir,
                                max_steps=MAX_STEPS,
                                outer_fold=outer_fold,
                                score_outer_test=True,
                            )
                        run_count += 1
    analysis = analyze(result_root, retained_shots)
    receipt = {
        "status": "phase1a_full_complete",
        "run_count": run_count,
        "conditions": list(CONDITIONS),
        "retained_shots": [str(value) for value in retained_shots],
        "draws": len(DRAW_SEEDS),
        "seeds": OPTIMIZATION_SEEDS,
        "warnings": 0,
        "decision": analysis["decision"],
    }
    write_json(result_root / "run_manifest.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "smoke", "profile", "full"], required=True)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    if args.mode == "audit":
        payload = run_audit(result_root)
    elif args.mode == "smoke":
        payload = run_smoke(result_root)
    elif args.mode == "profile":
        payload = run_profile(result_root)
    else:
        payload = run_full(result_root, resume=not args.no_resume)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
