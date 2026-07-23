from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXTENSION_ARCHITECTURE_ROOT, EXTENSION_FEATURES, EXTENSION_IMBALANCE_ROOT,
    EXTENSION_RESULT_ROOT, FEATURES, MANIFEST_PATH,
)
from .data import load_aligned_features, load_manifest, provenance_row, sha256_file
from .imbalance import consolidate_imbalance_results, run_imbalance_backbone
from .run import consolidate_full_results, run_backbone


EXTENSION_BACKBONES = tuple(EXTENSION_FEATURES)
ARCHITECTURE_SCOPE = (
    "Formal local controlled downstream extension for simple acoustic, HeAR, and "
    "OPERA-CT official-like features; not encoder-paper reproduction."
)
IMBALANCE_SCOPE = (
    "Formal local controlled loss extension for simple acoustic, HeAR, and OPERA-CT "
    "official-like features; not loss-paper or encoder-paper reproduction."
)


def audit_extension_inputs() -> Path:
    manifest = load_manifest()
    rows = []
    for backbone, config in EXTENSION_FEATURES.items():
        features, metadata = load_aligned_features(
            manifest,
            Path(config["path"]),
            config["key"],
            config["expected_dim"],
            allow_pickle_metadata=config.get("allow_pickle_metadata", False),
            expected_sha256=config.get("expected_sha256"),
        )
        with np.load(config["path"], allow_pickle=config.get("allow_pickle_metadata", False)) as archive:
            ids = archive["cycle_id"].astype(str)
            position = {value: index for index, value in enumerate(ids)}
            order = np.asarray([position[value] for value in manifest["cycle_id"]])
            labels_match = np.array_equal(
                archive["native_four_class_label"][order].astype(str),
                manifest["native_four_class_label"].astype(str).to_numpy(),
            ) and np.array_equal(
                archive["binary_label"][order].astype(str),
                manifest["binary_label"].astype(str).to_numpy(),
            )
            split_match = np.array_equal(
                archive["official_split"][order].astype(str),
                manifest["official_split"].astype(str).to_numpy(),
            )
            usable_all = bool(archive["usable"].all()) if "usable" in archive.files else True
            object_keys = [key for key in archive.files if archive[key].dtype == object]
            object_values_are_strings = all(
                all(isinstance(item, str) for item in archive[key].tolist())
                for key in object_keys
            )
        provenance = provenance_row(backbone, config, metadata)
        rows.append({
            **provenance,
            "representation": backbone,
            "rows": len(features),
            "feature_dim": features.shape[1],
            "cycle_id_unique": len(ids) == len(set(ids)),
            "cycle_id_set_matches_manifest": set(ids) == set(manifest["cycle_id"]),
            "labels_match_manifest_after_join": labels_match,
            "split_matches_manifest_after_join": split_match,
            "features_finite": bool(np.isfinite(features).all()),
            "usable_all": usable_all,
            "object_metadata_keys_json": json.dumps(object_keys),
            "object_metadata_values_are_strings": object_values_are_strings,
            "sha256_matches_registry": sha256_file(Path(config["path"])) == config["expected_sha256"],
            "official_reproduction_claim": False,
        })
    output = EXTENSION_RESULT_ROOT / "comparison/input_provenance_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("representation").to_csv(output, index=False)
    return output


def run_extension_architecture_backbone(backbone: str, mode: str) -> list[dict]:
    if backbone not in EXTENSION_BACKBONES:
        raise ValueError(backbone)
    return run_backbone(
        backbone,
        mode,
        result_base=EXTENSION_ARCHITECTURE_ROOT,
        protocol_version="formal-downstream-v2-extension",
        scope=ARCHITECTURE_SCOPE,
    )


def run_extension_architecture(mode: str) -> list[dict]:
    rows = []
    for backbone in EXTENSION_BACKBONES:
        rows.extend(run_extension_architecture_backbone(backbone, mode))
    if mode == "full":
        consolidate_full_results(
            list(EXTENSION_BACKBONES),
            result_base=EXTENSION_ARCHITECTURE_ROOT,
            result_filename="formal_extension_architecture_results.csv",
            summary_filename="formal_extension_architecture_summary.csv",
        )
        write_architecture_seed_deltas()
    return rows


def run_extension_imbalance_backbone(backbone: str, mode: str) -> list[dict]:
    if backbone not in EXTENSION_BACKBONES:
        raise ValueError(backbone)
    return run_imbalance_backbone(
        backbone,
        mode,
        result_base=EXTENSION_IMBALANCE_ROOT,
        protocol_version="formal-imbalance-loss-v2-extension",
        scope=IMBALANCE_SCOPE,
    )


def run_extension_imbalance(mode: str) -> list[dict]:
    rows = []
    for backbone in EXTENSION_BACKBONES:
        rows.extend(run_extension_imbalance_backbone(backbone, mode))
    if mode == "full":
        consolidate_imbalance_results(
            list(EXTENSION_BACKBONES),
            result_base=EXTENSION_IMBALANCE_ROOT,
            result_filename="formal_extension_imbalance_results.csv",
            summary_filename="formal_extension_imbalance_summary.csv",
        )
    return rows


def write_architecture_seed_deltas() -> Path:
    path = EXTENSION_ARCHITECTURE_ROOT / "comparison/formal_extension_architecture_results.csv"
    results = pd.read_csv(path)
    metrics = [
        "macro_f1", "weighted_f1", "uar", "both_recall", "normal_specificity", "icbhi_score",
    ]
    rows = []
    for (backbone, task), frame in results.groupby(["backbone", "task"]):
        lr = frame[frame["head"].eq("lr")].iloc[0]
        for _, row in frame[frame["head"].isin(["mlp1", "mlp2"])].iterrows():
            receipt = {"backbone": backbone, "task": task, "head": row["head"], "seed": row["seed"]}
            for metric in metrics:
                if pd.isna(row[metric]) or pd.isna(lr[metric]):
                    receipt[f"delta_{metric}_vs_lr"] = np.nan
                    receipt[f"win_{metric}_vs_lr"] = False
                else:
                    delta = row[metric] - lr[metric]
                    receipt[f"delta_{metric}_vs_lr"] = delta
                    receipt[f"win_{metric}_vs_lr"] = bool(delta > 0)
            rows.append(receipt)
    output = EXTENSION_ARCHITECTURE_ROOT / "comparison/seed_level_deltas.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_six_representation_receipt() -> Path:
    old_arch = pd.read_csv(
        EXTENSION_RESULT_ROOT.parent / "2026-07-12/comparison/formal_downstream_summary.csv"
    )
    new_arch = pd.read_csv(
        EXTENSION_ARCHITECTURE_ROOT / "comparison/formal_extension_architecture_summary.csv"
    )
    old_loss = pd.read_csv(
        EXTENSION_RESULT_ROOT.parent / "2026-07-12/imbalance/comparison/formal_imbalance_summary.csv"
    )
    new_loss = pd.read_csv(
        EXTENSION_IMBALANCE_ROOT / "comparison/formal_extension_imbalance_summary.csv"
    )
    architecture = pd.concat([old_arch, new_arch], ignore_index=True)
    losses = pd.concat([old_loss, new_loss], ignore_index=True)
    rows = []
    for representation in architecture["backbone"].unique():
        arch = architecture[architecture["backbone"].eq(representation)]
        loss = losses[losses["backbone"].eq(representation)]
        flat = arch[arch["task"].eq("flat4")].sort_values("macro_f1_mean", ascending=False).iloc[0]
        binary = arch[arch["task"].eq("binary")].sort_values("macro_f1_mean", ascending=False).iloc[0]
        best_loss = loss.sort_values("macro_f1_mean", ascending=False).iloc[0]
        best_both = loss.sort_values("both_recall_mean", ascending=False).iloc[0]
        best_score = loss.sort_values("icbhi_score_mean", ascending=False).iloc[0]
        rows.append({
            "representation": representation,
            "selection_note": "descriptive official-test summary only; not a training selection rule",
            "best_flat4_head_by_macro_f1": flat["head"],
            "best_flat4_head_macro_f1": flat["macro_f1_mean"],
            "best_flat4_head_uar": flat["uar_mean"],
            "best_flat4_head_both_recall": flat["both_recall_mean"],
            "best_binary_head_by_macro_f1": binary["head"],
            "best_binary_head_macro_f1": binary["macro_f1_mean"],
            "best_binary_head_uar": binary["uar_mean"],
            "best_loss_by_macro_f1": best_loss["loss"],
            "best_loss_macro_f1": best_loss["macro_f1_mean"],
            "best_loss_uar": best_loss["uar_mean"],
            "best_loss_both_recall": best_loss["both_recall_mean"],
            "best_loss_normal_specificity": best_loss["normal_specificity_mean"],
            "best_loss_icbhi_score": best_loss["icbhi_score_mean"],
            "best_loss_by_both_recall": best_both["loss"],
            "highest_both_recall": best_both["both_recall_mean"],
            "best_loss_by_icbhi_score": best_score["loss"],
            "highest_icbhi_score": best_score["icbhi_score_mean"],
        })
    output = EXTENSION_RESULT_ROOT / "comparison/six_representation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("representation").to_csv(output, index=False)
    return output
