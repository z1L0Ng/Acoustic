"""Corrected-HF harmonization regression using the Gate-B-selected R0 encoder."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    build_samples,
    sample_to_row,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import TASK_SPECS
from baseline.shared_compatible_head_harmonization.run import (
    COMPATIBLE_TASKS,
    CONDITIONS,
    HarmonizationModel,
    _gradient_route_checks,
    _ordered_sha,
    aggregate,
    analyze,
    smoke,
    train_full,
    write_gzip_csv,
    write_json,
)
from baseline.shared_compatible_head_harmonization.verify import _verify_pair


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "result/four_dataset_shared_compatible_head_harmonization/"
    "hf_proxy_fixed_v2"
)
SELECTED_CACHE = (
    ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
)
SELECTED_CACHE_SHA256 = (
    "f40ae7fe581457bc86d76b93b1ee811e7ea01bc5e098a6daa73db451f96d1b31"
)
GATE_B_SELECTION = (
    ROOT
    / "result/four_dataset_representation_attribution/hf_proxy_fixed_v2/"
    "comparison/encoder_selection.json"
)
ENCODER_IDENTITY = {
    "representation": "r0_pafa_icbhi_task_encoder",
    "cache_sha256": SELECTED_CACHE_SHA256,
    "encoder_frozen": True,
    "selection_caveat": "ICBHI official-test-selected PAFA task encoder",
    "claim_boundary": (
        "target-supervised frozen-feature evidence; not a clean neutral foundation "
        "representation or clean source-only generalization result"
    ),
}


def _gate_b_selection() -> dict[str, object]:
    selection = json.loads(GATE_B_SELECTION.read_text())
    if (
        selection["status"] != "encoder_selection_complete"
        or selection["selected_representation"]
        != ENCODER_IDENTITY["representation"]
        or selection["selected_cache_sha256"] != SELECTED_CACHE_SHA256
        or sha256_file(SELECTED_CACHE) != SELECTED_CACHE_SHA256
    ):
        raise RuntimeError("corrected Gate B R0 dependency failed")
    return selection


def data_audit(dataset_root: Path) -> tuple[list, np.ndarray]:
    selection = _gate_b_selection()
    folds = {}
    canonical = None
    for fold in range(5):
        samples, receipt = build_samples(dataset_root, fold)
        if (
            receipt["datasets"]["hf_lung"]["assignment_sha256"]
            != EXPECTED_HF_ASSIGNMENT_SHA256
        ):
            raise RuntimeError("corrected HF assignment failed")
        folds[str(fold)] = receipt
        if canonical is None:
            canonical = samples
            write_gzip_csv(
                RESULT_ROOT / "samples_fold_0.csv.gz",
                [sample_to_row(sample) for sample in samples],
            )
        elif [sample.sample_id for sample in samples] != [
            sample.sample_id for sample in canonical
        ]:
            raise RuntimeError("fold changed cache order")
    if canonical is None:
        raise RuntimeError("missing canonical samples")
    embeddings, cache_receipt = load_cache(SELECTED_CACHE, canonical)
    receipt = {
        "status": "hf_proxy_fixed_harmonization_data_audit_passed",
        "rows": len(canonical),
        "ordered_id_sha256": _ordered_sha(
            [sample.sample_id for sample in canonical]
        ),
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "gate_b_selection_receipt_sha256": sha256_file(GATE_B_SELECTION),
        "gate_b_selection": selection,
        "encoder_identity": ENCODER_IDENTITY,
        "cache_receipt": cache_receipt,
        "folds": folds,
    }
    write_json(RESULT_ROOT / "data_receipt.json", receipt)
    write_json(
        RESULT_ROOT / "protocol_override.json",
        {
            "status": "corrected_hf_selected_encoder_override_frozen",
            "selection_rule": "unchanged from representation attribution preregistration",
            "selected_encoder": ENCODER_IDENTITY,
            "harmonization_conditions": list(CONDITIONS),
            "parameter_mask_loss_selection_metric_policies": (
                "unchanged from baseline/shared_compatible_head_harmonization/"
                "protocol.json"
            ),
            "matrix_expansion": False,
        },
    )
    return canonical, embeddings


def _stamp(directory: Path) -> None:
    metrics_path = directory / "metrics.json"
    checkpoint_path = directory / "best.pth"
    metrics = json.loads(metrics_path.read_text())
    metrics["encoder_identity"] = ENCODER_IDENTITY
    write_json(metrics_path, metrics)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["encoder_identity"] = copy.deepcopy(ENCODER_IDENTITY)
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)


def run(dataset_root: Path, device: torch.device) -> dict[str, object]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    samples, embeddings = data_audit(dataset_root)
    smoke(samples, embeddings, dataset_root, RESULT_ROOT, device)
    for condition in CONDITIONS:
        _stamp(RESULT_ROOT / "smoke" / condition)
    train_full(dataset_root, RESULT_ROOT, samples, embeddings, device)
    for fold in range(5):
        for condition in CONDITIONS:
            _stamp(RESULT_ROOT / f"fold_{fold}" / condition)
    aggregate(RESULT_ROOT)
    return analyze(RESULT_ROOT)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify(dataset_root: Path) -> dict[str, object]:
    selection = _gate_b_selection()
    data = json.loads((RESULT_ROOT / "data_receipt.json").read_text())
    smoke_receipt = json.loads((RESULT_ROOT / "smoke_receipt.json").read_text())
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    canonical = samples_by_fold[0]
    values, _ = load_cache(SELECTED_CACHE, canonical)
    if (
        values.shape != (25_084, 768)
        or not np.isfinite(values).all()
        or data["encoder_identity"] != ENCODER_IDENTITY
        or data["hf_assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256
        or data["ordered_id_sha256"]
        != _ordered_sha([sample.sample_id for sample in canonical])
        or selection["selected_representation"]
        != "r0_pafa_icbhi_task_encoder"
    ):
        raise RuntimeError("Gate C data/cache/selection verification failed")
    route = _gradient_route_checks(torch.device("cpu"))
    if (
        smoke_receipt["status"] != "harmonization_real_data_smoke_passed"
        or not smoke_receipt["finite"]
        or not smoke_receipt["spr_label_free_then_terminal_join"]
        or not route["hf_kauh_missing_compatible_label_zero_gradient"]
        or route["h1_compatible_parameters"] != 1548
        or route["h2_compatible_parameters"] != 1548
    ):
        raise RuntimeError("Gate C smoke/routing verification failed")
    for condition in CONDITIONS:
        directory = RESULT_ROOT / "smoke" / condition
        metrics = json.loads((directory / "metrics.json").read_text())
        checkpoint = torch.load(directory / "best.pth", map_location="cpu")
        if (
            metrics["encoder_identity"] != ENCODER_IDENTITY
            or checkpoint["encoder_identity"] != ENCODER_IDENTITY
        ):
            raise RuntimeError("smoke encoder caveat missing")
        _verify_pair(directory, canonical, metrics)

    prediction_pairs = 0
    kauh_sets = []
    for fold, samples in enumerate(samples_by_fold):
        kauh_sets.append(
            {
                sample.sample_id
                for sample in samples
                if sample.dataset == "kauh" and sample.partition == "test"
            }
        )
        for condition in CONDITIONS:
            directory = RESULT_ROOT / f"fold_{fold}" / condition
            metrics = json.loads((directory / "metrics.json").read_text())
            checkpoint = torch.load(directory / "best.pth", map_location="cpu")
            model = HarmonizationModel(condition)
            if (
                metrics["encoder_identity"] != ENCODER_IDENTITY
                or checkpoint["encoder_identity"] != ENCODER_IDENTITY
                or metrics["parameters"]["compatible"]
                != model.compatible_parameter_count()
                or any("test" in json.dumps(row).lower() for row in metrics["history"])
                or "validation" not in json.dumps(metrics["selection"]).lower()
                or not all(
                    torch.isfinite(value).all()
                    for value in checkpoint["model"].values()
                    if torch.is_tensor(value)
                )
            ):
                raise RuntimeError("Gate C full checkpoint/selection gate failed")
            _verify_pair(directory, samples, metrics)
            prediction_pairs += 1
    if (
        len(set().union(*kauh_sets)) != 336
        or sum(map(len, kauh_sets)) != 336
        or any(
            kauh_sets[left] & kauh_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("Gate C KAUH OOF failed")

    before = json.loads((RESULT_ROOT / "harmonization_decision.json").read_text())
    recomputed = analyze(RESULT_ROOT)
    if before != recomputed:
        raise RuntimeError("Gate C decision is not independently reproducible")
    summary = _read_csv(RESULT_ROOT / "summary.csv")
    per_class = _read_csv(RESULT_ROOT / "per_class_summary.csv")
    if len(summary) != len(CONDITIONS) * (
        len(TASK_SPECS) + len(COMPATIBLE_TASKS)
    ):
        raise RuntimeError("Gate C summary row count failed")
    receipt = {
        "status": "hf_proxy_fixed_r0_harmonization_verified",
        "encoder_identity": ENCODER_IDENTITY,
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "prediction_pairs": prediction_pairs,
        "kauh_oof_rows": 336,
        "summary_rows": len(summary),
        "per_class_rows": len(per_class),
        "decision": recomputed["decision"],
        "gates": recomputed["gates"],
    }
    write_json(RESULT_ROOT / "verification.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["run", "verify", "all"], default="all")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    if args.phase in {"run", "all"}:
        run(dataset_root, torch.device(args.device))
    if args.phase in {"verify", "all"}:
        receipt = verify(dataset_root)
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
