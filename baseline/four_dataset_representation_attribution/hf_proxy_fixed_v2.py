"""Corrected-HF cached-feature regression for representation attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import (
    load_cache,
    sha256_file,
)
from baseline.four_dataset_frozen_encoder.verify import _verify_prediction_pair
from baseline.four_dataset_representation_attribution.run import (
    ALL_REPRESENTATIONS,
    CONDITION,
    EXPECTED_R0_SHA256,
    EXPECTED_ROWS,
    REPRESENTATIONS,
    R0,
    analyze,
    data_audit,
    ordered_id_sha256,
    train,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT / "result/four_dataset_representation_attribution/hf_proxy_fixed_v2"
)
CACHE_ROOT = ROOT / ".cache/four_dataset_representation_attribution"
R0_CACHE = ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
GATE_A_RECEIPT = (
    ROOT
    / "result/four_dataset_pafa_frozen_encoder/"
    "hf_proxy_fixed_v2/impact_receipt.json"
)


def _assert_gate_a() -> dict[str, object]:
    receipt = json.loads(GATE_A_RECEIPT.read_text())
    if (
        receipt["status"] != "hf_proxy_fix_and_impact_audit_verified"
        or receipt["hf_proxy"]["assignment_sha256"]
        != EXPECTED_HF_ASSIGNMENT_SHA256
        or receipt["tail_contract"]["training_allowed"]
    ):
        raise RuntimeError("Gate A dependency failed")
    return receipt


def run(dataset_root: Path, device: torch.device) -> dict[str, object]:
    _assert_gate_a()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    samples, receipt = data_audit(dataset_root, RESULT_ROOT)
    if (
        receipt["folds"]["0"]["datasets"]["hf_lung"]["assignment_sha256"]
        != EXPECTED_HF_ASSIGNMENT_SHA256
    ):
        raise RuntimeError("corrected data audit assignment failed")
    train(
        samples,
        dataset_root,
        R0_CACHE,
        CACHE_ROOT,
        RESULT_ROOT,
        device,
    )
    return analyze(RESULT_ROOT, CACHE_ROOT, R0_CACHE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify(dataset_root: Path) -> dict[str, object]:
    gate_a = _assert_gate_a()
    data = json.loads((RESULT_ROOT / "data_receipt.json").read_text())
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    canonical = samples_by_fold[0]
    ids = [sample.sample_id for sample in canonical]
    if (
        data["rows"] != EXPECTED_ROWS
        or data["ordered_id_sha256"] != ordered_id_sha256(ids)
        or any(
            fold["datasets"]["hf_lung"]["assignment_sha256"]
            != EXPECTED_HF_ASSIGNMENT_SHA256
            for fold in data["folds"].values()
        )
    ):
        raise RuntimeError("corrected data receipt verification failed")

    cache_receipts = {}
    cache_paths = {
        R0: R0_CACHE,
        REPRESENTATIONS[0]: CACHE_ROOT / REPRESENTATIONS[0] / "embeddings.npz",
        REPRESENTATIONS[1]: CACHE_ROOT / REPRESENTATIONS[1] / "embeddings.npz",
    }
    for representation, path in cache_paths.items():
        values, _ = load_cache(path, canonical)
        if values.shape != (EXPECTED_ROWS, 768) or not np.isfinite(values).all():
            raise RuntimeError(f"cache shape/finite failed: {representation}")
        digest = sha256_file(path)
        if representation == R0 and digest != EXPECTED_R0_SHA256:
            raise RuntimeError("R0 cache identity failed")
        if digest != gate_a["caches"][representation]["sha256"]:
            raise RuntimeError(f"Gate A cache identity changed: {representation}")
        cache_receipts[representation] = digest

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
        for representation in ALL_REPRESENTATIONS:
            directory = RESULT_ROOT / representation / f"fold_{fold}" / CONDITION
            metrics = json.loads((directory / "metrics.json").read_text())
            checkpoint = torch.load(directory / "best.pth", map_location="cpu")
            if (
                metrics["condition"] != CONDITION
                or any("test" in json.dumps(row).lower() for row in metrics["history"])
                or "validation" not in json.dumps(metrics["selection"]).lower()
                or not all(
                    torch.isfinite(value).all()
                    for value in checkpoint["model"].values()
                    if torch.is_tensor(value)
                )
            ):
                raise RuntimeError("training/selection/checkpoint gate failed")
            _verify_prediction_pair(
                directory / "predictions_label_free.csv.gz",
                directory / "predictions.csv.gz",
                samples,
                metrics["test_metrics"],
            )
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
        raise RuntimeError("KAUH OOF identity failed")

    before = json.loads(
        (RESULT_ROOT / "comparison/encoder_selection.json").read_text()
    )
    recomputed = analyze(RESULT_ROOT, CACHE_ROOT, R0_CACHE)
    if before != recomputed:
        raise RuntimeError("corrected encoder selection is not reproducible")
    results = _read_csv(RESULT_ROOT / "comparison/representation_results.csv")
    gaps = _read_csv(RESULT_ROOT / "comparison/task_gaps.csv")
    if len(results) != 18 or len(gaps) != 6:
        raise RuntimeError("corrected comparison row count failed")
    receipt = {
        "status": "hf_proxy_fixed_representation_regression_verified",
        "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        "rows": EXPECTED_ROWS,
        "cache_sha256": cache_receipts,
        "prediction_pairs": prediction_pairs,
        "kauh_oof_rows": 336,
        "task_rows": len(results),
        "gap_rows": len(gaps),
        "selected_representation": recomputed["selected_representation"],
        "selected_cache_sha256": recomputed["selected_cache_sha256"],
        "old_results_preserved": True,
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
