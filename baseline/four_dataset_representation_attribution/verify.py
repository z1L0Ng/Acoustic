"""Independent verification for four-dataset representation attribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import build_samples
from baseline.four_dataset_frozen_encoder.encoder import sha256_file
from baseline.four_dataset_frozen_encoder.verify import _verify_prediction_pair
from baseline.four_dataset_representation_attribution.run import (
    ALL_REPRESENTATIONS,
    CONDITION,
    DATASETS,
    EXPECTED_R0_SHA256,
    EXPECTED_ROWS,
    EXPERIMENT_ID,
    PROTOCOL_PATH,
    R0,
    REPRESENTATIONS,
    analyze,
    ordered_id_sha256,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_gate(result_root: Path, dataset_root: Path) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    data = json.loads((result_root / "data_receipt.json").read_text())
    smoke = json.loads((result_root / "smoke_receipt.json").read_text())
    profile = json.loads((result_root / "profile_receipt.json").read_text())
    samples, _ = build_samples(dataset_root, 0)
    if (
        data["status"] != "representation_attribution_data_audit_passed"
        or data["rows"] != EXPECTED_ROWS
        or data["protocol_sha256"] != sha256_file(PROTOCOL_PATH)
        or data["ordered_id_sha256"]
        != ordered_id_sha256([sample.sample_id for sample in samples])
    ):
        raise RuntimeError("data/prereg gate failed")
    if (
        smoke["status"] != "representation_attribution_smoke_passed"
        or set(smoke["representations"]) != set(ALL_REPRESENTATIONS)
        or not profile["all_new_representations_allowed"]
    ):
        raise RuntimeError("smoke/profile status failed")
    for representation, row in smoke["representations"].items():
        if (
            not row["finite_loss_gradient_checkpoint"]
            or not row["spr_label_free_then_terminal_join"]
        ):
            raise RuntimeError(f"smoke invariant failed: {representation}")
        directory = result_root / "smoke" / representation / CONDITION
        metrics = json.loads((directory / "metrics.json").read_text())
        _verify_prediction_pair(
            directory / "predictions_label_free.csv.gz",
            directory / "predictions.csv.gz",
            samples,
            metrics["test_metrics"],
        )
    for representation in REPRESENTATIONS:
        identity = smoke["representations"][representation]["encoder"]
        expected_loaded = (
            protocol["representations"][representation]["loaded_pretrained_tensors"]
        )
        if (
            identity["loaded_pretrained_tensors"] != expected_loaded
            or not identity["encoder_frozen"]
            or identity["source_task_states_present"]
        ):
            raise RuntimeError(f"initialization gate failed: {representation}")
        if representation.startswith("r1_"):
            if identity["final_state_digest"] != identity["pretrained_state_digest"]:
                raise RuntimeError("R1 did not equal AudioSet state")
        elif (
            identity["final_state_digest"] != identity["random_initial_state_digest"]
            or identity["final_state_digest"] == identity["pretrained_state_digest"]
            or identity["seed"] != 20260728
        ):
            raise RuntimeError("R2 random initialization identity failed")
    return {
        "status": "representation_attribution_gate_verified",
        "rows": EXPECTED_ROWS,
        "representations": len(ALL_REPRESENTATIONS),
        "smoke_prediction_pairs": len(ALL_REPRESENTATIONS),
        "profile_local_full_allowed": True,
    }


def verify_full(
    result_root: Path,
    cache_root: Path,
    dataset_root: Path,
    r0_cache: Path,
) -> dict[str, object]:
    gate = verify_gate(result_root, dataset_root)
    samples_by_fold = [build_samples(dataset_root, fold)[0] for fold in range(5)]
    canonical = samples_by_fold[0]
    canonical_ids = [sample.sample_id for sample in canonical]
    if sha256_file(r0_cache) != EXPECTED_R0_SHA256:
        raise RuntimeError("R0 immutable cache changed")
    for representation in REPRESENTATIONS:
        receipt = json.loads(
            (result_root / representation / "embedding_receipt.json").read_text()
        )
        cache_path = cache_root / representation / "embeddings.npz"
        archive = np.load(cache_path, allow_pickle=False)
        if (
            receipt["cache_sha256"] != sha256_file(cache_path)
            or archive["sample_ids"].astype(str).tolist() != canonical_ids
            or archive["embeddings"].shape != (EXPECTED_ROWS, 768)
            or not np.isfinite(archive["embeddings"]).all()
        ):
            raise RuntimeError(f"combined cache gate failed: {representation}")
        identity = receipt["encoder"]
        if representation.startswith("r1_"):
            if (
                identity["loaded_pretrained_tensors"] != 250
                or identity["final_state_digest"] != identity["pretrained_state_digest"]
            ):
                raise RuntimeError("R1 full initialization gate failed")
        elif (
            identity["loaded_pretrained_tensors"] != 0
            or identity["final_state_digest"] != identity["random_initial_state_digest"]
            or identity["final_state_digest"] == identity["pretrained_state_digest"]
        ):
            raise RuntimeError("R2 full initialization gate failed")
        final_values = dict(
            zip(canonical_ids, archive["embeddings"].astype(np.float32))
        )
        for dataset in DATASETS:
            shard_path = (
                cache_root / representation / "embedding_shards" / f"{dataset}.npz"
            )
            shard_receipt = json.loads(
                (
                    result_root
                    / representation
                    / "embedding_shards"
                    / f"{dataset}.json"
                ).read_text()
            )
            shard = np.load(shard_path, allow_pickle=False)
            ids = shard["sample_ids"].astype(str).tolist()
            expected = [
                sample.sample_id for sample in canonical if sample.dataset == dataset
            ]
            values = shard["embeddings"].astype(np.float32)
            if (
                ids != expected
                or sha256_file(shard_path) != shard_receipt["cache_sha256"]
                or not np.isfinite(values).all()
                or not all(
                    np.array_equal(value, final_values[sample_id])
                    for sample_id, value in zip(ids, values)
                )
            ):
                raise RuntimeError(f"shard gate failed: {representation}/{dataset}")
    prediction_pairs = 0
    kauh_sets = []
    for fold, samples in enumerate(samples_by_fold):
        kauh = {
            sample.sample_id
            for sample in samples
            if sample.dataset == "kauh" and sample.partition == "test"
        }
        kauh_sets.append(kauh)
        for representation in ALL_REPRESENTATIONS:
            directory = result_root / representation / f"fold_{fold}" / CONDITION
            metrics = json.loads((directory / "metrics.json").read_text())
            state = torch.load(directory / "best.pth", map_location="cpu")
            if (
                metrics["condition"] != CONDITION
                or any("test" in json.dumps(row).lower() for row in metrics["history"])
                or "validation" not in json.dumps(metrics["selection"]).lower()
                or not all(
                    torch.isfinite(value).all()
                    for value in state["model"].values()
                    if torch.is_tensor(value)
                )
            ):
                raise RuntimeError("training/selection state gate failed")
            _verify_prediction_pair(
                directory / "predictions_label_free.csv.gz",
                directory / "predictions.csv.gz",
                samples,
                metrics["test_metrics"],
            )
            prediction_pairs += 1
    if (
        len(set().union(*kauh_sets)) != 336
        or sum(len(values) for values in kauh_sets) != 336
        or any(
            kauh_sets[left] & kauh_sets[right]
            for left in range(5)
            for right in range(left)
        )
    ):
        raise RuntimeError("KAUH OOF split identity failed")
    before = json.loads(
        (result_root / "comparison" / "encoder_selection.json").read_text()
    )
    recomputed = analyze(result_root, cache_root, r0_cache)
    if before != recomputed:
        raise RuntimeError("encoder selection is not independently reproducible")
    if recomputed["selected_representation"] not in {R0, REPRESENTATIONS[0]}:
        raise RuntimeError("invalid selected representation")
    results = _read_csv(result_root / "comparison" / "representation_results.csv")
    gaps = _read_csv(result_root / "comparison" / "task_gaps.csv")
    if len(results) != 18 or len(gaps) != 6:
        raise RuntimeError("comparison row count failed")
    return {
        **gate,
        "status": "representation_attribution_full_verified",
        "full_prediction_pairs": prediction_pairs,
        "cache_rows": EXPECTED_ROWS,
        "new_cache_shards": len(REPRESENTATIONS) * len(DATASETS),
        "task_rows": len(results),
        "task_gap_rows": len(gaps),
        "selected_representation": recomputed["selected_representation"],
        "selected_cache_sha256": recomputed["selected_cache_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gate", "full"], default="full")
    parser.add_argument(
        "--result-root", type=Path, default=Path(f"result/{EXPERIMENT_ID}")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(f".cache/{EXPERIMENT_ID}")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--r0-cache",
        type=Path,
        default=Path(".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"),
    )
    args = parser.parse_args()
    receipt = (
        verify_gate(args.result_root.resolve(), args.dataset_root.resolve())
        if args.mode == "gate"
        else verify_full(
            args.result_root.resolve(),
            args.cache_root.resolve(),
            args.dataset_root.resolve(),
            args.r0_cache.resolve(),
        )
    )
    output = args.result_root / (
        "gate_verification.json" if args.mode == "gate" else "verification.json"
    )
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
