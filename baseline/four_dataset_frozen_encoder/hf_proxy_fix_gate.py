"""Verify the HF date-proxy correction and enumerate superseded evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    EXPECTED_HF_PARTITION_PROXY_COUNTS,
    EXPECTED_HF_SOURCE_PROXY_COUNTS,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import (
    load_cache,
    sha256_file,
)


EXPECTED_ROWS = 25_084
EXPECTED_HF_ROWS = 9_765
EXPECTED_HF_PARTITIONS = {
    "subtrain": 5_322,
    "validation": 2_487,
    "test": 1_956,
}
EXPECTED_OLD_HF_PROXY_COUNTS = {
    "subtrain": 104,
    "validation": 26,
    "test": 41,
}
CACHE_PATHS = {
    "r0_pafa_icbhi_task_encoder": Path(
        ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
    ),
    "r1_beats_as2m_audioset_only": Path(
        ".cache/four_dataset_representation_attribution/"
        "r1_beats_as2m_audioset_only/embeddings.npz"
    ),
    "r2_beats_random_init_sanity": Path(
        ".cache/four_dataset_representation_attribution/"
        "r2_beats_random_init_sanity/embeddings.npz"
    ),
}
IMPACTED_EVIDENCE = {
    "four_dataset_frozen_encoder_d0_d3": (
        "result/four_dataset_pafa_frozen_encoder"
    ),
    "representation_attribution_r0_r1_r2": (
        "result/four_dataset_representation_attribution"
    ),
    "shared_compatible_harmonization_h0_h1_h2": (
        "result/four_dataset_shared_compatible_head_harmonization"
    ),
    "model_design_cached_feature_general_specific_v1": (
        "result/model/2026-07-28/four_dataset_general_specific_v1"
    ),
}


def _sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _read_old_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("legacy sample manifest row count changed")
    return rows


def _contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    contract = payload["tail_eligibility_contract"]
    hf = contract["split_receipts"]["hf_lung"]
    if (
        contract["status"] != "tail_eligibility_management_accepted"
        or contract["baseline_t1_gate"]["training_allowed"]
        or contract["baseline_t1_gate"]["status"]
        != "blocked_pending_hf_proxy_fix_and_regression_verification"
        or hf["assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256
        or hf["legacy_130_41_proxy_grouping"]
        != "rejected_not_accepted_split_evidence"
    ):
        raise RuntimeError("tail eligibility management contract gate failed")
    return contract


def verify(
    dataset_root: Path,
    old_manifest: Path,
    contract_path: Path,
    output_path: Path,
) -> dict[str, object]:
    contract = _contract(contract_path)
    samples, receipt = build_samples(dataset_root, 0)
    if len(samples) != EXPECTED_ROWS:
        raise RuntimeError("corrected sample row count failed")
    hf_receipt = receipt["datasets"]["hf_lung"]
    if (
        hf_receipt["rows"] != EXPECTED_HF_ROWS
        or hf_receipt["partition"] != EXPECTED_HF_PARTITIONS
        or hf_receipt["source_date_proxy_counts"]
        != EXPECTED_HF_SOURCE_PROXY_COUNTS
        or hf_receipt["date_proxy_counts"]
        != EXPECTED_HF_PARTITION_PROXY_COUNTS
        or hf_receipt["assignment_sha256"] != EXPECTED_HF_ASSIGNMENT_SHA256
        or hf_receipt["group_overlap"] != 0
        or hf_receipt["date_proxy_identity"]
        != "deidentified grouping proxy; not patient_id"
    ):
        raise RuntimeError("corrected HF split receipt failed")

    old_rows = _read_old_rows(old_manifest)
    old_ids = [row["sample_id"] for row in old_rows]
    new_ids = [sample.sample_id for sample in samples]
    if old_ids != new_ids or len(new_ids) != len(set(new_ids)):
        raise RuntimeError("old/new ordered sample identity changed")
    old_by_id = {row["sample_id"]: row for row in old_rows}
    changed_partitions = []
    changed_groups = []
    for sample in samples:
        old = old_by_id[sample.sample_id]
        if (
            old["audio_path"] != sample.audio_path
            or old["crop_start_s"]
            != ("" if sample.crop_start_s is None else str(sample.crop_start_s))
            or old["crop_end_s"]
            != ("" if sample.crop_end_s is None else str(sample.crop_end_s))
            or json.loads(old["targets_json"]) != sample.targets
        ):
            raise RuntimeError(f"sample input/target changed: {sample.sample_id}")
        if old["partition"] != sample.partition:
            changed_partitions.append(sample.sample_id)
        if old["group_id"] != sample.group_id:
            changed_groups.append(sample.sample_id)
        if sample.dataset != "hf_lung" and (
            old["partition"] != sample.partition
            or old["group_id"] != sample.group_id
        ):
            raise RuntimeError("non-HF split/group changed")
    legacy_receipt = json.loads(
        (
            old_manifest.parent
            / "data_receipt.json"
        ).read_text()
    )["folds"]["0"]["datasets"]["hf_lung"]
    if legacy_receipt["date_proxy_counts"] != EXPECTED_OLD_HF_PROXY_COUNTS:
        raise RuntimeError("legacy rejected 130/41 grouping receipt changed")

    caches = {}
    for name, relative_path in CACHE_PATHS.items():
        path = relative_path.resolve()
        embeddings, _ = load_cache(path, samples)
        if (
            embeddings.shape != (EXPECTED_ROWS, 768)
            or not np.isfinite(embeddings).all()
        ):
            raise RuntimeError(f"cache finite/shape gate failed: {name}")
        caches[name] = {
            "path": str(relative_path),
            "sha256": sha256_file(path),
            "shape": list(embeddings.shape),
            "ordered_id_sha256": _sha_lines(new_ids),
            "sample_ids_exactly_aligned_after_fix": True,
            "audio_inputs_unchanged": True,
            "embedding_reextraction_required": False,
        }

    impacted = {
        name: {
            "path": path,
            "status": "pre_fix_exploratory_superseded_pending_corrected_regression",
            "reason": (
                "downstream validation/training membership used rejected mixed-format "
                "HF date-proxy grouping"
            ),
        }
        for name, path in IMPACTED_EVIDENCE.items()
    }
    assignment_rows = [
        (
            f"{sample.metadata['source_split']}:"
            f"{sample.sample_id.split(':', 2)[2]}\t"
            f"{sample.partition}\t"
            f"{str(sample.metadata['date_proxy']).replace('-', '')}"
        )
        for sample in samples
        if sample.dataset == "hf_lung"
    ]
    assignment_bytes = len(("\n".join(assignment_rows) + "\n").encode("utf-8"))
    if assignment_bytes != 502_441:
        raise RuntimeError("accepted assignment serialization length changed")

    result = {
        "status": "hf_proxy_fix_and_impact_audit_verified",
        "tail_contract": {
            "status": contract["status"],
            "training_allowed": contract["baseline_t1_gate"]["training_allowed"],
            "t1_gate": contract["baseline_t1_gate"]["status"],
            "accepted_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
        },
        "hf_proxy": {
            "identity": "deidentified date grouping proxy; not patient_id",
            "format_in_samples": "YYYY-MM-DD",
            "accepted_digest_proxy_format": "YYYYMMDD",
            "source_train_test_proxies": EXPECTED_HF_SOURCE_PROXY_COUNTS,
            "subtrain_validation_test_proxies": EXPECTED_HF_PARTITION_PROXY_COUNTS,
            "subtrain_validation_test_rows": EXPECTED_HF_PARTITIONS,
            "partition_overlap": 0,
            "assignment_rows": EXPECTED_HF_ROWS,
            "assignment_serialized_bytes": assignment_bytes,
            "assignment_sha256": hf_receipt["assignment_sha256"],
            "legacy_grouping": {
                "date_proxy_counts": EXPECTED_OLD_HF_PROXY_COUNTS,
                "status": "rejected_not_accepted_split_evidence",
            },
        },
        "alignment": {
            "rows": EXPECTED_ROWS,
            "ordered_id_sha256": _sha_lines(new_ids),
            "old_new_ordered_ids_equal": True,
            "old_new_audio_inputs_equal": True,
            "old_new_targets_equal": True,
            "changed_partition_rows": len(changed_partitions),
            "changed_partition_id_sha256": _sha_lines(changed_partitions),
            "changed_group_rows": len(changed_groups),
            "changed_group_id_sha256": _sha_lines(changed_groups),
        },
        "caches": caches,
        "impacted_evidence": impacted,
        "claim_boundary": (
            "old metrics remain immutable pre-fix exploratory evidence; only downstream "
            "regressions are superseded; frozen embeddings remain valid and read-only"
        ),
        "next_gate": "corrected_representation_regression_gate_b",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--old-manifest",
        type=Path,
        default=Path(
            "result/four_dataset_pafa_frozen_encoder/samples_fold_0.csv.gz"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/datasets/four_dataset_task_contract_draft_2026-07-28.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "result/four_dataset_pafa_frozen_encoder/"
            "hf_proxy_fixed_v2/impact_receipt.json"
        ),
    )
    args = parser.parse_args()
    result = verify(
        args.dataset_root.resolve(),
        args.old_manifest.resolve(),
        args.contract.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
