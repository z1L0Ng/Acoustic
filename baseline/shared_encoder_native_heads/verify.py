"""Independent package and generated smoke/profile verification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .protocol import (
    EXPERIMENT_ID,
    HF_AST_LEGACY_CANONICAL_SHA256,
    HF_AST_SHA256,
    ICBHI_MANIFEST,
    SPRSOUND_COMMIT,
    classification_metrics,
    validate_roots,
)


ROOT = Path(__file__).resolve().parents[2]


def package_check() -> dict[str, object]:
    required = [
        ROOT / "baseline/shared_encoder_native_heads/README.md",
        ROOT / "baseline/shared_encoder_native_heads/__init__.py",
        ROOT / "baseline/shared_encoder_native_heads/protocol.py",
        ROOT / "baseline/shared_encoder_native_heads/run.py",
        ROOT / "baseline/shared_encoder_native_heads/verify.py",
        ROOT / "baseline/shared_encoder_native_heads/protocol.json",
        ROOT / "experiments/icbhi_sprsound_shared_encoder_native_heads.yaml",
        ICBHI_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"package files missing: {missing}")
    subprocess.run(
        [
            "python",
            "-m",
            "py_compile",
            *[str(path) for path in required if path.suffix == ".py"],
        ],
        cwd=ROOT,
        check=True,
    )
    with (ROOT / "experiments/index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["experiment_id"] == EXPERIMENT_ID]
    if len(matches) != 1:
        raise RuntimeError("experiment registry identity/count gate failed")
    row = matches[0]
    if (
        row["definition_path"] != "experiments/icbhi_sprsound_shared_encoder_native_heads.yaml"
        or row["result_path"] != f"result/{EXPERIMENT_ID}"
    ):
        raise RuntimeError("experiment registry path gate failed")
    protocol = json.loads(
        (ROOT / "baseline/shared_encoder_native_heads/protocol.json").read_text()
    )
    if (
        protocol["encoder"]["checkpoint_source_sha256"] != HF_AST_SHA256
        or protocol["datasets"]["sprsound"]["source_commit"] != SPRSOUND_COMMIT
        or protocol["encoder"]["forbidden_initialization"] != "ICBHI task-selected checkpoint"
    ):
        raise RuntimeError("protocol identity/boundary gate failed")
    fixed_label_metrics, fixed_label_confusion = classification_metrics(
        np.asarray([0]), np.asarray([0]), ["present", "missing_a", "missing_b"]
    )
    if (
        fixed_label_confusion.shape != (3, 3)
        or not math.isclose(fixed_label_metrics["macro_f1"], 1 / 3)
        or not math.isclose(fixed_label_metrics["uar"], 1 / 3)
    ):
        raise RuntimeError("fixed native-label metric gate failed")
    return {"status": "package_verified", "required_files": len(required)}


def generated_check(result_root: Path, cache_root: Path, require_profile: bool) -> dict[str, object]:
    result, cache = validate_roots(result_root, cache_root)
    audit = json.loads((result / "protocol_and_data_receipt.json").read_text())
    smoke = json.loads((result / "smoke_receipt.json").read_text())
    asset = json.loads((result / "asset_receipt.json").read_text())
    if audit["status"] != "data_protocol_audit_passed":
        raise RuntimeError("data audit status failed")
    if (
        audit["datasets"]["icbhi_2017"]["partition"]
        != {"subtrain": 3055, "test": 2756, "validation": 1087}
        or audit["datasets"]["icbhi_2017"]["official_train_test_patient_overlap"]
        != ["156", "218"]
    ):
        raise RuntimeError("ICBHI split receipt failed")
    spr = audit["datasets"]["sprsound_biocas2022"]
    if (
        (spr["subtrain_events"], spr["validation_events"], spr["inter_events"], spr["intra_events"])
        != (5219, 1437, 1429, 1004)
        or spr["subtrain_validation_patient_overlap"] != 0
        or spr["train_inter_patient_overlap"] != 0
        or spr["train_intra_event_bearing_patient_overlap"] != 156
        or spr["train_intra_archive_patient_overlap"] != 162
    ):
        raise RuntimeError("SPRSound split receipt failed")
    if (
        asset["checkpoint_source_sha256"] != HF_AST_SHA256
        or asset["checkpoint_canonical_tensor_sha256"] != HF_AST_LEGACY_CANONICAL_SHA256
    ):
        raise RuntimeError("neutral AudioSet checkpoint receipt failed")
    if (
        smoke["status"] != "smoke_passed"
        or not smoke["losses_finite"]
        or not smoke["gradients_finite"]
        or not smoke["checkpoint_resume"]["reload_tensor_identity"]
    ):
        raise RuntimeError("smoke finite/resume gate failed")
    expected_confusions = {"icbhi_flat4": 4, "spr_binary": 6, "spr_seven": 6}
    for task, expected in expected_confusions.items():
        if smoke["validation"][task]["confusion_total"] != expected:
            raise RuntimeError(f"smoke confusion total failed for {task}")
    for partition in ("inter", "intra"):
        surface = smoke["label_free_test_forward"][partition]
        if surface["labels_loaded"]:
            raise RuntimeError(f"{partition} labels entered label-free smoke")
    output = {
        "status": "smoke_verified",
        "checkpoint_source_sha256": HF_AST_SHA256,
        "checkpoint_canonical_tensor_sha256": HF_AST_LEGACY_CANONICAL_SHA256,
        "result_root": str(result),
        "cache_root": str(cache),
    }
    if require_profile:
        profile = json.loads((result / "profile_receipt.json").read_text())
        finite_values = [
            profile["step_seconds"][key] for key in ("mean", "median", "min", "max")
        ] + [
            profile["projected_cpu_preprocessing_seconds"],
            profile["projected_cpu_training_seconds"],
            profile["projected_cpu_total_seconds"],
            profile["peak_rss_gib"],
        ]
        if (
            profile["profile_steps"] != 100
            or profile["sampling_policy"] != "source_proportional"
            or not profile["losses_finite"]
            or not profile["gradients_finite"]
            or not all(math.isfinite(float(value)) for value in finite_values)
        ):
            raise RuntimeError("profile completeness/finite gate failed")
        output["status"] = "profile_verified"
        output["profile_status"] = profile["status"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["package", "smoke", "profile"], required=True)
    parser.add_argument(
        "--result-root", type=Path, default=Path("result/icbhi_sprsound_shared_encoder_native_heads")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(".cache/icbhi_sprsound_shared_encoder_native_heads")
    )
    args = parser.parse_args()
    result = (
        package_check()
        if args.mode == "package"
        else generated_check(args.result_root, args.cache_root, args.mode == "profile")
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
