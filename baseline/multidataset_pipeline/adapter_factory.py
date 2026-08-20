"""Fail-closed factory and asset audit for P1-P5 production window encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .asset_manifest import (
    load_adapter_asset_manifest,
    load_p3_adapter_asset_manifest,
    load_p5_adapter_asset_manifest,
    manifest_asset_paths,
    p3_manifest_asset_paths,
    p5_manifest_asset_paths,
)

from .ast_window_encoder import (
    AST_CHECKPOINT_SHA256,
    AST_CHECKPOINT_SIZE_BYTES,
    AST_SOURCE_REVISION,
    build_ast_window_encoder,
)
from .beats_window_encoder import (
    BEATS_CHECKPOINT_SHA256,
    BEATS_CHECKPOINT_SIZE_BYTES,
    BEATS_SOURCE_REVISION,
    build_beats_window_encoder,
)
from .hear_window_encoder import (
    HEAR_CODE_LICENSE,
    HEAR_MODEL_LICENSE,
    HEAR_SOURCE_URL,
    build_hear_window_encoder,
)
from .panns_window_encoder import (
    PANNS_CHECKPOINT_NAME,
    PANNS_CHECKPOINT_SHA256,
    PANNS_CHECKPOINT_SIZE_BYTES,
    PANNS_SOURCE_LICENSE,
    PANNS_SOURCE_REVISION,
    PANNS_SOURCE_URL,
    build_panns_window_encoder,
)
from .window_encoder import (
    ProductionWindowEncoder,
    missing_asset_receipt,
    require_clean_source_revision,
    require_file_identity,
)


@dataclass(frozen=True)
class AdapterFactoryConfig:
    pipeline_id: str
    repo_root: Path
    device: str = "cpu"
    source_repo: Path | None = None
    source_revision: str | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None


def _at_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _manifest_bound_paths(config: AdapterFactoryConfig) -> tuple[Path, Path, Mapping[str, object]]:
    source, checkpoint, asset = manifest_asset_paths(config.repo_root, config.pipeline_id)
    if config.source_repo is not None and _at_root(config.repo_root, config.source_repo).resolve() != source.resolve():
        raise RuntimeError("P1/P2 source path must match the tracked canonical asset manifest")
    if config.checkpoint is not None and _at_root(config.repo_root, config.checkpoint).resolve() != checkpoint.resolve():
        raise RuntimeError("P1/P2 checkpoint path must match the tracked canonical asset manifest")
    if config.source_revision is not None and config.source_revision != asset["source_revision"]:
        raise RuntimeError("P1/P2 source revision differs from tracked asset manifest")
    if config.checkpoint_sha256 is not None and config.checkpoint_sha256 != asset["checkpoint_sha256"]:
        raise RuntimeError("P1/P2 checkpoint SHA differs from tracked asset manifest")
    return source, checkpoint, asset


def _p3_manifest_bound_paths(config: AdapterFactoryConfig) -> tuple[Path, Path, Mapping[str, object]]:
    source, checkpoint, asset = p3_manifest_asset_paths(config.repo_root)
    if (
        asset["source_revision"] != PANNS_SOURCE_REVISION
        or asset["checkpoint_sha256"] != PANNS_CHECKPOINT_SHA256
        or asset["checkpoint_size_bytes"] != PANNS_CHECKPOINT_SIZE_BYTES
    ):
        raise RuntimeError("P3 canonical asset manifest differs from the audited constants")
    supplied = {
        "source_repo": config.source_repo,
        "source_revision": config.source_revision,
        "checkpoint": config.checkpoint,
        "checkpoint_sha256": config.checkpoint_sha256,
    }
    expected = {
        "source_repo": source,
        "source_revision": asset["source_revision"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": asset["checkpoint_sha256"],
    }
    for key, value in supplied.items():
        if value is None:
            continue
        normalized = _at_root(config.repo_root, value).resolve() if key in {"source_repo", "checkpoint"} else value
        expected_value = expected[key].resolve() if isinstance(expected[key], Path) else expected[key]
        if normalized != expected_value:
            raise RuntimeError(f"P3 {key} differs from the tracked canonical asset manifest")
    return source, checkpoint, asset


def _p5_manifest_bound_paths(config: AdapterFactoryConfig) -> tuple[Path, Path, Mapping[str, object]]:
    from .opera_window_encoder import (
        OPERA_CHECKPOINT_SHA256,
        OPERA_CHECKPOINT_SIZE_BYTES,
        OPERA_SOURCE_REVISION,
    )
    source, checkpoint, asset = p5_manifest_asset_paths(config.repo_root)
    if (
        asset["source_revision"] != OPERA_SOURCE_REVISION
        or asset["checkpoint_sha256"] != OPERA_CHECKPOINT_SHA256
        or asset["checkpoint_size_bytes"] != OPERA_CHECKPOINT_SIZE_BYTES
    ):
        raise RuntimeError("P5 canonical asset manifest differs from the audited constants")
    supplied = {
        "source_repo": config.source_repo,
        "source_revision": config.source_revision,
        "checkpoint": config.checkpoint,
        "checkpoint_sha256": config.checkpoint_sha256,
    }
    expected = {
        "source_repo": source,
        "source_revision": asset["source_revision"],
        "checkpoint": checkpoint,
        "checkpoint_sha256": asset["checkpoint_sha256"],
    }
    for key, value in supplied.items():
        if value is None:
            continue
        normalized = _at_root(config.repo_root, value).resolve() if key in {"source_repo", "checkpoint"} else value
        expected_value = expected[key].resolve() if isinstance(expected[key], Path) else expected[key]
        if normalized != expected_value:
            raise RuntimeError(f"P5 {key} differs from the tracked canonical asset manifest")
    return source, checkpoint, asset


def build_production_adapter(config: AdapterFactoryConfig) -> ProductionWindowEncoder:
    root = config.repo_root.resolve()
    if config.pipeline_id == "P1":
        source, checkpoint, _ = _manifest_bound_paths(config)
        return build_ast_window_encoder(
            source,
            checkpoint,
            device=config.device,
        )
    if config.pipeline_id == "P2":
        source, checkpoint, _ = _manifest_bound_paths(config)
        return build_beats_window_encoder(
            source,
            checkpoint,
            device=config.device,
        )
    if config.pipeline_id == "P3":
        source, checkpoint, asset = _p3_manifest_bound_paths(config)
        return build_panns_window_encoder(
            source,
            str(asset["source_revision"]),
            checkpoint,
            str(asset["checkpoint_sha256"]),
            device=config.device,
        )
    if config.pipeline_id == "P4":
        if not all((config.checkpoint, config.source_revision, config.checkpoint_sha256)):
            raise RuntimeError(
                "P4 asset HOLD: accepted local SavedModel directory, immutable revision, "
                "and deterministic tree SHA256 are required; gated download is not allowed"
            )
        return build_hear_window_encoder(
            _at_root(root, config.checkpoint),
            config.source_revision,
            config.checkpoint_sha256,
            device=config.device,
        )
    if config.pipeline_id == "P5":
        from .opera_window_encoder import build_opera_window_encoder

        source, checkpoint, asset = _p5_manifest_bound_paths(config)
        return build_opera_window_encoder(
            source,
            str(asset["source_revision"]),
            checkpoint,
            str(asset["checkpoint_sha256"]),
            device=config.device,
        )
    raise ValueError("production adapter factory supports only P1-P5")


def audit_local_adapter_assets(repo_root: Path) -> dict[str, Mapping[str, object]]:
    """Read-only local asset status; this does not instantiate a large model."""

    root = repo_root.resolve()
    manifest = load_adapter_asset_manifest(root)
    output: dict[str, Mapping[str, object]] = {}
    expected_code = {
        "P1": ("AST", AST_SOURCE_REVISION, AST_CHECKPOINT_SHA256, AST_CHECKPOINT_SIZE_BYTES),
        "P2": ("BEATs", BEATS_SOURCE_REVISION, BEATS_CHECKPOINT_SHA256, BEATS_CHECKPOINT_SIZE_BYTES),
    }
    for pipeline_id in ("P1", "P2"):
        source, checkpoint, asset = manifest_asset_paths(root, pipeline_id)
        encoder, revision, sha256, size = expected_code[pipeline_id]
        if (
            asset["encoder_identity"] != encoder
            or asset["source_revision"] != revision
            or asset["checkpoint_sha256"] != sha256
            or asset["checkpoint_size_bytes"] != size
        ):
            raise RuntimeError(f"{pipeline_id} tracked asset manifest/code identity mismatch")
        try:
            source_receipt = require_clean_source_revision(source, revision)
            checkpoint_receipt = require_file_identity(
                checkpoint, sha256, expected_size_bytes=size
            )
            output[pipeline_id] = {
                "code_status": "READY",
                "asset_status": "READY_verified_local",
                "cpu_real_checkpoint_status": "READY_verified_real_checkpoint_smoke_2026-08-12",
                "cuda_status": "HOLD_waiting_L40_preflight",
                "source": source_receipt,
                "checkpoint": checkpoint_receipt,
                "asset_manifest": {
                    "schema_version": manifest["schema_version"],
                    "manifest_file_sha256": manifest["manifest_file_sha256"],
                    "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                    "canonical_source_path": asset["canonical_source_path"],
                    "canonical_checkpoint_path": asset["canonical_checkpoint_path"],
                    "server_provision_expectation": asset["server_provision_expectation"],
                },
                "experiment_result": False,
            }
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            output[pipeline_id] = {
                "code_status": "READY",
                "asset_status": "HOLD",
                "cpu_real_checkpoint_status": "HOLD",
                "cuda_status": "HOLD_waiting_L40_preflight",
                "reason": str(error),
                "asset_manifest": {
                    "schema_version": manifest["schema_version"],
                    "manifest_file_sha256": manifest["manifest_file_sha256"],
                    "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                    "canonical_source_path": asset["canonical_source_path"],
                    "canonical_checkpoint_path": asset["canonical_checkpoint_path"],
                    "server_provision_expectation": asset["server_provision_expectation"],
                },
                "experiment_result": False,
            }
    try:
        source, checkpoint, asset = p3_manifest_asset_paths(root)
        source_receipt = require_clean_source_revision(source, PANNS_SOURCE_REVISION)
        checkpoint_receipt = require_file_identity(
            checkpoint, PANNS_CHECKPOINT_SHA256,
            expected_size_bytes=PANNS_CHECKPOINT_SIZE_BYTES,
        )
        manifest = load_p3_adapter_asset_manifest(root)
        output["P3"] = {
            "code_status": "READY",
            "asset_status": "READY_verified_local",
            "source": source_receipt,
            "checkpoint": checkpoint_receipt,
            "required_dependency": asset["required_dependency"],
            "asset_manifest": {
                "schema_version": manifest["schema_version"],
                "manifest_file_sha256": manifest["manifest_file_sha256"],
                "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                "canonical_source_path": asset["canonical_source_path"],
                "canonical_checkpoint_path": asset["canonical_checkpoint_path"],
            },
            "experiment_result": False,
        }
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        output["P3"] = missing_asset_receipt(
            "PANNs_Cnn14", required_checkpoint=PANNS_CHECKPOINT_NAME,
            required_dependency="torchlibrosa==0.1.0",
            source_url=PANNS_SOURCE_URL, source_revision=PANNS_SOURCE_REVISION,
            license_name=PANNS_SOURCE_LICENSE,
        ) | {"reason": str(error)}
    output["P4"] = missing_asset_receipt(
        "HeAR",
        required_checkpoint="accepted local google/hear SavedModel 1.0.0 bundle",
        required_dependency="tensorflow + tf_keras serving runtime",
        source_url=HEAR_SOURCE_URL,
        source_revision="HOLD_until_gated_model_revision_is_accepted",
        license_name=f"code={HEAR_CODE_LICENSE}; model={HEAR_MODEL_LICENSE}",
    )
    try:
        from .opera_window_encoder import (
            OPERA_CHECKPOINT_NAME,
            OPERA_CHECKPOINT_SHA256,
            OPERA_CHECKPOINT_SIZE_BYTES,
            OPERA_SOURCE_LICENSE,
            OPERA_SOURCE_REVISION,
            OPERA_SOURCE_URL,
        )

        source, checkpoint, asset = p5_manifest_asset_paths(root)
        source_receipt = require_clean_source_revision(source, OPERA_SOURCE_REVISION)
        checkpoint_receipt = require_file_identity(
            checkpoint, OPERA_CHECKPOINT_SHA256,
            expected_size_bytes=OPERA_CHECKPOINT_SIZE_BYTES,
        )
        manifest = load_p5_adapter_asset_manifest(root)
        output["P5"] = {
            "code_status": "READY",
            "asset_status": "READY_verified_local",
            "source": source_receipt,
            "checkpoint": checkpoint_receipt,
            "required_dependency": asset["required_dependency"],
            "input_contract": asset["input_contract"],
            "standard_pretraining_overlap": asset["standard_pretraining_overlap"],
            "kauh_checkpoint_provenance": asset["kauh_checkpoint_provenance"],
            "scientific_role": asset["scientific_role"],
            "clean_generalization_claim_allowed": False,
            "asset_manifest": {
                "schema_version": manifest["schema_version"],
                "manifest_file_sha256": manifest["manifest_file_sha256"],
                "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                "canonical_source_path": asset["canonical_source_path"],
                "canonical_checkpoint_path": asset["canonical_checkpoint_path"],
            },
            "experiment_result": False,
        }
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        output["P5"] = missing_asset_receipt(
            "OPERA_CT",
            required_checkpoint=OPERA_CHECKPOINT_NAME,
            required_dependency="torchlibrosa==0.1.0; librosa==0.10.1; timm==1.0.3",
            source_url=OPERA_SOURCE_URL,
            source_revision=OPERA_SOURCE_REVISION,
            license_name=f"source={OPERA_SOURCE_LICENSE}; checkpoint=CC-BY-NC-4.0",
        ) | {
            "reason": str(error),
            "scientific_role": "overlap-aware reference only",
            "standard_pretraining_overlap": "ICBHI and HF Lung",
            "kauh_checkpoint_provenance": "unknown",
            "clean_generalization_claim_allowed": False,
        }
    return output
