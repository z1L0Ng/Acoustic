"""Fail-closed factory and asset audit for P1-P4 production window encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

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
    PANNS_SOURCE_LICENSE,
    PANNS_SOURCE_URL,
    build_panns_window_encoder,
)
from .window_encoder import (
    ProductionWindowEncoder,
    missing_asset_receipt,
    require_clean_source_revision,
    require_file_identity,
)


DEFAULT_AST_SOURCE = Path(
    ".cache/icbhi_sprsound_shared_encoder_native_heads/source/repo"
)
DEFAULT_AST_CHECKPOINT = Path(
    ".cache/icbhi_sprsound_shared_encoder_native_heads/checkpoints/"
    "hf_ast_legacy_compat.pth"
)
DEFAULT_BEATS_SOURCE = Path("result/pafa_sprsound_transfer_20260722_235659/source/repo")
DEFAULT_BEATS_CHECKPOINT = Path(
    ".cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt"
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


def build_production_adapter(config: AdapterFactoryConfig) -> ProductionWindowEncoder:
    root = config.repo_root.resolve()
    if config.pipeline_id == "P1":
        return build_ast_window_encoder(
            _at_root(root, config.source_repo or DEFAULT_AST_SOURCE),
            _at_root(root, config.checkpoint or DEFAULT_AST_CHECKPOINT),
            device=config.device,
        )
    if config.pipeline_id == "P2":
        return build_beats_window_encoder(
            _at_root(root, config.source_repo or DEFAULT_BEATS_SOURCE),
            _at_root(root, config.checkpoint or DEFAULT_BEATS_CHECKPOINT),
            device=config.device,
        )
    if config.pipeline_id == "P3":
        if not all(
            (config.source_repo, config.source_revision, config.checkpoint, config.checkpoint_sha256)
        ):
            raise RuntimeError(
                "P3 asset HOLD: source_repo, exact source_revision, checkpoint, and "
                "checkpoint_sha256 are all required; downloading is not allowed"
            )
        return build_panns_window_encoder(
            _at_root(root, config.source_repo),
            config.source_revision,
            _at_root(root, config.checkpoint),
            config.checkpoint_sha256,
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
    raise ValueError("production adapter factory supports only P1-P4")


def audit_local_adapter_assets(repo_root: Path) -> dict[str, Mapping[str, object]]:
    """Read-only local asset status; this does not instantiate a large model."""

    root = repo_root.resolve()
    output: dict[str, Mapping[str, object]] = {}
    for pipeline_id, source, revision, checkpoint, sha256, size in (
        (
            "P1",
            root / DEFAULT_AST_SOURCE,
            AST_SOURCE_REVISION,
            root / DEFAULT_AST_CHECKPOINT,
            AST_CHECKPOINT_SHA256,
            AST_CHECKPOINT_SIZE_BYTES,
        ),
        (
            "P2",
            root / DEFAULT_BEATS_SOURCE,
            BEATS_SOURCE_REVISION,
            root / DEFAULT_BEATS_CHECKPOINT,
            BEATS_CHECKPOINT_SHA256,
            BEATS_CHECKPOINT_SIZE_BYTES,
        ),
    ):
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
                "experiment_result": False,
            }
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            output[pipeline_id] = {
                "code_status": "READY",
                "asset_status": "HOLD",
                "cpu_real_checkpoint_status": "HOLD",
                "cuda_status": "HOLD_waiting_L40_preflight",
                "reason": str(error),
                "experiment_result": False,
            }
    output["P3"] = missing_asset_receipt(
        "PANNs_Cnn14",
        required_checkpoint=PANNS_CHECKPOINT_NAME,
        required_dependency="pinned official audioset_tagging_cnn + torchlibrosa",
        source_url=PANNS_SOURCE_URL,
        source_revision="HOLD_until_local_checkout_is_pinned",
        license_name=PANNS_SOURCE_LICENSE,
    )
    output["P4"] = missing_asset_receipt(
        "HeAR",
        required_checkpoint="accepted local google/hear SavedModel 1.0.0 bundle",
        required_dependency="tensorflow + tf_keras serving runtime",
        source_url=HEAR_SOURCE_URL,
        source_revision="HOLD_until_gated_model_revision_is_accepted",
        license_name=f"code={HEAR_CODE_LICENSE}; model={HEAR_MODEL_LICENSE}",
    )
    return output
