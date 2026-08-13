"""Tracked, fail-closed identities for production shared-window encoder assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


ASSET_MANIFEST_SCHEMA_VERSION = "multidataset_adapter_assets_v1"
ASSET_MANIFEST_RELATIVE_PATH = Path(
    "baseline/multidataset_pipeline/adapter_assets.json"
)
P3_ASSET_MANIFEST_SCHEMA_VERSION = "multidataset_p3_adapter_asset_v1"
P3_ASSET_MANIFEST_RELATIVE_PATH = Path(
    "baseline/multidataset_pipeline/p3_adapter_asset.json"
)
P5_ASSET_MANIFEST_SCHEMA_VERSION = "multidataset_p5_adapter_asset_v1"
P5_ASSET_MANIFEST_RELATIVE_PATH = Path(
    "baseline/multidataset_pipeline/p5_adapter_asset.json"
)
EXPECTED_PIPELINES = {"P1": "AST", "P2": "BEATs"}


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _canonical_cache_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ".cache":
        raise ValueError(f"{label} must be a canonical relative .cache path")
    return path


def load_adapter_asset_manifest(repo_root: Path) -> dict[str, object]:
    """Load and structurally validate the tracked manifest without touching assets."""

    path = repo_root.resolve() / ASSET_MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"tracked adapter asset manifest missing: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError("adapter asset manifest must be a JSON object")
    if payload.get("schema_version") != ASSET_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("adapter asset manifest schema mismatch")
    assets = payload.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(EXPECTED_PIPELINES):
        raise RuntimeError("adapter asset manifest must contain exactly P1 and P2")
    required = {
        "encoder_identity",
        "code_source_url",
        "source_revision",
        "source_license",
        "checkpoint_source",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_license",
        "canonical_source_path",
        "canonical_checkpoint_path",
        "server_provision_expectation",
    }
    normalized: dict[str, dict[str, object]] = {}
    for pipeline_id, expected_encoder in EXPECTED_PIPELINES.items():
        asset = assets[pipeline_id]
        if not isinstance(asset, Mapping) or set(asset) != required:
            raise RuntimeError(f"{pipeline_id} asset fields do not match frozen schema")
        if asset["encoder_identity"] != expected_encoder:
            raise RuntimeError(f"{pipeline_id} encoder identity mismatch")
        for field in (
            "code_source_url",
            "source_revision",
            "source_license",
            "checkpoint_source",
            "checkpoint_license",
            "server_provision_expectation",
        ):
            if not isinstance(asset[field], str) or not asset[field].strip():
                raise ValueError(f"{pipeline_id}.{field} must be non-empty")
        _require_sha256(asset["checkpoint_sha256"], f"{pipeline_id} checkpoint")
        if not isinstance(asset["checkpoint_size_bytes"], int) or asset[
            "checkpoint_size_bytes"
        ] <= 0:
            raise ValueError(f"{pipeline_id} checkpoint size must be positive")
        _canonical_cache_path(asset["canonical_source_path"], "source path")
        _canonical_cache_path(asset["canonical_checkpoint_path"], "checkpoint path")
        normalized[pipeline_id] = dict(asset)
    return {
        "schema_version": ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_identity_sha256": canonical_json_sha256(payload),
        "assets": normalized,
    }


def manifest_asset_paths(repo_root: Path, pipeline_id: str) -> tuple[Path, Path, Mapping[str, object]]:
    manifest = load_adapter_asset_manifest(repo_root)
    assets = manifest["assets"]
    if pipeline_id not in assets:
        raise KeyError(f"no production manifest asset for {pipeline_id}")
    asset = assets[pipeline_id]
    root = repo_root.resolve()
    return (
        root / str(asset["canonical_source_path"]),
        root / str(asset["canonical_checkpoint_path"]),
        asset,
    )


def load_p3_adapter_asset_manifest(repo_root: Path) -> dict[str, object]:
    """Load P3 independently so adding it cannot change P1/P2 cache identities."""

    path = repo_root.resolve() / P3_ASSET_MANIFEST_RELATIVE_PATH
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != P3_ASSET_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("P3 adapter asset manifest schema mismatch")
    asset = payload.get("asset")
    required = {
        "encoder_identity", "code_source_url", "source_revision", "source_license",
        "checkpoint_source", "checkpoint_sha256", "checkpoint_size_bytes",
        "checkpoint_license", "canonical_source_path", "canonical_checkpoint_path",
        "required_dependency", "server_provision_expectation",
    }
    if not isinstance(asset, Mapping) or set(asset) != required:
        raise RuntimeError("P3 adapter asset fields do not match the frozen schema")
    if asset["encoder_identity"] != "PANNs_Cnn14" or len(str(asset["source_revision"])) != 40:
        raise RuntimeError("P3 encoder/source revision identity mismatch")
    _require_sha256(asset["checkpoint_sha256"], "P3 checkpoint")
    if not isinstance(asset["checkpoint_size_bytes"], int) or asset["checkpoint_size_bytes"] <= 0:
        raise ValueError("P3 checkpoint size must be positive")
    _canonical_cache_path(asset["canonical_source_path"], "P3 source path")
    _canonical_cache_path(asset["canonical_checkpoint_path"], "P3 checkpoint path")
    for field in required - {"checkpoint_size_bytes"}:
        if not isinstance(asset[field], str) or not asset[field].strip():
            raise ValueError(f"P3.{field} must be non-empty")
    return {
        "schema_version": P3_ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_identity_sha256": canonical_json_sha256(payload),
        "asset": dict(asset),
    }


def p3_manifest_asset_paths(repo_root: Path) -> tuple[Path, Path, Mapping[str, object]]:
    manifest = load_p3_adapter_asset_manifest(repo_root)
    asset = manifest["asset"]
    root = repo_root.resolve()
    return (
        root / str(asset["canonical_source_path"]),
        root / str(asset["canonical_checkpoint_path"]),
        asset,
    )


def load_p5_adapter_asset_manifest(repo_root: Path) -> dict[str, object]:
    """Load the independent P5 overlap-aware reference asset identity."""

    path = repo_root.resolve() / P5_ASSET_MANIFEST_RELATIVE_PATH
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != P5_ASSET_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("P5 adapter asset manifest schema mismatch")
    asset = payload.get("asset")
    required = {
        "encoder_identity", "code_source_url", "source_revision", "source_license",
        "checkpoint_source", "checkpoint_repository_revision", "checkpoint_sha256",
        "checkpoint_size_bytes", "checkpoint_license", "canonical_source_path",
        "canonical_checkpoint_path", "required_dependency", "input_contract",
        "standard_pretraining_overlap", "kauh_checkpoint_provenance", "scientific_role",
        "server_provision_expectation",
    }
    if not isinstance(asset, Mapping) or set(asset) != required:
        raise RuntimeError("P5 adapter asset fields do not match the frozen schema")
    if asset["encoder_identity"] != "OPERA_CT":
        raise RuntimeError("P5 encoder identity mismatch")
    for field in ("source_revision", "checkpoint_repository_revision"):
        value = asset[field]
        if not isinstance(value, str) or len(value) != 40:
            raise RuntimeError(f"P5 {field} must be an exact Git revision")
    _require_sha256(asset["checkpoint_sha256"], "P5 checkpoint")
    if not isinstance(asset["checkpoint_size_bytes"], int) or asset["checkpoint_size_bytes"] <= 0:
        raise ValueError("P5 checkpoint size must be positive")
    _canonical_cache_path(asset["canonical_source_path"], "P5 source path")
    _canonical_cache_path(asset["canonical_checkpoint_path"], "P5 checkpoint path")
    for field in required - {"checkpoint_size_bytes"}:
        if not isinstance(asset[field], str) or not asset[field].strip():
            raise ValueError(f"P5.{field} must be non-empty")
    if (
        asset["standard_pretraining_overlap"] != "ICBHI and HF Lung"
        or asset["kauh_checkpoint_provenance"] != "unknown"
        or "overlap-aware reference" not in asset["scientific_role"]
    ):
        raise RuntimeError("P5 overlap/provenance scientific boundary changed")
    return {
        "schema_version": P5_ASSET_MANIFEST_SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_identity_sha256": canonical_json_sha256(payload),
        "asset": dict(asset),
    }


def p5_manifest_asset_paths(repo_root: Path) -> tuple[Path, Path, Mapping[str, object]]:
    manifest = load_p5_adapter_asset_manifest(repo_root)
    asset = manifest["asset"]
    root = repo_root.resolve()
    return (
        root / str(asset["canonical_source_path"]),
        root / str(asset["canonical_checkpoint_path"]),
        asset,
    )
