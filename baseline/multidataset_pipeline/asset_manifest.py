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
