"""Immutable frozen-encoder embedding cache for subtrain/validation only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn

from .asset_manifest import (
    load_adapter_asset_manifest,
    load_p3_adapter_asset_manifest,
    load_p5_adapter_asset_manifest,
)


EMBEDDING_CACHE_SCHEMA_VERSION = "frozen_window_embedding_cache_v2"
SERIALIZATION = "numpy_npy_little_endian_cpu_c_order"
ALLOWED_PARTITIONS = {"subtrain", "validation"}
FORBIDDEN_CACHE_PATH_PARTS = {"outer", "outer_test", "test", "terminal", "terminal-score"}
REQUIRED_ARTIFACTS = {
    "embeddings.npy",
    "unit_offsets.npy",
    "time_map.npy",
    "valid_samples.npy",
    "window_mask.npy",
}
SCHEMA_IDENTITY_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "serialization": SERIALIZATION,
            "artifacts": sorted(REQUIRED_ARTIFACTS),
            "partition_policy": sorted(ALLOWED_PARTITIONS),
            "packed_window_policy": "ordered_units_with_offsets_all_valid_windows_only",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _require_mapping_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} fields changed: expected {sorted(keys)}, got {sorted(value)}")


@dataclass(frozen=True)
class EmbeddingCacheIdentity:
    dataset_id: str
    dataset_release: str
    partition: str
    ordered_unit_ids: tuple[str, ...]
    data_identity_sha256: str
    preprocessing: Mapping[str, object]
    window_policy: Mapping[str, object]
    encoder_asset: Mapping[str, object]
    frontend_adapter_identity: Mapping[str, object]
    output_dtype: str
    code_identity_sha256: str
    code_dependency_sha256_by_path: Mapping[str, str]
    config_identity_sha256: str
    schema_identity_sha256: str = SCHEMA_IDENTITY_SHA256
    encoder_frozen: bool = True
    encoder_eval: bool = True
    deterministic_frontend: bool = True
    augmentation_enabled: bool = False
    serialization: str = SERIALIZATION

    @classmethod
    def from_tracked_asset(
        cls,
        *,
        repo_root: Path,
        pipeline_id: str,
        dataset_id: str,
        dataset_release: str,
        partition: str,
        ordered_unit_ids: tuple[str, ...],
        data_identity_sha256: str,
        preprocessing: Mapping[str, object],
        window_policy: Mapping[str, object],
        frontend_adapter_identity: Mapping[str, object],
        code_identity_sha256: str,
        code_dependency_sha256_by_path: Mapping[str, str],
        config_identity_sha256: str,
    ) -> "EmbeddingCacheIdentity":
        if pipeline_id == "P3":
            manifest = load_p3_adapter_asset_manifest(repo_root)
            asset = manifest["asset"]
        elif pipeline_id == "P5":
            manifest = load_p5_adapter_asset_manifest(repo_root)
            asset = manifest["asset"]
        else:
            manifest = load_adapter_asset_manifest(repo_root)
            if pipeline_id not in manifest["assets"]:
                raise ValueError("tracked cache assets cover only P1/P2/P3/P5")
            asset = manifest["assets"][pipeline_id]
        return cls(
            dataset_id=dataset_id,
            dataset_release=dataset_release,
            partition=partition,
            ordered_unit_ids=ordered_unit_ids,
            data_identity_sha256=data_identity_sha256,
            preprocessing=preprocessing,
            window_policy=window_policy,
            encoder_asset={
                "encoder_identity": asset["encoder_identity"],
                "source_url": asset["code_source_url"],
                "source_revision": asset["source_revision"],
                "checkpoint_sha256": asset["checkpoint_sha256"],
                "checkpoint_size_bytes": asset["checkpoint_size_bytes"],
                "license": f"source={asset['source_license']}; checkpoint={asset['checkpoint_license']}",
                "asset_manifest_identity_sha256": manifest[
                    "manifest_identity_sha256"
                ],
            },
            frontend_adapter_identity=frontend_adapter_identity,
            output_dtype="float32",
            code_identity_sha256=code_identity_sha256,
            code_dependency_sha256_by_path=code_dependency_sha256_by_path,
            config_identity_sha256=config_identity_sha256,
        )

    def validate(self) -> None:
        if self.dataset_id not in {"ICBHI", "SPRSound", "HF", "KAUH"}:
            raise ValueError("cache dataset must be one frozen native lane")
        if not self.dataset_release.strip():
            raise ValueError("cache dataset release identity is empty")
        if self.partition not in ALLOWED_PARTITIONS:
            raise PermissionError("embedding cache is forbidden for outer/test partitions")
        if not self.ordered_unit_ids or any(not value for value in self.ordered_unit_ids):
            raise ValueError("cache ordered unit IDs must be non-empty")
        if len(set(self.ordered_unit_ids)) != len(self.ordered_unit_ids):
            raise ValueError("cache ordered unit IDs contain duplicates")
        _require_sha256(self.data_identity_sha256, "cache data identity")
        _require_sha256(self.code_identity_sha256, "cache code identity")
        if (
            not self.code_dependency_sha256_by_path
            or any(
                not isinstance(path, str)
                or not path.startswith(
                    (
                        "baseline/multidataset_pipeline/",
                        "baseline/four_dataset_frozen_encoder/",
                    )
                )
                or path.startswith(("tests/", "docs/"))
                for path in self.code_dependency_sha256_by_path
            )
        ):
            raise ValueError("cache production code dependency map is empty or invalid")
        for path, digest in self.code_dependency_sha256_by_path.items():
            _require_sha256(digest, f"cache code dependency {path}")
        dependency_aggregate = hashlib.sha256(
            _canonical_json(dict(self.code_dependency_sha256_by_path))
        ).hexdigest()
        if dependency_aggregate != self.code_identity_sha256:
            raise RuntimeError("cache code aggregate does not match per-file dependency map")
        _require_sha256(self.config_identity_sha256, "cache config identity")
        if self.schema_identity_sha256 != SCHEMA_IDENTITY_SHA256:
            raise RuntimeError("cache schema identity changed")
        if self.output_dtype != "float32" or self.serialization != SERIALIZATION:
            raise RuntimeError("cache dtype/serialization contract changed")
        if (
            self.encoder_frozen is not True
            or self.encoder_eval is not True
            or self.deterministic_frontend is not True
            or self.augmentation_enabled is not False
        ):
            raise RuntimeError("cache requires frozen eval encoder and deterministic frontend")
        _require_mapping_keys(
            self.preprocessing,
            {"sample_rate", "resample_policy", "waveform_dtype"},
            "preprocessing",
        )
        if self.preprocessing["sample_rate"] != 16_000 or self.preprocessing[
            "waveform_dtype"
        ] != "float32":
            raise RuntimeError("cache preprocessing must preserve the 16 kHz float32 contract")
        if not str(self.preprocessing["resample_policy"]).strip():
            raise ValueError("cache resample policy is empty")
        _require_mapping_keys(
            self.window_policy,
            {
                "window_length_s",
                "window_stride_s",
                "tail_policy",
                "short_padding",
                "repeat_pad",
                "truncate",
                "mask_semantics",
            },
            "window policy",
        )
        if (
            self.window_policy["window_length_s"] != 2.0
            or self.window_policy["window_stride_s"] != 1.0
            or self.window_policy["tail_policy"]
            != "append_unique_end_aligned_window_when_stride_misses_tail"
            or self.window_policy["short_padding"] != "zero_pad_only"
            or self.window_policy["repeat_pad"] is not False
            or self.window_policy["truncate"] is not False
            or self.window_policy["mask_semantics"]
            != "window_mask_true_is_valid;time_map_source_seconds_half_open"
        ):
            raise RuntimeError("cache shared-window policy changed")
        _require_mapping_keys(
            self.encoder_asset,
            {
                "encoder_identity",
                "source_url",
                "source_revision",
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "license",
                "asset_manifest_identity_sha256",
            },
            "encoder asset",
        )
        _require_sha256(self.encoder_asset["checkpoint_sha256"], "encoder checkpoint")
        _require_sha256(
            self.encoder_asset["asset_manifest_identity_sha256"], "asset manifest identity"
        )
        if not isinstance(self.encoder_asset["checkpoint_size_bytes"], int) or self.encoder_asset[
            "checkpoint_size_bytes"
        ] <= 0:
            raise ValueError("encoder checkpoint size must be positive")
        for key in ("encoder_identity", "source_url", "source_revision", "license"):
            if not str(self.encoder_asset[key]).strip():
                raise ValueError(f"encoder asset {key} is empty")
        _require_mapping_keys(
            self.frontend_adapter_identity,
            {"frontend", "adapter", "identity_sha256"},
            "frontend/adapter identity",
        )
        if not str(self.frontend_adapter_identity["frontend"]).strip() or not str(
            self.frontend_adapter_identity["adapter"]
        ).strip():
            raise ValueError("frontend/adapter identity is empty")
        _require_sha256(
            self.frontend_adapter_identity["identity_sha256"], "frontend/adapter identity"
        )

    def payload(self) -> dict[str, object]:
        self.validate()
        return json.loads(json.dumps(asdict(self), sort_keys=True))

    def cache_key(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload())).hexdigest()


@dataclass(frozen=True)
class EmbeddingCachePayload:
    unit_ids: tuple[str, ...]
    embeddings: tuple[torch.Tensor, ...]
    time_maps: tuple[torch.Tensor, ...]
    valid_samples: tuple[torch.Tensor, ...]
    window_masks: tuple[torch.Tensor, ...]

    def validate(self, identity: EmbeddingCacheIdentity) -> None:
        identity.validate()
        if self.unit_ids != identity.ordered_unit_ids:
            raise RuntimeError("cache payload unit IDs are missing, duplicated, stale, or reordered")
        count = len(self.unit_ids)
        if any(
            len(values) != count
            for values in (
                self.embeddings,
                self.time_maps,
                self.valid_samples,
                self.window_masks,
            )
        ):
            raise ValueError("cache payload fields must align one-to-one with ordered units")
        dimension: int | None = None
        for index in range(count):
            embedding = self.embeddings[index].detach().cpu()
            time_map = self.time_maps[index].detach().cpu()
            valid = self.valid_samples[index].detach().cpu()
            mask = self.window_masks[index].detach().cpu()
            if embedding.ndim != 2 or embedding.shape[0] <= 0:
                raise ValueError("each cached unit requires non-empty [K,D] embeddings")
            if embedding.dtype != torch.float32 or not bool(torch.isfinite(embedding).all()):
                raise TypeError("cached embeddings must be finite float32")
            dimension = dimension or embedding.shape[1]
            if embedding.shape[1] != dimension:
                raise ValueError("cached embedding dimension changed between units")
            windows = embedding.shape[0]
            if time_map.shape != (windows, 2) or time_map.dtype != torch.float64:
                raise TypeError("cached time_map must be float64 [K,2]")
            if valid.shape != (windows,) or valid.dtype != torch.long:
                raise TypeError("cached valid_samples must be int64 [K]")
            if mask.shape != (windows,) or mask.dtype != torch.bool or not bool(mask.all()):
                raise TypeError("ragged cache stores all and only valid windows")
            if bool((valid <= 0).any()) or bool((valid > 32_000).any()):
                raise ValueError("cached valid_samples outside 2 s window contract")
            if not bool(torch.isfinite(time_map).all()) or bool(
                (time_map[:, 1] <= time_map[:, 0]).any()
            ):
                raise ValueError("cached source-time intervals are invalid")
            if windows > 1 and bool((time_map[1:, 0] <= time_map[:-1, 0]).any()):
                raise ValueError("cached source-time windows must be strictly ordered")


def _reject_terminal_cache_root(cache_root: Path) -> None:
    if cache_root.name.lower() in FORBIDDEN_CACHE_PATH_PARTS:
        raise PermissionError("outer/test cache paths are forbidden")


def _numpy_little_endian(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _packed_arrays(payload: EmbeddingCachePayload) -> dict[str, np.ndarray]:
    offsets = [0]
    for embedding in payload.embeddings:
        offsets.append(offsets[-1] + embedding.shape[0])
    return {
        "embeddings.npy": np.concatenate(
            [_numpy_little_endian(value, np.dtype("<f4")) for value in payload.embeddings]
        ),
        "unit_offsets.npy": np.asarray(offsets, dtype=np.dtype("<i8")),
        "time_map.npy": np.concatenate(
            [_numpy_little_endian(value, np.dtype("<f8")) for value in payload.time_maps]
        ),
        "valid_samples.npy": np.concatenate(
            [_numpy_little_endian(value, np.dtype("<i8")) for value in payload.valid_samples]
        ),
        "window_mask.npy": np.concatenate(
            [_numpy_little_endian(value, np.dtype("?")) for value in payload.window_masks]
        ),
    }


def _artifact_receipt(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "sha256": _sha256_path(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "count": int(array.shape[0]),
    }


def write_embedding_cache(
    cache_root: Path,
    identity: EmbeddingCacheIdentity,
    payload: EmbeddingCachePayload,
) -> dict[str, object]:
    """Atomically create one immutable cache directory; never overwrite an artifact."""

    _reject_terminal_cache_root(cache_root)
    payload.validate(identity)
    key = identity.cache_key()
    target = cache_root.resolve() / key
    if target.exists():
        loaded, receipt = load_embedding_cache(cache_root, identity)
        if not payloads_equal(payload, loaded):
            raise RuntimeError("existing cache key has different embedding bytes")
        return {**receipt, "cache_status": "hit_verified_existing"}
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = cache_root.resolve() / f".{key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        arrays = _packed_arrays(payload)
        artifacts: dict[str, object] = {}
        for name, array in arrays.items():
            path = staging / name
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            artifacts[name] = _artifact_receipt(path, array)
        manifest = {
            "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "identity": identity.payload(),
            "identity_sha256": key,
            "serialization": SERIALIZATION,
            "unit_count": len(payload.unit_ids),
            "ordered_unit_ids": list(payload.unit_ids),
            "embedding_dimension": payload.embeddings[0].shape[1],
            "total_valid_windows": sum(value.shape[0] for value in payload.embeddings),
            "artifacts": artifacts,
            "outer_test_cached": False,
        }
        manifest_path = staging / "cache_manifest.json"
        temporary_manifest = staging / "cache_manifest.json.tmp"
        temporary_manifest.write_bytes(_canonical_json(manifest) + b"\n")
        temporary_manifest.replace(manifest_path)
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    _, receipt = load_embedding_cache(cache_root, identity)
    return {**receipt, "cache_status": "miss_computed_and_written"}


def load_embedding_cache(
    cache_root: Path, identity: EmbeddingCacheIdentity
) -> tuple[EmbeddingCachePayload, dict[str, object]]:
    _reject_terminal_cache_root(cache_root)
    identity.validate()
    key = identity.cache_key()
    target = cache_root.resolve() / key
    if not target.is_dir():
        raise FileNotFoundError(f"embedding cache miss: {target}")
    actual_files = {path.name for path in target.iterdir()}
    expected_files = REQUIRED_ARTIFACTS | {"cache_manifest.json"}
    if actual_files != expected_files:
        raise RuntimeError("embedding cache is partial or contains stale extra artifacts")
    manifest_path = target / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EMBEDDING_CACHE_SCHEMA_VERSION
        or manifest.get("cache_key") != key
        or manifest.get("identity_sha256") != key
        or manifest.get("identity") != identity.payload()
        or manifest.get("ordered_unit_ids") != list(identity.ordered_unit_ids)
        or manifest.get("unit_count") != len(identity.ordered_unit_ids)
        or manifest.get("serialization") != SERIALIZATION
        or manifest.get("outer_test_cached") is not False
        or not isinstance(manifest.get("artifacts"), Mapping)
        or set(manifest["artifacts"]) != REQUIRED_ARTIFACTS
    ):
        raise RuntimeError("embedding cache manifest identity/schema mismatch")
    arrays: dict[str, np.ndarray] = {}
    for name in REQUIRED_ARTIFACTS:
        path = target / name
        expected = manifest["artifacts"][name]
        if (
            not isinstance(expected, Mapping)
            or path.stat().st_size != expected.get("size_bytes")
            or _sha256_path(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"embedding cache artifact corrupt: {name}")
        with path.open("rb") as handle:
            array = np.load(handle, allow_pickle=False)
        if (
            list(array.shape) != expected.get("shape")
            or array.dtype.str != expected.get("dtype")
            or int(array.shape[0]) != expected.get("count")
        ):
            raise RuntimeError(f"embedding cache artifact metadata mismatch: {name}")
        arrays[name] = array
    offsets = arrays["unit_offsets.npy"]
    total = int(manifest["total_valid_windows"])
    dimension = int(manifest["embedding_dimension"])
    if (
        offsets.dtype.str != "<i8"
        or offsets.shape != (len(identity.ordered_unit_ids) + 1,)
        or offsets[0] != 0
        or offsets[-1] != total
        or np.any(offsets[1:] <= offsets[:-1])
        or arrays["embeddings.npy"].shape != (total, dimension)
        or arrays["time_map.npy"].shape != (total, 2)
        or arrays["valid_samples.npy"].shape != (total,)
        or arrays["window_mask.npy"].shape != (total,)
    ):
        raise RuntimeError("embedding cache packed shape/offset contract failed")
    slices = [slice(int(offsets[index]), int(offsets[index + 1])) for index in range(len(offsets) - 1)]
    payload = EmbeddingCachePayload(
        unit_ids=identity.ordered_unit_ids,
        embeddings=tuple(torch.from_numpy(arrays["embeddings.npy"][value].copy()) for value in slices),
        time_maps=tuple(torch.from_numpy(arrays["time_map.npy"][value].copy()) for value in slices),
        valid_samples=tuple(torch.from_numpy(arrays["valid_samples.npy"][value].copy()) for value in slices),
        window_masks=tuple(torch.from_numpy(arrays["window_mask.npy"][value].copy()) for value in slices),
    )
    payload.validate(identity)
    receipt = {
        "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "cache_path": str(target),
        "manifest_sha256": _sha256_path(manifest_path),
        "unit_count": len(payload.unit_ids),
        "total_valid_windows": total,
        "embedding_dimension": dimension,
        "artifacts": manifest["artifacts"],
        "partition": identity.partition,
        "outer_test_cached": False,
        "integrity_verified": True,
    }
    return payload, receipt


def payloads_equal(left: EmbeddingCachePayload, right: EmbeddingCachePayload) -> bool:
    if left.unit_ids != right.unit_ids:
        return False
    for left_values, right_values in (
        (left.embeddings, right.embeddings),
        (left.time_maps, right.time_maps),
        (left.valid_samples, right.valid_samples),
        (left.window_masks, right.window_masks),
    ):
        if len(left_values) != len(right_values) or any(
            not torch.equal(a.detach().cpu(), b.detach().cpu())
            for a, b in zip(left_values, right_values)
        ):
            return False
    return True


class FrozenEmbeddingCache:
    """Compute on miss and prove an exact, device-independent hit on later calls."""

    def __init__(self, cache_root: Path) -> None:
        _reject_terminal_cache_root(cache_root)
        self.cache_root = cache_root

    def get_or_compute(
        self,
        identity: EmbeddingCacheIdentity,
        encoder: nn.Module,
        compute: Callable[[], EmbeddingCachePayload],
        *,
        frontend_deterministic: bool,
        augmentation_enabled: bool,
    ) -> tuple[EmbeddingCachePayload, dict[str, object]]:
        identity.validate()
        trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
        if trainable or encoder.training:
            raise RuntimeError("embedding cache requires a frozen encoder in eval mode")
        if not frontend_deterministic or augmentation_enabled:
            raise RuntimeError("embedding cache forbids stochastic frontend or augmentation")
        target = self.cache_root.resolve() / identity.cache_key()
        if target.exists():
            payload, receipt = load_embedding_cache(self.cache_root, identity)
            return payload, {**receipt, "cache_status": "hit_verified_existing"}
        payload = compute()
        if not isinstance(payload, EmbeddingCachePayload):
            raise TypeError("cache compute function returned the wrong contract type")
        payload.validate(identity)
        receipt = write_embedding_cache(self.cache_root, identity, payload)
        reloaded, verified = load_embedding_cache(self.cache_root, identity)
        if not payloads_equal(payload, reloaded):
            raise RuntimeError("cached embeddings are not numerically identical to uncached output")
        return reloaded, {**verified, "cache_status": receipt["cache_status"], "uncached_equivalence": "exact"}
