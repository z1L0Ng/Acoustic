"""Runner integration for immutable P1/P2 frozen window embeddings.

The full phase builds or verifies all four native lanes for both subtrain and
validation before update 1.  Subsequent batches reconstruct an exact
``SharedWindowEncoderOutput`` without decoding waveforms or invoking the frozen
encoder.  Outer/test partitions are structurally unavailable here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from .contracts import PREDICTION_UNITS
from .embedding_cache import (
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EmbeddingCacheIdentity,
    EmbeddingCachePayload,
    FrozenEmbeddingCache,
)
from .preflight import SharedWindowEncoderOutput
from .real_subtrain_provider import (
    TARGET_KEYS,
    FrozenNativeUnit,
    FrozenProviderIndex,
    NativeWindowBatch,
    load_native_window_batch,
)
from .window_encoder import ProductionWindowEncoder


RUNNER_CACHE_SET_SCHEMA_VERSION = "shared_window_runner_cache_set_v1"
RUNNER_CACHE_PIPELINES = {"P1": "AST", "P2": "BEATs"}
RUNNER_CACHE_PARTITIONS = ("subtrain", "validation")
RUNNER_CACHE_LANES = ("ICBHI", "SPRSound", "HF", "KAUH")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_set_sha256(repo_root: Path, pipeline_id: str) -> str:
    candidate = {
        "P1": "ast_window_encoder.py",
        "P2": "beats_window_encoder.py",
    }[pipeline_id]
    relatives = (
        "baseline/multidataset_pipeline/adapter_assets.json",
        "baseline/multidataset_pipeline/asset_manifest.py",
        "baseline/multidataset_pipeline/embedding_cache.py",
        "baseline/multidataset_pipeline/runner_embedding_cache.py",
        "baseline/multidataset_pipeline/sliding_window.py",
        "baseline/multidataset_pipeline/window_encoder.py",
        f"baseline/multidataset_pipeline/{candidate}",
    )
    receipt: dict[str, str] = {}
    for relative in relatives:
        path = repo_root.resolve() / relative
        if not path.is_file():
            raise FileNotFoundError(f"cache code identity file missing: {path}")
        receipt[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_json_sha256(receipt)


@dataclass(frozen=True)
class CachedNativeBatch:
    """Native targets/lineage paired with cached shared-window embeddings."""

    lane: str
    output: SharedWindowEncoderOutput
    targets: Mapping[str, torch.Tensor]
    hf_intervals: tuple[tuple[object, ...], ...] = ()
    hf_recording_states: tuple[object, ...] = ()

    def validate(self) -> None:
        if self.lane not in RUNNER_CACHE_LANES:
            raise ValueError("cached batch lane is outside the four native lanes")
        batch, windows, dimension = self.output.embeddings.shape
        if dimension != 768 or self.output.window_mask.shape != (batch, windows):
            raise ValueError("cached output must be [B,K,768] with bool [B,K] mask")
        if self.output.time_map.shape != (batch, windows, 2):
            raise ValueError("cached output time_map must be [B,K,2]")
        if self.output.window_mask.dtype != torch.bool or self.output.time_map.dtype != torch.float64:
            raise TypeError("cached output mask/time_map dtype changed")
        if not self.output.embeddings.dtype.is_floating_point or not bool(
            torch.isfinite(self.output.embeddings).all()
        ):
            raise TypeError("cached embeddings must be finite floating point")
        if bool(torch.count_nonzero(self.output.embeddings[~self.output.window_mask])):
            raise RuntimeError("cached padded window embeddings must remain exact zero")
        if len(self.output.sample_ids) != batch or any(
            value != self.lane for value in self.output.dataset_ids
        ):
            raise RuntimeError("cached native lineage changed")
        expected_targets = set(TARGET_KEYS[self.lane])
        if set(self.targets) != expected_targets:
            raise ValueError("cached native target keys changed")
        for target in self.targets.values():
            if target.shape != (batch,) or target.dtype != torch.long:
                raise TypeError("cached native class targets must be int64 [B]")
        if self.lane == "HF":
            if len(self.hf_intervals) != batch or len(self.hf_recording_states) != batch:
                raise ValueError("cached HF raw interval/state lineage changed")
        elif self.hf_intervals or self.hf_recording_states:
            raise ValueError("non-HF cached batches cannot carry HF interval lineage")


@dataclass(frozen=True)
class CachedLanePartition:
    partition: str
    lane: str
    units: tuple[FrozenNativeUnit, ...]
    payload: EmbeddingCachePayload
    receipt: Mapping[str, object]

    def batch(self, indices: Sequence[int], *, device: torch.device) -> CachedNativeBatch:
        if not indices:
            raise ValueError("cannot construct an empty cached native batch")
        if self.partition not in RUNNER_CACHE_PARTITIONS or self.lane not in RUNNER_CACHE_LANES:
            raise RuntimeError("cached lane/partition contract changed")
        if any(not isinstance(index, int) or not 0 <= index < len(self.units) for index in indices):
            raise IndexError("cached batch index outside frozen ordered units")
        selected_units = tuple(self.units[index] for index in indices)
        selected_embeddings = tuple(self.payload.embeddings[index] for index in indices)
        selected_time_maps = tuple(self.payload.time_maps[index] for index in indices)
        selected_masks = tuple(self.payload.window_masks[index] for index in indices)
        max_windows = max(value.shape[0] for value in selected_embeddings)
        embeddings = torch.zeros(
            len(indices), max_windows, 768, dtype=torch.float32, device=device
        )
        window_mask = torch.zeros(len(indices), max_windows, dtype=torch.bool, device=device)
        time_map = torch.zeros(
            len(indices), max_windows, 2, dtype=torch.float64, device=device
        )
        for row, (values, times, mask) in enumerate(
            zip(selected_embeddings, selected_time_maps, selected_masks)
        ):
            count = values.shape[0]
            if not bool(mask.all()) or mask.shape != (count,):
                raise RuntimeError("ragged cache must contain all and only valid windows")
            embeddings[row, :count].copy_(values.to(device))
            window_mask[row, :count] = True
            time_map[row, :count].copy_(times.to(device))
        output = SharedWindowEncoderOutput(
            embeddings=embeddings,
            window_mask=window_mask,
            time_map=time_map,
            encoder_identity=str(self.receipt["encoder_identity"]),
            sample_ids=tuple(unit.sample.sample_id for unit in selected_units),
            dataset_ids=tuple(self.lane for _ in selected_units),
            prediction_units=tuple(PREDICTION_UNITS[self.lane] for _ in selected_units),
        )
        targets = {
            key: torch.tensor(
                [int(unit.sample.targets[key]) for unit in selected_units],
                dtype=torch.long,
            )
            for key in TARGET_KEYS[self.lane]
        }
        result = CachedNativeBatch(
            lane=self.lane,
            output=output,
            targets=targets,
            hf_intervals=tuple(
                unit.hf_record.raw_intervals for unit in selected_units if unit.hf_record
            ),
            hf_recording_states=tuple(
                unit.hf_record.recording_state for unit in selected_units if unit.hf_record
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class RunnerEmbeddingCacheSet:
    pipeline_id: str
    entries: Mapping[tuple[str, str], CachedLanePartition]
    receipt: Mapping[str, object]

    def validate_complete(self) -> None:
        expected = {
            (partition, lane)
            for partition in RUNNER_CACHE_PARTITIONS
            for lane in RUNNER_CACHE_LANES
        }
        if self.pipeline_id not in RUNNER_CACHE_PIPELINES or set(self.entries) != expected:
            raise RuntimeError("full P1/P2 requires all eight lane/partition caches")
        if (
            self.receipt.get("schema_version") != RUNNER_CACHE_SET_SCHEMA_VERSION
            or self.receipt.get("pipeline_id") != self.pipeline_id
            or self.receipt.get("all_required_caches_complete") is not True
            or self.receipt.get("outer_test_cached") is not False
            or self.receipt.get("training_reads_waveforms_after_cache_gate") is not False
            or self.receipt.get("validation_reads_waveforms_after_cache_gate") is not False
        ):
            raise RuntimeError("runner embedding cache set receipt is incomplete")
        expected_identity = canonical_json_sha256(
            {key: value for key, value in self.receipt.items() if key != "receipt_sha256"}
        )
        if self.receipt.get("receipt_sha256") != expected_identity:
            raise RuntimeError("runner embedding cache set immutable receipt changed")

    def batch(
        self,
        partition: str,
        lane: str,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> CachedNativeBatch:
        self.validate_complete()
        return self.entries[(partition, lane)].batch(indices, device=device)


def _identity_for_lane(
    *,
    repo_root: Path,
    pipeline_id: str,
    config_identity_sha256: str,
    adapter: ProductionWindowEncoder,
    index: FrozenProviderIndex,
    lane: str,
) -> EmbeddingCacheIdentity:
    if index.partition not in RUNNER_CACHE_PARTITIONS:
        raise PermissionError("runner cache cannot bind an outer/test partition")
    adapter_receipt = adapter.receipt()
    frontend_adapter = {
        "frontend": {
            "encoder_identity": adapter.encoder_identity,
            "window_policy": "16k_source_time_2s_window_1s_stride",
            "provenance": adapter_receipt["provenance"],
        },
        "adapter": adapter_receipt["dimension_adapter"],
    }
    units = index.lanes[lane]
    dataset_release = canonical_json_sha256(
        {
            "lane": lane,
            "canonical_contract_identity": index.receipt["canonical_receipt"],
            "manifest_ordered_ids": index.receipt["data_identity"][
                "manifest_ordered_id_sha256_by_dataset"
            ][lane],
        }
    )
    return EmbeddingCacheIdentity.from_tracked_asset(
        repo_root=repo_root,
        pipeline_id=pipeline_id,
        dataset_id=lane,
        dataset_release=dataset_release,
        partition=index.partition,
        ordered_unit_ids=tuple(unit.sample.sample_id for unit in units),
        data_identity_sha256=str(index.receipt["data_identity_sha256"]),
        preprocessing={
            "sample_rate": 16_000,
            "resample_policy": "frozen_provider_source_rate_to_16000_no_repeat_no_truncate",
            "waveform_dtype": "float32",
        },
        window_policy={
            "window_length_s": 2.0,
            "window_stride_s": 1.0,
            "tail_policy": "append_unique_end_aligned_window_when_stride_misses_tail",
            "short_padding": "zero_pad_only",
            "repeat_pad": False,
            "truncate": False,
            "mask_semantics": "window_mask_true_is_valid;time_map_source_seconds_half_open",
        },
        frontend_adapter_identity={
            "frontend": json.dumps(frontend_adapter["frontend"], sort_keys=True),
            "adapter": json.dumps(frontend_adapter["adapter"], sort_keys=True),
            "identity_sha256": canonical_json_sha256(frontend_adapter),
        },
        code_identity_sha256=_file_set_sha256(repo_root, pipeline_id),
        config_identity_sha256=config_identity_sha256,
    )


def _compute_lane_payload(
    adapter: ProductionWindowEncoder,
    units: tuple[FrozenNativeUnit, ...],
    *,
    device: torch.device,
    batch_size: int,
    batch_loader: Callable[[Sequence[FrozenNativeUnit]], NativeWindowBatch],
) -> EmbeddingCachePayload:
    embeddings: list[torch.Tensor] = []
    time_maps: list[torch.Tensor] = []
    valid_samples: list[torch.Tensor] = []
    window_masks: list[torch.Tensor] = []
    adapter.eval()
    adapter.backend.eval()
    with torch.no_grad():
        for start in range(0, len(units), batch_size):
            batch = batch_loader(units[start : start + batch_size])
            if tuple(batch.windows.sample_ids) != tuple(
                unit.sample.sample_id for unit in units[start : start + batch_size]
            ):
                raise RuntimeError("cache builder batch loader changed ordered unit IDs")
            windows = batch.windows.to(device)
            output = adapter(windows)
            for row in range(len(output.sample_ids)):
                count = int(output.window_mask[row].sum())
                if count <= 0 or not bool(output.window_mask[row, :count].all()):
                    raise RuntimeError("cache builder received invalid window prefix")
                embeddings.append(output.embeddings[row, :count].detach().cpu().to(torch.float32))
                time_maps.append(output.time_map[row, :count].detach().cpu())
                valid_samples.append(batch.windows.valid_samples[row, :count].detach().cpu())
                window_masks.append(torch.ones(count, dtype=torch.bool))
    payload = EmbeddingCachePayload(
        unit_ids=tuple(unit.sample.sample_id for unit in units),
        embeddings=tuple(embeddings),
        time_maps=tuple(time_maps),
        valid_samples=tuple(valid_samples),
        window_masks=tuple(window_masks),
    )
    return payload


def build_or_load_runner_embedding_caches(
    *,
    repo_root: Path,
    cache_root: Path,
    pipeline_id: str,
    config_identity_sha256: str,
    adapter: ProductionWindowEncoder,
    indexes: Mapping[str, FrozenProviderIndex],
    device: torch.device,
    batch_size: int,
    batch_loader: Callable[[Sequence[FrozenNativeUnit]], NativeWindowBatch] = load_native_window_batch,
) -> RunnerEmbeddingCacheSet:
    """Build/verify all eight P1/P2 caches before any optimizer update."""

    if pipeline_id not in RUNNER_CACHE_PIPELINES:
        raise ValueError("production runner embedding cache currently supports P1/P2 only")
    if adapter.encoder_identity != RUNNER_CACHE_PIPELINES[pipeline_id]:
        raise RuntimeError("runner cache pipeline/adapter encoder identity mismatch")
    if set(indexes) != set(RUNNER_CACHE_PARTITIONS) or any(
        indexes[partition].partition != partition for partition in RUNNER_CACHE_PARTITIONS
    ):
        raise RuntimeError("runner cache requires exact subtrain and validation indexes")
    if batch_size <= 0:
        raise ValueError("runner cache batch size must be positive")
    cache = FrozenEmbeddingCache(cache_root)
    entries: dict[tuple[str, str], CachedLanePartition] = {}
    receipts: dict[str, dict[str, object]] = {}
    adapter.eval()
    adapter.backend.eval()
    for partition in RUNNER_CACHE_PARTITIONS:
        index = indexes[partition]
        receipts[partition] = {}
        for lane in RUNNER_CACHE_LANES:
            units = index.lanes[lane]
            identity = _identity_for_lane(
                repo_root=repo_root,
                pipeline_id=pipeline_id,
                config_identity_sha256=config_identity_sha256,
                adapter=adapter,
                index=index,
                lane=lane,
            )
            payload, cache_receipt = cache.get_or_compute(
                identity,
                adapter,
                lambda units=units: _compute_lane_payload(
                    adapter,
                    units,
                    device=device,
                    batch_size=batch_size,
                    batch_loader=batch_loader,
                ),
                frontend_deterministic=True,
                augmentation_enabled=False,
            )
            if payload.unit_ids != tuple(unit.sample.sample_id for unit in units):
                raise RuntimeError("runner cache ordered unit identity changed")
            entry_receipt = {
                **cache_receipt,
                "encoder_identity": adapter.encoder_identity,
                "lane": lane,
                "partition": partition,
                "ordered_unit_id_sha256": canonical_json_sha256(list(payload.unit_ids)),
                "data_identity_sha256": index.receipt["data_identity_sha256"],
                "identity_binding": {
                    "cache_key_is_full_identity_sha256": True,
                    "partition": identity.partition,
                    "dataset_release": identity.dataset_release,
                    "code_identity_sha256": identity.code_identity_sha256,
                    "config_identity_sha256": identity.config_identity_sha256,
                    "schema_identity_sha256": identity.schema_identity_sha256,
                    "asset_manifest_identity_sha256": identity.encoder_asset[
                        "asset_manifest_identity_sha256"
                    ],
                },
            }
            entries[(partition, lane)] = CachedLanePartition(
                partition=partition,
                lane=lane,
                units=units,
                payload=payload,
                receipt=entry_receipt,
            )
            receipts[partition][lane] = entry_receipt
    receipt: dict[str, object] = {
        "schema_version": RUNNER_CACHE_SET_SCHEMA_VERSION,
        "embedding_cache_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "encoder_identity": adapter.encoder_identity,
        "config_identity_sha256": config_identity_sha256,
        "required_partitions": list(RUNNER_CACHE_PARTITIONS),
        "required_lanes": list(RUNNER_CACHE_LANES),
        "entries": receipts,
        "all_required_caches_complete": True,
        "cache_build_completed_before_optimizer_updates": True,
        "training_reads_waveforms_after_cache_gate": False,
        "validation_reads_waveforms_after_cache_gate": False,
        "outer_test_cached": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    result = RunnerEmbeddingCacheSet(pipeline_id, entries, receipt)
    result.validate_complete()
    return result
