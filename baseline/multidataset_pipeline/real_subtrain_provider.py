"""Real, frozen-split waveform provider for shared-window engineering preflight.

This module is an adapter over ``four_dataset_frozen_encoder.data.build_samples``;
it does not create or modify a split.  The canonical loader necessarily audits
manifest metadata for every partition while reconstructing its accepted split,
but this provider exposes only ``subtrain`` or ``validation`` rows and never
loads terminal targets, outer/test waveforms, or evaluation results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from baseline.four_dataset_frozen_encoder.data import Sample, build_samples

from .beats_temporal import HFRawInterval
from .contracts import ObservationState, PREDICTION_UNITS, SAMPLE_RATE, WaveformSample
from .hf_data import HFSampleRecord, build_hf_manifest, load_hf_waveform
from .preflight import FOUR_DATASET_SUBTRAIN_UNITS
from .sliding_window import SlidingWindowBatch, collate_sliding_windows


LANE_BY_CANONICAL_DATASET = {
    "icbhi": "ICBHI",
    "sprsound": "SPRSound",
    "hf_lung": "HF",
    "kauh": "KAUH",
}
TARGET_KEYS = {
    "ICBHI": ("icbhi_flat4",),
    "SPRSound": ("spr_binary", "spr_seven"),
    "HF": (),
    "KAUH": ("kauh_raw9",),
}
PROVIDER_SCHEMA_VERSION = "real_frozen_provider_identity_v2"


def _sha_lines(values: Sequence[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    """Hash one JSON-only identity object with an explicit stable serialization."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_copy(value: Mapping[str, object]) -> dict[str, object]:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    copied = json.loads(payload)
    if not isinstance(copied, dict):
        raise TypeError("provider identity must serialize to a JSON object")
    return copied


def _require_bound_identity(value: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise RuntimeError(f"provider identity field missing: {'.'.join(path)}")
        current = current[key]
    if current is None:
        raise RuntimeError(f"provider identity field is null: {'.'.join(path)}")
    return current


def build_bound_data_identity(
    *,
    partition: str,
    kauh_outer_fold: int,
    manifest_ordered_id_sha256_by_dataset: Mapping[str, str],
    split_ordered_id_sha256_by_dataset: Mapping[str, str],
    canonical_identity: Mapping[str, object],
    hf_annotation_identity: Mapping[str, object],
) -> dict[str, object]:
    """Bind sample order, split authority, and HF annotation-tree identity."""

    required_canonical_paths = (
        ("status",),
        ("rows",),
        ("unique_ids",),
        ("ordered_id_sha256",),
        ("dataset_rows",),
        ("split_identity", "ICBHI", "manifest_sha256"),
        ("split_identity", "ICBHI", "partition"),
        ("split_identity", "ICBHI", "validation"),
        ("split_identity", "SPRSound", "source_commit"),
        ("split_identity", "SPRSound", "partition"),
        ("split_identity", "SPRSound", "validation"),
        ("split_identity", "SPRSound", "terminal_manifest_label_free"),
        ("split_identity", "HF", "assignment_sha256"),
        ("split_identity", "HF", "partition"),
        ("split_identity", "HF", "date_proxy_counts"),
        ("split_identity", "KAUH", "outer_fold"),
        ("split_identity", "KAUH", "partition"),
        ("split_identity", "KAUH", "partition_patients"),
        ("split_identity", "KAUH", "outer_test_ordered_id_sha256"),
    )
    required_hf_paths = tuple(
        (key,)
        for key in (
            "assignment_sha256",
            "ordered_record_sha256",
            "own_label_tree_sha256",
            "accepted_label_tree_sha256_reference",
            "label_tree_identity_status",
            "partition_proxy_counts",
            "recording_states",
            "interval_semantics",
            "gap_semantics",
            "explicit_negative_intervals",
        )
    )
    for path in required_canonical_paths:
        _require_bound_identity(canonical_identity, path)
    for path in required_hf_paths:
        _require_bound_identity(hf_annotation_identity, path)
    if set(manifest_ordered_id_sha256_by_dataset) != set(PREDICTION_UNITS):
        raise RuntimeError("manifest ID identities must cover exactly four native lanes")
    if set(split_ordered_id_sha256_by_dataset) != set(PREDICTION_UNITS):
        raise RuntimeError("split ID identities must cover exactly four native lanes")
    return {
        "provider_schema_version": PROVIDER_SCHEMA_VERSION,
        "canonical_json_serialization": (
            "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False);utf8"
        ),
        "canonical_split_authority": {
            "loader": "baseline.four_dataset_frozen_encoder.data.build_samples",
            "split_recomputed_by_provider": False,
        },
        "partition": partition,
        "kauh_outer_fold": kauh_outer_fold,
        "manifest_ordered_id_sha256_by_dataset": dict(
            manifest_ordered_id_sha256_by_dataset
        ),
        "split_ordered_id_sha256_by_dataset": dict(
            split_ordered_id_sha256_by_dataset
        ),
        "canonical_contract_identity": _canonical_json_copy(canonical_identity),
        "hf_annotation_identity": _canonical_json_copy(hf_annotation_identity),
    }


@dataclass(frozen=True)
class FrozenNativeUnit:
    """One accepted canonical row, optionally joined to an HF raw record."""

    lane: str
    sample: Sample
    hf_record: HFSampleRecord | None = None

    def __post_init__(self) -> None:
        if self.lane not in PREDICTION_UNITS:
            raise ValueError(f"unknown native lane: {self.lane}")
        if LANE_BY_CANONICAL_DATASET.get(self.sample.dataset) != self.lane:
            raise ValueError("canonical dataset/lane mismatch")
        if self.sample.partition not in {"subtrain", "validation"}:
            raise ValueError("provider rows must exclude outer/test")
        if (self.lane == "HF") != (self.hf_record is not None):
            raise ValueError("exactly the HF lane requires an HFSampleRecord")
        if self.hf_record is not None:
            if (
                self.hf_record.sample_id != self.sample.sample_id
                or self.hf_record.partition != self.sample.partition
                or self.hf_record.group_id != self.sample.group_id
            ):
                raise ValueError("HF canonical/raw-record join mismatch")


@dataclass(frozen=True)
class FrozenProviderIndex:
    partition: str
    lanes: Mapping[str, tuple[FrozenNativeUnit, ...]]
    receipt: Mapping[str, object]

    def unit(self, lane: str, index: int = 0) -> FrozenNativeUnit:
        if lane not in self.lanes:
            raise KeyError(lane)
        return self.lanes[lane][index]


@dataclass(frozen=True)
class NativeWindowBatch:
    """Homogeneous training batch plus native targets or HF raw intervals."""

    lane: str
    windows: SlidingWindowBatch
    targets: Mapping[str, torch.Tensor]
    hf_intervals: tuple[tuple[HFRawInterval, ...], ...] = ()
    hf_recording_states: tuple[ObservationState, ...] = ()

    def validate(self) -> None:
        self.windows.validate()
        batch_size = len(self.windows.sample_ids)
        if self.lane not in PREDICTION_UNITS:
            raise ValueError("unknown batch lane")
        if any(value != self.lane for value in self.windows.dataset_ids):
            raise ValueError("training batches must be lane-homogeneous")
        if any(
            lineage.get("partition") not in {"subtrain", "validation"}
            or lineage.get("outer_test_accessed") != "false"
            for lineage in self.windows.lineage
        ):
            raise RuntimeError("outer/test lineage isolation failed")
        expected = set(TARGET_KEYS[self.lane])
        if set(self.targets) != expected:
            raise ValueError("native target keys do not match lane")
        for value in self.targets.values():
            if value.shape != (batch_size,) or value.dtype != torch.long:
                raise TypeError("native class targets must be int64 [B]")
        if self.lane == "HF":
            if (
                len(self.hf_intervals) != batch_size
                or len(self.hf_recording_states) != batch_size
            ):
                raise ValueError("HF raw target lineage length mismatch")
        elif self.hf_intervals or self.hf_recording_states:
            raise ValueError("non-HF lanes cannot carry HF intervals")


def _asset_gate(dataset_root: Path) -> dict[str, Path]:
    roots = {
        "ICBHI": dataset_root / "icbhi_2017",
        "SPRSound": dataset_root / "sprsound",
        "HF": dataset_root / "hf_lung_v1" / "source_original",
        "KAUH": dataset_root / "kauh_fraiwan" / "source_original" / "audio_files",
    }
    missing = {lane: str(path) for lane, path in roots.items() if not path.is_dir()}
    if missing:
        raise FileNotFoundError(
            "real shared-window assets missing (downloads forbidden): "
            + json.dumps(missing, sort_keys=True)
        )
    return roots


def build_frozen_provider_index(
    dataset_root: Path,
    *,
    partition: str = "subtrain",
    kauh_outer_fold: int = 0,
    canonical_loader: Callable[[Path, int], tuple[list[Sample], dict[str, object]]] = build_samples,
    hf_loader: Callable[[Path], tuple[Sequence[HFSampleRecord], dict[str, object]]] = build_hf_manifest,
    enforce_real_counts: bool = True,
) -> FrozenProviderIndex:
    """Build a non-terminal view over the accepted four-dataset split contract."""

    if partition not in {"subtrain", "validation"}:
        raise ValueError("provider partition must be subtrain or validation; test is terminal-only")
    roots = _asset_gate(dataset_root) if enforce_real_counts else {
        "HF": dataset_root / "hf_lung_v1" / "source_original"
    }
    samples, canonical_receipt = canonical_loader(dataset_root, kauh_outer_fold)
    hf_records, hf_receipt = hf_loader(roots["HF"])
    hf_by_id = {record.sample_id: record for record in hf_records}
    if len(hf_by_id) != len(hf_records):
        raise RuntimeError("duplicate HF record ID")

    lanes: dict[str, list[FrozenNativeUnit]] = {lane: [] for lane in PREDICTION_UNITS}
    for sample in samples:
        if sample.partition != partition:
            continue
        lane = LANE_BY_CANONICAL_DATASET.get(sample.dataset)
        if lane is None:
            raise RuntimeError(f"unknown canonical dataset: {sample.dataset}")
        record = hf_by_id.get(sample.sample_id) if lane == "HF" else None
        if lane == "HF":
            if record is None:
                raise RuntimeError(f"HF raw record missing for {sample.sample_id}")
            expected_path = (roots["HF"] / record.wav_relative_path).resolve()
            if Path(sample.audio_path).resolve() != expected_path:
                raise RuntimeError(f"HF audio-path join mismatch for {sample.sample_id}")
        lanes[lane].append(FrozenNativeUnit(lane, sample, record))

    frozen_lanes = {
        lane: tuple(sorted(rows, key=lambda row: row.sample.sample_id))
        for lane, rows in lanes.items()
    }
    counts = {lane: len(rows) for lane, rows in frozen_lanes.items()}
    if any(value == 0 for value in counts.values()):
        raise RuntimeError(f"partition lacks a native lane: {counts}")
    if partition == "subtrain" and enforce_real_counts and counts != FOUR_DATASET_SUBTRAIN_UNITS:
        raise RuntimeError(
            f"frozen subtrain count gate failed: {counts} != {FOUR_DATASET_SUBTRAIN_UNITS}"
        )
    group_counts = {
        lane: len({unit.sample.group_id for unit in rows})
        for lane, rows in frozen_lanes.items()
    }
    manifest_ordered_id_sha256_by_dataset = {
        lane: _sha_lines(
            [
                sample.sample_id
                for sample in samples
                if LANE_BY_CANONICAL_DATASET[sample.dataset] == lane
            ]
        )
        for lane in PREDICTION_UNITS
    }
    split_ordered_id_sha256_by_dataset = {
        lane: _sha_lines([unit.sample.sample_id for unit in rows])
        for lane, rows in frozen_lanes.items()
    }
    canonical_datasets = canonical_receipt.get("datasets", {})
    canonical_identity = {
        "status": canonical_receipt.get("status"),
        "rows": canonical_receipt.get("rows"),
        "unique_ids": canonical_receipt.get("unique_ids"),
        "ordered_id_sha256": canonical_receipt.get("ordered_id_sha256"),
        "dataset_rows": canonical_receipt.get("dataset_rows"),
        "split_identity": {
            "ICBHI": {
                "manifest_sha256": canonical_datasets.get("icbhi", {}).get("manifest_sha256"),
                "partition": canonical_datasets.get("icbhi", {}).get("partition"),
                "validation": canonical_datasets.get("icbhi", {}).get("validation"),
            },
            "SPRSound": {
                "source_commit": canonical_datasets.get("sprsound", {}).get("source_commit"),
                "partition": {
                    "subtrain": canonical_datasets.get("sprsound", {}).get("subtrain_events"),
                    "validation": canonical_datasets.get("sprsound", {}).get("validation_events"),
                },
                "validation": canonical_datasets.get("sprsound", {}).get("validation"),
                "terminal_manifest_label_free": canonical_datasets.get("sprsound", {}).get(
                    "test_manifest_label_free"
                ),
            },
            "HF": {
                "assignment_sha256": canonical_datasets.get("hf_lung", {}).get("assignment_sha256"),
                "partition": canonical_datasets.get("hf_lung", {}).get("partition"),
                "date_proxy_counts": canonical_datasets.get("hf_lung", {}).get("date_proxy_counts"),
            },
            "KAUH": {
                "outer_fold": canonical_datasets.get("kauh", {}).get("outer_fold"),
                "partition": canonical_datasets.get("kauh", {}).get("partition"),
                "partition_patients": canonical_datasets.get("kauh", {}).get("partition_patients"),
                "outer_test_ordered_id_sha256": canonical_datasets.get("kauh", {}).get(
                    "outer_test_ordered_id_sha256"
                ),
            },
        },
    }
    hf_identity = {
        key: hf_receipt.get(key)
        for key in (
            "status",
            "recordings",
            "assignment_sha256",
            "ordered_record_sha256",
            "own_label_tree_sha256",
            "accepted_label_tree_sha256_reference",
            "label_tree_identity_status",
            "partition_proxy_counts",
            "recording_states",
            "interval_semantics",
            "gap_semantics",
            "explicit_negative_intervals",
        )
    }
    data_identity = build_bound_data_identity(
        partition=partition,
        kauh_outer_fold=kauh_outer_fold,
        manifest_ordered_id_sha256_by_dataset=manifest_ordered_id_sha256_by_dataset,
        split_ordered_id_sha256_by_dataset=split_ordered_id_sha256_by_dataset,
        canonical_identity=canonical_identity,
        hf_annotation_identity=hf_identity,
    )
    data_identity_sha256 = canonical_json_sha256(data_identity)
    independent_verifier_status = str(
        _require_bound_identity(hf_receipt, ("independent_verifier_status",))
    )
    if independent_verifier_status != "HOLD":
        raise RuntimeError("provider cannot upgrade independent verifier status")
    own_label_tree = str(hf_identity["own_label_tree_sha256"])
    accepted_label_tree = str(hf_identity["accepted_label_tree_sha256_reference"])
    receipt = {
        "status": "real_frozen_provider_inventory_passed",
        "partition": partition,
        "outer_test_accessed": False,
        "outer_test_samples_emitted": 0,
        "outer_test_waveforms_decoded": 0,
        "outer_test_scoring": False,
        "canonical_split_authority": (
            "baseline.four_dataset_frozen_encoder.data.build_samples"
        ),
        "split_recomputed_by_provider": False,
        "canonical_loader_manifest_audit_boundary": (
            "accepted loader reconstructs split assignments from source metadata; "
            "provider filters before waveform decoding and never calls terminal target loaders"
        ),
        "kauh_outer_fold": kauh_outer_fold,
        "lane_counts": counts,
        "group_counts": group_counts,
        "data_identity": data_identity,
        "data_identity_sha256": data_identity_sha256,
        "identity_binding_status": (
            "identity_bound_to_canonical_split_and_hf_annotation_v2"
        ),
        "independent_verifier_status": "HOLD",
        "hf_label_tree_equivalence_status": (
            "not_verified_equivalent_reference_differs"
            if own_label_tree != accepted_label_tree
            else "digest_match_only_equivalence_still_requires_independent_verifier"
        ),
        "sample_ids": {
            lane: [unit.sample.sample_id for unit in rows]
            for lane, rows in frozen_lanes.items()
        },
        "canonical_receipt": canonical_identity,
        "hf_manifest_receipt": hf_identity,
    }
    return FrozenProviderIndex(partition, frozen_lanes, receipt)


def _load_non_hf_waveform(unit: FrozenNativeUnit) -> tuple[WaveformSample, dict[str, object]]:
    try:
        import torchaudio
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "torchaudio is required for real waveform loading; inventory remains dependency-lazy"
        ) from error
    sample = unit.sample
    waveform, source_rate = torchaudio.load(sample.audio_path)
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise RuntimeError(f"source audio must be mono: {sample.audio_path}")
    waveform = waveform.squeeze(0).to(torch.float32)
    source_start_s = float(sample.crop_start_s or 0.0)
    if sample.crop_start_s is not None or sample.crop_end_s is not None:
        if sample.crop_start_s is None or sample.crop_end_s is None:
            raise RuntimeError("crop start/end must be both present or both absent")
        start = round(sample.crop_start_s * source_rate)
        end = round(sample.crop_end_s * source_rate)
        if not 0 <= start < end <= waveform.numel():
            raise RuntimeError(f"invalid canonical crop for {sample.sample_id}")
        waveform = waveform[start:end]
    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    waveform = waveform.contiguous().to(torch.float32)
    if waveform.numel() <= 0 or not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"invalid decoded waveform for {sample.sample_id}")
    source_end_s = source_start_s + waveform.numel() / SAMPLE_RATE
    lineage = {
        "partition": sample.partition,
        "outer_test_accessed": "false",
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "source_id": str(
            sample.metadata.get("recording_id", sample.metadata.get("patient_id", sample.sample_id))
        ),
        "source_audio_path": str(Path(sample.audio_path).resolve()),
        "source_sample_rate": str(source_rate),
        "resampler": (
            "identity_16000" if source_rate == SAMPLE_RATE else f"torchaudio_{source_rate}_to_16000"
        ),
    }
    for key in ("patient_id", "recording_id", "official_split", "source_partition", "filter_mode"):
        if key in sample.metadata:
            lineage[key] = str(sample.metadata[key])
    output = WaveformSample(
        waveform=waveform,
        sample_id=sample.sample_id,
        dataset_id=unit.lane,
        prediction_unit=PREDICTION_UNITS[unit.lane],
        source_start_s=source_start_s,
        source_end_s=source_end_s,
        lineage=lineage,
    )
    return output, {
        "sample_id": sample.sample_id,
        "lane": unit.lane,
        "source_sample_rate": source_rate,
        "output_samples": waveform.numel(),
        "output_sample_rate": SAMPLE_RATE,
        "repeat_pad": False,
        "truncate": False,
    }


def load_frozen_waveform(unit: FrozenNativeUnit) -> tuple[WaveformSample, dict[str, object]]:
    """Decode exactly one approved non-terminal unit at 16 kHz."""

    if unit.sample.partition not in {"subtrain", "validation"}:
        raise RuntimeError("outer/test waveform decode is forbidden in this provider")
    if unit.lane != "HF":
        return _load_non_hf_waveform(unit)
    if unit.hf_record is None:
        raise RuntimeError("HF record join missing")
    root = Path(unit.sample.audio_path).resolve()
    for _ in Path(unit.hf_record.wav_relative_path).parts:
        root = root.parent
    sample, receipt = load_hf_waveform(root, unit.hf_record)
    lineage = {
        **dict(sample.lineage),
        "outer_test_accessed": "false",
        "source_id": unit.hf_record.sample_id,
    }
    return WaveformSample(
        waveform=sample.waveform,
        sample_id=sample.sample_id,
        dataset_id=sample.dataset_id,
        prediction_unit=sample.prediction_unit,
        source_start_s=sample.source_start_s,
        source_end_s=sample.source_end_s,
        lineage=lineage,
    ), receipt


def load_native_window_batch(units: Sequence[FrozenNativeUnit]) -> NativeWindowBatch:
    if not units:
        raise ValueError("cannot load an empty native batch")
    lane = units[0].lane
    if any(unit.lane != lane for unit in units):
        raise ValueError("training batches must be lane-homogeneous")
    loaded = [load_frozen_waveform(unit)[0] for unit in units]
    windows = collate_sliding_windows(loaded)
    targets = {
        key: torch.tensor([int(unit.sample.targets[key]) for unit in units], dtype=torch.long)
        for key in TARGET_KEYS[lane]
    }
    output = NativeWindowBatch(
        lane=lane,
        windows=windows,
        targets=targets,
        hf_intervals=tuple(unit.hf_record.raw_intervals for unit in units if unit.hf_record),
        hf_recording_states=tuple(
            unit.hf_record.recording_state for unit in units if unit.hf_record
        ),
    )
    output.validate()
    return output


def build_real_subtrain_preflight_batches(
    pipeline_id: str,
    *,
    dataset_root: Path = Path("dataset/raw"),
) -> tuple[SlidingWindowBatch, ...]:
    """Return one real mixed-lane batch for the zero-update L40 preflight."""

    if pipeline_id not in {"P1", "P2", "P3", "P4", "P5"}:
        raise ValueError("real preflight provider supports P1-P5")
    index = build_frozen_provider_index(dataset_root, partition="subtrain")
    units = [index.unit(lane) for lane in PREDICTION_UNITS]
    samples = [load_frozen_waveform(unit)[0] for unit in units]
    batch = collate_sliding_windows(samples)
    if set(batch.dataset_ids) != set(PREDICTION_UNITS):
        raise RuntimeError("real preflight batch must cover all four lanes")
    return (batch,)


def cpu_loader_smoke(dataset_root: Path) -> dict[str, object]:
    """Decode one deterministic subtrain unit per lane; no model or optimizer."""

    index = build_frozen_provider_index(dataset_root, partition="subtrain")
    units = [index.unit(lane) for lane in PREDICTION_UNITS]
    loaded = [load_frozen_waveform(unit) for unit in units]
    batch = collate_sliding_windows([item[0] for item in loaded])
    return {
        "status": "real_subtrain_cpu_loader_contract_smoke_passed",
        "engineering_only": True,
        "performance_result": False,
        "decoded_units": 4,
        "decoded_units_by_lane": dict(Counter(batch.dataset_ids)),
        "sample_ids": list(batch.sample_ids),
        "lineage": [dict(value) for value in batch.lineage],
        "waveform_receipts": [item[1] for item in loaded],
        "window_receipt": batch.receipt(),
        "outer_test_accessed": False,
        "outer_test_waveforms_decoded": 0,
    }


def _inventory_summary(index: FrozenProviderIndex) -> dict[str, object]:
    receipt = dict(index.receipt)
    receipt["sample_ids"] = {
        lane: {
            "count": len(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
        }
        for lane, values in receipt["sample_ids"].items()
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--phase", choices=("inventory", "cpu-loader-smoke"), default="inventory")
    args = parser.parse_args()
    if args.phase == "inventory":
        receipt = _inventory_summary(
            build_frozen_provider_index(args.dataset_root, partition="subtrain")
        )
    else:
        receipt = cpu_loader_smoke(args.dataset_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
