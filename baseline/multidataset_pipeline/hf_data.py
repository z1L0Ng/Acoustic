"""HF_Lung_V1 immutable manifest, waveform, and P6 target adapters.

Source/task facts are bounded by:
docs/datasets/four_dataset_task_contract_review_2026-07-28.md, sections 5 and 13.3.
"""

from __future__ import annotations

import hashlib
import re
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from .beats_temporal import (
    HFRawInterval,
    HFTemporalSupervision,
    HFTargetPolicy,
    TokenAlignmentPolicy,
    raw_intervals_to_token_supervision,
)
from .contracts import ObservationState, WaveformSample


SEED = 20260728
HF_ROOT_RELATIVE = "dataset/raw/hf_lung_v1/source_original"
RAW_TOKENS = ("I", "E", "D", "Wheeze", "Rhonchi", "Stridor")
EXPECTED_TOKEN_COUNTS = {
    "I": 34_095,
    "E": 18_349,
    "D": 15_606,
    "Wheeze": 8_457,
    "Rhonchi": 4_740,
    "Stridor": 686,
}
EXPECTED_ASSIGNMENT_SHA256 = (
    "33387aa62ebcb8adbc1fba626e6d27f27a3121d3a686cda6fc075a7da106943e"
)
ACCEPTED_LABEL_TREE_SHA256_REFERENCE = (
    "31997bf3d5b43f3c959e681b7bee5b3f5c0bd1f320cac2943c9f99ac26861c1b"
)
ASSIGNMENT_DIGEST_SERIALIZATION = (
    "records in sorted(Path.rglob('*.wav')) order; "
    "source_split:stem<TAB>partition<TAB>YYYYMMDD<LF>"
)
ORDERED_RECORD_DIGEST_SERIALIZATION = (
    "records in sorted(Path.rglob('*.wav')) order; "
    "sample_id<TAB>source_split<TAB>partition<TAB>date_proxy<TAB>"
    "wav_relative_path<TAB>label_relative_path<TAB>recording_state<TAB>"
    "raw_token,start_s_9dp,end_s_9dp joined by semicolon<LF>"
)
OWN_LABEL_TREE_DIGEST_SERIALIZATION = (
    "labels sorted by POSIX relative path; "
    "relative_path<TAB>sha256(file_bytes)<LF>"
)
RESAMPLER_CONFIG = {
    "implementation": "torchaudio.functional.resample",
    "source_sample_rate": 4_000,
    "target_sample_rate": 16_000,
    "lowpass_filter_width": 6,
    "rolloff": 0.99,
    "resampling_method": "sinc_interp_hann",
    "beta": None,
    "repeat_pad": False,
    "truncate": False,
}


@dataclass(frozen=True)
class HFExpectedCounts:
    recordings: int
    source_split: dict[str, int]
    raw_intervals: int
    empty_label_files: int
    token_counts: dict[str, int]
    source_proxy_counts: dict[str, int] | None = None
    partition_proxy_counts: dict[str, int] | None = None
    assignment_sha256: str | None = None


REAL_EXPECTED_COUNTS = HFExpectedCounts(
    recordings=9_765,
    source_split={"train": 7_809, "test": 1_956},
    raw_intervals=81_933,
    empty_label_files=58,
    token_counts=EXPECTED_TOKEN_COUNTS,
    source_proxy_counts={"train": 118, "test": 39},
    partition_proxy_counts={"subtrain": 94, "validation": 24, "test": 39},
    assignment_sha256=EXPECTED_ASSIGNMENT_SHA256,
)


@dataclass(frozen=True)
class HFSampleRecord:
    """Immutable HF recording row; date proxy is grouping only, never patient ID."""

    sample_id: str
    source_split: str
    partition: str
    date_proxy: str
    group_id: str
    wav_relative_path: str
    label_relative_path: str
    raw_intervals: tuple[HFRawInterval, ...]
    recording_state: ObservationState

    def __post_init__(self) -> None:
        if self.source_split not in {"train", "test"}:
            raise ValueError("source_split must be train or test")
        if self.partition not in {"subtrain", "validation", "test"}:
            raise ValueError("invalid canonical partition")
        if self.source_split == "test" and self.partition != "test":
            raise ValueError("source test must remain canonical test")
        if self.source_split == "train" and self.partition == "test":
            raise ValueError("source train cannot become canonical test")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date_proxy):
            raise ValueError("date_proxy must be YYYY-MM-DD")
        if self.group_id != f"date_proxy:{self.date_proxy}":
            raise ValueError("group_id must identify a date proxy, not a patient")
        if Path(self.wav_relative_path).is_absolute() or Path(
            self.label_relative_path
        ).is_absolute():
            raise ValueError("HF record paths must be relative")
        if self.recording_state is ObservationState.EMPTY:
            if self.raw_intervals:
                raise ValueError("EMPTY label file cannot contain intervals")
        elif self.recording_state is ObservationState.OBSERVED:
            if not self.raw_intervals:
                raise ValueError("OBSERVED record must contain intervals")
        else:
            raise ValueError("manifest recording_state must be OBSERVED or EMPTY")


def parse_hms(value: str) -> float:
    """Parse strict HH:MM:SS(.fraction) without accepting malformed fields."""

    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"invalid HH:MM:SS value: {value!r}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3_600 + int(minutes) * 60 + float(seconds)


def canonical_date_proxy(filename: str) -> str:
    """Reproduce the accepted date grouping proxy; it is not patient identity."""

    if filename.startswith("steth_"):
        compact = filename.split("_")[1]
        if not re.fullmatch(r"\d{8}", compact):
            raise ValueError(f"cannot parse HF date proxy: {filename}")
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    match = re.match(r"trunc_(\d{4}-\d{2}-\d{2})-", filename)
    if not match:
        raise ValueError(f"cannot parse HF date proxy: {filename}")
    return match.group(1)


def parse_label_file(path: Path) -> tuple[HFRawInterval, ...]:
    rows: list[HFRawInterval] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: expected token start end")
        token, raw_start, raw_end = parts
        if token not in RAW_TOKENS:
            raise ValueError(f"{path}:{line_number}: unknown raw token {token!r}")
        start_s, end_s = parse_hms(raw_start), parse_hms(raw_end)
        if not 0 <= start_s < end_s <= 15.001:
            raise ValueError(
                f"{path}:{line_number}: interval outside 15-second contract"
            )
        rows.append(HFRawInterval(token, start_s, end_s))
    return tuple(rows)


def _paired_paths(root: Path) -> list[tuple[Path, Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    wavs = sorted(root.rglob("*.wav"))
    labels = sorted(root.rglob("*_label.txt"))
    wav_by_key = {
        (path.relative_to(root).parent.as_posix(), path.stem): path for path in wavs
    }
    label_by_key = {
        (
            path.relative_to(root).parent.as_posix(),
            path.name.removesuffix("_label.txt"),
        ): path
        for path in labels
    }
    if len(wav_by_key) != len(wavs) or len(label_by_key) != len(labels):
        raise RuntimeError("duplicate HF WAV or label pairing key")
    missing_labels = sorted(set(wav_by_key) - set(label_by_key))
    missing_wavs = sorted(set(label_by_key) - set(wav_by_key))
    if missing_labels or missing_wavs:
        raise RuntimeError(
            f"HF WAV/label pairing failed: missing_labels={missing_labels[:3]}, "
            f"missing_wavs={missing_wavs[:3]}"
        )
    return [
        (wav, label_by_key[key], key[1])
        for key, wav in sorted(
            wav_by_key.items(), key=lambda item: item[1]
        )
    ]


def _assignment_sha256(records: Sequence[HFSampleRecord]) -> str:
    rows = [
        f"{record.source_split}:{record.sample_id.split(':', 2)[2]}"
        f"\t{record.partition}\t{record.date_proxy.replace('-', '')}"
        for record in records
    ]
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _ordered_record_sha256(records: Sequence[HFSampleRecord]) -> str:
    rows = []
    for record in records:
        intervals = ";".join(
            f"{row.raw_token},{row.start_s:.9f},{row.end_s:.9f}"
            for row in record.raw_intervals
        )
        rows.append(
            "\t".join(
                (
                    record.sample_id,
                    record.source_split,
                    record.partition,
                    record.date_proxy,
                    record.wav_relative_path,
                    record.label_relative_path,
                    record.recording_state.value,
                    intervals,
                )
            )
        )
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _own_label_tree_sha256(root: Path, records: Sequence[HFSampleRecord]) -> str:
    rows = []
    for relative in sorted(record.label_relative_path for record in records):
        content_digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        rows.append(f"{relative}\t{content_digest}")
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def _assign_partitions(
    raw_rows: list[dict[str, object]], seed: int
) -> list[str]:
    if seed != SEED:
        raise ValueError(f"canonical HF split seed must remain {SEED}")
    train_indices = [
        index for index, row in enumerate(raw_rows) if row["source_split"] == "train"
    ]
    groups = np.asarray([raw_rows[index]["date_proxy"] for index in train_indices])
    if len(set(groups.tolist())) < 2:
        raise RuntimeError("canonical split needs at least two train date proxies")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, validation_relative = next(
        splitter.split(train_indices, groups=groups)
    )
    validation = {train_indices[index] for index in validation_relative}
    return [
        (
            "test"
            if row["source_split"] == "test"
            else "validation"
            if index in validation
            else "subtrain"
        )
        for index, row in enumerate(raw_rows)
    ]


def build_hf_manifest(
    root: Path,
    *,
    expected: HFExpectedCounts = REAL_EXPECTED_COUNTS,
    seed: int = SEED,
) -> tuple[tuple[HFSampleRecord, ...], dict[str, object]]:
    """Read labels only, reproduce the canonical assignment, and fail closed."""

    root = root.resolve()
    raw_rows: list[dict[str, object]] = []
    for wav, label, stem in _paired_paths(root):
        wav_relative = wav.relative_to(root)
        label_relative = label.relative_to(root)
        if wav_relative.parts[0] not in {"train", "test"}:
            raise RuntimeError(f"cannot infer source split: {wav_relative}")
        source_split = wav_relative.parts[0]
        intervals = parse_label_file(label)
        raw_rows.append(
            {
                "stem": stem,
                "source_split": source_split,
                "date_proxy": canonical_date_proxy(wav.name),
                "wav_relative": wav_relative.as_posix(),
                "label_relative": label_relative.as_posix(),
                "intervals": intervals,
            }
        )
    partitions = _assign_partitions(raw_rows, seed)
    records = tuple(
        HFSampleRecord(
            sample_id=f"hf:{row['source_split']}:{row['stem']}",
            source_split=str(row["source_split"]),
            partition=partition,
            date_proxy=str(row["date_proxy"]),
            group_id=f"date_proxy:{row['date_proxy']}",
            wav_relative_path=str(row["wav_relative"]),
            label_relative_path=str(row["label_relative"]),
            raw_intervals=tuple(row["intervals"]),
            recording_state=(
                ObservationState.OBSERVED
                if row["intervals"]
                else ObservationState.EMPTY
            ),
        )
        for row, partition in zip(raw_rows, partitions)
    )
    source_split = Counter(record.source_split for record in records)
    token_counts = Counter(
        interval.raw_token
        for record in records
        for interval in record.raw_intervals
    )
    raw_intervals = sum(token_counts.values())
    empty = sum(
        record.recording_state is ObservationState.EMPTY for record in records
    )
    source_groups = {
        split: {
            record.date_proxy for record in records if record.source_split == split
        }
        for split in ("train", "test")
    }
    partition_groups = {
        partition: {
            record.date_proxy for record in records if record.partition == partition
        }
        for partition in ("subtrain", "validation", "test")
    }
    if (
        partition_groups["subtrain"] & partition_groups["validation"]
        or (partition_groups["subtrain"] | partition_groups["validation"])
        & partition_groups["test"]
        or source_groups["train"] & source_groups["test"]
    ):
        raise RuntimeError("canonical HF date-proxy grouping overlap")
    assignment_sha256 = _assignment_sha256(records)
    actual = {
        "recordings": len(records),
        "source_split": dict(sorted(source_split.items())),
        "raw_intervals": raw_intervals,
        "empty_label_files": empty,
        "token_counts": dict(sorted(token_counts.items())),
        "source_proxy_counts": {
            key: len(value) for key, value in source_groups.items()
        },
        "partition_proxy_counts": {
            key: len(value) for key, value in partition_groups.items()
        },
        "assignment_sha256": assignment_sha256,
    }
    expected_values = {
        "recordings": expected.recordings,
        "source_split": expected.source_split,
        "raw_intervals": expected.raw_intervals,
        "empty_label_files": expected.empty_label_files,
        "token_counts": expected.token_counts,
    }
    for key, value in expected_values.items():
        if actual[key] != value:
            raise RuntimeError(
                f"HF manifest gate failed for {key}: {actual[key]} != {value}"
            )
    for key, value in (
        ("source_proxy_counts", expected.source_proxy_counts),
        ("partition_proxy_counts", expected.partition_proxy_counts),
        ("assignment_sha256", expected.assignment_sha256),
    ):
        if value is not None and actual[key] != value:
            raise RuntimeError(
                f"HF canonical gate failed for {key}: {actual[key]} != {value}"
            )

    own_label_digest = _own_label_tree_sha256(root, records)
    receipt = {
        "status": "hf_manifest_annotation_audit_passed",
        **actual,
        "seed": seed,
        "date_proxy_identity": "canonical grouping proxy; not patient_id",
        "prediction_unit": "15-second recording with raw intervals",
        "recording_states": {
            "observed": len(records) - empty,
            "empty": empty,
        },
        "interval_semantics": (
            "raw rows are positive intervals only; explicit negative intervals=0"
        ),
        "gap_semantics": "not_annotated; never raw negative or shared normal",
        "explicit_negative_intervals": 0,
        "assignment_digest_serialization": ASSIGNMENT_DIGEST_SERIALIZATION,
        "ordered_record_sha256": _ordered_record_sha256(records),
        "ordered_record_digest_serialization": (
            ORDERED_RECORD_DIGEST_SERIALIZATION
        ),
        "accepted_label_tree_sha256_reference": (
            ACCEPTED_LABEL_TREE_SHA256_REFERENCE
        ),
        "own_label_tree_sha256": own_label_digest,
        "own_label_tree_digest_serialization": (
            OWN_LABEL_TREE_DIGEST_SERIALIZATION
        ),
        "label_tree_identity_status": (
            "reference_not_reproduced_serialization_unknown"
        ),
        "own_digest_matches_accepted_reference": (
            own_label_digest == ACCEPTED_LABEL_TREE_SHA256_REFERENCE
        ),
        "independent_verifier_status": "HOLD",
    }
    return records, receipt


def audit_wave_headers(
    root: Path, records: Sequence[HFSampleRecord]
) -> dict[str, object]:
    """Read every WAV header without decoding all audio samples."""

    formats: Counter[tuple[int, int, int, int, str]] = Counter()
    for record in records:
        path = root / record.wav_relative_path
        with wave.open(str(path), "rb") as handle:
            values = (
                handle.getframerate(),
                handle.getnchannels(),
                handle.getnframes(),
                handle.getsampwidth(),
                handle.getcomptype(),
            )
        formats[values] += 1
    expected = (4_000, 1, 60_000, 2, "NONE")
    if set(formats) != {expected}:
        raise RuntimeError(f"HF WAV header contract failed: {dict(formats)}")
    return {
        "status": "hf_full_header_audit_passed",
        "headers_read": len(records),
        "sample_rate": 4_000,
        "channels": 1,
        "frames": 60_000,
        "sample_width_bytes": 2,
        "compression": "NONE",
        "duration_s": 15.0,
    }


def load_hf_waveform(root: Path, record: HFSampleRecord) -> tuple[
    WaveformSample, dict[str, object]
]:
    """Load one 4 kHz mono recording and explicitly resample to 16 kHz."""

    try:
        import torchaudio
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "torchaudio is required only for HF waveform loading; "
            "manifest import/audit remains dependency-lazy"
        ) from error
    path = root / record.wav_relative_path
    waveform, sample_rate = torchaudio.load(path)
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
    if sample_rate != 4_000 or waveform.shape != (1, 60_000):
        raise RuntimeError(
            f"HF source waveform must be mono [1,60000] at 4 kHz, got "
            f"{tuple(waveform.shape)} at {sample_rate}"
        )
    resampled = torchaudio.functional.resample(
        waveform,
        orig_freq=4_000,
        new_freq=16_000,
        lowpass_filter_width=6,
        rolloff=0.99,
        resampling_method="sinc_interp_hann",
        beta=None,
    ).squeeze(0)
    if resampled.dtype != torch.float32 or resampled.shape != (240_000,):
        raise RuntimeError(
            f"HF resampled waveform must be float32 [240000], got "
            f"{resampled.dtype} {tuple(resampled.shape)}"
        )
    sample = WaveformSample(
        waveform=resampled,
        sample_id=record.sample_id,
        dataset_id="HF",
        prediction_unit="recording_15s_with_intervals",
        source_start_s=0.0,
        source_end_s=15.0,
        sample_rate=16_000,
        lineage={
            "source_split": record.source_split,
            "partition": record.partition,
            "date_proxy": record.date_proxy,
            "group_id": record.group_id,
            "patient_id": "not_provided",
            "wav_relative_path": record.wav_relative_path,
            "label_relative_path": record.label_relative_path,
            "recording_state": record.recording_state.value,
            "resampler": "torchaudio.functional.resample_4000_to_16000",
        },
    )
    return sample, {
        "status": "hf_waveform_resample_smoke_passed",
        "sample_id": record.sample_id,
        "source_shape": [1, 60_000],
        "source_sample_rate": 4_000,
        "output_shape": [240_000],
        "output_sample_rate": 16_000,
        "source_time_s": [0.0, 15.0],
        "resampler_config": dict(RESAMPLER_CONFIG),
        "repeat_pad": False,
        "truncate": False,
    }


def records_to_p6_supervision(
    records: Sequence[HFSampleRecord],
    time_map: torch.Tensor,
    token_mask: torch.Tensor,
) -> HFTemporalSupervision:
    """Fixed P6 adapter: paper-native OVR with token-center alignment."""

    if len(records) != token_mask.shape[0]:
        raise ValueError("record and token batch size mismatch")
    supervision = raw_intervals_to_token_supervision(
        time_map,
        token_mask,
        [record.raw_intervals for record in records],
        [record.recording_state for record in records],
        policy=HFTargetPolicy.PAPER_NATIVE_RASTERIZED_OVR,
        alignment=TokenAlignmentPolicy.TOKEN_CENTER_IN_INTERVAL,
    )
    if (
        supervision.receipt["negative_semantics"]
        != "source_task_constructed_not_raw_normal"
        or supervision.receipt["shared_label_eligible"] is not False
    ):
        raise RuntimeError("P6 HF target claim boundary failed")
    return supervision
