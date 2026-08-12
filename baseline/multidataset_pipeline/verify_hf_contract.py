"""Read-only HF manifest/header/waveform engineering verifier; not independent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .beats_temporal import BEATsGeometry, build_time_map, exact_patch_masks
from .contracts import ObservationState, collate_waveforms
from .hf_data import (
    HF_ROOT_RELATIVE,
    audit_wave_headers,
    build_hf_manifest,
    load_hf_waveform,
    records_to_p6_supervision,
)


def smoke(root: Path, records) -> dict[str, object]:
    observed = next(
        record
        for record in records
        if record.recording_state is ObservationState.OBSERVED
    )
    empty = next(
        record
        for record in records
        if record.recording_state is ObservationState.EMPTY
    )
    samples_and_receipts = [
        load_hf_waveform(root, record) for record in (observed, empty)
    ]
    batch = collate_waveforms([item[0] for item in samples_and_receipts])
    geometry = BEATsGeometry()
    token_mask, _ = exact_patch_masks(
        batch.valid_samples, batch.waveform.shape[1], geometry
    )
    time_map = build_time_map(token_mask, batch.source_start_s, geometry)
    supervision = records_to_p6_supervision(
        (observed, empty), time_map, token_mask
    )
    return {
        "status": "hf_two_record_engineering_smoke_passed",
        "sample_ids": [observed.sample_id, empty.sample_id],
        "recording_states": [
            observed.recording_state.value,
            empty.recording_state.value,
        ],
        "waveform_shape": list(batch.waveform.shape),
        "valid_samples": batch.valid_samples.tolist(),
        "token_mask_shape": list(token_mask.shape),
        "valid_tokens": token_mask.sum(dim=1).tolist(),
        "targets_shape": list(supervision.targets.shape),
        "valid_supervision_values": int(supervision.valid_mask.sum()),
        "positive_values": int(supervision.targets[supervision.valid_mask].sum()),
        "target_receipt": dict(supervision.receipt),
        "waveform_receipts": [item[1] for item in samples_and_receipts],
        "model_or_performance_result": False,
        "independent_verifier_status": "HOLD",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(HF_ROOT_RELATIVE)
    )
    parser.add_argument(
        "--phase", choices=("audit", "headers", "smoke", "all"), default="audit"
    )
    args = parser.parse_args()
    records, manifest = build_hf_manifest(args.root)
    output: dict[str, object] = {"manifest": manifest}
    if args.phase in {"headers", "all"}:
        output["headers"] = audit_wave_headers(args.root, records)
    if args.phase in {"smoke", "all"}:
        output["smoke"] = smoke(args.root, records)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
