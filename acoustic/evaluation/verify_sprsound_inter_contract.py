"""Read-only verification of the exact SPRSound B0-compatible target contract."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from acoustic.evaluation.sprsound_inter import (
    EXPECTED_EVENTS,
    EXPECTED_ID_SHA256,
    EXPECTED_RAW_COUNTS,
    RAW_TO_FOUR,
    build_label_free_inter_manifest,
    id_sha256,
    load_scoring_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_label_free_inter_manifest(args.dataset_root)
    ids = [str(row["event_id"]) for row in manifest]
    scoring = load_scoring_labels(manifest)
    mapped = Counter(row["narrow_four_label"] for row in scoring.values())
    if len(ids) != EXPECTED_EVENTS or id_sha256(ids) != EXPECTED_ID_SHA256:
        raise RuntimeError("event contract mismatch")
    if dict(sorted(Counter(row["raw_event_label"] for row in scoring.values()).items())) != EXPECTED_RAW_COUNTS:
        raise RuntimeError("raw-label contract mismatch")
    if set(RAW_TO_FOUR.values()) != {"normal", "crackle", "wheeze", "both"}:
        raise RuntimeError("mapping contract mismatch")
    print(
        "sprsound_inter_contract_ok "
        f"events={len(ids)} id_sha256={id_sha256(ids)} mapped={dict(sorted(mapped.items()))}"
    )


if __name__ == "__main__":
    main()
