"""Materialize source-contract fbanks outside the tracked tree."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import preprocess_event_rows, read_csv, write_csv, write_json


def stratified_smoke_rows(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('inner_split', 'none')}::{row.get('raw_label', 'unlabelled')}"
        grouped[key].append(row)
    selected = []
    while len(selected) < limit and any(grouped.values()):
        for label in sorted(grouped):
            if grouped[label] and len(selected) < limit:
                selected.append(grouped[label].pop(0))
    return sorted(selected, key=lambda row: row["event_id"])


def materialize(rows: list[dict[str, str]], author_repo: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    features = preprocess_event_rows(rows, author_repo)
    array = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32, shape=(len(rows), 1, 798, 128)
    )
    index_rows = []
    for index, row in enumerate(rows):
        array[index] = features[row["event_id"]]
        index_rows.append(
            {
                "event_id": row["event_id"],
                "feature_index": index,
                "partition": row["partition"],
            }
        )
    array.flush()
    write_csv(output / "index.csv", index_rows)
    return {
        "rows": len(rows),
        "shape": [len(rows), 1, 798, 128],
        "dtype": "float32",
        "finite": bool(np.isfinite(np.asarray(array)).all()),
        "cache_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-inter-events", type=int, default=0)
    parser.add_argument("--cache-name", default="full")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    cache_root = args.cache_root.resolve()
    author_repo = (args.author_repo or result_root / "source" / "repo").resolve()
    train = read_csv(result_root / "data" / "train_events.csv")
    inter = read_csv(result_root / "data" / "inter_events_label_free.csv")
    train = stratified_smoke_rows(train, args.max_train_events)
    inter = sorted(inter, key=lambda row: row["event_id"])
    if args.max_inter_events > 0:
        inter = inter[: args.max_inter_events]
    base = cache_root / "c0_sprsound" / args.cache_name
    if base.exists():
        raise FileExistsError(f"cache is immutable; choose a new --cache-name: {base}")
    receipt = {
        "status": "fbank_cache_verified",
        "author_repo": str(author_repo),
        "preprocessing": "16 kHz; fade; 8 s repeat/truncate; 128 fbank; resize 798x128; no SpecAugment",
        "train": materialize(train, author_repo, base / "train"),
        "inter": materialize(inter, author_repo, base / "inter"),
        "train_selection": "all rows" if args.max_train_events <= 0 else "label-stratified deterministic smoke subset",
        "inter_selection": "all rows" if args.max_inter_events <= 0 else "first event IDs only; no target labels read",
    }
    write_json(base / "cache_receipt.json", receipt)
    write_json(result_root / "receipts" / f"cache_{args.cache_name}.json", receipt)
    print(f"c0_cache_ok name={args.cache_name} train={len(train)} inter={len(inter)}")


if __name__ == "__main__":
    main()
