"""Run the frozen-representation attribution experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from baseline.four_dataset_frozen_encoder.data import build_samples, sample_to_row
from baseline.four_dataset_frozen_encoder.encoder import (
    extract_embeddings,
    load_cache,
    save_cache,
    sha256_file,
)
from baseline.four_dataset_frozen_encoder.run import aggregate
from baseline.four_dataset_frozen_encoder.train import (
    BATCH_SIZE,
    EPOCHS,
    SharedNativeModel,
    TASK_SPECS,
    train_joint,
)
from baseline.four_dataset_representation_attribution.encoder import (
    REPRESENTATIONS,
    build_representation_encoder,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "four_dataset_representation_attribution"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
CONDITION = "d2_shared_adapter_dataset_balanced"
R0 = "r0_pafa_icbhi_task_encoder"
DATASETS = ("icbhi", "sprsound", "hf_lung", "kauh")
ALL_REPRESENTATIONS = (R0, *REPRESENTATIONS)
EXPECTED_ROWS = 25_084
EXPECTED_R0_SHA256 = (
    "f40ae7fe581457bc86d76b93b1ee811e7ea01bc5e098a6daa73db451f96d1b31"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ordered_id_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def validate_roots(result_root: Path, cache_root: Path) -> tuple[Path, Path]:
    result = result_root.resolve()
    cache = cache_root.resolve()
    if result.name != EXPERIMENT_ID or result.parent.name != "result":
        raise ValueError(f"result root must be result/{EXPERIMENT_ID}")
    if cache.name != EXPERIMENT_ID or cache.parent.name != ".cache":
        raise ValueError(f"cache root must be .cache/{EXPERIMENT_ID}")
    return result, cache


def configure_runtime(cache_root: Path, threads: int) -> None:
    for variable, relative in (
        ("NUMBA_CACHE_DIR", "runtime/numba"),
        ("MPLCONFIGDIR", "runtime/matplotlib"),
        ("XDG_CACHE_HOME", "runtime/xdg"),
    ):
        path = cache_root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    torch.set_num_threads(threads)


def data_audit(dataset_root: Path, result_root: Path) -> tuple[list, dict[str, object]]:
    folds: dict[str, object] = {}
    canonical = None
    for fold in range(5):
        samples, receipt = build_samples(dataset_root, fold)
        folds[str(fold)] = receipt
        if canonical is None:
            canonical = samples
            write_gzip_csv(
                result_root / "samples_fold_0.csv.gz",
                [sample_to_row(sample) for sample in samples],
            )
        elif [sample.sample_id for sample in samples] != [
            sample.sample_id for sample in canonical
        ]:
            raise RuntimeError("KAUH fold changed canonical embedding order")
    if canonical is None or len(canonical) != EXPECTED_ROWS:
        raise RuntimeError("canonical four-dataset row count failed")
    receipt = {
        "status": "representation_attribution_data_audit_passed",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "rows": len(canonical),
        "dataset_rows": dict(
            sorted(Counter(sample.dataset for sample in canonical).items())
        ),
        "ordered_id_sha256": ordered_id_sha256(
            [sample.sample_id for sample in canonical]
        ),
        "folds": folds,
    }
    write_json(result_root / "data_receipt.json", receipt)
    return canonical, receipt


def _r0_cache(
    path: Path, samples: list
) -> tuple[np.ndarray, dict[str, object]]:
    if sha256_file(path) != EXPECTED_R0_SHA256:
        raise RuntimeError("R0 PAFA cache SHA256 mismatch")
    embeddings, source_receipt = load_cache(path, samples)
    return embeddings, {
        "representation": R0,
        "cache_path": str(path.resolve()),
        "cache_sha256": EXPECTED_R0_SHA256,
        "shape": list(embeddings.shape),
        "ordered_id_sha256": ordered_id_sha256(
            [sample.sample_id for sample in samples]
        ),
        "read_only_reuse": True,
        "encoder_frozen": True,
        "source_task_states_present_during_extraction": True,
        "source_heads_used_for_downstream_prediction": False,
        "selection_caveat": "ICBHI official-test-selected",
        "source_receipt": source_receipt,
    }


def _selected_smoke_samples(samples: list) -> list:
    selected_ids: set[str] = set()
    for task, spec in TASK_SPECS.items():
        for partition in ("subtrain", "validation", "test"):
            candidates = [
                sample
                for sample in samples
                if sample.partition == partition
                and sample.dataset == spec["dataset"]
                and (
                    task in sample.targets
                    or (
                        partition == "test"
                        and sample.dataset == "sprsound"
                        and not sample.targets
                    )
                )
            ]
            selected_ids.update(sample.sample_id for sample in candidates[:8])
    return [sample for sample in samples if sample.sample_id in selected_ids]


def _subset_embeddings(
    samples: list, selected: list, embeddings: np.ndarray
) -> np.ndarray:
    by_id = {
        sample.sample_id: value for sample, value in zip(samples, embeddings)
    }
    return np.stack([by_id[sample.sample_id] for sample in selected])


def smoke(
    samples: list,
    source_repo: Path,
    backbone: Path,
    r0_cache_path: Path,
    dataset_root: Path,
    result_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    selected = _selected_smoke_samples(samples)
    r0_values, r0_receipt = _r0_cache(r0_cache_path, samples)
    representations: dict[str, object] = {}
    for representation in ALL_REPRESENTATIONS:
        if representation == R0:
            values = _subset_embeddings(samples, selected, r0_values)
            identity = r0_receipt
            extraction = {"embedding_shape": list(values.shape), "reuse": True}
        else:
            encoder, identity = build_representation_encoder(
                representation, source_repo, backbone, device
            )
            values, extraction = extract_embeddings(
                selected, source_repo, encoder, device, batch_size
            )
            del encoder
        output_dir = result_root / "smoke" / representation / CONDITION
        payload = train_joint(
            CONDITION,
            selected,
            values,
            output_dir,
            device,
            dataset_root,
        )
        checkpoint = torch.load(output_dir / "best.pth", map_location="cpu")
        finite = (
            np.isfinite(values).all()
            and all(np.isfinite(float(row["loss"])) for row in payload["history"])
            and all(
                torch.isfinite(value).all()
                for value in checkpoint["model"].values()
                if torch.is_tensor(value)
            )
        )
        if not finite:
            raise RuntimeError(f"non-finite smoke state: {representation}")
        representations[representation] = {
            "encoder": identity,
            "extraction": extraction,
            "finite_loss_gradient_checkpoint": True,
            "prediction_rows": payload["prediction_rows"],
            "spr_label_free_then_terminal_join": payload["terminal_label_join"][
                "spr_test_labels_loaded_after_label_free_write"
            ],
        }
    receipt = {
        "status": "representation_attribution_smoke_passed",
        "samples": len(selected),
        "dataset_rows": dict(
            sorted(Counter(sample.dataset for sample in selected).items())
        ),
        "tasks": list(TASK_SPECS),
        "condition": CONDITION,
        "representations": representations,
    }
    write_json(result_root / "smoke_receipt.json", receipt)
    return receipt


def _training_profile(samples: list, device: torch.device) -> dict[str, object]:
    model = SharedNativeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    values = torch.randn(BATCH_SIZE, 768, device=device)
    target = torch.randint(0, 4, (BATCH_SIZE,), device=device)
    started = time.perf_counter()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(values, "icbhi_flat4"), target
        )
        loss.backward()
        optimizer.step()
    seconds_per_step = (time.perf_counter() - started) / 100
    source_rows = {
        dataset: sum(
            sample.dataset == dataset
            and sample.partition == "subtrain"
            and bool(sample.targets)
            for sample in samples
        )
        for dataset in DATASETS
    }
    batches = {
        dataset: int(np.ceil(rows / BATCH_SIZE))
        for dataset, rows in source_rows.items()
    }
    updates = max(batches.values()) * len(DATASETS) * EPOCHS * 5
    return {
        "profile_steps": 100,
        "seconds_per_step": seconds_per_step,
        "projected_updates": updates,
        "projected_seconds": seconds_per_step * updates,
    }


def profile(
    samples: list,
    source_repo: Path,
    backbone: Path,
    result_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    training = _training_profile(samples, device)
    representation_receipts: dict[str, object] = {}
    for representation in REPRESENTATIONS:
        selected: list = []
        counts = Counter()
        for sample in samples:
            if counts[sample.dataset] < 50:
                selected.append(sample)
                counts[sample.dataset] += 1
        encoder, identity = build_representation_encoder(
            representation, source_repo, backbone, device
        )
        by_dataset: dict[str, object] = {}
        projected_seconds = 0.0
        for dataset in DATASETS:
            subset = [sample for sample in selected if sample.dataset == dataset]
            _, extraction = extract_embeddings(
                subset,
                source_repo,
                encoder,
                device,
                min(batch_size, len(subset)),
            )
            full_rows = sum(sample.dataset == dataset for sample in samples)
            estimate = extraction["runtime_seconds"] / len(subset) * full_rows
            projected_seconds += estimate
            by_dataset[dataset] = {
                "profile_rows": len(subset),
                "profile_seconds": extraction["runtime_seconds"],
                "full_rows": full_rows,
                "projected_seconds": estimate,
                "window_count_min": extraction["window_count_min"],
                "window_count_max": extraction["window_count_max"],
            }
        del encoder
        peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
        projected_total = projected_seconds + float(training["projected_seconds"])
        free_gib = shutil.disk_usage(result_root).free / 1024**3
        allowed = (
            projected_total / 60 <= 150
            and peak_rss_gib < 24
            and free_gib >= 2
        )
        representation_receipts[representation] = {
            "encoder": identity,
            "datasets": by_dataset,
            "training": training,
            "projected_total_minutes": projected_total / 60,
            "peak_rss_gib": peak_rss_gib,
            "free_disk_gib": free_gib,
            "decision": "local_full_allowed" if allowed else "hold",
        }
    receipt = {
        "status": "representation_attribution_profile_passed",
        "gate": {
            "per_representation_minutes_max": 150,
            "peak_rss_gib_max": 24,
            "free_disk_gib_min": 2,
        },
        "representations": representation_receipts,
        "all_new_representations_allowed": all(
            row["decision"] == "local_full_allowed"
            for row in representation_receipts.values()
        ),
    }
    write_json(result_root / "profile_receipt.json", receipt)
    return receipt


def _wall_guard(started: float, representation: str) -> None:
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    if elapsed > 150 * 60:
        raise RuntimeError(f"{representation} exceeded 150-minute extraction gate")
    if peak >= 24:
        raise RuntimeError(f"{representation} exceeded 24-GiB RSS gate")


def extract(
    samples: list,
    source_repo: Path,
    backbone: Path,
    cache_root: Path,
    result_root: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    profile_receipt = json.loads((result_root / "profile_receipt.json").read_text())
    if not profile_receipt["all_new_representations_allowed"]:
        raise RuntimeError("profile did not authorize local extraction")
    all_receipts: dict[str, object] = {}
    for representation in REPRESENTATIONS:
        started = time.perf_counter()
        encoder, identity = build_representation_encoder(
            representation, source_repo, backbone, device
        )
        shard_values: dict[str, np.ndarray] = {}
        shard_receipts: dict[str, object] = {}
        for dataset in DATASETS:
            subset = [sample for sample in samples if sample.dataset == dataset]
            shard_path = cache_root / representation / "embedding_shards" / f"{dataset}.npz"
            receipt_path = result_root / representation / "embedding_shards" / f"{dataset}.json"
            if shard_path.is_file() != receipt_path.is_file():
                raise RuntimeError(f"partial shard pair: {representation}/{dataset}")
            if shard_path.is_file():
                values, cached = load_cache(shard_path, subset)
                recorded = json.loads(receipt_path.read_text())
                if (
                    cached["cache_sha256"] != recorded["cache_sha256"]
                    or cached["encoder"]["final_state_digest"]
                    != identity["final_state_digest"]
                ):
                    raise RuntimeError("resumed shard identity mismatch")
                shard_values[dataset] = values
                shard_receipts[dataset] = {**recorded, "resumed": True}
                continue
            values, extraction = extract_embeddings(
                subset,
                source_repo,
                encoder,
                device,
                batch_size,
                guard=lambda: _wall_guard(started, representation),
            )
            cached = save_cache(
                shard_path,
                subset,
                values,
                {"encoder": identity, "extraction": extraction},
            )
            receipt = {
                **cached,
                "dataset": dataset,
                "shape": list(values.shape),
                "ordered_id_sha256": ordered_id_sha256(
                    [sample.sample_id for sample in subset]
                ),
                "representation_elapsed_seconds": time.perf_counter() - started,
                "resumed": False,
            }
            write_json(receipt_path, receipt)
            shard_values[dataset] = values
            shard_receipts[dataset] = receipt
        del encoder
        by_id = {
            sample.sample_id: value
            for dataset in DATASETS
            for sample, value in zip(
                [row for row in samples if row.dataset == dataset],
                shard_values[dataset],
            )
        }
        combined = np.stack([by_id[sample.sample_id] for sample in samples])
        if combined.shape != (EXPECTED_ROWS, 768) or not np.isfinite(combined).all():
            raise RuntimeError("combined representation cache failed")
        receipt = save_cache(
            cache_root / representation / "embeddings.npz",
            samples,
            combined,
            {
                "encoder": identity,
                "shards": shard_receipts,
                "combination_policy": "canonical sample order; no numeric transform",
                "actual_representation_wall_seconds": time.perf_counter() - started,
            },
        )
        write_json(result_root / representation / "embedding_receipt.json", receipt)
        all_receipts[representation] = receipt
    write_json(result_root / "embedding_receipts.json", all_receipts)
    return all_receipts


def _artifact_set(directory: Path) -> list[Path]:
    return [
        directory / "best.pth",
        directory / "metrics.json",
        directory / "predictions.csv.gz",
        directory / "predictions_label_free.csv.gz",
    ]


def train(
    samples: list,
    dataset_root: Path,
    r0_cache_path: Path,
    cache_root: Path,
    result_root: Path,
    device: torch.device,
) -> None:
    embeddings_by_representation = {
        R0: _r0_cache(r0_cache_path, samples)[0],
    }
    for representation in REPRESENTATIONS:
        embeddings_by_representation[representation] = load_cache(
            cache_root / representation / "embeddings.npz", samples
        )[0]
    for representation in ALL_REPRESENTATIONS:
        representation_root = result_root / representation
        for fold in range(5):
            fold_samples, fold_receipt = build_samples(dataset_root, fold)
            if [sample.sample_id for sample in fold_samples] != [
                sample.sample_id for sample in samples
            ]:
                raise RuntimeError("fold changed embedding identity")
            write_json(
                representation_root / f"fold_{fold}" / "data_receipt.json",
                fold_receipt,
            )
            directory = representation_root / f"fold_{fold}" / CONDITION
            present = [path.is_file() for path in _artifact_set(directory)]
            if any(present) and not all(present):
                raise RuntimeError(f"partial training artifacts: {directory}")
            if all(present):
                print(f"TRAIN_RESUMED representation={representation} fold={fold}", flush=True)
                continue
            train_joint(
                CONDITION,
                fold_samples,
                embeddings_by_representation[representation],
                directory,
                device,
                dataset_root,
            )
            print(f"TRAIN_COMPLETE representation={representation} fold={fold}", flush=True)
        aggregate(representation_root, [CONDITION])


def analyze(result_root: Path, cache_root: Path, r0_cache_path: Path) -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    rows: list[dict[str, object]] = []
    for representation in ALL_REPRESENTATIONS:
        with (result_root / representation / "summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                rows.append({"representation": representation, **row})
    write_csv(result_root / "comparison" / "representation_results.csv", rows)
    per_class: list[dict[str, object]] = []
    for representation in ALL_REPRESENTATIONS:
        with (result_root / representation / "per_class_summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                per_class.append({"representation": representation, **row})
    write_csv(result_root / "comparison" / "per_class_results.csv", per_class)
    by_key = {
        (row["representation"], row["task"]): row for row in rows
    }
    gaps = []
    task_assessments: dict[str, object] = {}
    band = float(
        protocol["preregistered_attribution_rule"]["practical_band_absolute"]
    )
    pafa_wins = 0
    pafa_losses = 0
    pafa_material_tasks = []
    pafa_less_stable_non_kauh = 0
    for task, spec in TASK_SPECS.items():
        r0 = by_key[(R0, task)]
        r1 = by_key[(REPRESENTATIONS[0], task)]
        r2 = by_key[(REPRESENTATIONS[1], task)]
        secondary_column = (
            "weighted_or_micro_f1_mean"
            if spec["kind"] == "multilabel"
            else "uar_mean"
        )
        secondary_std = (
            "weighted_or_micro_f1_sample_std"
            if spec["kind"] == "multilabel"
            else "uar_sample_std"
        )
        r1_r0_primary = float(r1["macro_f1_mean"]) - float(r0["macro_f1_mean"])
        r1_r0_secondary = float(r1[secondary_column]) - float(r0[secondary_column])
        r2_r1_primary = float(r2["macro_f1_mean"]) - float(r1["macro_f1_mean"])
        r2_r1_secondary = float(r2[secondary_column]) - float(r1[secondary_column])
        pafa_material = -r1_r0_primary >= band and -r1_r0_secondary >= -band
        audio_material = r1_r0_primary >= band and r1_r0_secondary >= -band
        if pafa_material:
            pafa_wins += 1
            pafa_material_tasks.append(task)
        if audio_material:
            pafa_losses += 1
        if task != "kauh_raw9" and r0["macro_f1_sample_std"] not in {"", None}:
            if (
                float(r0["macro_f1_sample_std"])
                - float(r1["macro_f1_sample_std"])
                > 0.01
            ):
                pafa_less_stable_non_kauh += 1
        close = (
            abs(r1_r0_primary) <= band and abs(r1_r0_secondary) <= band
        )
        task_assessments[task] = {
            "close": close,
            "pafa_material_win": pafa_material,
            "audioset_material_win": audio_material,
        }
        gaps.append(
            {
                "task": task,
                "primary_metric": "macro_f1",
                "secondary_metric": (
                    "micro_f1" if spec["kind"] == "multilabel" else "uar"
                ),
                "r1_minus_r0_primary": r1_r0_primary,
                "r1_minus_r0_secondary": r1_r0_secondary,
                "r2_minus_r1_primary": r2_r1_primary,
                "r2_minus_r1_secondary": r2_r1_secondary,
                "r0_primary_std": r0["macro_f1_sample_std"],
                "r1_primary_std": r1["macro_f1_sample_std"],
                "assessment": (
                    "close"
                    if close
                    else "pafa_material_win"
                    if pafa_material
                    else "audioset_material_win"
                    if audio_material
                    else "mixed_tradeoff"
                ),
            }
        )
    write_csv(result_root / "comparison" / "task_gaps.csv", gaps)
    select_pafa = (
        "icbhi_flat4" in pafa_material_tasks
        and len(pafa_material_tasks) >= 2
        and pafa_losses <= pafa_wins
        and pafa_less_stable_non_kauh <= 1
    )
    selected = R0 if select_pafa else REPRESENTATIONS[0]
    selected_cache = (
        r0_cache_path.resolve()
        if selected == R0
        else (cache_root / selected / "embeddings.npz").resolve()
    )
    receipt = {
        "status": "encoder_selection_complete",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "rule_applied_without_pooled_score": True,
        "task_assessments": task_assessments,
        "pafa_material_wins": pafa_wins,
        "pafa_material_losses": pafa_losses,
        "pafa_material_tasks": pafa_material_tasks,
        "pafa_less_stable_non_kauh_tasks": pafa_less_stable_non_kauh,
        "selected_representation": selected,
        "selected_cache_path": str(selected_cache),
        "selected_cache_sha256": sha256_file(selected_cache),
        "selection_reason": (
            "PAFA met every preregistered material-advantage gate"
            if select_pafa
            else "neutral AudioSet-only default because PAFA did not meet every preregistered material-advantage gate"
        ),
        "random_init_eligible": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_root / "comparison" / "encoder_selection.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["audit", "smoke", "profile", "extract", "train", "analyze", "all"],
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("result/pafa_sprsound_transfer_20260722_235659/source/repo"),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=Path(".cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt"),
    )
    parser.add_argument(
        "--r0-cache",
        type=Path,
        default=Path(".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"),
    )
    parser.add_argument(
        "--result-root", type=Path, default=Path(f"result/{EXPERIMENT_ID}")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(f".cache/{EXPERIMENT_ID}")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result_root, cache_root = validate_roots(args.result_root, args.cache_root)
    result_root.mkdir(parents=True, exist_ok=True)
    configure_runtime(cache_root, args.threads)
    device = torch.device(args.device)
    samples, _ = data_audit(args.dataset_root, result_root)
    if args.phase == "audit":
        return
    if args.phase in {"smoke", "all"}:
        smoke(
            samples,
            args.source_repo,
            args.backbone_checkpoint,
            args.r0_cache,
            args.dataset_root,
            result_root,
            device,
            args.batch_size,
        )
        if args.phase == "smoke":
            return
    if args.phase in {"profile", "all"}:
        profile(
            samples,
            args.source_repo,
            args.backbone_checkpoint,
            result_root,
            device,
            args.batch_size,
        )
        if args.phase == "profile":
            return
    if args.phase in {"extract", "all"}:
        extract(
            samples,
            args.source_repo,
            args.backbone_checkpoint,
            cache_root,
            result_root,
            device,
            args.batch_size,
        )
        if args.phase == "extract":
            return
    if args.phase in {"train", "all"}:
        train(
            samples,
            args.dataset_root,
            args.r0_cache,
            cache_root,
            result_root,
            device,
        )
        if args.phase == "train":
            return
    analyze(result_root, cache_root, args.r0_cache)


if __name__ == "__main__":
    main()
