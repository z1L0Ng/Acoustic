"""PAFA-repository BEATs+CE reproduction with separated selection modes.

This module deliberately exposes two different evidence contracts:

* ``author_test_selected`` evaluates the official test partition every epoch,
  matching the author repository's checkpoint-selection behavior.
* ``clean_validation_only`` builds a patient-grouped validation fold only from
  the official training recordings and touches the official test partition once
  after the checkpoint has been selected.

The implementation imports the author BEATs model and reproduces the source
waveform preprocessing: mono 16 kHz, recording fade, native cycle crop, then
front truncation or repeat padding to five seconds with a final fade-out.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np


LABEL_ORDER = ("normal", "crackle", "wheeze", "both")
AUTHOR_TEST_SELECTED = "author_test_selected"
CLEAN_VALIDATION_ONLY = "clean_validation_only"
MODES = (AUTHOR_TEST_SELECTED, CLEAN_VALIDATION_ONLY)
FUTURE_AUTHOR_SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ReproductionConfig:
    mode: str
    seed: int = 42
    sample_rate: int = 16_000
    desired_seconds: float = 5.0
    padding: str = "repeat_or_front_truncate"
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "Adam"
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6
    cosine: bool = True
    cosine_eta_min_ratio: float = 1e-3
    ema_beta: float = 0.5
    specaugment: bool = False
    model: str = "BEATs_iter3_plus_AS2M_full_finetune"
    task: str = "ICBHI_cycle_flat4_direct_softmax_CE"
    clean_split: str = "official_train_patient_grouped_5fold_fold0"
    clean_split_seed: int = 42
    clean_validation_fold: int = 0

    @property
    def evidence_label(self) -> str:
        if self.mode == AUTHOR_TEST_SELECTED:
            return "test_selected_reproduction_receipt"
        return "clean_validation_only_main"

    @property
    def selection_partition(self) -> str:
        if self.mode == AUTHOR_TEST_SELECTED:
            return "official_test"
        return "validation_from_official_train"

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unsupported mode: {self.mode}")
        frozen = ReproductionConfig(mode=self.mode)
        for field in (
            "sample_rate", "desired_seconds", "padding", "batch_size", "epochs",
            "optimizer", "learning_rate", "weight_decay", "cosine",
            "cosine_eta_min_ratio", "ema_beta", "specaugment", "model", "task",
        ):
            if getattr(self, field) != getattr(frozen, field):
                raise ValueError(f"frozen PAFA BEATs+CE field changed: {field}")


@dataclass(frozen=True)
class CycleRecord:
    sample_id: str
    recording_id: str
    group_id: str
    official_split: str
    start_s: float
    end_s: float
    ground_truth: int
    partition: str = "unassigned"


def source_contract() -> dict[str, object]:
    """Return the source-audited contract without loading data or a model."""

    return {
        "author_source": "PAFA repository scripts/beats_ce.sh and main.py",
        "task": "ICBHI cycle flat4 direct softmax CE",
        "label_order": list(LABEL_ORDER),
        "waveform": {
            "sample_rate": 16_000,
            "seconds": 5.0,
            "short": "repeat then front truncate to 80000 samples",
            "long": "front truncate to 80000 samples",
            "recording_and_padding_fade": True,
        },
        "training": {
            "encoder": "BEATs iter3+ AS2M full fine-tuning",
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": 5e-5,
            "weight_decay": 1e-6,
            "epochs": 100,
            "cosine": True,
            "ema_beta": 0.5,
            "specaugment": False,
            "first_local_seed": 42,
            "future_author_seed_interface": list(FUTURE_AUTHOR_SEEDS),
        },
        "selection_modes": {
            AUTHOR_TEST_SELECTED: (
                "official test Score every epoch; evidence is test-selected reproduction"
            ),
            CLEAN_VALIDATION_ONLY: (
                "patient-grouped validation from official train; official test once after selection"
            ),
        },
        "checkpoint_storage": {
            "policy": "overwrite one best model+classifier checkpoint when selection Score improves",
            "estimated_best_checkpoint_gb": 0.36,
            "recommended_free_gb_per_mode": 2,
            "resume": "best checkpoint is not optimizer-resumable",
        },
    }


def read_official_cycles(
    audio_dir: Path,
    author_repo: Path,
    official_splits: Sequence[str] = ("train", "test"),
) -> list[CycleRecord]:
    split_path = author_repo / "data" / "official_split.txt"
    recording_split: dict[str, str] = {}
    for raw in split_path.read_text().splitlines():
        if raw.strip():
            recording_id, split = raw.split()
            recording_split[recording_id] = split

    records: list[CycleRecord] = []
    included = set(official_splits)
    for recording_id in sorted(recording_split):
        if recording_split[recording_id] not in included:
            continue
        annotation = audio_dir / f"{recording_id}.txt"
        for cycle_index, raw in enumerate(annotation.read_text().splitlines()):
            if not raw.strip():
                continue
            start, end, crackle, wheeze = raw.split()
            crackle_i = int(crackle)
            wheeze_i = int(wheeze)
            records.append(
                CycleRecord(
                    sample_id=f"{recording_id}__cycle_{cycle_index:03d}",
                    recording_id=recording_id,
                    group_id=recording_id.split("_", 1)[0],
                    official_split=recording_split[recording_id],
                    start_s=float(start),
                    end_s=float(end),
                    ground_truth=crackle_i + 2 * wheeze_i,
                )
            )
    return records


def assign_partitions(
    records: Sequence[CycleRecord], config: ReproductionConfig
) -> list[CycleRecord]:
    config.validate()
    if config.mode == AUTHOR_TEST_SELECTED:
        return [
            replace(
                row,
                partition="subtrain" if row.official_split == "train" else "selection_test",
            )
            for row in records
        ]

    from sklearn.model_selection import StratifiedGroupKFold

    official_train = [row for row in records if row.official_split == "train"]
    labels = np.asarray([row.ground_truth for row in official_train])
    groups = np.asarray([row.group_id for row in official_train])
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.clean_split_seed,
    )
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    subtrain_indices, validation_indices = folds[config.clean_validation_fold]
    subtrain_ids = {official_train[index].sample_id for index in subtrain_indices}
    validation_ids = {official_train[index].sample_id for index in validation_indices}
    output = []
    for row in records:
        if row.official_split == "test":
            partition = "terminal_test"
        elif row.sample_id in subtrain_ids:
            partition = "subtrain"
        elif row.sample_id in validation_ids:
            partition = "validation"
        else:
            raise ValueError(f"unassigned official-train cycle: {row.sample_id}")
        output.append(replace(row, partition=partition))
    return output


def partition_summary(records: Sequence[CycleRecord]) -> dict[str, object]:
    output: dict[str, object] = {}
    for partition in sorted({row.partition for row in records}):
        current = [row for row in records if row.partition == partition]
        support = [sum(row.ground_truth == label for row in current) for label in range(4)]
        output[partition] = {
            "cycles": len(current),
            "recordings": len({row.recording_id for row in current}),
            "groups": len({row.group_id for row in current}),
            "support": dict(zip(LABEL_ORDER, support)),
        }
    return output


def metric_summary(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import confusion_matrix, f1_score

    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    confusion = confusion_matrix(ground_truth, prediction, labels=np.arange(4))
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(4, dtype=np.float64),
        where=support > 0,
    )
    specificity = float(recall[0])
    sensitivity = float(np.mean(recall[1:]))
    score = (specificity + sensitivity) / 2.0
    return {
        "sp": 100.0 * specificity,
        "se": 100.0 * sensitivity,
        "score": 100.0 * score,
        "macro_f1": 100.0 * float(
            f1_score(ground_truth, prediction, labels=np.arange(4), average="macro")
        ),
        "uar": 100.0 * float(np.mean(recall)),
        "support": dict(zip(LABEL_ORDER, [int(value) for value in support])),
        "per_class_recall": dict(
            zip(LABEL_ORDER, [100.0 * float(value) for value in recall])
        ),
        "confusion": confusion.tolist(),
    }


def _load_waveforms(
    records: Sequence[CycleRecord], audio_dir: Path, sample_rate: int, desired_seconds: float
) -> dict[str, "torch.Tensor"]:
    import torch
    import torchaudio
    from torchaudio import transforms as T

    target_samples = int(sample_rate * desired_seconds)
    fade_samples = int(sample_rate / 16)
    recording_fade = T.Fade(
        fade_in_len=fade_samples,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )
    repeat_fade = T.Fade(
        fade_in_len=0,
        fade_out_len=fade_samples,
        fade_shape="linear",
    )
    by_recording: dict[str, list[CycleRecord]] = {}
    for row in records:
        by_recording.setdefault(row.recording_id, []).append(row)

    waveforms: dict[str, torch.Tensor] = {}
    for recording_id, recording_rows in by_recording.items():
        waveform, source_rate = torchaudio.load(audio_dir / f"{recording_id}.wav")
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != sample_rate:
            waveform = T.Resample(source_rate, sample_rate)(waveform)
        waveform = recording_fade(waveform)
        for row in recording_rows:
            start = min(int(row.start_s * sample_rate), waveform.shape[-1])
            end = min(int(row.end_s * sample_rate), waveform.shape[-1])
            cycle = waveform[:, start:end]
            if cycle.shape[-1] > target_samples:
                cycle = cycle[:, :target_samples]
            else:
                repeats = int(np.ceil(target_samples / cycle.shape[-1]))
                cycle = cycle.repeat(1, repeats)[:, :target_samples]
                cycle = repeat_fade(cycle)
            waveforms[row.sample_id] = cycle.squeeze(0).to(torch.float32)
    return waveforms


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _build_model(author_repo: Path, checkpoint: Path, device: "torch.device"):
    import torch
    from torch import nn

    sys.path.insert(0, str(author_repo))
    try:
        from models.beats import BEATsTransferLearningModel
    finally:
        sys.path.pop(0)
    model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    ).to(device)
    classifier = nn.Linear(model.final_feat_dim, 4).to(device)
    return model, classifier


def _apply_author_ema(module, before: Mapping[str, "torch.Tensor"], beta: float) -> None:
    import torch

    with torch.no_grad():
        for name, current in module.state_dict().items():
            if current.is_floating_point():
                current.copy_(before[name] * beta + current * (1.0 - beta))


def _evaluate(loader, model, classifier, device) -> dict[str, object]:
    import torch

    model.eval()
    classifier.eval()
    sample_ids: list[str] = []
    filenames: list[str] = []
    group_ids: list[str] = []
    ground_truth: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for waveform, target, ids, recording_ids, groups in loader:
            waveform = waveform.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                features = model(waveform, training=False)
                output = classifier(features).mean(dim=1)
            sample_ids.extend(ids)
            filenames.extend(recording_ids)
            group_ids.extend(groups)
            ground_truth.append(target.numpy())
            logits.append(output.float().cpu().numpy())
    targets = np.concatenate(ground_truth)
    logit_values = np.concatenate(logits)
    shifted = logit_values - logit_values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    return {
        "sample_id": np.asarray(sample_ids),
        "filename": np.asarray(filenames),
        "group_id": np.asarray(group_ids),
        "ground_truth": targets,
        "logits": logit_values,
        "probabilities": probabilities,
        "prediction": predictions,
        "metrics": metric_summary(targets, predictions),
    }


def _save_predictions(path: Path, predictions: Mapping[str, object]) -> None:
    np.savez_compressed(
        path,
        **{key: value for key, value in predictions.items() if key != "metrics"},
    )


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def run(
    config: ReproductionConfig,
    audio_dir: Path,
    author_repo: Path,
    checkpoint: Path,
    output_base: Path,
    device_name: str,
    num_workers: int,
) -> dict[str, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    config.validate()
    _seed_everything(config.seed)
    mode_root = output_base / config.mode / f"seed_{config.seed}"
    if mode_root.exists() and any(mode_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {mode_root}")
    for child in ("checkpoints", "selection_predictions", "terminal"):
        (mode_root / child).mkdir(parents=True, exist_ok=True)

    selection_splits = (
        ("train", "test") if config.mode == AUTHOR_TEST_SELECTED else ("train",)
    )
    records = assign_partitions(
        read_official_cycles(audio_dir, author_repo, selection_splits),
        config,
    )
    config_payload = {
        **asdict(config),
        "evidence_label": config.evidence_label,
        "selection_partition": config.selection_partition,
        "label_order": list(LABEL_ORDER),
        "future_seed_interface": list(FUTURE_AUTHOR_SEEDS),
        "outer_test_policy": (
            "official test every epoch by author protocol"
            if config.mode == AUTHOR_TEST_SELECTED
            else "official test once after validation selection"
        ),
        "selection_phase_partition_summary": partition_summary(records),
        "terminal_test_at_selection_start": (
            "included_by_author_protocol"
            if config.mode == AUTHOR_TEST_SELECTED
            else "not_read_or_decoded"
        ),
    }
    (mode_root / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    )
    (mode_root / "split.jsonl").write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in records)
    )

    waveforms = _load_waveforms(
        records, audio_dir, config.sample_rate, config.desired_seconds
    )

    class CycleDataset(Dataset):
        def __init__(
            self,
            rows: Sequence[CycleRecord],
            waveform_store: Mapping[str, "torch.Tensor"],
        ):
            self.rows = list(rows)
            self.waveform_store = waveform_store

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            return (
                self.waveform_store[row.sample_id],
                row.ground_truth,
                row.sample_id,
                f"{row.recording_id}.wav",
                row.group_id,
            )

    generator = torch.Generator().manual_seed(config.seed)
    loaders = {}
    for partition in sorted({row.partition for row in records}):
        rows = [row for row in records if row.partition == partition]
        loaders[partition] = DataLoader(
            CycleDataset(rows, waveforms),
            batch_size=config.batch_size,
            shuffle=partition == "subtrain",
            drop_last=partition == "subtrain",
            num_workers=num_workers,
            pin_memory=True,
            generator=generator if partition == "subtrain" else None,
        )

    device = torch.device(device_name)
    model, classifier = _build_model(author_repo, checkpoint, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    selection_partition = (
        "selection_test" if config.mode == AUTHOR_TEST_SELECTED else "validation"
    )
    best_score = -np.inf
    best_epoch = 0
    best_metrics: Mapping[str, object] | None = None
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        learning_rate = config.learning_rate * (
            config.cosine_eta_min_ratio
            + (1.0 - config.cosine_eta_min_ratio)
            * (1.0 + np.cos(np.pi * epoch / config.epochs))
            / 2.0
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        classifier.train()
        losses = []
        for waveform, target, *_ in loaders["subtrain"]:
            waveform = waveform.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
            classifier_before = {
                key: value.detach().clone() for key, value in classifier.state_dict().items()
            }
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                features = model(waveform, training=True)
                logits = classifier(features).mean(dim=1)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _apply_author_ema(model, model_before, config.ema_beta)
            _apply_author_ema(classifier, classifier_before, config.ema_beta)
            losses.append(float(loss.detach().cpu()))

        predictions = _evaluate(
            loaders[selection_partition], model, classifier, device
        )
        prediction_path = mode_root / "selection_predictions" / f"epoch_{epoch:03d}.npz"
        _save_predictions(prediction_path, predictions)
        score = float(predictions["metrics"]["score"])
        selection_record = {
            "epoch": epoch,
            "selection_partition": config.selection_partition,
            "selection_metric": "ICBHI_Score=(Sp+Se)/2",
            "selection_score": score,
            "train_ce": float(np.mean(losses)),
            "learning_rate": learning_rate,
            "metrics": predictions["metrics"],
        }
        _append_jsonl(mode_root / "selection_log.jsonl", selection_record)
        if score > best_score:
            checkpoint_path = mode_root / "checkpoints" / "best_checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "config": config_payload,
                    "selection_score": score,
                },
                checkpoint_path,
            )
            best_score = score
            best_epoch = epoch
            best_metrics = predictions["metrics"]

    selected_checkpoint = mode_root / "checkpoints" / "best_checkpoint.pt"
    selected_state = torch.load(selected_checkpoint, map_location=device)
    model.load_state_dict(selected_state["model"])
    classifier.load_state_dict(selected_state["classifier"])
    if config.mode == CLEAN_VALIDATION_ONLY:
        terminal_records = [
            replace(row, partition="terminal_test")
            for row in read_official_cycles(
                audio_dir,
                author_repo,
                ("test",),
            )
        ]
        terminal_waveforms = _load_waveforms(
            terminal_records,
            audio_dir,
            config.sample_rate,
            config.desired_seconds,
        )
        terminal_loader = DataLoader(
            CycleDataset(terminal_records, terminal_waveforms),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        (mode_root / "terminal" / "official_test_split.jsonl").write_text(
            "".join(
                json.dumps(asdict(row), sort_keys=True) + "\n"
                for row in terminal_records
            )
        )
        terminal_predictions = _evaluate(
            terminal_loader, model, classifier, device
        )
        _save_predictions(
            mode_root / "terminal" / "official_test_predictions.npz",
            terminal_predictions,
        )
        (mode_root / "terminal" / "metrics.json").write_text(
            json.dumps(terminal_predictions["metrics"], indent=2, sort_keys=True) + "\n"
        )
        terminal_partition_summary = partition_summary(terminal_records)
    else:
        selected_prediction = (
            mode_root / "selection_predictions" / f"epoch_{best_epoch:03d}.npz"
        )
        shutil.copyfile(
            selected_prediction,
            mode_root / "terminal" / "official_test_predictions.npz",
        )
        (mode_root / "terminal" / "metrics.json").write_text(
            json.dumps(best_metrics, indent=2, sort_keys=True) + "\n"
        )
        terminal_partition_summary = partition_summary(
            [row for row in records if row.partition == "selection_test"]
        )

    summary = {
        "status": "complete",
        "evidence_label": config.evidence_label,
        "selected_epoch": best_epoch,
        "selected_score": best_score,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_metrics": best_metrics,
        "selection_partition": config.selection_partition,
        "official_test_access": (
            "every_epoch" if config.mode == AUTHOR_TEST_SELECTED else "once_after_selection"
        ),
        "checkpoint_role": "best-only selection checkpoint and terminal evaluation",
        "optimizer_resume": "not available from the best-only checkpoint",
        "terminal_partition_summary": terminal_partition_summary,
        "elapsed_seconds": time.time() - started,
    }
    (mode_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("result/reproduce/pafa_beats_ce"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe:
        print(json.dumps(source_contract(), indent=2, sort_keys=True))
        return
    missing = [
        name
        for name in ("mode", "audio_dir", "author_repo", "checkpoint")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"missing run arguments: {missing}")
    summary = run(
        ReproductionConfig(mode=args.mode, seed=args.seed),
        args.audio_dir,
        args.author_repo,
        args.checkpoint,
        args.output_base,
        args.device,
        args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
