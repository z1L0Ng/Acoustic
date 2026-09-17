"""DCASE-inspired ICBHI+SPRSound native-class-union runner.

The runner is complete but inert without ``--run``.  Source selection uses the
accepted internal grouped validation partitions; official tests are terminal.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from .pcmcl_source_transfer import build_pcmcl_single_5s
from .source_transfer_common import (
    AudioUnit,
    append_jsonl,
    binary_auroc,
    binary_metrics,
    icbhi_metrics,
    load_hf_cas_targets,
    load_hf_test_units,
    load_icbhi_units,
    load_kauh_units,
    load_single_5s,
    load_spr_inter_units,
    load_spr_targets,
    load_spr_train_units,
    prepare_run_directory,
    spr_metrics,
    update_early_stopping,
    write_json,
)


LABELS = (
    "N_ICBHI",
    "N_SPR",
    "Crackle",
    "Wheeze",
    "Both",
    "FineCrackle",
    "CoarseCrackle",
    "Rhonchi",
    "Stridor",
)
ICBHI_NATIVE = (0, 2, 3, 4)
SPR_NATIVE = (1, 7, 3, 8, 6, 5, 4)
HF_CAS = (3, 4, 7, 8)
SPR_RAW = (
    "Normal",
    "Rhonchi",
    "Wheeze",
    "Stridor",
    "Coarse Crackle",
    "Fine Crackle",
    "Wheeze+Crackle",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def union_target(dataset: str, raw_label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the approved nine-output target and observation mask."""

    target = torch.zeros(len(LABELS), dtype=torch.float32)
    mask = torch.zeros(len(LABELS), dtype=torch.bool)
    if dataset == "icbhi":
        native = {"normal": 0, "crackle": 2, "wheeze": 3, "both": 4}
        mask[list(ICBHI_NATIVE)] = True
        target[native[raw_label]] = 1
        return target, mask
    if dataset != "sprsound":
        raise ValueError(dataset)
    native = {
        "Normal": 1,
        "Rhonchi": 7,
        "Wheeze": 3,
        "Stridor": 8,
        "Coarse Crackle": 6,
        "Fine Crackle": 5,
        "Wheeze+Crackle": 4,
    }
    mask[list(SPR_NATIVE)] = True
    target[native[raw_label]] = 1
    # One-way specific-to-broad alias. Rhonchi/Stridor do not make the
    # Crackle-only class observed negative because co-occurrence is unresolved.
    if raw_label in {"Fine Crackle", "Coarse Crackle"}:
        mask[2] = True
        target[2] = 1
    elif raw_label in {"Normal", "Wheeze", "Wheeze+Crackle"}:
        mask[2] = True
        target[2] = 0
    return target, mask


def masked_node_bce(
    probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    values = F.binary_cross_entropy(probabilities, target, reduction="none")
    return values[mask.bool()].mean()


def official_attention_pool(
    frame_logits: torch.Tensor,
    attention_logits: torch.Tensor,
    class_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official DCASE class-softmax attention over sigmoid frame outputs."""

    frame_probabilities = torch.sigmoid(frame_logits)
    valid = None
    if class_mask is not None:
        valid = class_mask.bool().unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~valid, -1e30)
    attention = torch.softmax(attention_logits, dim=-1)
    if valid is not None:
        attention = attention.masked_fill(~valid, 0.0)
        frame_probabilities = frame_probabilities.masked_fill(~valid, 0.0)
    clip_probabilities = (
        (frame_probabilities * attention).sum(dim=1)
        / attention.sum(dim=1).clamp_min(1e-7)
    )
    if class_mask is not None:
        clip_probabilities = clip_probabilities.masked_fill(~class_mask.bool(), 0.0)
    return frame_probabilities, attention, clip_probabilities


def macro_multilabel_f1(
    scores: np.ndarray,
    target: np.ndarray,
    indices: Sequence[int],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    per_class = {}
    f1_values = []
    for index in indices:
        truth = target[:, index].astype(bool)
        prediction = scores[:, index] >= threshold
        tp = int((truth & prediction).sum())
        fp = int((~truth & prediction).sum())
        fn = int((truth & ~prediction).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[LABELS[index]] = {
            "positive_support": int(truth.sum()),
            "negative_support": int((~truth).sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "zero_division": 0,
        }
        f1_values.append(f1)
    return {"macro_f1": float(np.mean(f1_values)), "per_class": per_class}


def amplitude_to_db(value: torch.Tensor) -> torch.Tensor:
    return (20 * torch.log10(value.clamp_min(1e-5))).clamp(-50, 80)


class GLUBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, pool: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels * 2, 3, padding=1)
        self.norm = nn.BatchNorm2d(output_channels * 2)
        self.pool = nn.AvgPool2d(tuple(pool))
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(F.glu(self.norm(self.conv(value)), dim=1)))


class DCASEJointUnionCRNN(nn.Module):
    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        import torchaudio

        mel = config["logmel"]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(config["sample_rate"]),
            n_fft=int(mel["n_fft"]),
            win_length=int(mel["win_length"]),
            hop_length=int(mel["hop_length"]),
            f_min=float(mel["f_min"]),
            f_max=float(mel["f_max"]),
            n_mels=int(mel["n_mels"]),
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        channels = [int(value) for value in config["model"]["cnn_channels"]]
        pools = config["model"]["cnn_pooling"]
        blocks = []
        input_channels = 1
        frequency = int(mel["n_mels"])
        for output_channels, pool in zip(channels, pools):
            blocks.append(GLUBlock(input_channels, output_channels, pool, float(config["model"]["dropout"])))
            input_channels = output_channels
            frequency //= int(pool[1])
        if frequency != 1:
            raise ValueError("CNN pooling must reduce the mel axis to one bin")
        self.cnn = nn.Sequential(*blocks)
        fusion_dim = int(config["model"]["fusion_dim"])
        hidden = int(config["model"]["bigru_hidden"])
        self.fusion = nn.Linear(channels[-1] + 768, fusion_dim)
        self.rnn = nn.GRU(fusion_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(float(config["model"]["dropout"]))
        self.classifier = nn.Linear(hidden * 2, len(LABELS))
        self.attention = nn.Linear(hidden * 2, len(LABELS))

    def _logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        values = amplitude_to_db(self.mel(waveform))
        minimum = values.amin(dim=(1, 2), keepdim=True)
        maximum = values.amax(dim=(1, 2), keepdim=True)
        return (values - minimum) / (maximum - minimum).clamp_min(1e-7)

    def forward(
        self,
        waveform: torch.Tensor,
        beats_frames: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cnn = self.cnn(self._logmel(waveform).transpose(1, 2).unsqueeze(1))
        batch, channels, frames, frequency = cnn.shape
        if frequency != 1:
            raise RuntimeError("CNN mel axis did not reduce to one bin")
        cnn_frames = cnn.permute(0, 2, 1, 3).reshape(batch, frames, channels)
        aligned = F.adaptive_avg_pool1d(
            beats_frames.transpose(1, 2), frames
        ).transpose(1, 2)
        fused = self.fusion(torch.cat((cnn_frames, aligned), dim=-1))
        recurrent, _ = self.rnn(self.dropout(fused))
        recurrent = self.dropout(recurrent)
        frame_logits = self.classifier(recurrent)
        frame_probabilities, attention, clip_probabilities = official_attention_pool(
            frame_logits, self.attention(recurrent), class_mask
        )
        return {
            "frame_logits": frame_logits,
            "frame_probabilities": frame_probabilities,
            "attention": attention,
            "clip_probabilities": clip_probabilities,
        }


class JointCachedDataset(Dataset):
    def __init__(self, units: Sequence[AudioUnit], cache_dir: Path) -> None:
        self.units = tuple(units)
        ids = np.load(cache_dir / "ids.npy").astype(str)
        self.frames = np.load(cache_dir / "frames.npy", mmap_mode="r")
        self.index = {value: index for index, value in enumerate(ids.tolist())}

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        unit = self.units[index]
        raw = (
            ("normal", "crackle", "wheeze", "both")[int(unit.target)]
            if unit.dataset == "icbhi"
            else str(unit.metadata["raw_label"])
        )
        target, mask = union_target(unit.dataset, raw)
        return (
            load_single_5s(unit),
            torch.from_numpy(np.array(self.frames[self.index[unit.sample_id]], copy=True)).float(),
            target,
            mask,
            unit.sample_id,
        )


def _load_groups(repo_root: Path, config: Mapping[str, object]) -> dict[str, object]:
    return json.loads((repo_root / str(config["source_groups"])).read_text())


def _source_units(
    repo_root: Path, config: Mapping[str, object], seed: int
) -> dict[str, dict[str, list[AudioUnit]]]:
    groups = _load_groups(repo_root, config)
    icbhi = load_icbhi_units(repo_root, "train")
    icbhi_groups = groups["icbhi_by_seed"][str(seed)]
    icbhi_partition = {
        patient: partition
        for partition in ("subtrain", "validation")
        for patient in icbhi_groups[partition]
    }
    for unit in icbhi:
        if unit.group_id not in icbhi_partition:
            raise RuntimeError("ICBHI group missing from accepted source split")
    spr_partition = {
        patient: partition
        for partition in ("subtrain", "validation")
        for patient in groups["sprsound"][partition]
    }
    spr = load_spr_train_units(repo_root, spr_partition)
    return {
        "subtrain": {
            "icbhi": [row for row in icbhi if icbhi_partition[row.group_id] == "subtrain"],
            "sprsound": [row for row in spr if row.metadata["internal_partition"] == "subtrain"],
        },
        "validation": {
            "icbhi": [row for row in icbhi if icbhi_partition[row.group_id] == "validation"],
            "sprsound": [row for row in spr if row.metadata["internal_partition"] == "validation"],
        },
    }


def metadata_summary(
    repo_root: Path, config: Mapping[str, object], seed: int
) -> dict[str, object]:
    values = _source_units(repo_root, config, seed)
    output = {}
    for partition, datasets in values.items():
        output[partition] = {}
        for dataset, units in datasets.items():
            raw = [
                ("normal", "crackle", "wheeze", "both")[int(unit.target)]
                if dataset == "icbhi"
                else str(unit.metadata["raw_label"])
                for unit in units
            ]
            output[partition][dataset] = {
                "units": len(units),
                "patients": len({unit.group_id for unit in units}),
                "support": dict(sorted(Counter(raw).items())),
                "official_train_only": True,
            }
    return output


def _load_frozen_beats(repo_root: Path, config: Mapping[str, object], device: torch.device) -> nn.Module:
    source = repo_root / str(config["source_repo"])
    checkpoint = repo_root / str(config["initial_checkpoint"])
    sys.path.insert(0, str(source.resolve()))
    from models.beats import BEATsTransferLearningModel

    model = BEATsTransferLearningModel(
        num_target_classes=len(LABELS),
        model_path=str(checkpoint),
        ft_entire_network=False,
        spec_transform=None,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def _extract_temporal_frames(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    beats = model.beats
    fbank = beats.preprocess(waveform)
    patches = beats.patch_embedding(fbank.unsqueeze(1))
    batch, _, time_patches, frequency_patches = patches.shape
    flattened = patches.flatten(2).transpose(1, 2).contiguous()
    flattened = beats.layer_norm(flattened)
    if beats.post_extract_proj is not None:
        flattened = beats.post_extract_proj(flattened)
    flattened = beats.dropout_input(flattened)
    encoded, _ = beats.encoder(flattened, padding_mask=None)
    return encoded.reshape(batch, time_patches, frequency_patches, encoded.shape[-1]).mean(dim=2)


def _all_cache_units(repo_root: Path, config: Mapping[str, object]) -> dict[str, list[AudioUnit]]:
    sources = _source_units(repo_root, config, 0)
    icbhi_train = sources["subtrain"]["icbhi"] + sources["validation"]["icbhi"]
    spr_train = sources["subtrain"]["sprsound"] + sources["validation"]["sprsound"]
    hf_units = load_hf_test_units(repo_root)
    return {
        "icbhi_train": sorted(icbhi_train, key=lambda row: row.sample_id),
        "spr_train": sorted(spr_train, key=lambda row: row.sample_id),
        "icbhi_test": load_icbhi_units(repo_root, "test"),
        "spr_inter": load_spr_inter_units(repo_root, include_targets=False),
        "hf_test_windows": [
            AudioUnit(
                sample_id=f"{unit.sample_id}::window_{window}",
                dataset="hf_lung",
                audio_path=unit.audio_path,
                group_id=unit.group_id,
                start_s=window * 5.0,
                end_s=(window + 1) * 5.0,
                metadata=unit.metadata,
            )
            for unit in hf_units
            for window in range(3)
        ],
        "kauh_all": load_kauh_units(repo_root),
    }


def extract_frame_cache(repo_root: Path, config: Mapping[str, object], device: torch.device) -> dict[str, object]:
    root = repo_root / str(config["frame_cache_root"])
    beats = _load_frozen_beats(repo_root, config, device)
    report = {}
    for role, units in _all_cache_units(repo_root, config).items():
        output = root / role
        output.mkdir(parents=True, exist_ok=True)
        frames_file = output / "frames.npy"
        target = None
        started = time.perf_counter()
        for start in range(0, len(units), int(config["frame_extract_batch_size"])):
            current = units[start : start + int(config["frame_extract_batch_size"])]
            waveform = torch.stack([load_single_5s(unit) for unit in current]).to(device)
            with torch.no_grad():
                frames = _extract_temporal_frames(beats, waveform).cpu().numpy()
            if target is None:
                target = np.lib.format.open_memmap(
                    frames_file,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(units), frames.shape[1], frames.shape[2]),
                )
            target[start : start + len(current)] = frames
        np.save(output / "ids.npy", np.asarray([unit.sample_id for unit in units]))
        if target is not None:
            target.flush()
        report[role] = {
            "rows": len(units),
            "frame_shape": list(target.shape[1:]) if target is not None else None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    write_json(root / "extraction_summary.json", report)
    return report


def _predict(model: DCASEJointUnionCRNN, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    ids, scores, targets, masks = [], [], [], []
    with torch.no_grad():
        for waveform, frames, target, mask, sample_ids in loader:
            probabilities = model(
                waveform.to(device), frames.to(device), None
            )["clip_probabilities"]
            scores.append(probabilities.cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
            ids.extend(sample_ids)
    return {
        "sample_ids": np.asarray(ids),
        "scores": np.concatenate(scores),
        "targets": np.concatenate(targets),
        "masks": np.concatenate(masks),
    }


def _loader(units: Sequence[AudioUnit], cache_root: Path, role: str, batch_size: int) -> DataLoader:
    return DataLoader(JointCachedDataset(units, cache_root / role), batch_size=batch_size, shuffle=False)


def _selection(predictions: Mapping[str, dict[str, np.ndarray]]) -> dict[str, object]:
    icbhi = macro_multilabel_f1(predictions["icbhi"]["scores"], predictions["icbhi"]["targets"], ICBHI_NATIVE)
    spr = macro_multilabel_f1(predictions["sprsound"]["scores"], predictions["sprsound"]["targets"], SPR_NATIVE)
    return {
        "selection_score": 0.5 * float(icbhi["macro_f1"]) + 0.5 * float(spr["macro_f1"]),
        "icbhi_native4": icbhi,
        "sprsound_native7": spr,
        "threshold": 0.5,
        "source_weights": {"icbhi": 0.5, "sprsound": 0.5},
        "alias_outputs_counted_in_spr_native7": False,
    }


def _checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    best_score: float,
    best_epoch: int,
    no_improvement_epochs: int,
    stopped_early: bool,
    stop_reason: str | None,
    config: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "completed_epochs": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "selection_score": best_score,
            "best_epoch": best_epoch,
            "no_improvement_epochs": no_improvement_epochs,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "config": dict(config),
        },
        path,
    )


def _source_loader(dataset: JointCachedDataset, config: Mapping[str, object], seed: int) -> DataLoader:
    count = int(config["batches_per_source_per_epoch"]) * int(config["batch_size"])
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=count,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )


def train_source(
    repo_root: Path,
    config: dict[str, object],
    seed: int,
    device: torch.device,
    result_dir: Path,
    *,
    resume: Path | None,
) -> tuple[DCASEJointUnionCRNN, dict[str, object]]:
    _seed_everything(seed)
    cache_root = repo_root / str(config["frame_cache_root"])
    sources = _source_units(repo_root, config, seed)
    train_sets = {
        dataset: JointCachedDataset(units, cache_root / f"{dataset.split('sound')[0]}_train")
        for dataset, units in sources["subtrain"].items()
    }
    validation_loaders = {
        dataset: _loader(
            units,
            cache_root,
            "icbhi_train" if dataset == "icbhi" else "spr_train",
            int(config["batch_size"]),
        )
        for dataset, units in sources["validation"].items()
    }
    model = DCASEJointUnionCRNN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    start_epoch, best_score, best_epoch = 1, -1.0, 0
    no_improvement, completed, stopped, stop_reason = 0, 0, False, None
    if resume is not None:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_score, best_epoch = float(state["selection_score"]), int(state["best_epoch"])
        no_improvement = int(state["no_improvement_epochs"])
        completed, stopped, stop_reason = int(state["completed_epochs"]), bool(state["stopped_early"]), state["stop_reason"]
        best_state = torch.load(result_dir / "best_checkpoint.pt", map_location="cpu")
        if (
            int(best_state["epoch"]) != best_epoch
            or float(best_state["selection_score"]) != best_score
        ):
            raise RuntimeError("last checkpoint and best selection state disagree")
    started = time.perf_counter()
    epoch_range = range(start_epoch, int(config["epochs"]) + 1) if not stopped and completed < int(config["epochs"]) else ()
    for epoch in epoch_range:
        loaders = {
            dataset: iter(_source_loader(value, config, seed * 1000 + epoch * 10 + offset))
            for offset, (dataset, value) in enumerate(train_sets.items())
        }
        schedule = [dataset for dataset in ("icbhi", "sprsound") for _ in range(int(config["batches_per_source_per_epoch"]))]
        random.Random(seed * 1000 + epoch).shuffle(schedule)
        model.train()
        losses = []
        for dataset in schedule:
            waveform, frames, target, mask, _ = next(loaders[dataset])
            probabilities = model(
                waveform.to(device), frames.to(device), mask.to(device)
            )["clip_probabilities"]
            loss = masked_node_bce(probabilities, target.to(device), mask.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = {
            dataset: _predict(model, loader, device)
            for dataset, loader in validation_loaders.items()
        }
        selection = _selection(validation)
        state = update_early_stopping(
            score=float(selection["selection_score"]),
            best_score=best_score,
            no_improvement_epochs=no_improvement,
            patience=int(config["early_stopping_patience"]),
            min_delta=float(config["early_stopping_min_delta"]),
        )
        best_score = float(state["best_score"])
        no_improvement = int(state["no_improvement_epochs"])
        stopped = bool(state["stopped_early"])
        completed = epoch
        stop_reason = "patience_exhausted" if stopped else ("max_epochs_reached" if epoch == int(config["epochs"]) else None)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": selection,
            "improved": bool(state["improved"]),
            "no_improvement_epochs": no_improvement,
            "stopped_early": stopped,
            "stop_reason": stop_reason,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.perf_counter() - started) / 60,
        }
        append_jsonl(result_dir / "train_log.jsonl", record)
        if bool(state["improved"]):
            best_epoch = epoch
            np.savez_compressed(
                result_dir / "selected_validation_predictions.npz",
                icbhi_ids=validation["icbhi"]["sample_ids"],
                icbhi_scores=validation["icbhi"]["scores"],
                icbhi_targets=validation["icbhi"]["targets"],
                spr_ids=validation["sprsound"]["sample_ids"],
                spr_scores=validation["sprsound"]["scores"],
                spr_targets=validation["sprsound"]["targets"],
            )
            _checkpoint(
                result_dir / "best_checkpoint.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                best_epoch=best_epoch,
                no_improvement_epochs=0,
                stopped_early=False,
                stop_reason=None,
                config=config,
            )
        scheduler.step()
        _checkpoint(
            result_dir / "last_checkpoint.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_score=best_score,
            best_epoch=best_epoch,
            no_improvement_epochs=no_improvement,
            stopped_early=stopped,
            stop_reason=stop_reason,
            config=config,
        )
        if stopped:
            break
    if not (result_dir / "best_checkpoint.pt").is_file():
        raise RuntimeError("joint source run produced no best checkpoint")
    selected = torch.load(result_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(selected["model"])
    return model, {
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "completed_epochs": completed,
        "max_epochs": int(config["epochs"]),
        "stopped_early": stopped,
        "stop_reason": stop_reason,
        "no_improvement_epochs": no_improvement,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
    }


def evaluate_terminal(
    repo_root: Path,
    config: Mapping[str, object],
    result_dir: Path,
    model: DCASEJointUnionCRNN,
    device: torch.device,
) -> dict[str, object]:
    cache_root = repo_root / str(config["frame_cache_root"])
    batch_size = int(config["batch_size"])
    icbhi_units = load_icbhi_units(repo_root, "test")
    icbhi = _predict(model, _loader(icbhi_units, cache_root, "icbhi_test", batch_size), device)
    icbhi_prediction = icbhi["scores"][:, ICBHI_NATIVE].argmax(axis=1)
    np.savez_compressed(result_dir / "icbhi_terminal_predictions.npz", **icbhi, predictions=icbhi_prediction)

    spr_units = load_spr_inter_units(repo_root, include_targets=False)
    spr = _predict(model, _loader(spr_units, cache_root, "spr_inter", batch_size), device)
    spr_raw_prediction = spr["scores"][:, SPR_NATIVE].argmax(axis=1)
    spr_prediction = (spr_raw_prediction != 0).astype(np.int64)
    np.savez_compressed(result_dir / "spr_predictions_label_free.npz", sample_ids=spr["sample_ids"], scores=spr["scores"], raw7_predictions=spr_raw_prediction, predictions=spr_prediction)
    spr_targets_by_id = load_spr_targets(spr_units)
    spr_target = np.asarray([spr_targets_by_id[str(value)]["binary"] for value in spr["sample_ids"]])
    np.savez_compressed(result_dir / "spr_predictions_scored.npz", sample_ids=spr["sample_ids"], scores=spr["scores"], raw7_predictions=spr_raw_prediction, predictions=spr_prediction, targets=spr_target)

    hf_units = load_hf_test_units(repo_root)
    hf_windows = _all_cache_units(repo_root, config)["hf_test_windows"]
    hf = _predict(model, _loader(hf_windows, cache_root, "hf_test_windows", batch_size), device)
    window_score = hf["scores"][:, HF_CAS].max(axis=1)
    recording_score = window_score.reshape(len(hf_units), 3).max(axis=1)
    hf_ids = np.asarray([unit.sample_id for unit in hf_units])
    np.savez_compressed(result_dir / "hf_predictions_label_free.npz", sample_ids=hf_ids, window_ids=hf["sample_ids"].reshape(len(hf_units), 3), window_union_scores=window_score.reshape(len(hf_units), 3), window_native_scores=hf["scores"].reshape(len(hf_units), 3, len(LABELS)), recording_scores=recording_score)
    hf_targets = load_hf_cas_targets(hf_units)
    hf_mask = np.asarray([value in hf_targets for value in hf_ids])
    hf_target = np.asarray([hf_targets[value]["positive"] for value in hf_ids[hf_mask]])
    hf_score = recording_score[hf_mask]
    np.savez_compressed(result_dir / "hf_predictions_scored.npz", sample_ids=hf_ids[hf_mask], scores=hf_score, targets=hf_target)

    kauh_units = load_kauh_units(repo_root)
    kauh = _predict(model, _loader(kauh_units, cache_root, "kauh_all", batch_size), device)
    np.savez_compressed(result_dir / "kauh_view_predictions.npz", sample_ids=kauh["sample_ids"], patient_ids=np.asarray([unit.group_id for unit in kauh_units]), views=np.asarray([unit.metadata["view"] for unit in kauh_units]), native_scores=kauh["scores"][:, ICBHI_NATIVE])
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(kauh_units):
        by_patient[unit.group_id].append(index)
    patient_ids, predictions, targets, mean_scores, consistency = [], [], [], [], []
    for patient in sorted(by_patient, key=lambda value: int(value[1:])):
        indices = by_patient[patient]
        units = [kauh_units[index] for index in indices]
        if units[0].target is None:
            continue
        vector = kauh["scores"][indices][:, ICBHI_NATIVE].mean(axis=0)
        prediction = int(vector.argmax())
        view_predictions = kauh["scores"][indices][:, ICBHI_NATIVE].argmax(axis=1)
        patient_ids.append(patient)
        mean_scores.append(vector)
        predictions.append(int(prediction != 0))
        targets.append(int(units[0].target))
        consistency.append(bool(np.all(view_predictions == view_predictions[0])))
    np.savez_compressed(result_dir / "kauh_patient_predictions.npz", patient_ids=np.asarray(patient_ids), mean_native_scores=np.asarray(mean_scores), predictions=np.asarray(predictions), targets=np.asarray(targets))

    metrics = {
        "icbhi": icbhi_metrics(np.asarray([int(unit.target) for unit in icbhi_units]), icbhi_prediction),
        "sprsound": spr_metrics(spr_target, spr_prediction),
        "hf": {"eligible_recordings": int(len(hf_target)), "positive": int(hf_target.sum()), "negative": int((1-hf_target).sum()), "cas_auroc": binary_auroc(hf_target, hf_score), "score": "max window max(Wheeze,Both,Rhonchi,Stridor)"},
        "kauh": {**binary_metrics(np.asarray(targets), np.asarray(predictions)), "eligible_patients": len(patient_ids), "filter_view_consistency": float(np.mean(consistency)), "score": "mean B/D/E native4 scores, argmax, regroup"},
    }
    write_json(result_dir / "metrics.json", metrics)
    return metrics


def run(repo_root: Path, config_path: Path, seed: int, device_name: str, resume: Path | None) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    config["seed"] = seed
    result_dir = repo_root / str(config["output_root"]) / f"seed_{seed}"
    prepare_run_directory(result_dir, config, resume)
    metadata = metadata_summary(repo_root, config, seed)
    write_json(result_dir / "source_split_summary.json", metadata)
    model, selection = train_source(repo_root, config, seed, torch.device(device_name), result_dir, resume=resume)
    metrics = evaluate_terminal(repo_root, config, result_dir, model, torch.device(device_name))
    summary = {"status": "complete_joint_native_union", "method": config["method"], "seed": seed, "source_split": metadata, "selection": selection, "metrics": metrics, "hf_kauh_training": False}
    write_json(result_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("baseline/frozen_method_baselines/dcase_joint_union_run.json"))
    parser.add_argument("--phase", choices=("metadata", "extract-frames", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 42))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "NOT_RUN", "phase": args.phase, "seed": args.seed}, indent=2))
        return
    config = json.loads(args.config.read_text())
    repo_root = args.repo_root.resolve()
    if args.phase == "metadata":
        print(
            json.dumps(
                {str(seed): metadata_summary(repo_root, config, seed) for seed in (0, 1, 42)},
                indent=2,
            )
        )
    elif args.phase == "extract-frames":
        print(json.dumps(extract_frame_cache(repo_root, config, torch.device(args.device)), indent=2))
    else:
        if args.seed is None:
            raise ValueError("--seed is required for train")
        print(json.dumps(run(repo_root, args.config, args.seed, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
