"""Historical ICBHI-only DCASE runner; superseded by joint native union."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .pcmcl_source_transfer import build_pcmcl_single_5s
from .source_transfer_common import (
    AudioUnit,
    append_jsonl,
    binary_auroc,
    binary_metrics,
    icbhi_metrics,
    load_hf_cas_targets,
    load_hf_test_units,
    load_hf_windows,
    load_icbhi_units,
    load_kauh_units,
    load_single_5s,
    load_spr_inter_units,
    load_spr_targets,
    prepare_run_directory,
    spr_metrics,
    update_early_stopping,
    write_json,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amplitude_to_db(value: torch.Tensor) -> torch.Tensor:
    """Amplitude dB with the clamp applied after the 20*log10 conversion."""

    return (20 * torch.log10(value.clamp_min(1e-5))).clamp(-50, 80)


def pooled_frequency_bins(n_mels: int, pooling: Sequence[Sequence[int]]) -> int:
    value = n_mels
    for _, frequency in pooling:
        value //= int(frequency)
    return value


class GLUBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, pool: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels * 2, 3, padding=1)
        self.norm = nn.BatchNorm2d(output_channels * 2)
        self.pool = nn.AvgPool2d(tuple(pool))
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(F.glu(self.norm(self.conv(value)), dim=1)))


class DCASEFlat4CRNN(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
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
        if pooled_frequency_bins(int(mel["n_mels"]), pools) != 1:
            raise ValueError("DCASE CNN pooling must reduce the 128-mel axis to one bin")
        blocks = []
        input_channels = 1
        for output_channels, pool in zip(channels, pools):
            blocks.append(GLUBlock(input_channels, output_channels, pool, float(config["model"]["dropout"])))
            input_channels = output_channels
        self.cnn = nn.Sequential(*blocks)
        fusion_dim = int(config["model"]["fusion_dim"])
        hidden = int(config["model"]["bigru_hidden"])
        self.fusion = nn.Linear(channels[-1] + 768, fusion_dim)
        self.rnn = nn.GRU(fusion_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(float(config["model"]["dropout"]))
        self.classifier = nn.Linear(hidden * 2, 4)
        self.attention = nn.Linear(hidden * 2, 4)

    def _logmel(self, waveform: torch.Tensor) -> torch.Tensor:
        values = self.mel(waveform).clamp_min(1e-5)
        values = amplitude_to_db(values)
        minimum = values.amin(dim=(1, 2), keepdim=True)
        maximum = values.amax(dim=(1, 2), keepdim=True)
        return (values - minimum) / (maximum - minimum).clamp_min(1e-7)

    def forward(self, waveform: torch.Tensor, beats_frames: torch.Tensor) -> dict[str, torch.Tensor]:
        # MelSpectrogram returns [B,Mel,T]; the CNN contract is [B,1,T,Mel].
        cnn = self.cnn(self._logmel(waveform).transpose(1, 2).unsqueeze(1))
        batch, channels, frames, frequency = cnn.shape
        if frequency != 1:
            raise RuntimeError("DCASE CNN did not reduce the mel-frequency axis to one bin")
        cnn_frames = cnn.permute(0, 2, 1, 3).reshape(batch, frames, channels * frequency)
        aligned = F.adaptive_avg_pool1d(
            beats_frames.transpose(1, 2), frames
        ).transpose(1, 2)
        fused = self.fusion(torch.cat((cnn_frames, aligned), dim=-1))
        recurrent, _ = self.rnn(self.dropout(fused))
        recurrent = self.dropout(recurrent)
        frame_logits = self.classifier(recurrent)
        attention = torch.softmax(self.attention(recurrent), dim=1)
        clip_logits = (frame_logits * attention).sum(dim=1)
        return {"frame_logits": frame_logits, "clip_logits": clip_logits}


class CachedFrameDataset(Dataset):
    def __init__(self, units: Sequence[AudioUnit], cache_dir: Path) -> None:
        self.units = tuple(units)
        ids = np.load(cache_dir / "ids.npy").astype(str)
        self.frames = np.load(cache_dir / "frames.npy", mmap_mode="r")
        self.index = {value: index for index, value in enumerate(ids.tolist())}

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        unit = self.units[index]
        return (
            load_single_5s(unit),
            torch.from_numpy(
                np.array(self.frames[self.index[unit.sample_id]], copy=True)
            ).float(),
            int(unit.target or 0),
            unit.sample_id,
        )


def _load_frozen_beats(
    repo_root: Path, config: dict[str, object], device: torch.device
) -> nn.Module:
    source = repo_root / str(config["source_repo"])
    checkpoint = repo_root / str(config["initial_checkpoint"])
    sys.path.insert(0, str(source.resolve()))
    from models.beats import BEATsTransferLearningModel

    model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(checkpoint),
        ft_entire_network=False,
        spec_transform=None,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def _extract_temporal_frames(model: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Restore the BEATs time/frequency patch grid and mean only frequency."""

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
    return encoded.reshape(
        batch, time_patches, frequency_patches, encoded.shape[-1]
    ).mean(dim=2)


def _role_units(repo_root: Path) -> dict[str, list[AudioUnit]]:
    return {
        "icbhi_train": load_icbhi_units(repo_root, "train"),
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
            for unit in load_hf_test_units(repo_root)
            for window in range(3)
        ],
        "kauh_all": load_kauh_units(repo_root),
    }


def extract_frame_cache(
    repo_root: Path, config: dict[str, object], device: torch.device
) -> dict[str, object]:
    cache_root = repo_root / str(config["frame_cache_root"])
    beats = _load_frozen_beats(repo_root, config, device)
    report = {}
    for role, units in _role_units(repo_root).items():
        output_dir = cache_root / role
        output_dir.mkdir(parents=True, exist_ok=True)
        ids = np.asarray([unit.sample_id for unit in units])
        frame_file = output_dir / "frames.npy"
        target = None
        started = time.perf_counter()
        for start in range(0, len(units), int(config["frame_extract_batch_size"])):
            current = units[start : start + int(config["frame_extract_batch_size"])]
            waveform = torch.stack([load_single_5s(unit) for unit in current]).to(device)
            with torch.no_grad():
                frames = _extract_temporal_frames(beats, waveform).cpu().numpy()
            if target is None:
                target = np.lib.format.open_memmap(
                    frame_file,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(units), frames.shape[1], frames.shape[2]),
                )
            target[start : start + len(current)] = frames
        np.save(output_dir / "ids.npy", ids)
        if target is not None:
            target.flush()
        report[role] = {
            "rows": len(units),
            "frame_shape": list(target.shape[1:]) if target is not None else None,
            "elapsed_seconds": time.perf_counter() - started,
        }
    write_json(cache_root / "extraction_summary.json", report)
    return report


def _predict(
    model: DCASEFlat4CRNN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    ids, probabilities, targets = [], [], []
    with torch.no_grad():
        for waveform, frames, target, sample_ids in loader:
            logits = model(waveform.to(device), frames.to(device))["clip_logits"]
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            targets.append(target.numpy())
            ids.extend(sample_ids)
    return {
        "sample_ids": np.asarray(ids),
        "probabilities": np.concatenate(probabilities),
        "targets": np.concatenate(targets),
    }


def _checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    score: float,
    best_epoch: int,
    no_improvement_epochs: int,
    completed_epochs: int,
    stopped_early: bool,
    stop_reason: str | None,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "selection_score": score,
            "best_epoch": best_epoch,
            "no_improvement_epochs": no_improvement_epochs,
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "config": config,
        },
        path,
    )


def train_source(
    repo_root: Path,
    config: dict[str, object],
    seed: int,
    device: torch.device,
    result_dir: Path,
    *,
    resume: Path | None,
) -> tuple[DCASEFlat4CRNN, dict[str, object]]:
    _seed_everything(seed)
    cache_root = repo_root / str(config["frame_cache_root"])
    train_units = load_icbhi_units(repo_root, "train")
    test_units = load_icbhi_units(repo_root, "test")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CachedFrameDataset(train_units, cache_root / "icbhi_train"),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    test_loader = DataLoader(
        CachedFrameDataset(test_units, cache_root / "icbhi_test"),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    model = DCASEFlat4CRNN(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    start_epoch, best_score, best_epoch = 1, -1.0, 0
    no_improvement_epochs = 0
    completed_epochs = 0
    stopped_early = False
    stop_reason = None
    if resume is not None:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["selection_score"])
        best_epoch = int(state["best_epoch"])
        no_improvement_epochs = int(state["no_improvement_epochs"])
        completed_epochs = int(state["completed_epochs"])
        stopped_early = bool(state["stopped_early"])
        stop_reason = state["stop_reason"]
        if best_epoch > 0:
            best_state = torch.load(result_dir / "best_checkpoint.pt", map_location="cpu")
            if (
                int(best_state["epoch"]) != best_epoch
                or float(best_state["selection_score"]) != best_score
            ):
                raise RuntimeError("last checkpoint and best selection state disagree")
    started = time.perf_counter()
    epoch_range = (
        range(start_epoch, int(config["epochs"]) + 1)
        if not stopped_early and completed_epochs < int(config["epochs"])
        else ()
    )
    for epoch in epoch_range:
        model.train()
        losses = []
        for waveform, frames, target, _ in train_loader:
            logits = model(waveform.to(device), frames.to(device))["clip_logits"]
            loss = F.cross_entropy(logits, target.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        source = _predict(model, test_loader, device)
        prediction = source["probabilities"].argmax(axis=1)
        metrics = icbhi_metrics(source["targets"], prediction)
        score = float(metrics["icbhi_score"])
        early = update_early_stopping(
            score=score,
            best_score=best_score,
            no_improvement_epochs=no_improvement_epochs,
            patience=int(config["early_stopping_patience"]),
            min_delta=float(config["early_stopping_min_delta"]),
            eligible=True,
        )
        best_score = float(early["best_score"])
        no_improvement_epochs = int(early["no_improvement_epochs"])
        stopped_early = bool(early["stopped_early"])
        completed_epochs = epoch
        stop_reason = (
            "patience_exhausted"
            if stopped_early
            else ("max_epochs_reached" if epoch == int(config["epochs"]) else None)
        )
        append_jsonl(
            result_dir / "train_log.jsonl",
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "source_test_metrics": metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_minutes": (time.perf_counter() - started) / 60,
                "improved": bool(early["improved"]),
                "no_improvement_epochs": no_improvement_epochs,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
            },
        )
        if bool(early["improved"]):
            best_epoch = epoch
            _checkpoint(
                result_dir / "best_checkpoint.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                score=score,
                best_epoch=epoch,
                no_improvement_epochs=0,
                completed_epochs=epoch,
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
            score=best_score,
            best_epoch=best_epoch,
            no_improvement_epochs=no_improvement_epochs,
            completed_epochs=completed_epochs,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            config=config,
        )
        if stopped_early:
            break
    if not (result_dir / "best_checkpoint.pt").is_file():
        raise RuntimeError("source run produced no eligible best checkpoint")
    selected = torch.load(result_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(selected["model"])
    return model, {
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "completed_epochs": completed_epochs,
        "max_epochs": int(config["epochs"]),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "no_improvement_epochs": no_improvement_epochs,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "resume_semantics": (
            "model/optimizer/scheduler/epoch/best selection restored; full Python/NumPy/"
            "DataLoader RNG state is not restored, so resume is not bitwise identical"
        ),
    }


def _loader(units: Sequence[AudioUnit], cache_root: Path, role: str, batch_size: int) -> DataLoader:
    return DataLoader(
        CachedFrameDataset(units, cache_root / role),
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate_fixed_transfer(
    repo_root: Path,
    config: dict[str, object],
    result_dir: Path,
    model: DCASEFlat4CRNN,
    device: torch.device,
) -> dict[str, object]:
    cache_root = repo_root / str(config["frame_cache_root"])
    batch_size = int(config["batch_size"])
    icbhi_units = load_icbhi_units(repo_root, "test")
    icbhi = _predict(model, _loader(icbhi_units, cache_root, "icbhi_test", batch_size), device)
    icbhi_prediction = icbhi["probabilities"].argmax(axis=1)
    np.savez_compressed(
        result_dir / "icbhi_selected_predictions.npz",
        **icbhi,
        predictions=icbhi_prediction,
    )

    spr_units = load_spr_inter_units(repo_root, include_targets=False)
    spr = _predict(model, _loader(spr_units, cache_root, "spr_inter", batch_size), device)
    spr_score = 1.0 - spr["probabilities"][:, 0]
    spr_prediction = (spr_score >= 0.5).astype(np.int64)
    np.savez_compressed(
        result_dir / "spr_predictions_label_free.npz",
        sample_ids=spr["sample_ids"], probabilities=spr["probabilities"], scores=spr_score,
        predictions=spr_prediction,
    )
    spr_targets = load_spr_targets(spr_units)
    spr_target = np.asarray([spr_targets[str(value)]["binary"] for value in spr["sample_ids"]])
    np.savez_compressed(
        result_dir / "spr_predictions_scored.npz",
        sample_ids=spr["sample_ids"], probabilities=spr["probabilities"], scores=spr_score,
        predictions=spr_prediction, targets=spr_target,
    )

    hf_units = load_hf_test_units(repo_root)
    hf_window_units = _role_units(repo_root)["hf_test_windows"]
    hf = _predict(model, _loader(hf_window_units, cache_root, "hf_test_windows", batch_size), device)
    hf_window_score = hf["probabilities"][:, 2] + hf["probabilities"][:, 3]
    hf_recording_score = hf_window_score.reshape(len(hf_units), 3).max(axis=1)
    hf_ids = np.asarray([unit.sample_id for unit in hf_units])
    np.savez_compressed(
        result_dir / "hf_predictions_label_free.npz",
        sample_ids=hf_ids,
        window_sample_ids=hf["sample_ids"].reshape(len(hf_units), 3),
        window_class_probabilities=hf["probabilities"].reshape(len(hf_units), 3, 4),
        maximum_window_p_wheeze_plus_p_both=hf_recording_score,
    )
    hf_targets = load_hf_cas_targets(hf_units)
    hf_mask = np.asarray([value in hf_targets for value in hf_ids])
    hf_target = np.asarray([hf_targets[value]["positive"] for value in hf_ids[hf_mask]])
    hf_score = hf_recording_score[hf_mask]
    np.savez_compressed(
        result_dir / "hf_predictions_scored.npz",
        sample_ids=hf_ids[hf_mask], scores=hf_score, targets=hf_target,
    )

    kauh_units = load_kauh_units(repo_root)
    kauh = _predict(model, _loader(kauh_units, cache_root, "kauh_all", batch_size), device)
    view_score = 1.0 - kauh["probabilities"][:, 0]
    np.savez_compressed(
        result_dir / "kauh_view_predictions.npz",
        sample_ids=kauh["sample_ids"],
        patient_ids=np.asarray([unit.group_id for unit in kauh_units]),
        views=np.asarray([str(unit.metadata["view"]) for unit in kauh_units]),
        class_probabilities=kauh["probabilities"],
        abnormal_scores=view_score,
    )
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(kauh_units):
        by_patient[unit.group_id].append(index)
    patient_ids, patient_scores, patient_predictions, patient_targets, consistency = [], [], [], [], []
    for patient in sorted(by_patient, key=lambda value: int(value[1:])):
        indices = by_patient[patient]
        units = [kauh_units[index] for index in indices]
        if units[0].target is None:
            continue
        score = float(view_score[indices].mean())
        view_prediction = view_score[indices] >= 0.5
        patient_ids.append(patient)
        patient_scores.append(score)
        patient_predictions.append(int(score >= 0.5))
        patient_targets.append(int(units[0].target))
        consistency.append(bool(np.all(view_prediction == view_prediction[0])))
    np.savez_compressed(
        result_dir / "kauh_patient_predictions.npz",
        patient_ids=np.asarray(patient_ids), scores=np.asarray(patient_scores),
        predictions=np.asarray(patient_predictions), targets=np.asarray(patient_targets),
    )

    metrics = {
        "icbhi": icbhi_metrics(icbhi["targets"], icbhi_prediction),
        "sprsound": spr_metrics(spr_target, spr_prediction),
        "hf": {
            "eligible_recordings": int(len(hf_target)),
            "positive": int(hf_target.sum()),
            "negative": int((1 - hf_target).sum()),
            "cas_auroc": binary_auroc(hf_target, hf_score),
            "score": "max over windows of p_Wheeze + p_Both",
        },
        "kauh": {
            **binary_metrics(np.asarray(patient_targets), np.asarray(patient_predictions)),
            "eligible_patients": len(patient_ids),
            "filter_view_consistency": float(np.mean(consistency)),
            "score": "mean B/D/E (1-p_Normal), threshold 0.5",
        },
    }
    write_json(result_dir / "metrics.json", metrics)
    return metrics


def run(repo_root: Path, config_path: Path, seed: int, device: str, resume: Path | None) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    config["seed"] = seed
    result_dir = repo_root / str(config["output_root"]) / f"seed_{seed}"
    prepare_run_directory(result_dir, config, resume)
    model, selection = train_source(
        repo_root, config, seed, torch.device(device), result_dir, resume=resume
    )
    metrics = evaluate_fixed_transfer(repo_root, config, result_dir, model, torch.device(device))
    summary = {
        "status": "complete_test_selected_source_transfer",
        "method": config["method"],
        "seed": seed,
        "selection": selection,
        "metrics": metrics,
        "target_training": False,
        "target_selection": False,
        "target_threshold_fit": False,
    }
    write_json(result_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path,
        default=Path("baseline/frozen_method_baselines/dcase_source_run.json"),
    )
    parser.add_argument("--phase", choices=("extract-frames", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 42))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "NOT_RUN", "phase": args.phase, "seed": args.seed}, indent=2))
        return
    raise RuntimeError(
        "historical DCASE runner is superseded; use dcase_joint_union_runner"
    )
    config = json.loads(args.config.read_text())
    if args.phase == "extract-frames":
        print(json.dumps(extract_frame_cache(args.repo_root.resolve(), config, torch.device(args.device)), indent=2))
        return
    if args.seed is None:
        raise ValueError("--seed is required for train phase")
    print(json.dumps(run(args.repo_root.resolve(), args.config, args.seed, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
