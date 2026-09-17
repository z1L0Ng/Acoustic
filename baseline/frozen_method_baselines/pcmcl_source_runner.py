"""Complete PC-MCL ICBHI-source training and fixed-transfer runner.

Nothing runs without ``--run``.  The runner is intentionally test-selected to
match the audited PC-MCL source code and historical benchmark evidence label.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .pcmcl_source_transfer import (
    NCW_ORDER,
    PCMCLSourceHeads,
    SOURCE_THRESHOLD,
    build_pcmcl_pair_5s,
    build_pcmcl_single_5s,
    compose_icbhi_ncw_targets,
    hf_maximum_window_p_w,
    icbhi_flat4_from_ncw,
    kauh_patient_from_ncw_views,
    pcmcl_source_loss,
    spr_binary_from_ncw,
)
from .source_transfer_common import (
    AudioUnit,
    aggregate_seed_metrics,
    append_jsonl,
    binary_auroc,
    binary_metrics,
    icbhi_metrics,
    load_beats_transfer_class,
    load_hf_cas_targets,
    load_hf_test_units,
    load_hf_windows,
    load_icbhi_units,
    load_kauh_units,
    load_single_5s,
    load_spr_inter_units,
    load_spr_targets,
    load_waveform,
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


def _ncw(label: int) -> torch.Tensor:
    values = torch.zeros(3, dtype=torch.float32)
    if label == 0:
        values[0] = 1
    elif label == 1:
        values[1] = 1
    elif label == 2:
        values[2] = 1
    elif label == 3:
        values[1:] = 1
    return values


class SourceSpecAugment(nn.Module):
    """Dependency-light form of the audited icbhi_ast_sup masking policy."""

    def __init__(self, *, mask_value: str = "mean") -> None:
        super().__init__()
        self.mask_value = mask_value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # BEATs supplies [B,1,T,F].
        output = value.clone()
        fill = output.mean() if self.mask_value == "mean" else output.new_zeros(())
        time_frames, frequency_bins = output.shape[2], output.shape[3]
        for _ in range(2):
            width = int(torch.randint(0, min(48, frequency_bins) + 1, ()).item())
            start = int(torch.randint(0, frequency_bins - width + 1, ()).item())
            output[..., start : start + width] = fill
        for _ in range(2):
            width = int(torch.randint(0, min(160, time_frames) + 1, ()).item())
            start = int(torch.randint(0, time_frames - width + 1, ()).item())
            output[:, :, start : start + width, :] = fill
        return output


class PCMCLTrainDataset(Dataset):
    def __init__(
        self,
        units: Sequence[AudioUnit],
        *,
        seed: int,
        mixing_probability: float,
        patient_probability: float,
    ) -> None:
        self.units = tuple(units)
        self.seed = seed
        self.epoch = 0
        self.mixed_count = int(len(units) * mixing_probability)
        self.patient_count = int(len(units) * patient_probability)
        self.by_patient: dict[str, list[int]] = defaultdict(list)
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, unit in enumerate(units):
            self.by_patient[unit.group_id].append(index)
            self.by_class[int(unit.target)].append(index)
        self.mixing_combinations = (
            (0, 1), (0, 2), (1, 2),
            (0, 0), (1, 1), (2, 2),
            (0, 3), (3, 1), (3, 2), (3, 3),
        )
        self.patient_profiles = {
            patient: tuple(
                torch.stack([_ncw(int(units[index].target)) for index in indices])
                .amax(dim=0)
                .int()
                .tolist()
            )
            for patient, indices in self.by_patient.items()
        }
        self.profile_patients: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for patient, profile in self.patient_profiles.items():
            self.profile_patients[profile].append(patient)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.units) + self.mixed_count + self.patient_count

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed * 10_000_019 + self.epoch * 100_003 + index)

    def _pair(self, first_index: int, second_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self.units[first_index], self.units[second_index]
        waveform = build_pcmcl_pair_5s(load_waveform(first), load_waveform(second))
        target = compose_icbhi_ncw_targets(
            _ncw(int(first.target)), _ncw(int(second.target))
        )
        return waveform, target

    def __getitem__(self, index: int):
        rng = self._rng(index)
        source_count = len(self.units)
        if index < source_count:
            unit = self.units[index]
            return (
                load_single_5s(unit),
                _ncw(int(unit.target)),
                -1,
                False,
                unit.sample_id,
            )
        if index < source_count + self.mixed_count:
            first_class, second_class = rng.choice(self.mixing_combinations)
            if first_class == second_class:
                first, second = rng.sample(self.by_class[first_class], 2)
            else:
                first = rng.choice(self.by_class[first_class])
                second = rng.choice(self.by_class[second_class])
            waveform, target = self._pair(first, second)
            return waveform, target, -1, False, f"mix:{first}:{second}"

        positive = rng.random() < 0.5
        patients_with_pairs = [key for key, values in self.by_patient.items() if len(values) >= 2]
        if positive and patients_with_pairs:
            patient = rng.choice(patients_with_pairs)
            first, second = rng.sample(self.by_patient[patient], 2)
            patient_target = 1
        else:
            eligible_profiles = [
                profile for profile, patients in self.profile_patients.items() if len(patients) >= 2
            ]
            if eligible_profiles:
                first_patient, second_patient = rng.sample(
                    self.profile_patients[rng.choice(eligible_profiles)], 2
                )
            else:
                first_patient, second_patient = rng.sample(list(self.by_patient), 2)
            first = rng.choice(self.by_patient[first_patient])
            second = rng.choice(self.by_patient[second_patient])
            patient_target = 0
        waveform, target = self._pair(first, second)
        return waveform, target, patient_target, True, f"patient:{first}:{second}"


class SingleUnitDataset(Dataset):
    def __init__(self, units: Sequence[AudioUnit]) -> None:
        self.units = tuple(units)

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int):
        unit = self.units[index]
        return load_single_5s(unit), int(unit.target or 0), unit.sample_id


def _load_encoder(repo_root: Path, config: dict[str, object], device: torch.device) -> nn.Module:
    source = repo_root / str(config["source_repo"])
    checkpoint = repo_root / str(config["initial_checkpoint"])
    BEATsTransferLearningModel = load_beats_transfer_class(source)

    transform = None
    if bool(config["specaugment"]):
        transform = SourceSpecAugment(mask_value=str(config["specaugment_mask"]))
    return BEATsTransferLearningModel(
        num_target_classes=3,
        model_path=str(checkpoint),
        ft_entire_network=True,
        spec_transform=transform,
    ).to(device)


def _features(encoder: nn.Module, waveform: torch.Tensor, *, training: bool) -> torch.Tensor:
    values = encoder(waveform, training=training)
    return values.mean(dim=1) if values.ndim == 3 else values


def _predict_ncw(
    encoder: nn.Module,
    heads: PCMCLSourceHeads,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    encoder.eval()
    heads.eval()
    ids, probabilities, targets = [], [], []
    with torch.no_grad():
        for waveform, target, sample_ids in loader:
            output = heads(_features(encoder, waveform.to(device), training=False))
            probabilities.append(torch.sigmoid(output["pathology_logits"]).cpu().numpy())
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
    encoder: nn.Module,
    heads: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    selection_score: float,
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
            "encoder": encoder.state_dict(),
            "heads": heads.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "selection_score": selection_score,
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
) -> tuple[nn.Module, PCMCLSourceHeads, dict[str, object]]:
    _seed_everything(seed)
    train_units = load_icbhi_units(repo_root, "train")
    test_units = load_icbhi_units(repo_root, "test")
    dataset = PCMCLTrainDataset(
        train_units,
        seed=seed,
        mixing_probability=float(config["mixing_probability"]),
        patient_probability=float(config["patient_probability"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    test_loader = DataLoader(
        SingleUnitDataset(test_units),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    encoder = _load_encoder(repo_root, config, device)
    heads = PCMCLSourceHeads().to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *heads.parameters()],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(value) for value in config["lr_milestones"]],
        gamma=float(config["lr_decay"]),
    )
    start_epoch, best_score, best_epoch = 1, -1.0, 0
    no_improvement_epochs = 0
    completed_epochs = 0
    stopped_early = False
    stop_reason = None
    if resume is not None:
        state = torch.load(resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        heads.load_state_dict(state["heads"])
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
    log_path = result_dir / "train_log.jsonl"
    started = time.perf_counter()
    epoch_range = (
        range(start_epoch, int(config["epochs"]) + 1)
        if not stopped_early and completed_epochs < int(config["epochs"])
        else ()
    )
    for epoch in epoch_range:
        dataset.set_epoch(epoch)
        encoder.train()
        heads.train()
        losses = []
        for waveform, pathology_target, patient_target, patient_eligible, _ in train_loader:
            waveform = waveform.to(device)
            output = heads(_features(encoder, waveform, training=True))
            loss = pcmcl_source_loss(
                pathology_logits=output["pathology_logits"],
                pathology_target=pathology_target.to(device),
                patient_logits=output["patient_logits"],
                patient_target=patient_target.to(device),
                patient_eligibility=patient_eligible.to(device),
                patient_weight=float(config["patient_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss["total"].backward()
            optimizer.step()
            losses.append(float(loss["total"].detach()))
        source = _predict_ncw(encoder, heads, test_loader, device)
        prediction = icbhi_flat4_from_ncw(
            torch.from_numpy(source["probabilities"]), threshold=float(config["threshold"])
        ).numpy()
        metrics = icbhi_metrics(source["targets"], prediction)
        score = float(metrics["icbhi_score"])
        early = update_early_stopping(
            score=score,
            best_score=best_score,
            no_improvement_epochs=no_improvement_epochs,
            patience=int(config["early_stopping_patience"]),
            min_delta=float(config["early_stopping_min_delta"]),
            eligible=float(metrics["sensitivity"]) > 0.001,
            enabled=bool(config["early_stopping"]) and not bool(config["run_full_epochs"]),
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
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "source_test_metrics": metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_minutes": (time.perf_counter() - started) / 60,
            "improved": bool(early["improved"]),
            "no_improvement_epochs": no_improvement_epochs,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }
        append_jsonl(log_path, record)
        if bool(early["improved"]):
            best_epoch = epoch
            _checkpoint(
                result_dir / "best_checkpoint.pt",
                epoch=epoch,
                encoder=encoder,
                heads=heads,
                optimizer=optimizer,
                scheduler=scheduler,
                selection_score=score,
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
            encoder=encoder,
            heads=heads,
            optimizer=optimizer,
            scheduler=scheduler,
            selection_score=best_score,
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
    encoder.load_state_dict(selected["encoder"])
    heads.load_state_dict(selected["heads"])
    return encoder, heads, {
        "selected_epoch": int(selected["epoch"]),
        "selected_score": float(selected["selection_score"]),
        "completed_epochs": completed_epochs,
        "max_epochs": int(config["epochs"]),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "no_improvement_epochs": no_improvement_epochs,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "best_epoch_observed_in_current_process": best_epoch,
        "resume_semantics": (
            "model/optimizer/scheduler/epoch/best selection restored; full Python/NumPy/"
            "DataLoader RNG state is not restored, so resume is not bitwise identical"
        ),
    }


def _predict_units(
    units: Sequence[AudioUnit],
    encoder: nn.Module,
    heads: PCMCLSourceHeads,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(SingleUnitDataset(units), batch_size=batch_size, shuffle=False)
    result = _predict_ncw(encoder, heads, loader, device)
    return result["sample_ids"], result["probabilities"]


def evaluate_fixed_transfer(
    repo_root: Path,
    result_dir: Path,
    encoder: nn.Module,
    heads: PCMCLSourceHeads,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    source_units = load_icbhi_units(repo_root, "test")
    source_ids, source_prob = _predict_units(source_units, encoder, heads, device, batch_size)
    source_target = np.asarray([int(unit.target) for unit in source_units])
    source_pred = icbhi_flat4_from_ncw(torch.from_numpy(source_prob)).numpy()
    np.savez_compressed(
        result_dir / "icbhi_selected_predictions.npz",
        sample_ids=source_ids,
        probabilities=source_prob,
        targets=source_target,
        predictions=source_pred,
    )

    spr_units = load_spr_inter_units(repo_root, include_targets=False)
    spr_ids, spr_prob = _predict_units(spr_units, encoder, heads, device, batch_size)
    spr_pred = spr_binary_from_ncw(torch.from_numpy(spr_prob)).numpy()
    np.savez_compressed(
        result_dir / "spr_predictions_label_free.npz",
        sample_ids=spr_ids,
        probabilities=spr_prob,
        predictions=spr_pred,
    )
    spr_targets_by_id = load_spr_targets(spr_units)
    spr_target = np.asarray([spr_targets_by_id[str(value)]["binary"] for value in spr_ids])
    np.savez_compressed(
        result_dir / "spr_predictions_scored.npz",
        sample_ids=spr_ids,
        probabilities=spr_prob,
        predictions=spr_pred,
        targets=spr_target,
    )

    hf_units = load_hf_test_units(repo_root)
    hf_window_units = [
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
    ]
    _, hf_window_probabilities = _predict_units(
        hf_window_units, encoder, heads, device, batch_size
    )
    hf_ids = [unit.sample_id for unit in hf_units]
    hf_scores = hf_maximum_window_p_w(
        torch.from_numpy(hf_window_probabilities.reshape(len(hf_units), 3, 3))
    ).numpy()
    np.savez_compressed(
        result_dir / "hf_predictions_label_free.npz",
        sample_ids=np.asarray(hf_ids),
        window_sample_ids=np.asarray(
            [unit.sample_id for unit in hf_window_units]
        ).reshape(len(hf_units), 3),
        window_ncw_probabilities=hf_window_probabilities.reshape(len(hf_units), 3, 3),
        maximum_window_p_w=hf_scores,
    )
    hf_targets_by_id = load_hf_cas_targets(hf_units)
    hf_mask = np.asarray([value in hf_targets_by_id for value in hf_ids])
    hf_target = np.asarray([hf_targets_by_id[value]["positive"] for value in np.asarray(hf_ids)[hf_mask]])
    hf_score = hf_scores[hf_mask]
    np.savez_compressed(
        result_dir / "hf_predictions_scored.npz",
        sample_ids=np.asarray(hf_ids)[hf_mask],
        scores=hf_score,
        targets=hf_target,
    )

    kauh_units = load_kauh_units(repo_root)
    kauh_ids, kauh_prob = _predict_units(kauh_units, encoder, heads, device, batch_size)
    kauh_view_score = kauh_prob[:, 1:3].max(axis=1)
    np.savez_compressed(
        result_dir / "kauh_view_predictions.npz",
        sample_ids=kauh_ids,
        patient_ids=np.asarray([unit.group_id for unit in kauh_units]),
        views=np.asarray([str(unit.metadata["view"]) for unit in kauh_units]),
        ncw_probabilities=kauh_prob,
        abnormal_scores=kauh_view_score,
    )
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(kauh_units):
        by_patient[unit.group_id].append(index)
    patient_ids, patient_scores, patient_predictions, patient_targets = [], [], [], []
    consistency = []
    for patient in sorted(by_patient, key=lambda value: int(value[1:])):
        indices = by_patient[patient]
        units = [kauh_units[index] for index in indices]
        if units[0].target is None:
            continue
        view = torch.from_numpy(kauh_prob[indices]).unsqueeze(0)
        score, prediction = kauh_patient_from_ncw_views(view)
        view_prediction = (view[0, :, 1:3].max(dim=-1).values >= SOURCE_THRESHOLD).long()
        patient_ids.append(patient)
        patient_scores.append(float(score.item()))
        patient_predictions.append(int(prediction.item()))
        patient_targets.append(int(units[0].target))
        consistency.append(bool((view_prediction == view_prediction[0]).all()))
    np.savez_compressed(
        result_dir / "kauh_patient_predictions.npz",
        patient_ids=np.asarray(patient_ids),
        scores=np.asarray(patient_scores),
        predictions=np.asarray(patient_predictions),
        targets=np.asarray(patient_targets),
    )
    metrics = {
        "icbhi": icbhi_metrics(source_target, source_pred),
        "sprsound": spr_metrics(spr_target, spr_pred),
        "hf": {
            "eligible_recordings": int(len(hf_target)),
            "positive": int(hf_target.sum()),
            "negative": int((1 - hf_target).sum()),
            "cas_auroc": binary_auroc(hf_target, hf_score),
            "score": "maximum-window p_W",
        },
        "kauh": {
            **binary_metrics(np.asarray(patient_targets), np.asarray(patient_predictions)),
            "eligible_patients": len(patient_ids),
            "filter_view_consistency": float(np.mean(consistency)),
            "score": "mean B/D/E max(p_C,p_W), threshold 0.5",
        },
    }
    write_json(result_dir / "metrics.json", metrics)
    return metrics


def run(repo_root: Path, config_path: Path, seed: int, device: str, resume: Path | None) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    config["seed"] = seed
    result_dir = repo_root / str(config["output_root"]) / f"seed_{seed}"
    prepare_run_directory(result_dir, config, resume)
    encoder, heads, selection = train_source(
        repo_root, config, seed, torch.device(device), result_dir, resume=resume
    )
    metrics = evaluate_fixed_transfer(
        repo_root,
        result_dir,
        encoder,
        heads,
        torch.device(device),
        int(config["batch_size"]),
    )
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
        "--config",
        type=Path,
        default=Path("baseline/frozen_method_baselines/pcmcl_source_run.json"),
    )
    parser.add_argument("--seed", type=int, choices=(0, 1, 42), required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "NOT_RUN", "config": str(args.config), "seed": args.seed}, indent=2))
        return
    print(json.dumps(run(args.repo_root.resolve(), args.config, args.seed, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
