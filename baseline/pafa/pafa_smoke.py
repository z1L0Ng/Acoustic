"""PAFA preprocessing, BEATs forward, and patient-loss compatibility smoke."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as audio_transforms


LABELS = ["normal", "crackle", "wheeze", "both"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_repeated_patient_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["official_split"] == "train"]
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    selected = []
    for patient in sorted(by_patient):
        if len(by_patient[patient]) >= 2:
            selected.extend(by_patient[patient][:2])
        if len(selected) == 8:
            break
    assert len(selected) == 8
    assert all(row["patient_id"] == row["recording_id"].split("_", 1)[0] for row in selected)
    assert all(count == 2 for count in {pid: sum(row["patient_id"] == pid for row in selected) for pid in {row["patient_id"] for row in selected}}.values())
    return selected


def preprocess(rows: list[dict[str, str]], author_repo: Path, args: SimpleNamespace) -> torch.Tensor:
    sys.path.insert(0, str(author_repo))
    from util.icbhi_util import cut_pad_sample_torchaudio

    output = []
    for row in rows:
        waveform, sample_rate = torchaudio.load(row["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != args.sample_rate:
            waveform = audio_transforms.Resample(sample_rate, args.sample_rate)(waveform)
        fade_samples = int(args.sample_rate / 16)
        waveform = audio_transforms.Fade(
            fade_in_len=fade_samples,
            fade_out_len=fade_samples,
            fade_shape="linear",
        )(waveform)
        start = int(float(row["cycle_start_s"]) * args.sample_rate)
        end = int(float(row["cycle_end_s"]) * args.sample_rate)
        cycle = waveform[:, min(start, waveform.shape[1]) : min(end, waveform.shape[1])]
        cycle = cut_pad_sample_torchaudio(cycle, args).squeeze(0)
        output.append(cycle)
    batch = torch.stack(output)
    assert batch.shape == (8, 80000)
    assert torch.isfinite(batch).all()
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args_cli = parser.parse_args()
    if args_cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args_cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args_cli.device}")
    manifest = args_cli.manifest.resolve()
    author_repo = args_cli.author_repo.resolve()
    checkpoint = args_cli.checkpoint.resolve()
    work_dir = args_cli.work_dir.resolve()
    output_dir = args_cli.output_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    sys.path.insert(0, str(author_repo))

    from method.pafa import PAFALoss, ProjectionHead
    from models.beats import BEATsTransferLearningModel
    from util.misc import update_moving_average

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    rows = select_repeated_patient_rows(manifest)
    start = time.perf_counter()
    batch = preprocess(
        rows,
        author_repo,
        SimpleNamespace(sample_rate=16000, desired_length=5, pad_types="repeat"),
    )
    preprocessing_seconds = time.perf_counter() - start
    batch = batch.to(device)
    class_labels = torch.tensor(
        [LABELS.index(row["native_four_class_label"]) for row in rows],
        device=device,
    )
    patient_ids = torch.tensor([int(row["patient_id"]) for row in rows], device=device)

    model = BEATsTransferLearningModel(
        num_target_classes=4,
        model_path=str(checkpoint),
        ft_entire_network=True,
        spec_transform=None,
    )
    classifier = torch.nn.Linear(model.final_feat_dim, 4)
    projector = ProjectionHead(
        input_dim=model.final_feat_dim,
        hidden_dim=None,
        output_dim=768,
        attention=True,
        norm_type="ln",
        proj_type="end2end",
    )
    model = model.to(device)
    classifier = classifier.to(device)
    projector = projector.to(device)
    model.train()
    classifier.train()
    projector.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
        lr=5e-5,
        weight_decay=1e-6,
    )
    step_times = []
    profile_start = time.perf_counter()
    for _ in range(args_cli.steps):
        model_before = deepcopy(model.state_dict())
        classifier_before = deepcopy(classifier.state_dict())
        projector_before = deepcopy(projector.state_dict())
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        frame_features = model(batch, training=True)
        logits = classifier(frame_features).mean(dim=1)
        projected = projector(frame_features)
        classification_loss = torch.nn.functional.cross_entropy(logits, class_labels)
        pafa_loss = PAFALoss()(projected, patient_ids, lambda_pcsl=50.0, lambda_gpal=0.0005)
        loss = classification_loss + pafa_loss
        loss.backward()
        optimizer.step()
        model = update_moving_average(0.5, model, model_before)
        classifier = update_moving_average(0.5, classifier, classifier_before)
        projector = update_moving_average(0.5, projector, projector_before)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - start)
    profile_wall_seconds = time.perf_counter() - profile_start
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    )
    assert torch.isfinite(loss)
    assert gradients_finite

    np.savez_compressed(
        output_dir / "pafa_smoke_8_outputs.npz",
        cycle_id=np.asarray([row["cycle_id"] for row in rows]),
        patient_id=patient_ids.detach().cpu().numpy(),
        label=class_labels.detach().cpu().numpy(),
        frame_features=frame_features.detach().cpu().numpy(),
        projected=projected.detach().cpu().numpy(),
        logits=logits.detach().cpu().numpy(),
    )
    payload = {
        "status": "passed_with_unverified_official_named_checkpoint_mirror",
        "official_checkpoint_identity_verified": False,
        "official_onedrive_probe": "HTTP 403",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cycles": 8,
        "cycle_ids": [row["cycle_id"] for row in rows],
        "patient_ids": patient_ids.detach().cpu().tolist(),
        "patient_id_policy": "int(recording filename token before first underscore), exact author loader rule",
        "unique_patients": len(set(patient_ids.detach().cpu().tolist())),
        "cycles_per_patient": 2,
        "waveform_shape": list(batch.shape),
        "frame_feature_shape": list(frame_features.shape),
        "projected_shape": list(projected.shape),
        "logits_shape": list(logits.shape),
        "preprocessing": "author 16 kHz mono, recording fade, cycle crop, 5 s repeat+fade, --nospec",
        "classification_loss": float(classification_loss.item()),
        "pafa_loss": float(pafa_loss.item()),
        "total_loss": float(loss.item()),
        "loss_finite": True,
        "gradients_finite": gradients_finite,
        "preprocessing_seconds": preprocessing_seconds,
        "steps": args_cli.steps,
        "cpu_training_step_seconds": step_times,
        "mean_cpu_training_step_seconds": float(np.mean(step_times)),
        "projected_100_step_seconds": float(np.mean(step_times) * 100),
        "profile_wall_seconds_including_state_copy_and_ema": profile_wall_seconds,
        "process_max_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "ema_beta": 0.5,
        "compatibility_changes": ["CPU device handling", "explicit cycle ID/output export"],
    }
    receipt_name = "pafa_smoke_8.json" if args_cli.steps == 1 else f"pafa_profile_{args_cli.steps}.json"
    (output_dir / receipt_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
