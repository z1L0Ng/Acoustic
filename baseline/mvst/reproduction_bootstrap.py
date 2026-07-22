"""Fresh-checkout bootstrap for the author-code MVST server package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

from baseline.common.official_reproduction_bootstrap import (
    MethodSpec,
    build_manifest,
    chicago_timestamp,
    clone_source,
    discover_audio_dir,
    ensure_symlink,
    materialize_checkpoint,
    prepare_result_root_for_bootstrap,
    sha256,
)
from baseline.common.storage_safe_retention import (
    install_author_checkpoint_writer,
    install_author_training_retention,
)


METHOD = "mvst"
MINIMUM_COMMIT = "4172524a0e5d7b792de248820439f30874e2ae6d"
ENVIRONMENT_NAME = "acoustic-mvst-r4"
VIEWS = ["16", "32", "64", "128", "256"]
SPEC = MethodSpec(
    method=METHOD,
    repo_url="https://github.com/wentaoheunnc/MVST.git",
    commit="51f93fa6ffa580d0819ccb59f861582927937264",
    checkpoint_filename="audioset_16_16_0.4422.pth",
    checkpoint_url="https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1",
    checkpoint_sha256="dc71a6d4d07aeb7e746547f72a141f404e4c167d660bf003179f3865e06a970c",
    checkpoint_provenance=(
        "current author-hosted MVST runtime artifact; historical paper-run byte identity unpublished"
    ),
    support_subdir="data",
)
RUN_ROOT_PATTERN = re.compile(r"^mvst_\d{8}_\d{6}$")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"MVST patch anchor {label!r} count={text.count(old)}")
    return text.replace(old, new)


def install_compatibility_patch(source: Path) -> dict:
    """Make all five author encoder checkpoints interruption-safe."""
    patched_files = []
    for view in VIEWS:
        misc_path = source / view / "util" / "misc.py"
        misc = misc_path.read_text()
        old_save = """def save_model(model, optimizer, args, epoch, save_file, classifier):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'classifier': classifier.state_dict()
    }

    torch.save(state, save_file)
    del state
"""
        new_save = """def save_model(model, optimizer, args, epoch, save_file, classifier,
               scaler=None, best_acc=None, best_model=None):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'classifier': classifier.state_dict(),
        'best_acc': best_acc,
        'best_model': best_model,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    torch.save(state, save_file)
    del state
"""
        misc = _replace_once(misc, old_save, new_save, f"{view} checkpoint state")
        misc_path.write_text(misc)
        patched_files.append(f"{view}/util/misc.py")

        main_path = source / view / "main.py"
        main = main_path.read_text()
        old_resume = """    if args.resume:
        if os.path.isfile(args.resume):
            print(\"=> loading checkpoint '{}'\".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']\x20
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print(\"=> loaded checkpoint '{}' (epoch {})\".format(args.resume, checkpoint['epoch']))
        else:
            print(\"=> no checkpoint found at '{}'\".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
"""
        new_resume = """    # Compatibility patch: restore every trainable/evaluation state.
    scaler = torch.cuda.amp.GradScaler()
    if args.resume:
        if os.path.isfile(args.resume):
            print(\"=> loading checkpoint '{}'\".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint.get('scaler') is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            best_acc = checkpoint.get('best_acc') or best_acc
            if checkpoint.get('best_model') is None:
                raise ValueError('resume checkpoint lacks historical best_model')
            best_model = checkpoint['best_model']
            print(\"=> loaded checkpoint '{}' (epoch {})\".format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError(\"no checkpoint found at '{}'\".format(args.resume))
    else:
        args.start_epoch = 1
"""
        main = _replace_once(main, old_resume, new_resume, f"{view} safe resume")
        old_call = "save_model(model, optimizer, args, epoch, save_file, classifier)"
        new_call = "save_model(model, optimizer, args, epoch, save_file, classifier, scaler=scaler, best_acc=best_acc, best_model=best_model)"
        if main.count(old_call) != 3:
            raise ValueError(f"MVST {view} save call count={main.count(old_call)}")
        main = main.replace(old_call, new_call)
        validation_anchor = """            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
<SOURCE_INDENTED_BLANK>
            # save a checkpoint of model and classifier when the best score is updated
""".replace("<SOURCE_INDENTED_BLANK>", " " * 12)
        validation_replacement = """            best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)

            # Compatibility patch: one rolling complete checkpoint avoids 50 large epoch files.
            save_model(model, optimizer, args, epoch, os.path.join(args.save_folder, 'last.pth'), classifier, scaler=scaler, best_acc=best_acc, best_model=best_model)
<SOURCE_INDENTED_BLANK>
            # save a checkpoint of model and classifier when the best score is updated
""".replace("<SOURCE_INDENTED_BLANK>", " " * 12)
        main = _replace_once(
            main, validation_anchor, validation_replacement, f"{view} rolling resume checkpoint"
        )
        main_path.write_text(main)
        install_author_checkpoint_writer(misc_path)
        install_author_training_retention(
            main_path,
            "save_model(model, optimizer, args, epoch, os.path.join(args.save_folder, 'last.pth'), classifier, scaler=scaler, best_acc=best_acc, best_model=best_model)",
        )
        patched_files.append(f"{view}/main.py")

    command = ["git", "diff", "--no-ext-diff", "--", *patched_files]
    diff = subprocess.run(command, cwd=source, check=True, stdout=subprocess.PIPE).stdout
    if not diff:
        raise ValueError("MVST compatibility patch produced an empty diff")
    return {
        "scope": "complete interruption-safe state and bounded best-plus-last retention for each author encoder",
        "files": patched_files,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_size_bytes": len(diff),
        "semantic_invariants": [
            "author random file split, model, preprocessing, loss, optimizer, test selection, and metric unchanged",
            "resume restores model, classifier, optimizer, scaler, epoch, best score, and historical best-model buffer from one rolling last.pth",
            "each encoder retains its independently author-selected best.pth and atomically replaced last.pth with SHA256 receipts",
            "cycle IDs are added only by the maintained extraction path before fusion",
        ],
    }


def validate_roots(project_root: Path, result_root: Path, cache_root: Path | None) -> tuple[Path, Path]:
    canonical = (project_root / "result").resolve()
    result_root = result_root.resolve()
    if result_root.parent != canonical or not RUN_ROOT_PATTERN.fullmatch(result_root.name):
        raise ValueError(f"result root must match {canonical}/mvst_YYYYMMDD_HHMMSS")
    cache_root = (cache_root or result_root / "cache").resolve()
    try:
        cache_root.relative_to(result_root)
    except ValueError as error:
        raise ValueError("cache root must be inside the timestamped result root") from error
    return result_root, cache_root


def author_split_receipt(manifest: Path) -> dict:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recordings = sorted({row["recording_id"] for row in rows})
    indices = list(range(len(recordings)))
    random.Random(1).shuffle(indices)
    boundary = int(len(indices) * 0.6)
    split_recordings = {
        "train": {recordings[index] for index in indices[:boundary]},
        "test": {recordings[index] for index in indices[boundary:]},
    }
    result = {
        "algorithm": "sorted recording IDs; random.Random(1).shuffle(indices); first int(920*0.6) train",
        "recording_counts": {split: len(ids) for split, ids in split_recordings.items()},
        "cycle_counts": {},
        "label_counts": {},
        "ordered_cycle_id_sha256": {},
    }
    for split, ids in split_recordings.items():
        selected = sorted(
            (row for row in rows if row["recording_id"] in ids),
            key=lambda row: (row["recording_id"], int(row["cycle_id"].rsplit("_", 1)[-1])),
        )
        result["cycle_counts"][split] = len(selected)
        result["label_counts"][split] = {
            label: sum(row["native_four_class_label"] == label for row in selected)
            for label in ["normal", "crackle", "wheeze", "both"]
        }
        joined = "\n".join(row["cycle_id"] for row in selected).encode()
        result["ordered_cycle_id_sha256"][split] = hashlib.sha256(joined).hexdigest()
    if result["recording_counts"] != {"train": 552, "test": 368}:
        raise ValueError(f"MVST author recording split mismatch: {result}")
    if result["cycle_counts"] != {"train": 4213, "test": 2685}:
        raise ValueError(f"MVST author cycle split mismatch: {result}")
    return result


def build_adapter(audio_dir: Path, checkpoint: Path, root: Path) -> dict:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    ensure_symlink(data / "icbhi_dataset", audio_dir)
    pretrained = root / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    ensure_symlink(pretrained / SPEC.checkpoint_filename, checkpoint)
    wav_count = len(list(audio_dir.glob("*.wav")))
    annotation_count = len([
        path for path in audio_dir.glob("*.txt") if len(path.stem.split("_")) >= 5
    ])
    if wav_count != 920 or annotation_count != 920:
        raise ValueError("MVST raw adapter must expose 920 WAV and annotation files")
    return {
        "portable_run": str(root.resolve()),
        "raw_audio_target": str(audio_dir.resolve()),
        "raw_policy": "read-only directory symlink; author sorted/random(1) loader preserved",
        "wav_count": wav_count,
        "annotation_count": annotation_count,
        "checkpoint_target": str(checkpoint.resolve()),
    }


def bootstrap(
    project_root: Path,
    dataset_root: Path,
    result_root: Path,
    cache_root: Path | None,
    checkpoint_path: Path | None,
    checkpoint_url: str | None,
    device: str,
) -> dict:
    project_root = project_root.resolve()
    dataset_root = dataset_root.resolve()
    result_root, cache_root = validate_roots(project_root, result_root, cache_root)
    environment_gate = prepare_result_root_for_bootstrap(result_root, METHOD)
    cache_root.mkdir(parents=True, exist_ok=True)
    source = result_root / "source" / "repo"
    source_receipt = clone_source(SPEC, source)
    compatibility = install_compatibility_patch(source)
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    data = build_manifest(dataset_root, manifest)
    data["author_repo_split"] = author_split_receipt(manifest)
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    checkpoint_receipt = materialize_checkpoint(
        SPEC, checkpoint, checkpoint_path, checkpoint_url
    )
    adapter = build_adapter(discover_audio_dir(dataset_root), checkpoint, result_root / "portable_run")
    contract = project_root / "codex" / "2026-07-21" / "paper_contracts" / "mvst.json"
    environment_spec = project_root / "baseline" / "mvst" / "environment.linux-cu118.yml"
    receipt = {
        "status": "fresh_checkout_bootstrap_ready",
        "created_at": chicago_timestamp(),
        "timezone": "America/Chicago",
        "method": METHOD,
        "minimum_compatible_commit": MINIMUM_COMMIT,
        "classification": "author-code fixed-random-file-split reproduction; not official ICBHI split",
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "result_root": str(result_root),
        "cache_root": str(cache_root),
        "device_requested": device,
        "source": source_receipt,
        "compatibility_patch": compatibility,
        "data": data,
        "checkpoint": checkpoint_receipt,
        "adapter": adapter,
        "environment_gate": {
            "path": str(environment_gate), "sha256": sha256(environment_gate),
        },
        "paper_contract": {"path": str(contract), "sha256": sha256(contract)},
        "environment_spec": {
            "name": ENVIRONMENT_NAME,
            "path": str(environment_spec),
            "project_relative_path": str(environment_spec.relative_to(project_root)),
            "sha256": sha256(environment_spec),
        },
        "environment": {
            "HF_HOME": str(cache_root / "huggingface"),
            "TORCH_HOME": str(cache_root / "torch"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"),
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
        },
    }
    receipts = result_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "bootstrap_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def verify_bootstrap(result_root: Path) -> dict:
    result_root = result_root.resolve()
    receipt = json.loads((result_root / "receipts" / "bootstrap_receipt.json").read_text())
    source = result_root / "source" / "repo"
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    patched_files = receipt["compatibility_patch"]["files"]
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", *patched_files], cwd=source,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if receipt.get("method") != METHOD or head != SPEC.commit:
        errors.append("method_or_source_commit")
    if hashlib.sha256(diff).hexdigest() != receipt["compatibility_patch"]["diff_sha256"]:
        errors.append("compatibility_patch_diff")
    if sha256(checkpoint) != SPEC.checkpoint_sha256:
        errors.append("checkpoint_sha256")
    if sha256(manifest) != receipt["data"]["manifest_sha256"]:
        errors.append("manifest_sha256")
    if len(rows) != 6898 or len({row["cycle_id"] for row in rows}) != 6898:
        errors.append("manifest_rows_or_ids")
    if receipt["data"]["author_repo_split"]["cycle_counts"] != {"train": 4213, "test": 2685}:
        errors.append("author_split_counts")
    environment_spec = Path(receipt["environment_spec"]["path"])
    if receipt.get("minimum_compatible_commit") != MINIMUM_COMMIT:
        errors.append("minimum_compatible_commit")
    if receipt["environment_spec"].get("name") != ENVIRONMENT_NAME:
        errors.append("environment_name")
    if not environment_spec.is_file() or sha256(environment_spec) != receipt["environment_spec"]["sha256"]:
        errors.append("environment_spec_sha256")
    environment_gate = Path(receipt["environment_gate"]["path"])
    if not environment_gate.is_file() or sha256(environment_gate) != receipt["environment_gate"]["sha256"]:
        errors.append("environment_gate_sha256")
    portable = result_root / "portable_run"
    if not (portable / "data" / "icbhi_dataset").is_symlink():
        errors.append("data_adapter")
    if not (portable / "pretrained_models" / SPEC.checkpoint_filename).is_symlink():
        errors.append("checkpoint_adapter")
    result = {
        "status": "verified" if not errors else "failed",
        "method": METHOD,
        "minimum_compatible_commit": MINIMUM_COMMIT,
        "environment_name": ENVIRONMENT_NAME,
        "result_root": str(result_root),
        "errors": errors,
        "cycles": len(rows),
        "unique_cycle_ids": len({row["cycle_id"] for row in rows}),
        "author_split_cycle_counts": receipt["data"]["author_repo_split"]["cycle_counts"],
        "source_commit": head,
        "checkpoint_sha256": sha256(checkpoint),
    }
    (result_root / "receipts" / "bootstrap_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if errors:
        raise ValueError(f"MVST bootstrap verification failed: {errors}")
    return result
