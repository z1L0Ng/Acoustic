"""Fresh-checkout bootstrap for two explicitly bounded ADD-RSC tracks."""

from __future__ import annotations

import csv
import hashlib
import json
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


METHOD = "add_rsc"
MINIMUM_COMMIT = "4172524a0e5d7b792de248820439f30874e2ae6d"
ENVIRONMENT_NAME = "acoustic-addrsc-r4"
TRACKS = ["paper_declared_reconstruction", "author_repo_default_official_like"]
SPEC = MethodSpec(
    method=METHOD,
    repo_url="https://github.com/deegy666/ADD-RSC.git",
    commit="e2b0f8213cb7ca451ef28757cb1329e17469fe72",
    checkpoint_filename="audioset_10_10_0.4593.pth",
    checkpoint_url="https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1",
    checkpoint_sha256="dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f",
    checkpoint_provenance=(
        "current AST author-runtime compatibility artifact; ADD-RSC only links an HF model and publishes no task checkpoint"
    ),
    support_subdir="data",
)
RUN_ROOT_PATTERN = re.compile(r"^add_rsc_\d{8}_\d{6}$")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"ADD-RSC patch anchor {label!r} count={text.count(old)}")
    return text.replace(old, new)


def install_compatibility_patch(source: Path) -> dict:
    """Add explicit split tracks, device portability, and complete resume state."""
    model_path = source / "models" / "adapt_diff_denoise.py"
    model_text = model_path.read_text()
    model_text = _replace_once(
        model_text,
        "nn.LayerNorm(norm_shape, elementwise_affine=True).to('cuda:0')",
        "nn.LayerNorm(norm_shape, elementwise_affine=True).to(attention.device)",
        "device-safe attention LayerNorm",
    )
    model_path.write_text(model_text)

    dataset_path = source / "util" / "icbhi_dataset.py"
    dataset = dataset_path.read_text()
    old_filenames = """        filenames = os.listdir(data_folder)
        filenames =set([f.strip().split('.')[0] for f in filenames if '.wav' in f or '.txt' in f])
        filenames = sorted(filenames)

        patient_dict = {}
        indices = [i for i, file in enumerate(filenames)]
        random.Random(1).shuffle(indices)
        train_size = int(len(indices) * 0.6)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        train_files = [filenames[i] for i in train_idx]
        test_files = [filenames[i] for i in test_idx]
        for f in train_files:
            if train_flag:
                patient_dict[f] = 'train'
        for f in test_files:
            if not train_flag:
                patient_dict[f] = 'test'
"""
    new_filenames = """        # Compatibility adapter: WAV names are the authoritative recording list.
        filenames = sorted(f.rsplit('.', 1)[0] for f in os.listdir(data_folder) if f.endswith('.wav'))

        patient_dict = {}
        if args.split_protocol == 'paper_declared_official':
            split_rows = [line.split() for line in open(os.path.join(data_folder, 'official_split.txt')).read().splitlines() if line.strip()]
            split_by_id = {recording_id: split for recording_id, split in split_rows}
            stable_split = {'_'.join(recording_id.split('_')[:4]): split for recording_id, split in split_by_id.items()}
            for recording_id in filenames:
                split = split_by_id.get(recording_id, stable_split.get('_'.join(recording_id.split('_')[:4])))
                if split is None:
                    raise ValueError('recording absent from official split: {}'.format(recording_id))
                if (train_flag and split == 'train') or (not train_flag and split == 'test'):
                    patient_dict[recording_id] = split
        else:
            indices = list(range(len(filenames)))
            random.Random(1).shuffle(indices)
            train_size = int(len(indices) * 0.6)
            selected = indices[:train_size] if train_flag else indices[train_size:]
            for index in selected:
                patient_dict[filenames[index]] = 'train' if train_flag else 'test'
"""
    dataset = _replace_once(dataset, old_filenames, new_filenames, "explicit split protocols")
    dataset_path.write_text(dataset)

    misc_path = source / "util" / "misc.py"
    misc = misc_path.read_text()
    old_save = """def save_model(model, bias_denoise_encoder, optimizer, args, epoch, save_file, classifier):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'bias_denoise_encoder': bias_denoise_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'classifier': classifier.state_dict()
    }

    torch.save(state, save_file)
    del state
"""
    new_save = """def save_model(model, bias_denoise_encoder, optimizer, args, epoch, save_file,
               classifier, best_acc=None):
    print('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'bias_denoise_encoder': bias_denoise_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'classifier': classifier.state_dict(),
        'best_acc': best_acc,
    }

    torch.save(state, save_file)
    del state
"""
    misc = _replace_once(misc, old_save, new_save, "complete checkpoint state")
    misc_path.write_text(misc)

    main_path = source / "main.py"
    main = main_path.read_text()
    parser_anchor = """    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
"""
    parser_replacement = parser_anchor + """    parser.add_argument('--split_protocol', type=str, default='author_repo_random_file',
                        choices=['author_repo_random_file', 'paper_declared_official'],
                        help='explicit bounded execution track; the repo default remains random file split')
"""
    main = _replace_once(main, parser_anchor, parser_replacement, "split protocol CLI")
    old_resume = """    if args.resume:
        if os.path.isfile(args.resume):
            print(\"=> loading checkpoint '{}'\".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']\x20
            bias_denoise_encoder.load_state_dict(checkpoint['bias_denoise_encoder'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print(\"=> loaded checkpoint '{}' (epoch {})\".format(args.resume, checkpoint['epoch']))
        else:
            print(\"=> no checkpoint found at '{}'\".format(args.resume))
    else:
        args.start_epoch = 1
"""
    new_resume = """    if args.resume:
        if os.path.isfile(args.resume):
            print(\"=> loading checkpoint '{}'\".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            bias_denoise_encoder.load_state_dict(checkpoint['bias_denoise_encoder'])
            model.load_state_dict(checkpoint['model'])
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint.get('best_acc') or best_acc
            best_model = [deepcopy(model.state_dict()), deepcopy(bias_denoise_encoder.state_dict()), deepcopy(classifier.state_dict())]
            print(\"=> loaded checkpoint '{}' (epoch {})\".format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError(\"no checkpoint found at '{}'\".format(args.resume))
    else:
        args.start_epoch = 1
"""
    main = _replace_once(main, old_resume, new_resume, "safe resume block")
    old_call = "save_model(model, bias_denoise_encoder, optimizer, args, epoch, save_file, classifier)"
    new_call = "save_model(model, bias_denoise_encoder, optimizer, args, epoch, save_file, classifier, best_acc=best_acc)"
    if main.count(old_call) != 3:
        raise ValueError(f"ADD-RSC save call count={main.count(old_call)}")
    main = main.replace(old_call, new_call)
    validation_anchor = """            best_acc, best_model, save_bool = validate(val_loader, model, bias_denoise_encoder, classifier, criterion, args, best_acc, best_model)
<SOURCE_INDENTED_BLANK>
            # save a checkpoint of model and classifier when the best score is updated
""".replace("<SOURCE_INDENTED_BLANK>", " " * 12)
    validation_replacement = """            best_acc, best_model, save_bool = validate(val_loader, model, bias_denoise_encoder, classifier, criterion, args, best_acc, best_model)

            # Compatibility patch: one rolling complete resume checkpoint.
            save_model(model, bias_denoise_encoder, optimizer, args, epoch, os.path.join(args.save_folder, 'last.pth'), classifier, best_acc=best_acc)
<SOURCE_INDENTED_BLANK>
            # save a checkpoint of model and classifier when the best score is updated
""".replace("<SOURCE_INDENTED_BLANK>", " " * 12)
    main = _replace_once(main, validation_anchor, validation_replacement, "rolling resume checkpoint")
    main_path.write_text(main)

    files = ["main.py", "util/icbhi_dataset.py", "util/misc.py", "models/adapt_diff_denoise.py"]
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", *files], cwd=source,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if not diff:
        raise ValueError("ADD-RSC compatibility patch produced an empty diff")
    return {
        "scope": "explicit dual split tracks, device portability, complete rolling resume",
        "files": files,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_size_bytes": len(diff),
        "semantic_boundaries": [
            "author_repo_default_official_like preserves the sorted random.Random(1) split",
            "paper_declared_reconstruction intentionally changes split, n_mels, and weight decay per paper and is not author-repo faithful",
            "model, ADD loss algebra, alpha=0.02 scale, epsilon=0.2 smoothing, and test-selected metric remain unchanged",
        ],
    }


def validate_roots(project_root: Path, result_root: Path, cache_root: Path | None) -> tuple[Path, Path]:
    canonical = (project_root / "result").resolve()
    result_root = result_root.resolve()
    if result_root.parent != canonical or not RUN_ROOT_PATTERN.fullmatch(result_root.name):
        raise ValueError(f"result root must match {canonical}/add_rsc_YYYYMMDD_HHMMSS")
    cache_root = (cache_root or result_root / "cache").resolve()
    try:
        cache_root.relative_to(result_root)
    except ValueError as error:
        raise ValueError("cache root must be inside the timestamped result root") from error
    return result_root, cache_root


def repo_random_split(manifest: Path) -> dict:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    recordings = sorted({row["recording_id"] for row in rows})
    indices = list(range(len(recordings)))
    random.Random(1).shuffle(indices)
    boundary = int(len(indices) * 0.6)
    selected = {
        "train": {recordings[index] for index in indices[:boundary]},
        "test": {recordings[index] for index in indices[boundary:]},
    }
    receipt = {"recording_counts": {}, "cycle_counts": {}, "ordered_cycle_id_sha256": {}}
    for split, ids in selected.items():
        ordered = sorted(
            (row for row in rows if row["recording_id"] in ids),
            key=lambda row: (row["recording_id"], int(row["cycle_id"].rsplit("_", 1)[-1])),
        )
        receipt["recording_counts"][split] = len(ids)
        receipt["cycle_counts"][split] = len(ordered)
        receipt["ordered_cycle_id_sha256"][split] = hashlib.sha256(
            "\n".join(row["cycle_id"] for row in ordered).encode()
        ).hexdigest()
    if receipt["recording_counts"] != {"train": 552, "test": 368} or receipt["cycle_counts"] != {"train": 4213, "test": 2685}:
        raise ValueError(f"ADD-RSC repo split mismatch: {receipt}")
    receipt["algorithm"] = "sorted recording IDs; random.Random(1); first 60% train"
    return receipt


def build_adapter(dataset_root: Path, audio_dir: Path, checkpoint: Path, root: Path) -> dict:
    target = root / "data" / "icbhi_dataset"
    target.mkdir(parents=True, exist_ok=True)
    for wav in sorted(audio_dir.glob("*.wav")):
        ensure_symlink(target / wav.name, wav)
        ensure_symlink(target / wav.with_suffix(".txt").name, wav.with_suffix(".txt"))
    split_candidates = list(dataset_root.rglob("ICBHI_challenge_train_test.txt"))
    if len(split_candidates) != 1:
        raise ValueError(f"expected one official split file, found {split_candidates}")
    ensure_symlink(target / "official_split.txt", split_candidates[0])
    pretrained = root / "pretrained_models"
    pretrained.mkdir(parents=True, exist_ok=True)
    ensure_symlink(pretrained / SPEC.checkpoint_filename, checkpoint)
    return {
        "portable_run": str(root.resolve()),
        "raw_audio_target": str(audio_dir.resolve()),
        "raw_policy": "920 read-only WAV/TXT symlinks plus read-only official split symlink",
        "wav_count": len(list(target.glob("*.wav"))),
        "cycle_annotation_count": len([p for p in target.glob("*.txt") if len(p.stem.split("_")) >= 5]),
        "official_split": str((target / "official_split.txt").resolve()),
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
    data["repo_random_split"] = repo_random_split(manifest)
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    checkpoint_receipt = materialize_checkpoint(
        SPEC, checkpoint, checkpoint_path, checkpoint_url
    )
    adapter = build_adapter(
        dataset_root, discover_audio_dir(dataset_root), checkpoint, result_root / "portable_run"
    )
    contract = project_root / "codex" / "2026-07-21" / "paper_contracts" / "add_rsc.json"
    environment_spec = project_root / "baseline" / "add_rsc" / "environment.linux-cu121.yml"
    protocols = {
        "paper_declared_reconstruction": {
            "split_protocol": "paper_declared_official", "n_mels": 64,
            "weight_decay": 0.1, "expected_test_cycles": 2756,
            "classification": "bounded paper-declared reconstruction; not official faithful",
        },
        "author_repo_default_official_like": {
            "split_protocol": "author_repo_random_file", "n_mels": 128,
            "weight_decay": 1e-6, "expected_test_cycles": 2685,
            "classification": "author-code default execution; target 65.53 not directly comparable",
        },
    }
    receipt = {
        "status": "fresh_checkout_bootstrap_ready",
        "created_at": chicago_timestamp(), "timezone": "America/Chicago",
        "method": METHOD, "minimum_compatible_commit": MINIMUM_COMMIT,
        "classification": "no unique faithful protocol; two bounded tracks only",
        "project_root": str(project_root), "dataset_root": str(dataset_root),
        "result_root": str(result_root), "cache_root": str(cache_root),
        "device_requested": device, "source": source_receipt,
        "compatibility_patch": compatibility, "data": data,
        "checkpoint": checkpoint_receipt, "adapter": adapter, "protocols": protocols,
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
            "HF_HOME": str(cache_root / "huggingface"), "TORCH_HOME": str(cache_root / "torch"),
            "NUMBA_CACHE_DIR": str(cache_root / "numba"), "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
        },
    }
    receipts = result_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "bootstrap_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def verify_bootstrap(result_root: Path) -> dict:
    result_root = result_root.resolve()
    receipt = json.loads((result_root / "receipts" / "bootstrap_receipt.json").read_text())
    source = result_root / "source" / "repo"
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    files = receipt["compatibility_patch"]["files"]
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", *files], cwd=source,
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
    if receipt["data"]["repo_random_split"]["cycle_counts"] != {"train": 4213, "test": 2685}:
        errors.append("repo_random_split")
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
    adapter = result_root / "portable_run" / "data" / "icbhi_dataset"
    if len(list(adapter.glob("*.wav"))) != 920 or not (adapter / "official_split.txt").is_symlink():
        errors.append("data_adapter")
    result = {
        "status": "verified" if not errors else "failed", "method": METHOD,
        "minimum_compatible_commit": MINIMUM_COMMIT, "result_root": str(result_root),
        "environment_name": ENVIRONMENT_NAME,
        "errors": errors, "cycles": len(rows),
        "unique_cycle_ids": len({row["cycle_id"] for row in rows}),
        "official_test_cycles": 2756, "author_repo_test_cycles": 2685,
        "source_commit": head, "checkpoint_sha256": sha256(checkpoint),
    }
    (result_root / "receipts" / "bootstrap_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if errors:
        raise ValueError(f"ADD-RSC bootstrap verification failed: {errors}")
    return result
