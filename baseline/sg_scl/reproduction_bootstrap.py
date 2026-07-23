"""Fresh-checkout bootstrap for the SG-SCL server package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from baseline.common.official_reproduction_bootstrap import (
    MethodSpec,
    build_adapter,
    build_manifest,
    chicago_timestamp,
    clone_source,
    materialize_checkpoint,
    prepare_result_root_for_bootstrap,
    sha256,
)
from baseline.common.storage_safe_retention import (
    install_author_checkpoint_writer,
    install_author_training_retention,
)


METHOD = "sg_scl"
MINIMUM_COMMIT = "4172524a0e5d7b792de248820439f30874e2ae6d"
ENVIRONMENT_NAME = "acoustic-sgscl-r4"
SPEC = MethodSpec(
    method=METHOD,
    repo_url="https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning.git",
    commit="66564609595090b61540595d3d27764c00553086",
    checkpoint_filename="audioset_10_10_0.4593.pth",
    checkpoint_url="https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1",
    checkpoint_sha256="dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f",
    checkpoint_provenance=(
        "current author-hosted runtime artifact; historical serialized-byte identity unresolved"
    ),
    support_subdir="data/icbhi_dataset",
)
RUN_ROOT_PATTERN = re.compile(r"^sg_scl_\d{8}_\d{6}$")
DEVICES = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"SG-SCL patch anchor {label!r} count={text.count(old)}")
    return text.replace(old, new)


def install_compatibility_patch(source: Path) -> dict:
    """Apply device portability, safe resume, and checkpoint receipt changes."""
    mcl_path = source / "method" / "mcl.py"
    mcl = mcl_path.read_text()
    mcl = _replace_once(
        mcl,
        "mask = torch.eye(batch_size, dtype=torch.float32).cuda()",
        "mask = torch.eye(batch_size, dtype=torch.float32, device=projection1.device)",
        "MetaCL identity mask device",
    )
    mcl = _replace_once(
        mcl,
        "mask = torch.eq(meta_labels, meta_labels.T).float().cuda()",
        "mask = torch.eq(meta_labels, meta_labels.T).float().to(projection1.device)",
        "MetaCL label mask device",
    )
    mcl = _replace_once(
        mcl,
        "torch.arange(batch_size * contrast_count).view(-1, 1).cuda()",
        "torch.arange(batch_size * contrast_count, device=projection1.device).view(-1, 1)",
        "MetaCL diagonal mask device",
    )
    mcl_path.write_text(mcl)

    misc_path = source / "util" / "misc.py"
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
               projector=None, scaler=None, best_acc=None, best_model=None):
    print('==> Saving...')
    class_classifier = classifier[0] if isinstance(classifier, (list, tuple)) else classifier
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'classifier': class_classifier.state_dict(),
        'best_acc': best_acc,
        'best_model': best_model,
    }
    if isinstance(classifier, (list, tuple)) and len(classifier) > 1:
        state['domain_classifier'] = classifier[1].state_dict()
    if projector is not None:
        state['projector'] = projector.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()

    torch.save(state, save_file)
    del state
"""
    misc = _replace_once(misc, old_save, new_save, "complete checkpoint state")
    misc_path.write_text(misc)

    main_path = source / "main.py"
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
    new_resume = """    # Compatibility patch: preserve all trainable state for interruption-safe resume.
    scaler = torch.cuda.amp.GradScaler()
    if args.resume:
        if os.path.isfile(args.resume):
            print(\"=> loading checkpoint '{}'\".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            classifier[0].load_state_dict(checkpoint['classifier']) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(checkpoint['classifier'])
            if args.domain_adaptation or args.domain_adaptation2:
                classifier[1].load_state_dict(checkpoint['domain_classifier'])
            projector.load_state_dict(checkpoint['projector'])
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
    main = _replace_once(main, old_resume, new_resume, "safe resume block")
    old_call = "save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)"
    new_call = "save_model(model, optimizer, args, epoch, save_file, classifier, projector=projector, scaler=scaler, best_acc=best_acc, best_model=best_model)"
    if main.count(old_call) != 3:
        raise ValueError(f"SG-SCL save call count={main.count(old_call)}")
    main = main.replace(old_call, new_call)
    main_path.write_text(main)
    install_author_checkpoint_writer(misc_path)
    install_author_training_retention(
        main_path,
        "save_model(model, optimizer, args, epoch, os.path.join(args.save_folder, 'last.pth'), classifier, projector=projector, scaler=scaler, best_acc=best_acc, best_model=best_model)",
    )

    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", "main.py", "method/mcl.py", "util/misc.py"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not diff:
        raise ValueError("SG-SCL compatibility patch produced an empty diff")
    return {
        "scope": "non-semantic device allocation, complete checkpoint state, safe resume, and bounded best-plus-last retention",
        "files": ["main.py", "method/mcl.py", "util/misc.py"],
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_size_bytes": len(diff),
        "semantic_invariants": [
            "model, loss algebra, preprocessing, split, hyperparameters, test selection, and metric unchanged",
            "MetaCL tensors are allocated on the input tensor device instead of hard-coded CUDA",
            "resume restores model, both classifiers, projector, optimizer, scaler, epoch, best score, and historical best-model buffer",
            "the original save_bool condition selects best.pth; last.pth is atomically replaced each epoch with a SHA256 receipt",
        ],
    }


def validate_roots(project_root: Path, result_root: Path, cache_root: Path | None) -> tuple[Path, Path]:
    canonical = (project_root / "result").resolve()
    result_root = result_root.resolve()
    if result_root.parent != canonical or not RUN_ROOT_PATTERN.fullmatch(result_root.name):
        raise ValueError(f"result root must match {canonical}/sg_scl_YYYYMMDD_HHMMSS")
    cache_root = (cache_root or result_root / "cache").resolve()
    try:
        cache_root.relative_to(result_root)
    except ValueError as error:
        raise ValueError("cache root must be inside the timestamped result root") from error
    return result_root, cache_root


def device_receipt(manifest: Path) -> dict:
    cycle_counts = {split: {device: 0 for device in DEVICES} for split in ["train", "test"]}
    recordings = {split: {device: set() for device in DEVICES} for split in ["train", "test"]}
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        device = Path(row["audio_path"]).stem.split("_")[-1]
        if device not in DEVICES:
            raise ValueError(f"unmapped SG-SCL device: {device}")
        split = row["official_split"]
        cycle_counts[split][device] += 1
        recordings[split][device].add(row["recording_id"])
    recording_counts = {
        split: {device: len(ids) for device, ids in values.items() if ids}
        for split, values in recordings.items()
    }
    cycle_counts = {
        split: {device: count for device, count in values.items() if count}
        for split, values in cycle_counts.items()
    }
    return {
        "extraction": "final underscore-delimited recording filename token; identical to author loader",
        "mapping": {device: index for index, device in enumerate(DEVICES)},
        "unknown_cycles": 0,
        "cycle_counts_by_split_and_device": cycle_counts,
        "recording_counts_by_split_and_device": recording_counts,
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
    data["device_receipt"] = device_receipt(manifest)
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    checkpoint_receipt = materialize_checkpoint(
        SPEC, checkpoint, checkpoint_path, checkpoint_url
    )
    adapter = build_adapter(
        SPEC, source, Path(data["audio_dir"]), checkpoint, result_root / "portable_run"
    )

    contract = project_root / "baseline" / "sg_scl" / "paper_contract.json"
    environment_spec = project_root / "baseline" / "sg_scl" / "environment.linux-cu118.yml"
    receipt = {
        "status": "fresh_checkout_bootstrap_ready",
        "created_at": chicago_timestamp(),
        "timezone": "America/Chicago",
        "method": METHOD,
        "minimum_compatible_commit": MINIMUM_COMMIT,
        "classification": "metadata-aware official-like reproduction; official-test-selected; not audio-only",
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
    checkpoint = result_root / "checkpoints" / SPEC.checkpoint_filename
    manifest = result_root / "manifest" / "icbhi_2017_cycle_manifest.csv"
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", "main.py", "method/mcl.py", "util/misc.py"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
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
    if receipt["data"]["device_receipt"]["unknown_cycles"] != 0:
        errors.append("device_mapping")
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
    if len(list((portable / "data" / "icbhi_dataset" / "audio_test_data").glob("*.wav"))) != 920:
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
        "unknown_device_cycles": receipt["data"]["device_receipt"]["unknown_cycles"],
        "source_commit": head,
        "checkpoint_sha256": sha256(checkpoint),
    }
    (result_root / "receipts" / "bootstrap_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if errors:
        raise ValueError(f"SG-SCL bootstrap verification failed: {errors}")
    return result
