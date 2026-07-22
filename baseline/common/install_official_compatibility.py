"""Install receipted, non-semantic save/resume fixes in pinned author sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def replace_exact(path: Path, old: str, new: str, expected_count: int = 1) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != expected_count:
        raise ValueError(f"unexpected patch context in {path}: expected={expected_count} observed={count}")
    path.write_text(text.replace(old, new))


def install_patch_mix(source: Path) -> None:
    misc = source / "util" / "misc.py"
    main = source / "main.py"
    replace_exact(
        misc,
        "def save_model(model, optimizer, args, epoch, save_file, classifier):",
        "def save_model(model, optimizer, args, epoch, save_file, classifier, projector=None, best_acc=None, best_model=None):",
    )
    replace_exact(
        misc,
        "        'classifier': classifier.state_dict()\n",
        "        'classifier': classifier.state_dict(),\n"
        "        'projector': projector.state_dict() if projector is not None else None,\n"
        "        'best_acc': best_acc,\n"
        "        'best_model': best_model\n",
    )
    replace_exact(
        main,
        "                save_model(model, optimizer, args, epoch, save_file, classifier)",
        "                save_model(model, optimizer, args, epoch, save_file, classifier, projector, best_acc, best_model)",
        expected_count=2,
    )
    replace_exact(
        main,
        "        save_model(model, optimizer, args, epoch, save_file, classifier)",
        "        save_model(model, optimizer, args, epoch, save_file, classifier, projector, best_acc, best_model)",
    )
    replace_exact(
        main,
        "            model.load_state_dict(checkpoint['model'])\n"
        "            optimizer.load_state_dict(checkpoint['optimizer'])\n"
        "            args.start_epoch += 1",
        "            model.load_state_dict(checkpoint['model'])\n"
        "            classifier.load_state_dict(checkpoint['classifier'])\n"
        "            if checkpoint.get('projector') is None:\n"
        "                raise ValueError('resume checkpoint lacks projector state')\n"
        "            projector.load_state_dict(checkpoint['projector'])\n"
        "            optimizer.load_state_dict(checkpoint['optimizer'])\n"
        "            if checkpoint.get('best_acc') is None:\n"
        "                raise ValueError('resume checkpoint lacks historical best_acc')\n"
        "            best_acc = checkpoint['best_acc']\n"
        "            if checkpoint.get('best_model') is None:\n"
        "                raise ValueError('resume checkpoint lacks historical best_model')\n"
        "            best_model = checkpoint['best_model']\n"
        "            args.start_epoch += 1",
    )


def install_pafa(source: Path) -> None:
    misc = source / "util" / "misc.py"
    main = source / "main.py"
    replace_exact(
        misc,
        "def save_model(model, optimizer, args, epoch, save_file, classifiers, projector=None):",
        "def save_model(model, optimizer, args, epoch, save_file, classifiers, projector=None, best_acc=None, best_model=None):",
    )
    replace_exact(
        misc,
        "    if args.method == 'pccl':",
        "    if args.method in ['pccl', 'pafa']:",
    )
    replace_exact(
        misc,
        "            'optimizer': optimizer.state_dict(),\n            'args': args",
        "            'optimizer': optimizer.state_dict(),\n"
        "            'args': args,\n"
        "            'best_acc': best_acc,\n"
        "            'best_model': best_model",
        expected_count=2,
    )
    replace_exact(
        main,
        "                save_model(model, optimizer, args, epoch, save_file,  classifier, projector)",
        "                save_model(model, optimizer, args, epoch, save_file, classifier, projector, best_acc, best_model)",
    )
    replace_exact(
        main,
        "                save_model(model, optimizer, args, epoch, save_file, classifier, projector)",
        "                save_model(model, optimizer, args, epoch, save_file, classifier, projector, best_acc, best_model)",
    )
    replace_exact(
        main,
        "        save_model(model, optimizer, args, epoch, save_file, classifier, projector)",
        "        save_model(model, optimizer, args, epoch, save_file, classifier, projector, best_acc, best_model)",
    )
    replace_exact(
        main,
        "            model.load_state_dict(checkpoint['model'])\n"
        "            optimizer.load_state_dict(checkpoint['optimizer'])\n"
        "            args.start_epoch += 1",
        "            model.load_state_dict(checkpoint['model'])\n"
        "            classifier.load_state_dict(checkpoint['classifier'])\n"
        "            if checkpoint.get('projector') is None:\n"
        "                raise ValueError('resume checkpoint lacks projector state')\n"
        "            projector.load_state_dict(checkpoint['projector'])\n"
        "            optimizer.load_state_dict(checkpoint['optimizer'])\n"
        "            if checkpoint.get('best_acc') is None:\n"
        "                raise ValueError('resume checkpoint lacks historical best_acc')\n"
        "            best_acc = checkpoint['best_acc']\n"
        "            if checkpoint.get('best_model') is None:\n"
        "                raise ValueError('resume checkpoint lacks historical best_model')\n"
        "            best_model = checkpoint['best_model']\n"
        "            args.start_epoch += 1",
    )


def install_compatibility_patch(method: str, source: Path) -> dict:
    if method == "patch_mix_cl":
        install_patch_mix(source)
    elif method == "pafa":
        install_pafa(source)
    else:
        raise ValueError(method)
    subprocess.run(["git", "diff", "--check"], cwd=source, check=True)
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--", "main.py", "util/misc.py"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if not diff:
        raise ValueError("compatibility patch produced no diff")
    return {
        "status": "installed",
        "scope": "checkpoint save/resume state only; uninterrupted training semantics unchanged",
        "files": ["main.py", "util/misc.py"],
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_bytes": len(diff),
        "resume_boundary": "only checkpoints created by this compatibility patch are resumable",
    }
