"""Storage-bounded checkpoint retention for author-code reproductions."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


POLICY_VERSION = "best_plus_atomic_rolling_last_v1"


_AUTHOR_HELPER = r'''

def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_checkpoint(state, save_file, epoch):
    save_file = os.path.abspath(save_file)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    temporary = '{}.tmp.{}'.format(save_file, os.getpid())
    replaced_existing = os.path.exists(save_file)
    torch.save(state, temporary)
    temporary_sha256 = _checkpoint_sha256(temporary)
    os.replace(temporary, save_file)
    final_sha256 = _checkpoint_sha256(save_file)
    if final_sha256 != temporary_sha256:
        raise RuntimeError('checkpoint checksum changed during atomic replacement')
    receipt = {
        'policy_version': 'best_plus_atomic_rolling_last_v1',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'epoch': int(epoch),
        'role': os.path.splitext(os.path.basename(save_file))[0],
        'path': save_file,
        'bytes': os.path.getsize(save_file),
        'sha256': final_sha256,
        'replaced_existing': replaced_existing,
        'selection_policy_unchanged': True,
    }
    receipt_path = os.path.join(os.path.dirname(save_file), 'retention_receipts.jsonl')
    with open(receipt_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"retention patch anchor {label!r} count={count}")
    return text.replace(old, new)


def install_author_checkpoint_writer(misc_path: Path) -> None:
    """Make one generated author save_model writer atomic and receipted."""
    text = misc_path.read_text()
    if "def _atomic_save_checkpoint(" in text:
        raise ValueError(f"retention writer already installed in {misc_path}")
    text = _replace_once(
        text,
        "import torch\n",
        "import torch\nimport datetime\nimport hashlib\n" + _AUTHOR_HELPER + "\n",
        f"{misc_path} imports",
    )
    save_anchor = "    torch.save(state, save_file)\n"
    save_count = text.count(save_anchor)
    if save_count not in {1, 2}:
        raise ValueError(f"retention checkpoint writer count={save_count} in {misc_path}")
    text = text.replace(
        save_anchor,
        "    _atomic_save_checkpoint(state, save_file, epoch)\n",
    )
    misc_path.write_text(text)


def install_author_training_retention(main_path: Path, rolling_save_call: str) -> None:
    """Retain one author-selected best checkpoint and one complete rolling last."""
    text = main_path.read_text()
    if "'last.pth'" not in text:
        anchor = "            # save a checkpoint of model and classifier when the best score is updated\n"
        replacement = (
            "            # Storage compatibility: one complete, atomic rolling resume checkpoint.\n"
            f"            {rolling_save_call}\n\n"
            + anchor
        )
        text = _replace_once(text, anchor, replacement, f"{main_path} rolling last")

    text = _replace_once(
        text,
        "'best_epoch_{}.pth'.format(epoch)",
        "'best.pth'",
        f"{main_path} selected best path",
    )
    periodic_pattern = re.compile(
        r"\n[ \t]*if epoch % args\.save_freq == 0:\n"
        r"[ \t]*save_file = os\.path\.join\(args\.save_folder, 'epoch_\{\}\.pth'\.format\(epoch\)\)\n"
        r"[ \t]*save_model\([^\n]+\)\n"
    )
    text, count = periodic_pattern.subn(
        "\n            # Periodic epoch files are superseded by the verified rolling last.pth.\n",
        text,
    )
    if count != 1:
        raise ValueError(f"retention periodic checkpoint block count={count} in {main_path}")
    if "best_epoch_" in text or "'epoch_{}.pth'" in text:
        raise ValueError(f"unbounded checkpoint path remains in {main_path}")
    main_path.write_text(text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(state: dict[str, Any], path: Path, epoch: int, role: str) -> dict[str, Any]:
    """Atomically replace a tracked runtime checkpoint and append its receipt."""
    import torch

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    replaced_existing = path.exists()
    torch.save(state, temporary)
    temporary_sha256 = _sha256(temporary)
    os.replace(temporary, path)
    final_sha256 = _sha256(path)
    if final_sha256 != temporary_sha256:
        raise RuntimeError("checkpoint checksum changed during atomic replacement")
    receipt = {
        "policy_version": POLICY_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": final_sha256,
        "replaced_existing": replaced_existing,
        "selection_policy_unchanged": True,
    }
    receipt_path = path.parent / "retention_receipts.jsonl"
    with receipt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
