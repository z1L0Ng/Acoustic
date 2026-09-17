"""Finite-value checks for PC-MCL updates, predictions, and saved run status."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def require_finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {context}")


def backward_and_step(
    loss: torch.Tensor, optimizer: torch.optim.Optimizer, context: str
) -> None:
    """Stop before updating on invalid loss/gradients; do not clip valid gradients."""

    require_finite(loss.detach(), f"loss ({context})")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norms = [
        parameter.grad.detach().norm()
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    require_finite(torch.stack(gradient_norms), f"gradient norms ({context})")
    optimizer.step()


def nonfinite_training_issue(result_dir: Path) -> str | None:
    """Read existing metadata without rewriting failed or historical artifacts."""

    summary_path = result_dir / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "failed_nonfinite":
            return str(summary["error"])
    log_path = result_dir / "train_log.jsonl"
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not math.isfinite(float(row["train_loss"])):
                return f"non-finite training loss at epoch {row['epoch']}"
    return None
