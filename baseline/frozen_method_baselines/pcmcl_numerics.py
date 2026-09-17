"""Finite-value checks for PC-MCL updates, predictions, and saved run status."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


class NonFiniteTrainingError(FloatingPointError):
    """Numerical failure with compact, JSON-safe localization metadata."""

    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def require_finite(value: torch.Tensor, context: str) -> None:
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    raise NonFiniteTrainingError(
        f"non-finite {context}",
        {
            "stage": "tensor_check",
            "context": context,
            "shape": list(value.shape),
            "nonfinite_elements": int((~finite).sum().item()),
            "total_elements": int(value.numel()),
        },
    )


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    context: str,
    *,
    named_parameters: list[tuple[str, torch.nn.Parameter]] | None = None,
    loss_components: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Stop before an invalid update and report the first affected tensor.

    Finite gradients are left unchanged. Parameter values are not scanned after
    every update; if an optimizer update corrupts them, the next loss/logit
    check stops the run before another checkpoint or result summary is written.
    """

    diagnostic_base = dict(metadata or {})
    diagnostic_base["context"] = context
    diagnostic_base["learning_rates"] = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    component_values: dict[str, float] = {}
    for name, component in (loss_components or {}).items():
        try:
            require_finite(component.detach(), f"{name} loss ({context})")
        except NonFiniteTrainingError as error:
            error.diagnostics.update(diagnostic_base)
            error.diagnostics["loss_components"] = component_values
            raise
        component_values[name] = float(component.detach().item())
    try:
        require_finite(loss.detach(), f"total loss ({context})")
    except NonFiniteTrainingError as error:
        error.diagnostics.update(diagnostic_base)
        error.diagnostics["loss_components"] = component_values
        raise

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if named_parameters is None:
        named_parameters = [
            (f"optimizer_parameter_{index}", parameter)
            for index, parameter in enumerate(
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        ]
    gradients = []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        gradients.append((name, parameter.grad.detach()))
    if not gradients:
        raise RuntimeError(f"no gradients produced ({context})")
    finite_flags = [torch.isfinite(gradient).all() for _, gradient in gradients]
    if not bool(torch.stack(finite_flags).all()):
        for (name, gradient), finite in zip(gradients, finite_flags):
            if bool(finite):
                continue
            finite_elements = torch.isfinite(gradient)
            raise NonFiniteTrainingError(
                f"non-finite gradient ({context}), parameter={name}",
                {
                    **diagnostic_base,
                    "stage": "gradient",
                    "parameter": name,
                    "shape": list(gradient.shape),
                    "nonfinite_elements": int((~finite_elements).sum().item()),
                    "total_elements": int(gradient.numel()),
                    "loss_components": component_values,
                },
            )
    max_abs_gradient = float(
        torch.stack([gradient.abs().max() for _, gradient in gradients]).max().item()
    )
    optimizer.step()
    return {
        "max_abs_gradient": max_abs_gradient,
        "gradient_tensors": len(gradients),
        "learning_rates": diagnostic_base["learning_rates"],
    }


def _first_nonfinite_number(value: object, path: str = "") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            issue = _first_nonfinite_number(child, f"{path}.{key}" if path else str(key))
            if issue is not None:
                return issue
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issue = _first_nonfinite_number(child, f"{path}[{index}]")
            if issue is not None:
                return issue
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return path
    return None


def nonfinite_training_issue(result_dir: Path) -> str | None:
    """Read existing metadata without rewriting failed or historical artifacts."""

    summary_path = result_dir / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "failed_nonfinite":
            return str(summary["error"])
        issue = _first_nonfinite_number(summary)
        if issue is not None:
            return f"non-finite summary number at {issue}"
    log_path = result_dir / "train_log.jsonl"
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            issue = _first_nonfinite_number(row)
            if issue is not None:
                return f"non-finite training log value at epoch {row.get('epoch')}: {issue}"
    return None
