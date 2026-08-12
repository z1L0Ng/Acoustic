"""Approval-gated P1-P4 shared-window training assembly.

Importing this module is inert.  ``preflight`` is inventory-only.  ``smoke`` and
``full`` require a matching immutable approval receipt; ``terminal-score`` also
requires explicit outer/test authorization and an independently supplied scorer.
Engineering smokes are never model-performance results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .adapter_factory import AdapterFactoryConfig, audit_local_adapter_assets, build_production_adapter
from .joint_native import JOINT_LANES, SEED, JointNativeProjector
from .preflight import (
    FOUR_DATASET_SUBTRAIN_UNITS,
    P1_P5_BATCH_SIZE,
    P1_P5_SELECTION_RULE,
    P1_P5_UPDATE_BUDGET,
    P1_P5_VALIDATION_INTERVAL_UPDATES,
    PIPELINE_ENCODERS,
    SharedWindowCoreConfig,
    hf_masked_channel_balanced_bce,
    select_validation_checkpoint,
    source_proportional_validation_selection_loss,
)
from .real_subtrain_provider import (
    FrozenProviderIndex,
    NativeWindowBatch,
    build_frozen_provider_index,
    load_native_window_batch,
)
from .sliding_window import hf_window_supervision, masked_mean_window_embeddings
from .window_encoder import ProductionWindowEncoder


RUNNER_SCHEMA_VERSION = "shared_window_training_v3"
VALIDATION_SELECTION_SCHEMA_VERSION = "validation_selection_v1"
OPTIMIZER_POLICY_STATUS = "proposed_benchmark_policy"
OPTIMIZER_POLICY_REFERENCE = "baseline/shared_encoder_native_heads/protocol.json"
OPTIMIZER_NAME = "Adam"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-6
SCHEDULE = "cosine_per_update_no_warmup"
INITIALIZATION = (
    "torch.manual_seed(20260728) before candidate dimension adapter, shared biased "
    "Linear(768,256), and native Linear heads; torch.nn.Linear.reset_parameters"
)
ALLOWED_PHASES = ("preflight", "smoke", "full", "terminal-score")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{label} must be a full 64-character SHA256")
    return value.lower()


def structured_state_sha256(value: object) -> str:
    """Deterministically hash nested tensor/state objects without pickle."""

    digest = hashlib.sha256()

    def visit(current: object) -> None:
        if isinstance(current, torch.Tensor):
            tensor = current.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("utf-8") + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode("utf-8") + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(current, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(current, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(current[key])
        elif isinstance(current, tuple):
            digest.update(b"tuple\0")
            for item in current:
                visit(item)
        elif isinstance(current, list):
            digest.update(b"list\0")
            for item in current:
                visit(item)
        elif current is None:
            digest.update(b"none\0")
        elif isinstance(current, bool):
            digest.update(b"bool\0" + (b"1" if current else b"0"))
        elif isinstance(current, int):
            digest.update(b"int\0" + str(current).encode("ascii") + b"\0")
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("state hash refuses non-finite floats")
            digest.update(b"float\0" + current.hex().encode("ascii") + b"\0")
        elif isinstance(current, str):
            digest.update(b"str\0" + current.encode("utf-8") + b"\0")
        else:
            raise TypeError(f"unsupported checkpoint state type: {type(current).__name__}")

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingRunnerConfig:
    pipeline_id: str
    dataset_root: Path
    run_root: Path
    phase: str = "preflight"
    seed: int = SEED
    batch_size: int = P1_P5_BATCH_SIZE
    update_budget: int = P1_P5_UPDATE_BUDGET
    validation_interval_updates: int = P1_P5_VALIDATION_INTERVAL_UPDATES
    selection_rule: str = P1_P5_SELECTION_RULE
    optimizer: str = OPTIMIZER_NAME
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    schedule: str = SCHEDULE
    kauh_outer_fold: int = 0

    @classmethod
    def frozen(
        cls,
        pipeline_id: str,
        repo_root: Path,
        *,
        phase: str = "preflight",
    ) -> "TrainingRunnerConfig":
        return cls(
            pipeline_id=pipeline_id,
            dataset_root=repo_root / "dataset" / "raw",
            run_root=(
                repo_root
                / "result"
                / "reproduce"
                / f"{pipeline_id}_shared_window_seed{SEED}"
            ),
            phase=phase,
        )

    def validate(self) -> None:
        if self.pipeline_id not in {"P1", "P2", "P3", "P4"}:
            raise ValueError("training runner supports P1-P4")
        if self.phase not in ALLOWED_PHASES:
            raise ValueError("invalid runner phase")
        if (
            self.seed != SEED
            or self.batch_size != P1_P5_BATCH_SIZE
            or self.update_budget != P1_P5_UPDATE_BUDGET
            or self.validation_interval_updates != P1_P5_VALIDATION_INTERVAL_UPDATES
            or self.selection_rule != P1_P5_SELECTION_RULE
        ):
            raise ValueError("frozen seed/batch/budget/selection contract changed")
        if (
            self.optimizer != OPTIMIZER_NAME
            or self.learning_rate != LEARNING_RATE
            or self.weight_decay != WEIGHT_DECAY
            or self.schedule != SCHEDULE
        ):
            raise ValueError("proposed optimizer benchmark policy changed")
        expected_suffix = f"{self.pipeline_id}_shared_window_seed{SEED}"
        if self.run_root.name != expected_suffix:
            raise ValueError("run_root must use the frozen pipeline/seed identity")
        SharedWindowCoreConfig(
            pipeline_id=self.pipeline_id,
            encoder_identity=PIPELINE_ENCODERS[self.pipeline_id],
            split_digest="provided_by_frozen_manifest_receipt",
        ).validate()

    def normalized(self) -> dict[str, object]:
        values = asdict(self)
        values["dataset_root"] = str(self.dataset_root.resolve())
        values["run_root"] = str(self.run_root.resolve())
        return values

    def sha256(self) -> str:
        values = self.normalized()
        values.pop("phase")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optimizer_policy_receipt() -> dict[str, object]:
    return {
        "status": OPTIMIZER_POLICY_STATUS,
        "reference": OPTIMIZER_POLICY_REFERENCE,
        "optimizer": OPTIMIZER_NAME,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "schedule": SCHEDULE,
        "initialization": INITIALIZATION,
        "outer_test_used_to_select_policy": False,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_and_validate_approval(
    path: Path,
    config: TrainingRunnerConfig,
    *,
    expected_data_identity_sha256: str | None = None,
) -> dict[str, object]:
    """Require a phase-specific approval file without accepting wildcard authority."""

    config.validate()
    if config.phase == "preflight":
        raise ValueError("preflight does not accept or require an approval receipt")
    if not path.is_file():
        raise FileNotFoundError(f"approval receipt missing: {path}")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    required = {
        "status",
        "pipeline_id",
        "phase",
        "config_sha256",
        "data_identity_sha256",
        "authorized_by",
        "outer_test_authorized",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise ValueError(f"approval receipt missing fields: {missing}")
    if receipt["status"] != "approved":
        raise PermissionError("runner approval status is not approved")
    if receipt["pipeline_id"] != config.pipeline_id or receipt["phase"] != config.phase:
        raise PermissionError("approval pipeline/phase mismatch")
    if receipt["config_sha256"] != config.sha256():
        raise PermissionError("approval is not bound to this exact runner config")
    if (
        not isinstance(receipt["data_identity_sha256"], str)
        or len(receipt["data_identity_sha256"]) != 64
        or (
            expected_data_identity_sha256 is not None
            and receipt["data_identity_sha256"] != expected_data_identity_sha256
        )
    ):
        raise PermissionError(
            "approval is not bound to the frozen data authority/annotation identity"
        )
    if not str(receipt["authorized_by"]).strip():
        raise PermissionError("approval authority is empty")
    terminal = config.phase == "terminal-score"
    if bool(receipt["outer_test_authorized"]) is not terminal:
        raise PermissionError(
            "outer_test_authorized must be false for smoke/full and true only for terminal-score"
        )
    return {
        **receipt,
        "approval_receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def combined_data_identity_sha256(
    subtrain: FrozenProviderIndex,
    validation: FrozenProviderIndex,
) -> str:
    payload = {
        "subtrain": subtrain.receipt["data_identity_sha256"],
        "validation": validation.receipt["data_identity_sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SourceProportionalBatchPlanner:
    """Deterministic homogeneous-lane batch schedule with resumable RNG state."""

    def __init__(
        self,
        lane_counts: Mapping[str, int],
        *,
        batch_size: int = P1_P5_BATCH_SIZE,
        seed: int = SEED,
    ) -> None:
        if dict(lane_counts) != FOUR_DATASET_SUBTRAIN_UNITS:
            raise ValueError("source counts must match the frozen subtrain inventory")
        if batch_size != P1_P5_BATCH_SIZE or seed != SEED:
            raise ValueError("planner batch size/seed changed")
        self.lane_counts = dict(lane_counts)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.probabilities = torch.tensor(
            [self.lane_counts[lane] for lane in JOINT_LANES], dtype=torch.float64
        )
        self.probabilities /= self.probabilities.sum()
        self.orders = {
            lane: torch.randperm(self.lane_counts[lane], generator=self.generator).tolist()
            for lane in JOINT_LANES
        }
        self.positions = {lane: 0 for lane in JOINT_LANES}
        self.draws = 0

    def next(self) -> tuple[str, tuple[int, ...]]:
        lane_index = int(
            torch.multinomial(self.probabilities, 1, replacement=True, generator=self.generator)
        )
        lane = JOINT_LANES[lane_index]
        indices = []
        while len(indices) < self.batch_size:
            position = self.positions[lane]
            if position == len(self.orders[lane]):
                self.orders[lane] = torch.randperm(
                    self.lane_counts[lane], generator=self.generator
                ).tolist()
                position = 0
            take = min(self.batch_size - len(indices), len(self.orders[lane]) - position)
            indices.extend(self.orders[lane][position : position + take])
            self.positions[lane] = position + take
        self.draws += 1
        return lane, tuple(indices)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "lane_counts": self.lane_counts,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state().clone(),
            "orders": {lane: list(values) for lane, values in self.orders.items()},
            "positions": dict(self.positions),
            "draws": self.draws,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("lane_counts") != self.lane_counts
            or state.get("batch_size") != self.batch_size
        ):
            raise RuntimeError("sampler resume contract mismatch")
        self.generator.set_state(state["generator_state"])
        self.orders = {lane: list(state["orders"][lane]) for lane in JOINT_LANES}
        self.positions = {lane: int(state["positions"][lane]) for lane in JOINT_LANES}
        self.draws = int(state["draws"])


def assemble_trainable_modules(
    adapter: ProductionWindowEncoder,
    *,
    device: torch.device,
) -> JointNativeProjector:
    """Freeze the candidate backbone and expose only declared trainable modules."""

    for parameter in adapter.backend.parameters():
        parameter.requires_grad_(False)
    adapter.backend.eval()
    model = JointNativeProjector().to(device)
    adapter.to(device)
    return model


def trainable_scope_receipt(
    adapter: ProductionWindowEncoder,
    model: JointNativeProjector,
) -> dict[str, object]:
    frozen = {
        f"adapter.backend.{name}": parameter.numel()
        for name, parameter in adapter.backend.named_parameters()
    }
    dimension = {
        f"adapter.dimension_adapter.{name}": parameter.numel()
        for name, parameter in adapter.dimension_adapter.named_parameters()
        if parameter.requires_grad
    }
    projector = {
        f"model.projector.{name}": parameter.numel()
        for name, parameter in model.projector.named_parameters()
        if parameter.requires_grad
    }
    heads = {
        f"model.heads.{name}.{parameter_name}": parameter.numel()
        for name, module in model.heads.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    forbidden = [
        name for name, parameter in adapter.backend.named_parameters() if parameter.requires_grad
    ]
    if forbidden:
        raise RuntimeError(f"frozen encoder exposes trainable parameters: {forbidden}")
    return {
        "candidate_encoder": {"scope": "frozen", "parameters": sum(frozen.values())},
        "candidate_dimension_adapter": {
            "scope": "trainable_only_for_P3_P4" if dimension else "identity_no_parameters",
            "named_parameters": dimension,
        },
        "shared_biased_linear_768_to_256": {"named_parameters": projector},
        "dataset_native_heads": {"named_parameters": heads},
        "hf_uses_same_shared_projector": True,
        "trainable_parameters": sum(dimension.values()) + sum(projector.values()) + sum(heads.values()),
    }


def build_optimizer(
    adapter: ProductionWindowEncoder,
    model: JointNativeProjector,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    groups = []
    receipt = []
    candidates = (
        ("candidate_dimension_adapter", list(adapter.dimension_adapter.parameters())),
        ("shared_projector", list(model.projector.parameters())),
        ("dataset_native_heads", list(model.heads.parameters())),
    )
    for name, parameters in candidates:
        values = [parameter for parameter in parameters if parameter.requires_grad]
        if values:
            groups.append({"params": values, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY})
            receipt.append({"name": name, "parameters": sum(value.numel() for value in values)})
    optimizer = torch.optim.Adam(groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer, {
        **optimizer_policy_receipt(),
        "parameter_groups": receipt,
        "frozen_encoder_in_optimizer": False,
    }


def native_batch_loss(
    adapter: nn.Module,
    model: JointNativeProjector,
    batch: NativeWindowBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Compute one native-task loss; no optimizer step is performed here."""

    batch.validate()
    windows = batch.windows.to(device)
    output = adapter(windows)
    if batch.lane == "HF":
        logits = model(output.embeddings, "HF")["temporal4"]
        supervision = hf_window_supervision(
            windows, batch.hf_intervals, batch.hf_recording_states
        )
        loss, receipt = hf_masked_channel_balanced_bce(
            logits,
            supervision.targets.to(device),
            supervision.observation_mask.to(device),
            supervision.valid_mask.to(device),
        )
        return loss, {
            "lane": "HF",
            "native_task_losses": {"HF_temporal4": float(loss.detach())},
            "target_receipt": supervision.receipt,
            "loss_receipt": receipt,
        }
    pooled = masked_mean_window_embeddings(output.embeddings, output.window_mask)
    logits = model(pooled, batch.lane)
    targets = {key: value.to(device) for key, value in batch.targets.items()}
    if batch.lane == "ICBHI":
        loss = F.cross_entropy(logits["flat4"], targets["icbhi_flat4"])
        losses = {"ICBHI_flat4": float(loss.detach())}
    elif batch.lane == "SPRSound":
        binary = F.cross_entropy(logits["binary"], targets["spr_binary"])
        raw7 = F.cross_entropy(logits["raw7"], targets["spr_seven"])
        loss = (binary + raw7) / 2
        losses = {
            "SPRSound_binary": float(binary.detach()),
            "SPRSound_raw7": float(raw7.detach()),
        }
    elif batch.lane == "KAUH":
        loss = F.cross_entropy(logits["raw9"], targets["kauh_raw9"])
        losses = {"KAUH_raw9": float(loss.detach())}
    else:
        raise ValueError(f"unsupported lane: {batch.lane}")
    return loss, {"lane": batch.lane, "native_task_losses": losses}


def save_training_checkpoint(
    path: Path,
    *,
    config: TrainingRunnerConfig,
    update: int,
    adapter: ProductionWindowEncoder,
    model: JointNativeProjector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    planner: SourceProportionalBatchPlanner,
    validation_history: Sequence[Mapping[str, object]],
    data_identity_sha256: str,
    approval_receipt_sha256: str,
    selection_scalar: float,
) -> dict[str, object]:
    if (
        not 0 <= update <= config.update_budget
        or update % config.validation_interval_updates
    ):
        raise ValueError("checkpoint update outside frozen budget")
    data_identity_sha256 = _require_sha256(
        data_identity_sha256, "checkpoint data identity"
    )
    approval_receipt_sha256 = _require_sha256(
        approval_receipt_sha256, "checkpoint approval receipt"
    )
    if not math.isfinite(selection_scalar):
        raise ValueError("checkpoint selection scalar must be finite")
    component_states = {
        "dimension_adapter": adapter.dimension_adapter.state_dict(),
        "joint_native_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "planner": planner.state_dict(),
    }
    component_state_sha256 = {
        name: structured_state_sha256(state)
        for name, state in component_states.items()
    }
    payload = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "config": config.normalized(),
        "config_sha256": config.sha256(),
        "data_identity_sha256": data_identity_sha256,
        "approval_receipt_sha256": approval_receipt_sha256,
        "update": update,
        "selection_scalar": float(selection_scalar),
        "component_state_sha256": component_state_sha256,
        "dimension_adapter_state": component_states["dimension_adapter"],
        "joint_native_state": component_states["joint_native_model"],
        "optimizer_state": component_states["optimizer"],
        "scheduler_state": component_states["scheduler"],
        "planner_state": component_states["planner"],
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "validation_history": list(validation_history),
        "outer_test_accessed": False,
        "native_metrics_only": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    artifact_sha256 = sha256_path(path)
    receipt = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": artifact_sha256,
        "update": update,
        "selection_scalar": float(selection_scalar),
        "config_sha256": config.sha256(),
        "data_identity_sha256": data_identity_sha256,
        "approval_receipt_sha256": approval_receipt_sha256,
        "component_state_sha256": component_state_sha256,
        "outer_test_accessed": False,
        "native_metrics_only": True,
    }
    _write_json(path.with_suffix(".receipt.json"), receipt)
    return receipt


def load_training_checkpoint(
    path: Path,
    *,
    config: TrainingRunnerConfig,
    adapter: ProductionWindowEncoder,
    model: JointNativeProjector,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    planner: SourceProportionalBatchPlanner,
    expected_data_identity_sha256: str,
    expected_checkpoint_sha256: str,
    expected_approval_receipt_sha256: str,
) -> tuple[int, list[Mapping[str, object]], Mapping[str, object]]:
    expected_checkpoint_sha256 = _require_sha256(
        expected_checkpoint_sha256, "expected checkpoint"
    )
    expected_approval_receipt_sha256 = _require_sha256(
        expected_approval_receipt_sha256, "expected approval receipt"
    )
    expected_data_identity_sha256 = _require_sha256(
        expected_data_identity_sha256, "expected data identity"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_checkpoint_sha256 = sha256_path(path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("checkpoint byte SHA256 mismatch before deserialization")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    update = payload.get("update")
    if (
        payload.get("schema_version") != RUNNER_SCHEMA_VERSION
        or payload.get("config_sha256") != config.sha256()
        or payload.get("data_identity_sha256") != expected_data_identity_sha256
        or payload.get("approval_receipt_sha256")
        != expected_approval_receipt_sha256
        or payload.get("outer_test_accessed") is not False
        or payload.get("native_metrics_only") is not True
        or not isinstance(update, int)
        or not 0 <= update <= config.update_budget
        or update % config.validation_interval_updates
    ):
        raise RuntimeError("checkpoint resume identity/isolation gate failed")
    expected_components = payload.get("component_state_sha256")
    component_states = {
        "dimension_adapter": payload.get("dimension_adapter_state"),
        "joint_native_model": payload.get("joint_native_state"),
        "optimizer": payload.get("optimizer_state"),
        "scheduler": payload.get("scheduler_state"),
        "planner": payload.get("planner_state"),
    }
    if not isinstance(expected_components, Mapping) or set(expected_components) != set(
        component_states
    ):
        raise RuntimeError("checkpoint component hash schema failed")
    actual_components = {
        name: structured_state_sha256(state)
        for name, state in component_states.items()
    }
    if dict(expected_components) != actual_components:
        raise RuntimeError("checkpoint component state hash mismatch")
    adapter.dimension_adapter.load_state_dict(payload["dimension_adapter_state"])
    model.load_state_dict(payload["joint_native_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    planner.load_state_dict(payload["planner_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_rng_state"])
    selection_scalar = payload.get("selection_scalar")
    if not isinstance(selection_scalar, (int, float)) or not math.isfinite(
        float(selection_scalar)
    ):
        raise RuntimeError("checkpoint selection scalar is invalid")
    receipt = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": actual_checkpoint_sha256,
        "update": update,
        "selection_scalar": float(selection_scalar),
        "config_sha256": config.sha256(),
        "data_identity_sha256": expected_data_identity_sha256,
        "approval_receipt_sha256": expected_approval_receipt_sha256,
        "component_state_sha256": actual_components,
        "outer_test_accessed": False,
        "native_metrics_only": True,
    }
    return update, list(payload["validation_history"]), receipt


def write_validation_selection_receipt(
    path: Path,
    *,
    config: TrainingRunnerConfig,
    data_identity_sha256: str,
    full_approval_receipt_sha256: str,
    candidates: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Persist the exact validation-selected checkpoint artifact contract."""

    data_identity_sha256 = _require_sha256(
        data_identity_sha256, "selection data identity"
    )
    full_approval_receipt_sha256 = _require_sha256(
        full_approval_receipt_sha256, "selection full approval"
    )
    if not candidates:
        raise ValueError("validation selection requires checkpoint candidates")
    normalized = []
    for candidate in candidates:
        required = {
            "schema_version",
            "path",
            "size_bytes",
            "sha256",
            "update",
            "selection_scalar",
            "config_sha256",
            "data_identity_sha256",
            "approval_receipt_sha256",
            "component_state_sha256",
            "outer_test_accessed",
            "native_metrics_only",
        }
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"validation checkpoint receipt missing fields: {missing}")
        candidate_path = Path(str(candidate["path"]))
        candidate_sha256 = _require_sha256(
            str(candidate["sha256"]), "validation checkpoint"
        )
        update = candidate["update"]
        scalar = candidate["selection_scalar"]
        if (
            candidate["schema_version"] != RUNNER_SCHEMA_VERSION
            or candidate["config_sha256"] != config.sha256()
            or candidate["data_identity_sha256"] != data_identity_sha256
            or candidate["approval_receipt_sha256"]
            != full_approval_receipt_sha256
            or candidate["outer_test_accessed"] is not False
            or candidate["native_metrics_only"] is not True
            or not isinstance(update, int)
            or not 0 < update <= config.update_budget
            or update % config.validation_interval_updates
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
        ):
            raise RuntimeError("validation checkpoint candidate contract failed")
        if (
            not candidate_path.is_file()
            or candidate_path.stat().st_size != int(candidate["size_bytes"])
            or sha256_path(candidate_path) != candidate_sha256
        ):
            raise RuntimeError("validation checkpoint candidate artifact changed")
        normalized.append(dict(candidate))
    selected_update, selected_scalar = select_validation_checkpoint(
        [
            (int(candidate["update"]), float(candidate["selection_scalar"]))
            for candidate in normalized
        ]
    )
    selected_candidates = [
        candidate
        for candidate in normalized
        if candidate["update"] == selected_update
        and float(candidate["selection_scalar"]) == selected_scalar
    ]
    if len(selected_candidates) != 1:
        raise RuntimeError("validation selection does not identify one exact artifact")
    document = {
        "schema_version": VALIDATION_SELECTION_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "pipeline_id": config.pipeline_id,
        "config_sha256": config.sha256(),
        "data_identity_sha256": data_identity_sha256,
        "full_approval_receipt_sha256": full_approval_receipt_sha256,
        "selection_rule": config.selection_rule,
        "candidates": normalized,
        "selected_update": selected_update,
        "selected_checkpoint": selected_candidates[0],
        "outer_test_accessed": False,
        "reported_as_pooled_performance": False,
    }
    _write_json(path, document)
    return {
        **document,
        "selection_receipt_artifact": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        },
    }


def _load_verified_selection_receipt(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    expected_sha256 = _require_sha256(expected_sha256, "selection receipt")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_path(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("validation selection receipt SHA256 mismatch")
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise TypeError("validation selection receipt must be a JSON object")
    return document, {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": actual_sha256,
    }


def preflight_receipt(config: TrainingRunnerConfig, repo_root: Path) -> dict[str, object]:
    """Read frozen inventories only; create no model, optimizer, run directory, or update."""

    config.validate()
    if config.phase != "preflight":
        raise ValueError("preflight_receipt requires phase=preflight")
    index = build_frozen_provider_index(
        config.dataset_root, partition="subtrain", kauh_outer_fold=config.kauh_outer_fold
    )
    validation_index = build_frozen_provider_index(
        config.dataset_root, partition="validation", kauh_outer_fold=config.kauh_outer_fold
    )
    data_identity_sha256 = combined_data_identity_sha256(index, validation_index)
    return {
        "status": "shared_window_training_preflight_ready",
        "engineering_only": True,
        "performance_result": False,
        "config": config.normalized(),
        "config_sha256": config.sha256(),
        "data_identity_sha256": data_identity_sha256,
        "optimizer_policy": optimizer_policy_receipt(),
        "provider_inventory": {
            key: value for key, value in index.receipt.items() if key != "sample_ids"
        },
        "adapter_assets": audit_local_adapter_assets(repo_root)[config.pipeline_id],
        "trainable_scope_declared": (
            "frozen candidate encoder; P3/P4 trainable dimension adapter; shared biased "
            "Linear(768,256); dataset-native heads"
        ),
        "optimizer_created": False,
        "optimizer_updates": 0,
        "run_root_created": False,
        "outer_test_accessed": False,
    }


def _validation_native_losses(
    adapter: ProductionWindowEncoder,
    model: JointNativeProjector,
    index: FrozenProviderIndex,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    totals = {key: 0.0 for key in (
        "ICBHI_flat4", "SPRSound_binary", "SPRSound_raw7", "HF_temporal4", "KAUH_raw9"
    )}
    weights = {key: 0 for key in totals}
    adapter.eval()
    model.eval()
    with torch.no_grad():
        for lane in JOINT_LANES:
            rows = index.lanes[lane]
            for start in range(0, len(rows), batch_size):
                batch = load_native_window_batch(rows[start : start + batch_size])
                _, receipt = native_batch_loss(adapter, model, batch, device=device)
                weight = len(batch.windows.sample_ids)
                for task, value in receipt["native_task_losses"].items():
                    totals[task] += float(value) * weight
                    weights[task] += weight
    adapter.train()
    adapter.backend.eval()
    model.train()
    if any(value == 0 for value in weights.values()):
        raise RuntimeError("validation task denominator is zero")
    return {key: totals[key] / weights[key] for key in totals}


def run_approved_training(
    config: TrainingRunnerConfig,
    *,
    repo_root: Path,
    approval_path: Path,
    adapter_config: AdapterFactoryConfig,
    resume: Path | None = None,
    resume_sha256: str | None = None,
) -> dict[str, object]:
    """Run future approved smoke/full training; callers control authorization file."""

    config.validate()
    if config.phase not in {"smoke", "full"}:
        raise ValueError("training execution accepts only smoke/full")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device(adapter_config.device)
    train_index = build_frozen_provider_index(
        config.dataset_root, partition="subtrain", kauh_outer_fold=config.kauh_outer_fold
    )
    validation_index = build_frozen_provider_index(
        config.dataset_root, partition="validation", kauh_outer_fold=config.kauh_outer_fold,
        enforce_real_counts=True,
    )
    data_identity_sha256 = combined_data_identity_sha256(train_index, validation_index)
    approval = load_and_validate_approval(
        approval_path,
        config,
        expected_data_identity_sha256=data_identity_sha256,
    )
    approval_receipt_sha256 = str(approval["approval_receipt_sha256"])
    if (resume is None) != (resume_sha256 is None):
        raise PermissionError("--resume and --resume-sha256 must be provided together")
    if resume is not None and config.phase != "full":
        raise PermissionError("resume is permitted only for an approved full phase")
    adapter = build_production_adapter(adapter_config)
    model = assemble_trainable_modules(adapter, device=device)
    scope = trainable_scope_receipt(adapter, model)
    optimizer, optimizer_receipt = build_optimizer(adapter, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.update_budget
    )
    planner = SourceProportionalBatchPlanner(FOUR_DATASET_SUBTRAIN_UNITS)
    start_update = 0
    validation_history: list[Mapping[str, object]] = []
    validation_checkpoint_receipts: list[Mapping[str, object]] = []
    if resume is not None:
        start_update, validation_history, resumed_checkpoint_receipt = load_training_checkpoint(
            resume,
            config=config,
            adapter=adapter,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            planner=planner,
            expected_data_identity_sha256=data_identity_sha256,
            expected_checkpoint_sha256=str(resume_sha256),
            expected_approval_receipt_sha256=approval_receipt_sha256,
        )
        validation_checkpoint_receipts.extend(
            item["checkpoint_receipt"]
            for item in validation_history
            if isinstance(item, Mapping) and "checkpoint_receipt" in item
        )
        if start_update % config.validation_interval_updates == 0:
            validation_checkpoint_receipts.append(resumed_checkpoint_receipt)
            validation_history = [
                {
                    **item,
                    "checkpoint_receipt": resumed_checkpoint_receipt,
                }
                if isinstance(item, Mapping)
                and item.get("update") == start_update
                and "checkpoint_receipt" not in item
                else item
                for item in validation_history
            ]
    config.run_root.mkdir(parents=True, exist_ok=True)
    _write_json(config.run_root / "config.json", config.normalized())
    _write_json(config.run_root / "approval_receipt.json", approval)
    _write_json(config.run_root / "trainable_scope_receipt.json", scope)
    _write_json(config.run_root / "optimizer_receipt.json", optimizer_receipt)
    maximum = 1 if config.phase == "smoke" else config.update_budget
    adapter.train()
    adapter.backend.eval()
    model.train()
    last_loss_receipt: Mapping[str, object] | None = None
    for update in range(start_update + 1, maximum + 1):
        lane, indices = planner.next()
        batch = load_native_window_batch([train_index.lanes[lane][i] for i in indices])
        optimizer.zero_grad(set_to_none=True)
        loss, last_loss_receipt = native_batch_loss(adapter, model, batch, device=device)
        loss.backward()
        if any(parameter.grad is not None for parameter in adapter.backend.parameters()):
            raise RuntimeError("frozen encoder received a gradient")
        optimizer.step()
        scheduler.step()
        _append_jsonl(
            config.run_root / "train_log.jsonl",
            {
                "update": update,
                "lane": lane,
                "native_task_losses": last_loss_receipt["native_task_losses"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "engineering_only": config.phase == "smoke",
                "outer_test_accessed": False,
            },
        )
        if config.phase == "full" and update % config.validation_interval_updates == 0:
            native_losses = _validation_native_losses(
                adapter, model, validation_index, device=device, batch_size=config.batch_size
            )
            scalar, selection_receipt = source_proportional_validation_selection_loss(native_losses)
            validation_history.append({
                "update": update,
                "native_losses_by_dataset_task": native_losses,
                "selection_scalar": scalar,
                "selection_receipt": selection_receipt,
            })
            _append_jsonl(
                config.run_root / "validation_log.jsonl", validation_history[-1]
            )
            checkpoint_receipt = save_training_checkpoint(
                config.run_root / "checkpoints" / f"update_{update:06d}.pt",
                config=config,
                update=update,
                adapter=adapter,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                planner=planner,
                validation_history=validation_history,
                data_identity_sha256=data_identity_sha256,
                approval_receipt_sha256=approval_receipt_sha256,
                selection_scalar=scalar,
            )
            validation_checkpoint_receipts.append(checkpoint_receipt)
            validation_history[-1] = {
                **validation_history[-1],
                "checkpoint_receipt": checkpoint_receipt,
            }
    validation_selection_receipt = None
    if config.phase == "full":
        validation_selection_receipt = write_validation_selection_receipt(
            config.run_root / "validation_selection_receipt.json",
            config=config,
            data_identity_sha256=data_identity_sha256,
            full_approval_receipt_sha256=approval_receipt_sha256,
            candidates=validation_checkpoint_receipts,
        )
    final_receipt = {
        "status": (
            "engineering_smoke_completed_not_performance_result"
            if config.phase == "smoke"
            else "approved_training_completed_validation_only_selection"
        ),
        "pipeline_id": config.pipeline_id,
        "phase": config.phase,
        "approval": approval,
        "data_identity_sha256": data_identity_sha256,
        "scope": scope,
        "optimizer": optimizer_receipt,
        "updates_completed": maximum,
        "last_native_loss_receipt": last_loss_receipt,
        "validation_history": validation_history,
        "validation_selection_receipt": validation_selection_receipt,
        "metrics_reporting": "per_dataset_per_native_task_only",
        "cross_dataset_pooled_performance": False,
        "outer_test_accessed": False,
    }
    _write_json(config.run_root / "run_receipt.json", final_receipt)
    return final_receipt


def terminal_score_gate(
    config: TrainingRunnerConfig,
    approval_path: Path,
    selection_receipt_path: Path,
    expected_selection_receipt_sha256: str,
    selected_checkpoint: Path,
    scorer: Callable[[Path], Mapping[str, object]] | None = None,
) -> Mapping[str, object]:
    """Keep terminal scoring isolated until an independently verified scorer is supplied."""

    if config.phase != "terminal-score":
        raise ValueError("terminal_score_gate requires terminal-score phase")
    selection, selection_artifact = _load_verified_selection_receipt(
        selection_receipt_path, expected_selection_receipt_sha256
    )
    required = {
        "schema_version",
        "runner_schema_version",
        "pipeline_id",
        "config_sha256",
        "data_identity_sha256",
        "full_approval_receipt_sha256",
        "selection_rule",
        "candidates",
        "selected_update",
        "selected_checkpoint",
        "outer_test_accessed",
        "reported_as_pooled_performance",
    }
    missing = sorted(required - set(selection))
    if missing:
        raise ValueError(f"validation selection receipt missing fields: {missing}")
    data_identity_sha256 = _require_sha256(
        str(selection["data_identity_sha256"]), "terminal data identity"
    )
    full_approval_receipt_sha256 = _require_sha256(
        str(selection["full_approval_receipt_sha256"]), "full approval receipt"
    )
    candidates = selection["candidates"]
    selected = selection["selected_checkpoint"]
    if (
        selection["schema_version"] != VALIDATION_SELECTION_SCHEMA_VERSION
        or selection["runner_schema_version"] != RUNNER_SCHEMA_VERSION
        or selection["pipeline_id"] != config.pipeline_id
        or selection["config_sha256"] != config.sha256()
        or selection["selection_rule"] != config.selection_rule
        or selection["outer_test_accessed"] is not False
        or selection["reported_as_pooled_performance"] is not False
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(selected, Mapping)
    ):
        raise RuntimeError("validation selection identity/rule/isolation gate failed")
    candidate_required = {
        "schema_version",
        "path",
        "size_bytes",
        "sha256",
        "update",
        "selection_scalar",
        "config_sha256",
        "data_identity_sha256",
        "approval_receipt_sha256",
        "component_state_sha256",
        "outer_test_accessed",
        "native_metrics_only",
    }
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate_required - set(candidate):
            raise RuntimeError("validation selection candidate schema failed")
        if (
            candidate["schema_version"] != RUNNER_SCHEMA_VERSION
            or candidate["config_sha256"] != config.sha256()
            or candidate["data_identity_sha256"] != data_identity_sha256
            or candidate["approval_receipt_sha256"]
            != full_approval_receipt_sha256
            or candidate["outer_test_accessed"] is not False
            or candidate["native_metrics_only"] is not True
        ):
            raise RuntimeError("validation selection candidate binding failed")
    if candidate_required - set(selected):
        raise RuntimeError("selected checkpoint receipt schema failed")
    chosen_update, chosen_scalar = select_validation_checkpoint(
        [
            (int(candidate["update"]), float(candidate["selection_scalar"]))
            for candidate in candidates
        ]
    )
    if (
        selection["selected_update"] != chosen_update
        or selected.get("update") != chosen_update
        or float(selected.get("selection_scalar")) != chosen_scalar
        or sum(dict(candidate) == dict(selected) for candidate in candidates)
        != 1
    ):
        raise RuntimeError("validation selection winner was altered")
    selected_expected_path = Path(str(selected["path"])).resolve()
    if selected_checkpoint.resolve() != selected_expected_path:
        raise RuntimeError("terminal checkpoint is not the exact validation-selected artifact")
    selected_sha256 = _require_sha256(
        str(selected["sha256"]), "selected checkpoint"
    )
    if (
        not selected_checkpoint.is_file()
        or selected_checkpoint.stat().st_size != int(selected["size_bytes"])
        or sha256_path(selected_checkpoint) != selected_sha256
    ):
        raise RuntimeError("selected checkpoint byte identity failed")

    terminal_approval = load_and_validate_approval(
        approval_path,
        config,
        expected_data_identity_sha256=data_identity_sha256,
    )
    if (
        terminal_approval.get("selection_receipt_sha256")
        != selection_artifact["sha256"]
    ):
        raise PermissionError(
            "terminal approval is not bound to the exact validation selection receipt"
        )

    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    update = payload.get("update")
    payload_selection_scalar = payload.get("selection_scalar")
    if (
        payload.get("schema_version") != RUNNER_SCHEMA_VERSION
        or payload.get("config_sha256") != config.sha256()
        or payload.get("data_identity_sha256") != data_identity_sha256
        or payload.get("approval_receipt_sha256")
        != full_approval_receipt_sha256
        or payload.get("outer_test_accessed") is not False
        or payload.get("native_metrics_only") is not True
        or update != chosen_update
        or not isinstance(update, int)
        or not 0 <= update <= config.update_budget
        or not isinstance(payload_selection_scalar, (int, float))
        or not math.isfinite(float(payload_selection_scalar))
        or float(payload_selection_scalar) != chosen_scalar
    ):
        raise RuntimeError("selected checkpoint internal binding failed")
    component_states = {
        "dimension_adapter": payload.get("dimension_adapter_state"),
        "joint_native_model": payload.get("joint_native_state"),
        "optimizer": payload.get("optimizer_state"),
        "scheduler": payload.get("scheduler_state"),
        "planner": payload.get("planner_state"),
    }
    actual_components = {
        name: structured_state_sha256(state)
        for name, state in component_states.items()
    }
    if (
        payload.get("component_state_sha256") != actual_components
        or selected.get("component_state_sha256") != actual_components
    ):
        raise RuntimeError("selected checkpoint component state identity failed")
    if scorer is None:
        raise RuntimeError(
            "terminal scorer HOLD: supply an independently verified per-native-task scorer; "
            "this runner will not infer or pool outer/test metrics"
        )
    receipt = dict(scorer(selected_checkpoint))
    forbidden = {"pooled_score", "cross_dataset_score", "global_score"}
    if forbidden & set(receipt):
        raise RuntimeError("terminal scorer returned a forbidden pooled score")
    return {
        "status": "terminal_native_task_scoring_complete",
        "selection_receipt_artifact": selection_artifact,
        "selected_checkpoint": dict(selected),
        "terminal_approval_receipt_sha256": terminal_approval[
            "approval_receipt_sha256"
        ],
        "native_metrics_by_dataset_task": receipt,
        "cross_dataset_pooled_performance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=("P1", "P2", "P3", "P4"), required=True)
    parser.add_argument("--phase", choices=ALLOWED_PHASES, default="preflight")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--approval-receipt", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    config = TrainingRunnerConfig.frozen(args.pipeline, args.repo_root, phase=args.phase)
    if args.phase == "preflight":
        receipt = preflight_receipt(config, args.repo_root)
    elif args.phase in {"smoke", "full"}:
        if args.approval_receipt is None:
            raise PermissionError("smoke/full require --approval-receipt")
        if (args.resume is None) != (args.resume_sha256 is None):
            raise PermissionError(
                "--resume and --resume-sha256 must be provided together"
            )
        receipt = run_approved_training(
            config,
            repo_root=args.repo_root,
            approval_path=args.approval_receipt,
            resume=args.resume,
            resume_sha256=args.resume_sha256,
            adapter_config=AdapterFactoryConfig(
                pipeline_id=args.pipeline,
                repo_root=args.repo_root,
                device=args.device,
                source_repo=args.source_repo,
                source_revision=args.source_revision,
                checkpoint=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
            ),
        )
    else:
        raise RuntimeError(
            "terminal-score is not exposed through the generic CLI; invoke terminal_score_gate "
            "with an independently verified scorer after explicit terminal approval"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
