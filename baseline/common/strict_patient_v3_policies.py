from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .losses import build_loss, effective_number_weights, label_counts
from .models import seed_everything


POLICIES = (
    "unweighted_ce",
    "class_weighted_ce",
    "focal_loss",
    "logit_adjusted_ce",
    "ldam_drw",
    "crt",
)


class StrictMLP2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_1: int,
        hidden_2: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_2, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden(features))


class LogitAdjustedLoss(nn.Module):
    def __init__(self, priors: np.ndarray, tau: float = 1.0):
        super().__init__()
        if np.any(priors <= 0) or not np.isclose(priors.sum(), 1.0):
            raise ValueError(f"Invalid class priors: {priors.tolist()}")
        self.register_buffer("adjustment", tau * torch.log(torch.from_numpy(priors.astype(np.float32))))
        self.tau = tau

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Menon et al. train-time adjustment uses logits + tau * log(class prior).
        return F.cross_entropy(logits + self.adjustment, target)


class LDAMLoss(nn.Module):
    def __init__(
        self,
        counts: np.ndarray,
        max_margin: float = 0.5,
        scale: float = 30.0,
        weights: np.ndarray | None = None,
    ):
        super().__init__()
        margins = 1.0 / np.sqrt(np.sqrt(counts.astype(np.float64)))
        margins *= max_margin / margins.max()
        self.register_buffer("margins", torch.from_numpy(margins.astype(np.float32)))
        self.register_buffer(
            "weights",
            None if weights is None else torch.from_numpy(weights.astype(np.float32)),
        )
        self.scale = scale

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        adjusted = logits.clone()
        row = torch.arange(len(target), device=logits.device)
        adjusted[row, target] -= self.margins[target]
        return F.cross_entropy(self.scale * adjusted, target, weight=self.weights)


@dataclass
class PolicyResult:
    prediction: np.ndarray
    probability: np.ndarray
    parameter_count: int
    trainable_parameter_count: int
    runtime_seconds: float
    best_epoch: int
    best_validation_macro_f1: float
    stage1_best_epoch: int | None
    curve: list[dict]
    state_dict: dict
    details: dict
    loss_finite: bool
    gradient_finite: bool
    warnings: list[str]


def _indices(y: np.ndarray, labels: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels)}
    return np.asarray([mapping[value] for value in y], dtype=np.int64)


def _loader(
    x: np.ndarray,
    y_index: np.ndarray,
    batch_size: int,
    seed: int,
    *,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y_index)),
        batch_size=min(batch_size, len(x)),
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
    )


def _predict_probability(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    probabilities = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            probabilities.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _validation_macro_f1(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: list[str],
    batch_size: int,
) -> float:
    probability = _predict_probability(model, x_val, batch_size)
    prediction = np.asarray([labels[index] for index in probability.argmax(axis=1)])
    return float(f1_score(y_val, prediction, labels=labels, average="macro", zero_division=0))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    frozen_hidden: bool = False,
) -> tuple[float, bool, bool]:
    model.train()
    if frozen_hidden:
        model.hidden.eval()
    values = []
    loss_finite = True
    gradient_finite = True
    for batch_x, batch_y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_x), batch_y)
        if not torch.isfinite(loss):
            loss_finite = False
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            gradient_finite = False
            raise FloatingPointError("Non-finite training gradient")
        optimizer.step()
        values.append(float(loss.detach()))
    return float(np.mean(values)), loss_finite, gradient_finite


def _finish_result(
    model: StrictMLP2,
    x_test: np.ndarray,
    labels: list[str],
    protocol,
    start: float,
    best_epoch: int,
    best_score: float,
    curve: list[dict],
    details: dict,
    *,
    stage1_best_epoch: int | None = None,
    loss_finite: bool = True,
    gradient_finite: bool = True,
) -> PolicyResult:
    probability = _predict_probability(model, x_test, protocol.batch_size)
    prediction = np.asarray([labels[index] for index in probability.argmax(axis=1)])
    return PolicyResult(
        prediction=prediction,
        probability=probability,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        runtime_seconds=float(time.perf_counter() - start),
        best_epoch=best_epoch,
        best_validation_macro_f1=best_score,
        stage1_best_epoch=stage1_best_epoch,
        curve=curve,
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        details=details,
        loss_finite=loss_finite,
        gradient_finite=gradient_finite,
        warnings=[],
    )


def _standard_loss(
    policy: str,
    reference_y: np.ndarray,
    labels: list[str],
) -> tuple[nn.Module, dict]:
    if policy in {"unweighted_ce", "class_weighted_ce", "focal_loss"}:
        loss_fn, spec = build_loss(policy, reference_y, labels, torch.device("cpu"))
        return loss_fn, spec.__dict__
    if policy == "logit_adjusted_ce":
        counts = label_counts(reference_y, labels)
        priors = counts / counts.sum()
        tau = 1.0
        return LogitAdjustedLoss(priors, tau=tau), {
            "name": policy,
            "class_counts": counts.tolist(),
            "class_priors": priors.tolist(),
            "tau": tau,
            "training_adjustment": "logits + tau * log(inner-train class prior)",
            "inference_adjustment": "none; argmax raw model logits",
        }
    raise ValueError(policy)


def run_standard_policy(
    policy: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    labels: list[str],
    protocol,
    seed: int,
    reference_y: np.ndarray,
    smoke: bool,
) -> PolicyResult:
    seed_everything(seed)
    start = time.perf_counter()
    model = StrictMLP2(
        x_train.shape[1], len(labels), protocol.hidden_1, protocol.hidden_2, protocol.dropout
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    loss_fn, loss_spec = _standard_loss(policy, reference_y, labels)
    train_index = _indices(y_train, labels)
    loader = _loader(x_train, train_index, protocol.batch_size, seed)
    max_epochs = 2 if smoke else protocol.max_epochs
    patience = 2 if smoke else protocol.patience
    best_score, best_epoch, stale = -1.0, 0, 0
    best_state = None
    curve = []
    loss_finite = True
    gradient_finite = True
    for epoch in range(1, max_epochs + 1):
        train_loss, finite_loss, finite_gradient = _train_epoch(model, loader, optimizer, loss_fn)
        loss_finite &= finite_loss
        gradient_finite &= finite_gradient
        score = _validation_macro_f1(model, x_val, y_val, labels, protocol.batch_size)
        curve.append({
            "stage": "joint_training",
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_f1": score,
        })
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return _finish_result(
        model,
        x_test,
        labels,
        protocol,
        start,
        best_epoch,
        best_score,
        curve,
        {"policy": policy, "loss_spec": loss_spec, "schedule": "formal-v1 early stopping"},
        loss_finite=loss_finite,
        gradient_finite=gradient_finite,
    )


def run_ldam_drw(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    labels: list[str],
    protocol,
    seed: int,
    reference_y: np.ndarray,
    smoke: bool,
) -> PolicyResult:
    seed_everything(seed)
    start = time.perf_counter()
    model = StrictMLP2(
        x_train.shape[1], len(labels), protocol.hidden_1, protocol.hidden_2, protocol.dropout
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    counts = label_counts(reference_y, labels)
    drw_weights = effective_number_weights(counts, beta=0.9999)
    max_epochs = 4 if smoke else protocol.max_epochs
    switch_after_epoch = 2 if smoke else 40
    loader = _loader(x_train, _indices(y_train, labels), protocol.batch_size, seed)
    best_score, best_epoch, best_state = -1.0, 0, None
    curve = []
    loss_finite = True
    gradient_finite = True
    for epoch in range(1, max_epochs + 1):
        stage2 = epoch > switch_after_epoch
        loss_fn = LDAMLoss(
            counts,
            max_margin=0.5,
            scale=30.0,
            weights=drw_weights if stage2 else None,
        )
        train_loss, finite_loss, finite_gradient = _train_epoch(model, loader, optimizer, loss_fn)
        loss_finite &= finite_loss
        gradient_finite &= finite_gradient
        score = _validation_macro_f1(model, x_val, y_val, labels, protocol.batch_size)
        curve.append({
            "stage": "ldam_drw" if stage2 else "ldam_pre_drw",
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_f1": score,
        })
        # The selected checkpoint must be after DRW activates.
        if stage2 and score > best_score + 1e-12:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise AssertionError("LDAM-DRW never entered the deferred reweighting stage")
    model.load_state_dict(best_state)
    margins = 1.0 / np.sqrt(np.sqrt(counts.astype(np.float64)))
    margins *= 0.5 / margins.max()
    details = {
        "policy": "ldam_drw",
        "class_counts": counts.tolist(),
        "margin_formula": "Delta_c = C / n_c^(1/4), normalized so max margin=0.5",
        "margins": margins.tolist(),
        "scale": 30.0,
        "drw_beta": 0.9999,
        "drw_weights": drw_weights.tolist(),
        "switch_after_epoch": switch_after_epoch,
        "stage2_first_epoch": switch_after_epoch + 1,
        "forced_schedule": True,
        "checkpoint_selection": "best validation macro F1 within DRW stage only",
    }
    return _finish_result(
        model,
        x_test,
        labels,
        protocol,
        start,
        best_epoch,
        best_score,
        curve,
        details,
        loss_finite=loss_finite,
        gradient_finite=gradient_finite,
    )


def _train_crt_stage(
    model: StrictMLP2,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: list[str],
    protocol,
    seed: int,
    stage: str,
    smoke: bool,
    *,
    class_balanced_sampler: bool,
) -> tuple[dict, int, float, list[dict], bool, bool]:
    train_index = _indices(y_train, labels)
    sampler = None
    if class_balanced_sampler:
        counts = np.bincount(train_index, minlength=len(labels))
        sample_weights = 1.0 / counts[train_index]
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_index),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loader = _loader(
        x_train,
        train_index,
        protocol.batch_size,
        seed,
        sampler=sampler,
    )
    parameters = model.classifier.parameters() if class_balanced_sampler else model.parameters()
    optimizer = torch.optim.AdamW(
        parameters, lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    max_epochs = 2 if smoke else protocol.max_epochs
    patience = 2 if smoke else protocol.patience
    best_score, best_epoch, stale = -1.0, 0, 0
    best_state = None
    curve = []
    loss_finite = True
    gradient_finite = True
    for epoch in range(1, max_epochs + 1):
        train_loss, finite_loss, finite_gradient = _train_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            frozen_hidden=class_balanced_sampler,
        )
        loss_finite &= finite_loss
        gradient_finite &= finite_gradient
        score = _validation_macro_f1(model, x_val, y_val, labels, protocol.batch_size)
        curve.append({
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_f1": score,
        })
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    return best_state, best_epoch, best_score, curve, loss_finite, gradient_finite


def run_crt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    labels: list[str],
    protocol,
    seed: int,
    reference_y: np.ndarray,
    smoke: bool,
) -> PolicyResult:
    seed_everything(seed)
    start = time.perf_counter()
    model = StrictMLP2(
        x_train.shape[1], len(labels), protocol.hidden_1, protocol.hidden_2, protocol.dropout
    )
    stage1 = _train_crt_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        labels,
        protocol,
        seed,
        "crt_stage1_natural",
        smoke,
        class_balanced_sampler=False,
    )
    stage1_state, stage1_epoch, stage1_score, stage1_curve, loss_ok1, grad_ok1 = stage1
    model.load_state_dict(stage1_state)
    hidden_before = copy.deepcopy(model.hidden.state_dict())
    classifier_before = copy.deepcopy(model.classifier.state_dict())

    torch.manual_seed(seed + 1_000_000)
    model.classifier.reset_parameters()
    classifier_reinitialized = any(
        not torch.equal(classifier_before[key], model.classifier.state_dict()[key])
        for key in classifier_before
    )
    for parameter in model.hidden.parameters():
        parameter.requires_grad = False
    hidden_frozen = not any(parameter.requires_grad for parameter in model.hidden.parameters())

    stage2 = _train_crt_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        labels,
        protocol,
        seed + 2_000_000,
        "crt_stage2_classifier_balanced",
        smoke,
        class_balanced_sampler=True,
    )
    stage2_state, stage2_epoch, stage2_score, stage2_curve, loss_ok2, grad_ok2 = stage2
    model.load_state_dict(stage2_state)
    hidden_unchanged = all(
        torch.equal(hidden_before[key], model.hidden.state_dict()[key]) for key in hidden_before
    )
    counts = label_counts(reference_y, labels)
    details = {
        "policy": "crt",
        "stage1": {
            "sampling": "instance-balanced natural sampling",
            "loss": "unweighted CE",
            "optimizer": "AdamW formal-v1 settings",
            "max_epochs": 2 if smoke else protocol.max_epochs,
            "patience": 2 if smoke else protocol.patience,
            "best_epoch": stage1_epoch,
            "best_validation_macro_f1": stage1_score,
        },
        "stage2": {
            "hidden_frozen": hidden_frozen,
            "classifier_reinitialized": classifier_reinitialized,
            "hidden_unchanged_after_training": hidden_unchanged,
            "hidden_mode_during_stage2": "eval (dropout disabled)",
            "sampling": "replacement sampling with per-example weight 1 / inner-train class count",
            "class_counts": counts.tolist(),
            "loss": "unweighted CE",
            "optimizer": "AdamW on classifier parameters only; formal-v1 lr/wd",
            "max_epochs": 2 if smoke else protocol.max_epochs,
            "patience": 2 if smoke else protocol.patience,
            "best_epoch": stage2_epoch,
            "best_validation_macro_f1": stage2_score,
        },
    }
    if not (classifier_reinitialized and hidden_frozen and hidden_unchanged):
        raise AssertionError(f"cRT stage invariant failed: {details['stage2']}")
    return _finish_result(
        model,
        x_test,
        labels,
        protocol,
        start,
        stage2_epoch,
        stage2_score,
        [*stage1_curve, *stage2_curve],
        details,
        stage1_best_epoch=stage1_epoch,
        loss_finite=loss_ok1 and loss_ok2,
        gradient_finite=grad_ok1 and grad_ok2,
    )


def run_policy(
    policy: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    labels: list[str],
    protocol,
    seed: int,
    reference_y: np.ndarray,
    smoke: bool,
) -> PolicyResult:
    if policy in {"unweighted_ce", "class_weighted_ce", "focal_loss", "logit_adjusted_ce"}:
        return run_standard_policy(
            policy,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            labels,
            protocol,
            seed,
            reference_y,
            smoke,
        )
    if policy == "ldam_drw":
        return run_ldam_drw(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            labels,
            protocol,
            seed,
            reference_y,
            smoke,
        )
    if policy == "crt":
        return run_crt(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            labels,
            protocol,
            seed,
            reference_y,
            smoke,
        )
    raise ValueError(policy)
