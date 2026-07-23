from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import build_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_lr(x_train, y_train, x_test, labels, protocol, seed):
    start = time.perf_counter()
    model = LogisticRegression(
        C=protocol.lr_c,
        class_weight="balanced",
        max_iter=protocol.lr_max_iter,
        solver=protocol.lr_solver,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train.astype(np.float64, copy=False), y_train)
    warning_text = [f"{type(item.message).__name__}: {item.message}" for item in caught]
    return {
        "prediction": model.predict(x_test.astype(np.float64, copy=False)),
        "parameter_count": int(model.coef_.size + model.intercept_.size),
        "runtime_seconds": float(time.perf_counter() - start),
        "warnings": warning_text,
        "converged": not any(isinstance(item.message, ConvergenceWarning) for item in caught),
        "model": model,
    }


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, head: str, dropout: float,
        hidden_1: int, hidden_2: int,
    ):
        super().__init__()
        if head == "mlp1":
            layers = [
                nn.Linear(input_dim, hidden_1), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_1, output_dim),
            ]
        elif head == "mlp2":
            layers = [
                nn.Linear(input_dim, hidden_1), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_2, output_dim),
            ]
        else:
            raise ValueError(head)
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


@dataclass
class MLPResult:
    prediction: np.ndarray
    parameter_count: int
    runtime_seconds: float
    warnings: list[str]
    best_epoch: int
    best_validation_macro_f1: float
    curve: list[dict]
    state_dict: dict
    class_weights: list[float]
    loss_spec: dict
    loss_finite: bool
    gradient_finite: bool


def _predict(model, x, batch_size, device):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def run_mlp(
    x_train, y_train, x_val, y_val, x_test, labels, protocol, seed, head,
    smoke=False, loss_name="class_weighted_ce", loss_reference_y=None,
):
    seed_everything(seed)
    device = torch.device("cpu")
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_index = np.asarray([label_to_index[value] for value in y_train], dtype=np.int64)
    model = MLP(
        x_train.shape[1], len(labels), head, protocol.dropout,
        protocol.hidden_1, protocol.hidden_2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    loss_reference_y = y_train if loss_reference_y is None else loss_reference_y
    loss_fn, loss_spec = build_loss(loss_name, loss_reference_y, labels, device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_index)),
        batch_size=min(protocol.batch_size, len(x_train)), shuffle=True, generator=generator,
    )
    max_epochs = 2 if smoke else protocol.max_epochs
    patience = 2 if smoke else protocol.patience
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    curve = []
    start = time.perf_counter()
    loss_finite = True
    gradient_finite = True
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            if not torch.isfinite(loss):
                loss_finite = False
                raise FloatingPointError(f"Non-finite {loss_name} loss")
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                gradient_finite = False
                raise FloatingPointError(f"Non-finite {loss_name} gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        val_index = _predict(model, x_val, protocol.batch_size, device)
        val_pred = np.asarray([labels[index] for index in val_index])
        score = float(f1_score(y_val, val_pred, labels=labels, average="macro", zero_division=0))
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    test_index = _predict(model, x_test, protocol.batch_size, device)
    prediction = np.asarray([labels[index] for index in test_index])
    return MLPResult(
        prediction=prediction,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        runtime_seconds=float(time.perf_counter() - start),
        warnings=[], best_epoch=best_epoch, best_validation_macro_f1=best_score,
        curve=curve, state_dict=best_state, class_weights=loss_spec.class_weights or [],
        loss_spec=loss_spec.__dict__, loss_finite=loss_finite, gradient_finite=gradient_finite,
    )
