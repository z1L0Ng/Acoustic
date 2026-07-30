"""Run the corrected support-aware classifier-only retraining control."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from baseline.four_dataset_frozen_encoder.data import (
    EXPECTED_HF_ASSIGNMENT_SHA256,
    KAUH_LABELS,
    Sample,
    build_samples,
)
from baseline.four_dataset_frozen_encoder.encoder import load_cache, sha256_file
from baseline.four_dataset_frozen_encoder.train import (
    BATCH_SIZE,
    SEED,
    SharedNativeModel,
    TASK_SPECS,
    _multiclass_metrics,
    _prior_receipt,
    _save_outputs,
    _task_indices,
)
from baseline.four_dataset_representation_attribution.run import (
    CONDITION as D2_CONDITION,
    R0,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result/four_dataset_support_aware_crt"
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
CONTRACT_PATH = (
    ROOT / "docs/datasets/four_dataset_task_contract_draft_2026-07-28.json"
)
CACHE_PATH = ROOT / ".cache/four_dataset_pafa_frozen_encoder/embeddings.npz"
CACHE_SHA256 = (
    "f40ae7fe581457bc86d76b93b1ee811e7ea01bc5e098a6daa73db451f96d1b31"
)
GATE_A = (
    ROOT
    / "result/four_dataset_pafa_frozen_encoder/hf_proxy_fixed_v2/"
    "impact_receipt.json"
)
GATE_B_ROOT = (
    ROOT
    / "result/four_dataset_representation_attribution/hf_proxy_fixed_v2"
)
GATE_C = (
    ROOT
    / "result/four_dataset_shared_compatible_head_harmonization/"
    "hf_proxy_fixed_v2/verification.json"
)
T0 = "t0_corrected_r0_d2_reference"
T1 = "t1_support_aware_classifier_retraining"
TASK_ID_MAP = {
    "spr_event_binary": "spr_binary",
    "spr_event_seven": "spr_seven",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module: torch.nn.Module) -> str:
    return _tensor_digest(module.state_dict())


def _read_dependencies() -> tuple[dict[str, object], dict[str, object]]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    payload = json.loads(CONTRACT_PATH.read_text())
    contract = payload["tail_eligibility_contract"]
    counts = Counter(
        row["eligibility"] for row in contract["label_assignments"]
    )
    if (
        protocol["status"] != "preregistered_before_t1_outer_test_scoring"
        or contract["status"] != "tail_eligibility_management_accepted"
        or counts
        != {
            "primary_evaluable": 16,
            "diagnostic_only": 5,
            "not_evaluable": 7,
        }
    ):
        raise RuntimeError("accepted tail contract gate failed")
    gate_a = json.loads(GATE_A.read_text())
    gate_b = json.loads((GATE_B_ROOT / "verification.json").read_text())
    selection = json.loads(
        (GATE_B_ROOT / "comparison/encoder_selection.json").read_text()
    )
    gate_c = json.loads(GATE_C.read_text())
    if (
        gate_a["status"] != "hf_proxy_fix_and_impact_audit_verified"
        or gate_a["hf_proxy"]["assignment_sha256"]
        != EXPECTED_HF_ASSIGNMENT_SHA256
        or gate_b["status"]
        != "hf_proxy_fixed_representation_regression_verified"
        or selection["selected_representation"] != R0
        or selection["selected_cache_sha256"] != CACHE_SHA256
        or gate_c["status"] != "hf_proxy_fixed_r0_harmonization_verified"
        or gate_c["decision"]
        != "hold_or_negative_representation_harmonization_not_supported"
        or sha256_file(CACHE_PATH) != CACHE_SHA256
    ):
        raise RuntimeError("Gate A/B/C dependency failed")
    return protocol, contract


def _eligibility(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    tasks: dict[str, dict[str, object]] = {}
    for row in contract["label_assignments"]:
        task = TASK_ID_MAP.get(str(row["task_id"]), str(row["task_id"]))
        current = tasks.setdefault(
            task,
            {"primary": [], "diagnostic": [], "not_evaluable": [], "support": {}},
        )
        state = str(row["eligibility"])
        bucket = {
            "primary_evaluable": "primary",
            "diagnostic_only": "diagnostic",
            "not_evaluable": "not_evaluable",
        }[state]
        label = str(row["label"])
        current[bucket].append(label)
        support = row["support"]
        if "subtrain" in support:
            positive = int(support["subtrain"][0])
        else:
            positive = int(support["full_release"][0])
        current["support"][label] = positive
    if set(tasks) != set(TASK_SPECS):
        raise RuntimeError("tail contract task mapping failed")
    for task, values in tasks.items():
        primary = list(values["primary"])
        maximum = max(int(values["support"][label]) for label in primary)
        tail = [
            label
            for label in primary
            if int(values["support"][label]) < maximum
        ]
        values["tail"] = tail or primary
    return tasks


def _adapter_hidden(
    adapter: torch.nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, len(embeddings), 4096):
            outputs.append(
                adapter(
                    torch.from_numpy(embeddings[start : start + 4096]).to(device)
                )
                .cpu()
                .numpy()
            )
    hidden = np.concatenate(outputs).astype(np.float32)
    if hidden.shape != (len(embeddings), 256) or not np.isfinite(hidden).all():
        raise RuntimeError("frozen adapter output failed")
    return hidden


def _balanced_multiclass_draw(
    samples: list[Sample],
    task: str,
    epoch: int,
) -> tuple[list[int], dict[str, object]]:
    indices = _task_indices(samples, task, "subtrain")
    by_class: dict[int, list[int]] = {}
    for index in indices:
        by_class.setdefault(int(samples[index].targets[task]), []).append(index)
    observed = sorted(by_class)
    if not observed:
        raise RuntimeError(f"no observed class for {task}")
    rng = np.random.default_rng(SEED + 1000 * epoch + list(TASK_SPECS).index(task))
    labels = np.resize(np.asarray(observed, dtype=int), len(indices))
    rng.shuffle(labels)
    drawn = [
        int(rng.choice(by_class[int(label)]))
        for label in labels
    ]
    return drawn, {
        "eligible_rows": len(indices),
        "observed_classes": observed,
        "source_class_counts": {
            str(label): len(by_class[label]) for label in observed
        },
        "draw_class_counts": dict(
            sorted(Counter(map(int, labels)).items())
        ),
        "replacement": True,
    }


def _balanced_binary_draw(
    samples: list[Sample],
    task: str,
    label_index: int,
    epoch: int,
) -> tuple[list[int], dict[str, object]]:
    indices = _task_indices(samples, task, "subtrain")
    by_value = {
        value: [
            index
            for index in indices
            if int(samples[index].targets[task][label_index]) == value
        ]
        for value in (0, 1)
    }
    if not by_value[0] or not by_value[1]:
        raise RuntimeError(f"missing observed polarity: {task}[{label_index}]")
    seed = SEED + 10_000 * (label_index + 1) + 1000 * epoch + list(
        TASK_SPECS
    ).index(task)
    rng = np.random.default_rng(seed)
    labels = np.resize(np.asarray([0, 1], dtype=int), len(indices))
    rng.shuffle(labels)
    drawn = [int(rng.choice(by_value[int(label)])) for label in labels]
    return drawn, {
        "eligible_rows": len(indices),
        "observed_negative_rows": len(by_value[0]),
        "observed_positive_rows": len(by_value[1]),
        "draw_negative_rows": int(np.sum(labels == 0)),
        "draw_positive_rows": int(np.sum(labels == 1)),
        "replacement": True,
        "unknown_not_annotated_rows_omitted": True,
    }


def _multiclass_validation(
    head: torch.nn.Module,
    hidden: np.ndarray,
    samples: list[Sample],
    task: str,
    device: torch.device,
) -> dict[str, object]:
    indices = _task_indices(samples, task, "validation")
    with torch.inference_mode():
        logits = head(torch.from_numpy(hidden[indices]).to(device))
        predicted = logits.argmax(dim=1).cpu().numpy()
    target = np.asarray([samples[index].targets[task] for index in indices])
    return _multiclass_metrics(
        target.astype(int),
        predicted.astype(int),
        list(TASK_SPECS[task]["labels"]),
        task,
    )


def _train_multiclass(
    head: torch.nn.Linear,
    hidden: np.ndarray,
    samples: list[Sample],
    task: str,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    best_score = -math.inf
    best_state = None
    history = []
    selection = {}
    sampling = {}
    for epoch in range(1, 6):
        drawn, sampling_receipt = _balanced_multiclass_draw(samples, task, epoch)
        sampling[str(epoch)] = sampling_receipt
        head.train()
        losses = []
        for start in range(0, len(drawn), BATCH_SIZE):
            indices = drawn[start : start + BATCH_SIZE]
            values = torch.from_numpy(hidden[indices]).to(device)
            target = torch.tensor(
                [int(samples[index].targets[task]) for index in indices],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(values), target)
            loss.backward()
            if not all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ):
                raise RuntimeError(f"non-finite classifier gradient: {task}")
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _multiclass_validation(
            head, hidden, samples, task, device
        )
        score = float(validation["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_macro_f1": score,
            }
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
            selection = {"epoch": epoch, "validation_macro_f1": score}
    if best_state is None:
        raise RuntimeError(f"no classifier state selected: {task}")
    head.load_state_dict(best_state)
    return history, selection, sampling


def _binary_validation_score(
    row: torch.nn.Linear,
    hidden: np.ndarray,
    samples: list[Sample],
    task: str,
    label_index: int,
    device: torch.device,
) -> float:
    indices = _task_indices(samples, task, "validation")
    with torch.inference_mode():
        probability = torch.sigmoid(
            row(torch.from_numpy(hidden[indices]).to(device)).squeeze(1)
        ).cpu().numpy()
    target = np.asarray(
        [samples[index].targets[task][label_index] for index in indices],
        dtype=int,
    )
    predicted = (probability >= 0.5).astype(int)
    return float(
        f1_score(
            target,
            predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )


def _train_multilabel(
    head: torch.nn.Linear,
    hidden: np.ndarray,
    samples: list[Sample],
    task: str,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    history = {}
    selection = {}
    sampling = {}
    for label_index, label in enumerate(TASK_SPECS[task]["labels"]):
        row = torch.nn.Linear(256, 1).to(device)
        with torch.no_grad():
            row.weight.copy_(head.weight[label_index : label_index + 1])
            row.bias.copy_(head.bias[label_index : label_index + 1])
        optimizer = torch.optim.Adam(row.parameters(), lr=1e-3)
        best_score = -math.inf
        best_state = None
        label_history = []
        label_sampling = {}
        for epoch in range(1, 6):
            drawn, sampling_receipt = _balanced_binary_draw(
                samples, task, label_index, epoch
            )
            label_sampling[str(epoch)] = sampling_receipt
            row.train()
            losses = []
            for start in range(0, len(drawn), BATCH_SIZE):
                indices = drawn[start : start + BATCH_SIZE]
                values = torch.from_numpy(hidden[indices]).to(device)
                target = torch.tensor(
                    [
                        float(samples[index].targets[task][label_index])
                        for index in indices
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = row(values).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target
                )
                loss.backward()
                if not all(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in row.parameters()
                ):
                    raise RuntimeError(
                        f"non-finite classifier gradient: {task}/{label}"
                    )
                optimizer.step()
                losses.append(float(loss.detach()))
            score = _binary_validation_score(
                row, hidden, samples, task, label_index, device
            )
            label_history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_binary_macro_f1": score,
                }
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(row.state_dict())
                selection[label] = {
                    "epoch": epoch,
                    "validation_binary_macro_f1": score,
                }
        if best_state is None:
            raise RuntimeError(f"no classifier row selected: {task}/{label}")
        row.load_state_dict(best_state)
        with torch.no_grad():
            head.weight[label_index].copy_(row.weight[0])
            head.bias[label_index].copy_(row.bias[0])
        history[label] = label_history
        sampling[label] = label_sampling
    return history, selection, sampling


def _load_t0_model(fold: int) -> tuple[SharedNativeModel, dict[str, object]]:
    path = (
        GATE_B_ROOT
        / R0
        / f"fold_{fold}"
        / D2_CONDITION
        / "best.pth"
    )
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["condition"] != D2_CONDITION:
        raise RuntimeError("T0 source checkpoint condition failed")
    model = SharedNativeModel()
    loaded = model.load_state_dict(checkpoint["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("T0 source state failed")
    return model, {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "selection": checkpoint["selection"],
        "adapter_digest": _module_digest(model.adapter),
        "source_heads_digest": _module_digest(model.heads),
    }


def _fresh_model_with_frozen_adapter(
    source: SharedNativeModel,
) -> tuple[SharedNativeModel, dict[str, object]]:
    set_seed()
    model = SharedNativeModel()
    model.adapter.load_state_dict(source.adapter.state_dict(), strict=True)
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(False)
    model.adapter.eval()
    identity = {
        "source_adapter_digest": _module_digest(source.adapter),
        "initial_adapter_digest": _module_digest(model.adapter),
        "source_heads_digest": _module_digest(source.heads),
        "reinitialized_heads_digest": _module_digest(model.heads),
        "adapter_trainable_parameters": sum(
            parameter.numel()
            for parameter in model.adapter.parameters()
            if parameter.requires_grad
        ),
        "head_trainable_parameters": sum(
            parameter.numel() for parameter in model.heads.parameters()
        ),
    }
    if (
        identity["source_adapter_digest"] != identity["initial_adapter_digest"]
        or identity["source_heads_digest"]
        == identity["reinitialized_heads_digest"]
        or identity["adapter_trainable_parameters"] != 0
    ):
        raise RuntimeError("adapter freeze/head reinitialization gate failed")
    return model, identity


def train_fold(
    fold: int,
    samples: list[Sample],
    embeddings: np.ndarray,
    dataset_root: Path,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    source, source_receipt = _load_t0_model(fold)
    model, identity = _fresh_model_with_frozen_adapter(source)
    model = model.to(device)
    hidden = _adapter_hidden(model.adapter, embeddings, device)
    history = {}
    selection = {}
    sampling = {}
    for task, spec in TASK_SPECS.items():
        head = model.heads[task]
        if spec["kind"] == "multiclass":
            task_history, task_selection, task_sampling = _train_multiclass(
                head, hidden, samples, task, device
            )
        else:
            task_history, task_selection, task_sampling = _train_multilabel(
                head, hidden, samples, task, device
            )
        history[task] = task_history
        selection[task] = task_selection
        sampling[task] = task_sampling
    if _module_digest(model.adapter) != identity["source_adapter_digest"]:
        raise RuntimeError("frozen adapter changed during classifier retraining")
    payload = _save_outputs(
        output_dir,
        T1,
        model,
        samples,
        embeddings,
        history,
        selection,
        _prior_receipt(samples),
        device,
        dataset_root,
    )
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "best.pth"
    payload.update(
        {
            "encoder_identity": {
                "representation": R0,
                "cache_sha256": CACHE_SHA256,
                "selection_caveat": (
                    "ICBHI official-test-selected PAFA task encoder"
                ),
                "target_supervised": True,
            },
            "classifier_retraining": {
                **identity,
                "final_adapter_digest": _module_digest(model.adapter),
                "final_heads_digest": _module_digest(model.heads),
                "source_checkpoint": source_receipt,
                "sampling": sampling,
                "epochs": 5,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "seed": SEED,
            },
        }
    )
    write_json(metrics_path, payload)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["encoder_identity"] = payload["encoder_identity"]
    checkpoint["classifier_retraining"] = {
        key: value
        for key, value in payload["classifier_retraining"].items()
        if key != "sampling"
    }
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    return payload


def smoke(
    dataset_root: Path,
    samples: list[Sample],
    embeddings: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    source, source_receipt = _load_t0_model(0)
    model, identity = _fresh_model_with_frozen_adapter(source)
    model = model.to(device)
    hidden = _adapter_hidden(model.adapter, embeddings, device)
    checks = {}
    for task, spec in TASK_SPECS.items():
        head = model.heads[task]
        if spec["kind"] == "multiclass":
            indices, sampling = _balanced_multiclass_draw(samples, task, 1)
            indices = indices[: min(BATCH_SIZE, len(indices))]
            logits = head(torch.from_numpy(hidden[indices]).to(device))
            target = torch.tensor(
                [int(samples[index].targets[task]) for index in indices],
                dtype=torch.long,
                device=device,
            )
            loss = torch.nn.functional.cross_entropy(logits, target)
        else:
            label_checks = []
            loss_values = []
            for label_index, label in enumerate(spec["labels"]):
                indices, sampling = _balanced_binary_draw(
                    samples, task, label_index, 1
                )
                indices = indices[: min(BATCH_SIZE, len(indices))]
                logits = head(torch.from_numpy(hidden[indices]).to(device))[
                    :, label_index
                ]
                target = torch.tensor(
                    [
                        float(samples[index].targets[task][label_index])
                        for index in indices
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                loss_values.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target
                    )
                )
                label_checks.append(
                    {
                        "label": label,
                        "eligible_rows": sampling["eligible_rows"],
                        "unknown_not_annotated_rows_omitted": sampling[
                            "unknown_not_annotated_rows_omitted"
                        ],
                    }
                )
            loss = torch.stack(loss_values).mean()
            sampling = {"labels": label_checks}
        model.zero_grad(set_to_none=True)
        loss.backward()
        if (
            not torch.isfinite(loss)
            or any(parameter.grad is not None for parameter in model.adapter.parameters())
            or not all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            )
        ):
            raise RuntimeError(f"smoke gradient routing failed: {task}")
        checks[task] = {
            "loss": float(loss.detach()),
            "sampling": sampling,
            "adapter_gradient": "none",
            "head_gradient_finite": True,
        }
    if _module_digest(model.adapter) != identity["source_adapter_digest"]:
        raise RuntimeError("smoke changed frozen adapter")
    receipt = {
        "status": "support_aware_crt_smoke_passed",
        "outer_test_loaded": False,
        "encoder_identity": {
            "representation": R0,
            "cache_sha256": CACHE_SHA256,
            "selection_caveat": "ICBHI official-test-selected PAFA task encoder",
        },
        "source_checkpoint": source_receipt,
        "identity": identity,
        "tasks": checks,
        "finite_loss_gradient": True,
        "missing_not_annotated_omitted": True,
    }
    write_json(RESULT_ROOT / "smoke_receipt.json", receipt)
    return receipt


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _condition_dir(condition: str, fold: int) -> Path:
    if condition == T0:
        return GATE_B_ROOT / R0 / f"fold_{fold}" / D2_CONDITION
    return RESULT_ROOT / f"fold_{fold}" / T1


def _test_metrics(condition: str, fold: int) -> dict[str, object]:
    return json.loads(
        (_condition_dir(condition, fold) / "metrics.json").read_text()
    )["test_metrics"]


def _eligible_metrics(
    metrics: dict[str, object],
    eligibility: dict[str, object],
) -> tuple[float, float]:
    primary = list(eligibility["primary"])
    tail = list(eligibility["tail"])
    per_class = metrics["per_class"]
    return (
        float(np.mean([float(per_class[label]["f1"]) for label in primary])),
        float(np.mean([float(per_class[label]["recall"]) for label in tail])),
    )


def aggregate(
    contract: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    eligibility = _eligibility(contract)
    summary = []
    per_class_rows = []
    for condition in (T0, T1):
        for task, spec in TASK_SPECS.items():
            if task == "kauh_raw9":
                rows = []
                for fold in range(5):
                    rows.extend(
                        row
                        for row in _read_predictions(
                            _condition_dir(condition, fold)
                            / "predictions.csv.gz"
                        )
                        if row["task"] == task
                    )
                ids = [row["sample_id"] for row in rows]
                if len(ids) != 336 or len(ids) != len(set(ids)):
                    raise RuntimeError("KAUH OOF coverage failed")
                target = np.asarray(
                    [json.loads(row["true_json"]) for row in rows], dtype=int
                )
                predicted = np.asarray(
                    [json.loads(row["pred_json"]) for row in rows], dtype=int
                )
                runs = [
                    _multiclass_metrics(
                        target, predicted, KAUH_LABELS, "kauh_raw9"
                    )
                ]
                evaluation = "five-fold patient-grouped aggregate OOF"
            else:
                runs = [
                    _test_metrics(condition, fold)[task] for fold in range(5)
                ]
                evaluation = "five KAUH-fold-conditioned outer/test runs"
            eligible_values = [
                _eligible_metrics(run, eligibility[task]) for run in runs
            ]
            row = {
                "condition": condition,
                "task": task,
                "dataset": spec["dataset"],
                "evaluation": evaluation,
                "runs": len(runs),
                "rows": runs[0]["rows"],
                "eligible_macro_f1_mean": float(
                    np.mean([value[0] for value in eligible_values])
                ),
                "eligible_macro_f1_sample_std": (
                    float(np.std([value[0] for value in eligible_values], ddof=1))
                    if len(runs) > 1
                    else None
                ),
                "tail_recall_mean": float(
                    np.mean([value[1] for value in eligible_values])
                ),
                "tail_recall_sample_std": (
                    float(np.std([value[1] for value in eligible_values], ddof=1))
                    if len(runs) > 1
                    else None
                ),
                "primary_labels_json": json.dumps(eligibility[task]["primary"]),
                "tail_labels_json": json.dumps(eligibility[task]["tail"]),
                "diagnostic_labels_json": json.dumps(
                    eligibility[task]["diagnostic"]
                ),
                "not_evaluable_labels_json": json.dumps(
                    eligibility[task]["not_evaluable"]
                ),
            }
            for metric in (
                "macro_f1",
                "weighted_f1",
                "micro_f1",
                "uar",
                "native_score",
                "specificity",
            ):
                row[f"{metric}_mean"] = None
                row[f"{metric}_sample_std"] = None
            metric_names = (
                ("macro_f1", "micro_f1")
                if spec["kind"] == "multilabel"
                else (
                    "macro_f1",
                    "weighted_f1",
                    "uar",
                    "native_score",
                    "specificity",
                )
            )
            for metric in metric_names:
                available = [
                    float(run[metric]) for run in runs if metric in run
                ]
                row[f"{metric}_mean"] = (
                    float(np.mean(available)) if available else None
                )
                row[f"{metric}_sample_std"] = (
                    float(np.std(available, ddof=1))
                    if len(available) > 1
                    else None
                )
            summary.append(row)
            for label in spec["labels"]:
                class_runs = [run["per_class"][label] for run in runs]
                per_class_rows.append(
                    {
                        "condition": condition,
                        "task": task,
                        "dataset": spec["dataset"],
                        "label": label,
                        "eligibility": (
                            "primary_evaluable"
                            if label in eligibility[task]["primary"]
                            else "diagnostic_only"
                            if label in eligibility[task]["diagnostic"]
                            else "not_evaluable"
                        ),
                        "tail_label": label in eligibility[task]["tail"],
                        "support": class_runs[0]["support"],
                        "precision_mean": float(
                            np.mean(
                                [float(value["precision"]) for value in class_runs]
                            )
                        ),
                        "recall_mean": float(
                            np.mean(
                                [float(value["recall"]) for value in class_runs]
                            )
                        ),
                        "f1_mean": float(
                            np.mean([float(value["f1"]) for value in class_runs])
                        ),
                    }
                )
    write_csv(RESULT_ROOT / "summary.csv", summary)
    write_csv(RESULT_ROOT / "per_class_summary.csv", per_class_rows)
    return summary, per_class_rows


def analyze(summary: list[dict[str, object]]) -> dict[str, object]:
    by_key = {
        (str(row["condition"]), str(row["task"])): row for row in summary
    }
    tasks = {}
    material = []
    guardrails = []
    for task, spec in TASK_SPECS.items():
        t0 = by_key[(T0, task)]
        t1 = by_key[(T1, task)]
        eligible_gap = float(t1["eligible_macro_f1_mean"]) - float(
            t0["eligible_macro_f1_mean"]
        )
        tail_gap = float(t1["tail_recall_mean"]) - float(t0["tail_recall_mean"])
        constrained = (
            ["micro_f1"]
            if spec["kind"] == "multilabel"
            else ["weighted_f1", "native_score", "specificity"]
        )
        regressions = {}
        for metric in constrained:
            key = f"{metric}_mean"
            if t0.get(key) is not None and t1.get(key) is not None:
                regressions[metric] = float(t1[key]) - float(t0[key])
        material_improvement = max(eligible_gap, tail_gap) >= 0.03
        guardrail_pass = all(value >= -0.03 for value in regressions.values())
        if material_improvement:
            material.append(task)
        guardrails.append(guardrail_pass)
        tasks[task] = {
            "eligible_macro_f1_delta": eligible_gap,
            "tail_recall_delta": tail_gap,
            "constrained_metric_deltas": regressions,
            "material_improvement": material_improvement,
            "guardrail_pass": guardrail_pass,
        }
    go = len(material) >= 2 and all(guardrails)
    receipt = {
        "status": "support_aware_crt_analysis_complete",
        "comparison": "T1 classifier-only retraining minus corrected R0+D2 T0",
        "tasks": tasks,
        "material_improvement_tasks": material,
        "material_improvement_count": len(material),
        "all_regression_guardrails_pass": all(guardrails),
        "decision": (
            "go_support_aware_classifier_retraining"
            if go
            else "hold_or_negative_support_aware_classifier_retraining"
        ),
        "claim_boundary": (
            "single-seed target-supervised frozen-feature control with an "
            "ICBHI official-test-selected PAFA encoder; no statistical "
            "significance or solved-imbalance claim"
        ),
    }
    write_json(RESULT_ROOT / "decision.json", receipt)
    return receipt


def run_full(
    dataset_root: Path,
    samples_by_fold: list[list[Sample]],
    embeddings: np.ndarray,
    device: torch.device,
) -> None:
    for fold, samples in enumerate(samples_by_fold):
        directory = RESULT_ROOT / f"fold_{fold}" / T1
        artifacts = [
            directory / "best.pth",
            directory / "metrics.json",
            directory / "predictions_label_free.csv.gz",
            directory / "predictions.csv.gz",
        ]
        present = [path.is_file() for path in artifacts]
        if all(present):
            print(f"TRAIN_RESUMED fold={fold}", flush=True)
            continue
        if any(present):
            raise RuntimeError(f"partial T1 artifact set: {directory}")
        temporary = directory.with_name(directory.name + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"stale atomic training directory: {temporary}")
        train_fold(
            fold,
            samples,
            embeddings,
            dataset_root,
            device,
            temporary,
        )
        temporary.replace(directory)
        print(f"TRAIN_COMPLETE fold={fold}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["smoke", "full", "analyze", "all"], default="all"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    protocol, contract = _read_dependencies()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        RESULT_ROOT / "preregistration.json",
        {
            "status": protocol["status"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "tail_contract_sha256": sha256_file(CONTRACT_PATH),
            "hf_assignment_sha256": EXPECTED_HF_ASSIGNMENT_SHA256,
            "source_contract_training_allowed_field": contract[
                "baseline_t1_gate"
            ]["training_allowed"],
            "runtime_unblock_evidence": (
                "Gate A/B/C verified and management explicitly authorized Gate D "
                "after accepting the corrected R0 selection"
            ),
            "outer_test_metrics_read_by_this_run": False,
        },
    )
    samples_by_fold = [
        build_samples(dataset_root, fold)[0] for fold in range(5)
    ]
    canonical = samples_by_fold[0]
    embeddings, _ = load_cache(CACHE_PATH, canonical)
    device = torch.device(args.device)
    if args.phase in {"smoke", "all"}:
        smoke(dataset_root, canonical, embeddings, device)
        if args.phase == "smoke":
            return
    if args.phase in {"full", "all"}:
        run_full(
            dataset_root, samples_by_fold, embeddings, device
        )
        if args.phase == "full":
            return
    summary, _ = aggregate(contract)
    decision = analyze(summary)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
