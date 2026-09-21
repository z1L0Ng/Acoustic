"""Fixed native-head HF/KAUH post-hoc for Native+C/W checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_numpy_metric_compat() -> None:
    try:
        import sklearn  # noqa: F401
        return
    except Exception:
        for name in list(sys.modules):
            if name == "sklearn" or name.startswith("sklearn.") or name == "scipy" or name.startswith("scipy."):
                del sys.modules[name]

    metrics = types.ModuleType("sklearn.metrics")
    model_selection = types.ModuleType("sklearn.model_selection")

    def confusion_matrix(y_true, y_pred, labels=None):
        labels = list(np.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred)))) if labels is None else labels)
        index = {label: position for position, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for truth, prediction in zip(y_true, y_pred):
            matrix[index[truth], index[prediction]] += 1
        return matrix

    def precision_recall_fscore_support(y_true, y_pred, labels=None, zero_division=0):
        matrix = confusion_matrix(y_true, y_pred, labels)
        support = matrix.sum(axis=1).astype(np.int64)
        predicted = matrix.sum(axis=0).astype(np.int64)
        true_positive = np.diag(matrix).astype(np.float64)
        precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0)
        recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
        return precision, recall, f1, support

    def f1_score(y_true, y_pred, labels=None, average=None, zero_division=0):
        values = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)[2]
        return float(values.mean()) if average == "macro" else values

    def recall_score(y_true, y_pred, labels=None, average=None, zero_division=0):
        values = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)[1]
        return float(values.mean()) if average == "macro" else values

    def roc_auc_score(y_true, score):
        y_true = np.asarray(y_true, dtype=np.int64)
        score = np.asarray(score, dtype=np.float64)
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        ranks = np.arange(1, len(score) + 1, dtype=np.float64)
        start = 0
        while start < len(score):
            end = start + 1
            while end < len(score) and sorted_score[end] == sorted_score[start]:
                end += 1
            ranks[start:end] = ranks[start:end].mean()
            start = end
        rank_sum = ranks[y_true[order] == 1].sum()
        positive = float((y_true == 1).sum())
        negative = float((y_true == 0).sum())
        return float((rank_sum - positive * (positive + 1.0) / 2.0) / (positive * negative))

    def average_precision_score(y_true, score):
        y_true = np.asarray(y_true, dtype=np.int64)
        order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
        target = y_true[order]
        cumulative = np.cumsum(target)
        positions = np.arange(1, len(target) + 1)
        positive = int(target.sum())
        return float((cumulative[target == 1] / positions[target == 1]).sum() / positive) if positive else 0.0

    class StratifiedGroupKFold:
        def __init__(self, *args, **kwargs):
            pass

        def split(self, *args, **kwargs):
            raise RuntimeError("StratifiedGroupKFold is unavailable in post-hoc inference-only mode")

    class GroupShuffleSplit:
        def __init__(self, *args, **kwargs):
            pass

        def split(self, *args, **kwargs):
            raise RuntimeError("GroupShuffleSplit is unavailable in post-hoc inference-only mode")

    metrics.confusion_matrix = confusion_matrix
    metrics.precision_recall_fscore_support = precision_recall_fscore_support
    metrics.f1_score = f1_score
    metrics.recall_score = recall_score
    metrics.roc_auc_score = roc_auc_score
    metrics.average_precision_score = average_precision_score
    model_selection.StratifiedGroupKFold = StratifiedGroupKFold
    model_selection.GroupShuffleSplit = GroupShuffleSplit
    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    sklearn.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
    metrics.__spec__ = ModuleSpec("sklearn.metrics", loader=None)
    model_selection.__spec__ = ModuleSpec("sklearn.model_selection", loader=None)
    sklearn.metrics = metrics
    sklearn.model_selection = model_selection
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = metrics
    sys.modules["sklearn.model_selection"] = model_selection


_install_numpy_metric_compat()

from baseline.pafa import native_only_external_posthoc as base


REPO = Path("/Users/zilongzeng/Research/Acoustic")
SOURCE_ROOT = REPO / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/native_attributes"
SOURCE_SEED42 = REPO / "result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/native_attributes/seed_42"
OUTPUT_ROOT = REPO / "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_attributes_external_posthoc"
SEEDS = (0, 1, 42)
EVIDENCE_LABEL = "complete_native_attributes_fixed_native_head_external_posthoc"
ORIGINAL_BENCHMARK_CONFIG = base.BenchmarkConfig


def _selected_source(repo_root: Path, seed: int) -> dict[str, object]:
    run_dir = SOURCE_SEED42 if seed == 42 else SOURCE_ROOT / f"seed_{seed}"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    if summary.get("status") != "complete_test_selected_benchmark_control":
        raise RuntimeError(f"Native+C/W source seed {seed} is not complete")
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "best_checkpoint.pt",
        "selected_epoch": int(summary["selected_epoch"]),
        "source_status": summary["status"],
    }


def _native_attributes_config(*args, **kwargs):
    kwargs["variant"] = "native_attributes"
    return ORIGINAL_BENCHMARK_CONFIG(*args, **kwargs)


def _annotate_output(output_dir: Path, seed: int, source: dict[str, object]) -> None:
    for name in ("config.json", "run_summary.json", "hf_cas_metrics.json", "kauh_patient_metrics.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        payload.update(
            {
                "condition": "Native+C/W",
                "condition_label": "LSAA-N w/C+W",
                "seed": seed,
                "source_run": str(source["run_dir"]),
                "checkpoint": str(source["checkpoint"]),
                "source_head": "icbhi_native",
                "evidence_label": EVIDENCE_LABEL,
                "source_evidence_boundary": "ICBHI-official-test-selected Native+C/W benchmark control checkpoint; fixed-head external post-hoc only",
            }
        )
        if name == "run_summary.json":
            payload["status"] = EVIDENCE_LABEL
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stat(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def _aggregate(output_root: Path) -> dict[str, object]:
    rows = [json.loads((output_root / f"seed_{seed}/run_summary.json").read_text()) for seed in SEEDS]
    summary = {
        "status": EVIDENCE_LABEL + "_three_seed",
        "condition": "Native+C/W",
        "condition_label": "LSAA-N w/C+W",
        "seeds": list(SEEDS),
        "n": 3,
        "sample_sd_ddof": 1,
        "selected_epochs": {str(row["seed"]): int(row["selected_epoch"]) for row in rows},
        "hf_cas_auroc": _stat([float(row["hf_cas"]["hf_cas_auroc"]) for row in rows]),
        "kauh_patient_ba": _stat([float(row["kauh_patient"]["kauh_patient_ba"]) for row in rows]),
        "support": {
            "hf": [
                {key: int(row["hf_cas"][key]) for key in ("support", "positive", "negative")}
                for row in rows
            ],
            "kauh_compatible_patients": [int(row["kauh_patient"]["rows"]) for row in rows],
        },
        "readout": {
            "hf": "ICBHI native softmax P(Wheeze)+P(Both), maximum over three existing 5-s windows",
            "kauh": "ICBHI native softmax 1-P(Normal), B/D/E patient mean, >0.5 abnormal, tie Normal",
        },
        "zero_target_boundary": {
            "target_training": False,
            "target_selection": False,
            "target_threshold_fit": False,
        },
        "claim_boundary": "Fixed-head external post-hoc diagnostic; does not modify Native+C/W training, selection, native predictions, or primary summary.",
        "per_seed": rows,
    }
    (output_root / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    hf = summary["hf_cas_auroc"]
    kauh = summary["kauh_patient_ba"]
    lines = [
        "# LSAA-N w/C+W fixed-head HF/KAUH post-hoc",
        "",
        "Evidence: fixed ICBHI-test-selected Native+C/W checkpoints; post-hoc external diagnostic only.",
        "",
        "Readout: trained ICBHI native four-class softmax head. HF score is P(Wheeze)+P(Both), max over three existing 5-s windows; KAUH score is 1-P(Normal), averaged over B/D/E per compatible patient, >0.5 abnormal and tie Normal.",
        "",
        "| Seed | Selected epoch | HF CAS AUROC (%) | KAUH patient BA (%) |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | {100*row['hf_cas']['hf_cas_auroc']:.2f} | {100*row['kauh_patient']['kauh_patient_ba']:.2f} |")
    lines.extend([
        "",
        f"Mean ± sample SD: HF CAS AUROC **{100*hf['mean']:.2f}±{100*hf['sample_sd']:.2f}%**; KAUH patient BA **{100*kauh['mean']:.2f}±{100*kauh['sample_sd']:.2f}%**.",
        "",
        "Support: HF 957 eligible recordings (661/296 CAS positive/negative); KAUH 86 compatible patients. This does not use the unsupervised auxiliary C/W heads and does not change Native+C/W primary native results.",
        "",
    ])
    (output_root / "multiseed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"status": "READY_FOR_USER_START", "source_root": str(SOURCE_ROOT), "output_root": str(args.output_root)}, indent=2))
        return
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite Native+C/W post-hoc output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    base._selected_source = _selected_source
    base.BenchmarkConfig = _native_attributes_config
    for seed in SEEDS:
        source = _selected_source(args.repo_root, seed)
        output_dir = args.output_root / f"seed_{seed}"
        result = base.run_seed(args.repo_root, seed, output_dir, args.device)
        _annotate_output(output_dir, seed, source)
        print(json.dumps({"seed": seed, "selected_epoch": source["selected_epoch"], "output_dir": str(output_dir), "status": result["status"]}, sort_keys=True), flush=True)
    summary = _aggregate(args.output_root)
    print(json.dumps({"status": summary["status"], "output_root": str(args.output_root), "hf_cas_auroc": summary["hf_cas_auroc"], "kauh_patient_ba": summary["kauh_patient_ba"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
