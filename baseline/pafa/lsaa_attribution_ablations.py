"""Approved Native-only and LSAA-without-PAFA attribution runs.

The module reuses the historical benchmark-control and formal JH2 runners. It
does nothing unless ``--run`` is supplied. Queues are fresh-only and stop on the
first error; they never retry or change a scientific parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from baseline.pafa import jh2_main_multiseed_external as external
from baseline.pafa import joint_hierarchy_main_multiseed as main_runner
from baseline.pafa import table2_benchmark_controls as benchmark
from baseline.pafa.joint_hierarchy import PAFAJointHierarchyConfig, _write_json


SEEDS = (0, 1, 42)
VARIANTS = ("native_only", "lsaa_without_pafa")
OUTPUT_ROOT_RELATIVE = Path(
    "result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918"
)


def _root(repo_root: Path, output_root: Path | None) -> Path:
    requested = output_root or OUTPUT_ROOT_RELATIVE
    return requested if requested.is_absolute() else repo_root / requested


def _without_pafa_config(
    repo_root: Path,
    seed: int,
    output_root: Path,
    device: str,
    cpu_threads: int,
) -> PAFAJointHierarchyConfig:
    return PAFAJointHierarchyConfig(
        repo_root=repo_root,
        author_repo=repo_root / "result/pafa_sprsound_transfer_20260722_235659/source/repo",
        checkpoint=repo_root
        / ".cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt",
        icbhi_audio_dir=repo_root
        / "dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database",
        output_dir=output_root / "lsaa_without_pafa" / f"seed_{seed}",
        device=device,
        seed=seed,
        cpu_threads=cpu_threads,
    )


def config_payload(
    repo_root: Path,
    variant: str,
    seed: int,
    *,
    output_root: Path | None = None,
    device: str = "cuda",
    cpu_threads: int = 4,
) -> dict[str, object]:
    root = _root(repo_root, output_root)
    if variant == "native_only":
        config = benchmark.default_config(
            repo_root,
            variant,
            seed,
            device=device,
            cpu_threads=cpu_threads,
            output_root=root,
        )
        return config.to_dict()
    config = _without_pafa_config(repo_root, seed, root, device, cpu_threads)
    return main_runner._config_payload(config, main_runner.WITHOUT_PAFA_MODE)


def run_seed(
    repo_root: Path,
    variant: str,
    seed: int,
    *,
    output_root: Path | None,
    device: str,
    cpu_threads: int,
) -> dict[str, object]:
    root = _root(repo_root, output_root)
    if variant == "native_only":
        config = benchmark.default_config(
            repo_root,
            variant,
            seed,
            device=device,
            cpu_threads=cpu_threads,
            output_root=root,
        )
        return benchmark.run_benchmark_control(config)

    config = _without_pafa_config(repo_root, seed, root, device, cpu_threads)
    training = main_runner.run_seed(config, main_runner.WITHOUT_PAFA_MODE)
    external_result = external.run_seed(
        repo_root,
        seed,
        config.output_dir / "external_hf_kauh",
        main_relative=root / "lsaa_without_pafa",
        device_name=device,
    )
    summary = {
        **training,
        "status": "complete_lsaa_without_pafa_native_hf_kauh_terminal",
        "external_evaluation": {
            "status": external_result["status"],
            "hf_metrics_path": external_result["hf_metrics_path"],
            "hf_cas_metrics_path": external_result["hf_cas_metrics_path"],
            "kauh_metrics_path": external_result["kauh_metrics_path"],
            "hf_cas_auroc": external_result["hf_cas_auroc"],
            "hf_cas_support": external_result["hf_cas_support"],
            "kauh_patient_ba": external_result["kauh_patient_ba"],
            "kauh_patient_support": external_result["kauh_patient_support"],
            "threshold_tuning": False,
            "checkpoint_selection": False,
        },
    }
    _write_json(config.output_dir / "run_summary.json", summary)
    return summary


def _stat(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite attribution metric cannot be aggregated")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def _nested(row: Mapping[str, object], path: Sequence[str]) -> object:
    value: object = row
    for key in path:
        value = value[key]
    return value


def aggregate(
    repo_root: Path, variant: str, *, output_root: Path | None = None
) -> dict[str, object]:
    root = _root(repo_root, output_root) / variant
    rows = [
        json.loads((root / f"seed_{seed}" / "run_summary.json").read_text())
        for seed in SEEDS
    ]
    expected_status = (
        "complete_test_selected_benchmark_control"
        if variant == "native_only"
        else "complete_lsaa_without_pafa_native_hf_kauh_terminal"
    )
    if any(row.get("status") != expected_status for row in rows):
        raise RuntimeError("all three seeds must complete before attribution aggregation")

    if variant == "native_only":
        metric_paths = {
            "icbhi_score": ("terminal", "per_native_task", "icbhi", "average_score"),
            "spr_official_score": (
                "terminal",
                "per_native_task",
                "sprsound",
                "official_score",
            ),
        }
        external_summary: dict[str, object] = {
            "status": "not_applicable",
            "reason": "C/W heads receive no supervision; no HF/readout is reported",
        }
    else:
        metric_paths = {
            "icbhi_score": ("selected_metrics", "icbhi_flat4", "official_score"),
            "spr_official_score": (
                "selected_metrics",
                "sprsound_inter_task1_1",
                "official_score",
            ),
            "spr_mean_cw_auroc": (
                "selected_metrics",
                "sprsound_cw_auroc",
                "mean_cw_auroc",
            ),
        }
        hf_rows = [
            json.loads(Path(row["external_evaluation"]["hf_metrics_path"]).read_text())
            for row in rows
        ]
        kauh_rows = [
            json.loads(Path(row["external_evaluation"]["kauh_metrics_path"]).read_text())
            for row in rows
        ]
        hf_scalars = [external._hf_scalars(row) for row in hf_rows]
        kauh_scalars = [external._kauh_scalars(row) for row in kauh_rows]
        hf_cas_rows = [
            json.loads(
                Path(row["external_evaluation"]["hf_cas_metrics_path"]).read_text()
            )
            for row in rows
        ]
        external_summary = {
            "primary": {
                "hf_cas_auroc": _stat(
                    [float(row["hf_cas_auroc"]) for row in hf_cas_rows]
                ),
                "hf_cas_support": {
                    key: [int(row[key]) for row in hf_cas_rows]
                    for key in ("support", "positive", "negative")
                },
                "kauh_patient_ba": _stat(
                    [
                        float(
                            row["patient_level_after_BDE_probability_mean"][
                                "level1_binary"
                            ]["average_score"]
                        )
                        for row in kauh_rows
                    ]
                ),
                "kauh_compatible_patient_support": [
                    int(
                        row["patient_level_after_BDE_probability_mean"][
                            "level1_binary"
                        ]["rows"]
                    )
                    for row in kauh_rows
                ],
            },
            "hf_d_w_diagnostic": {
                key: _stat([row[key] for row in hf_scalars])
                for key in sorted(hf_scalars[0])
            },
            "kauh_secondary": {
                key: _stat([row[key] for row in kauh_scalars])
                for key in sorted(kauh_scalars[0])
            },
        }

    summary = {
        "status": f"complete_{variant}_three_seed_attribution",
        "variant": variant,
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "sample_sd_ddof": 1,
        "metrics": {
            name: _stat([float(_nested(row, path)) for row in rows])
            for name, path in metric_paths.items()
        },
        "per_seed": rows,
        "external": external_summary,
        "evidence_boundary": "test-selected three-seed attribution diagnostic",
    }
    _write_json(root / "multiseed_summary.json", summary)
    return summary


def run_queue(
    repo_root: Path,
    variant: str,
    *,
    output_root: Path | None,
    device: str,
    cpu_threads: int,
) -> dict[str, object]:
    for seed in SEEDS:
        run_seed(
            repo_root,
            variant,
            seed,
            output_root=output_root,
            device=device,
            cpu_threads=cpu_threads,
        )
    return aggregate(repo_root, variant, output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--queue", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.aggregate:
        print(json.dumps(aggregate(repo_root, args.variant, output_root=args.output_root), indent=2))
        return
    if args.queue:
        if not args.run:
            print(json.dumps({
                "status": "CODE_READY_NOT_RUN",
                "variant": args.variant,
                "seeds": list(SEEDS),
                "output_root": str(_root(repo_root, args.output_root)),
            }, indent=2))
            return
        print(json.dumps(run_queue(
            repo_root,
            args.variant,
            output_root=args.output_root,
            device=args.device,
            cpu_threads=args.cpu_threads,
        ), indent=2))
        return
    if args.seed is None:
        parser.error("--seed is required unless --queue or --aggregate is used")
    if not args.run:
        print(json.dumps({
            "status": "CODE_READY_NOT_RUN",
            "variant": args.variant,
            "seed": args.seed,
            "config": config_payload(
                repo_root,
                args.variant,
                args.seed,
                output_root=args.output_root,
                device=args.device,
                cpu_threads=args.cpu_threads,
            ),
        }, indent=2))
        return
    print(json.dumps(run_seed(
        repo_root,
        args.variant,
        args.seed,
        output_root=args.output_root,
        device=args.device,
        cpu_threads=args.cpu_threads,
    ), indent=2))


if __name__ == "__main__":
    main()
