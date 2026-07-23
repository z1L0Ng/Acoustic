# Acoustic

Research workspace for respiratory-sound benchmarking, cross-dataset evaluation,
and project-authored acoustic models. The repository separates maintained code,
experiment definitions, source data, generated results, and research notes so a
local Mac and a training server can share one Git history without sharing caches.

## Repository map

| Path | Purpose |
|---|---|
| `acoustic/` | Shared data, evaluation, training, and utility code. |
| `baseline/` | Existing methods, paper reproductions, and controlled baseline comparisons. |
| `model/` | Project-authored model implementations only. |
| `experiments/` | Small, extensible experiment definitions and the project experiment index. |
| `checkpoints/` | At most one publishable best checkpoint per registered experiment. |
| `dataset/` | Read-only raw data, lightweight processed metadata, and acquisition/audit scripts. |
| `tests/` | Project-structure and reproducibility checks. |
| `server/` | The single local-to-server operating contract. |
| `docs/` | Maintained project, dataset, baseline, experiment, model, and work-plan documentation. |
| `result/` | Generated run artifacts; ignored by Git and indexed locally. |
| `.cache/` | Rebuildable checkpoints, source clones, features, and preprocessing caches; ignored. |

`results/`, `codex/`, and tracked `tmp/` are retired. Active outputs must use
`result/<experiment_id>/`; historical decisions are condensed into `docs/` and
Git history remains the audit trail.

A registered experiment may publish one `checkpoints/<experiment_id>/best.*`
file when it is no larger than GitHub's 100 MiB regular-file limit. All rolling,
resume, optimizer, and oversized checkpoints remain outside Git.

## Current evidence

- The Patch-Mix author ICBHI checkpoint was reproduced exactly at checkpoint
  inference level: Score `62.1708` versus the author-posted `62.17`.
- The same frozen checkpoint on SPRSound BioCAS2022 inter events achieved binary
  Score `59.38` and narrow-four Score `51.15`. This is exploratory transfer,
  not a degradation claim, until a matched target-trained reference is complete.
- Controlled frozen-feature and strict patient-grouped results show that stronger
  heads and long-tail losses move the minority/specificity operating point, but
  no tested policy solves the minority `both` class without tradeoffs.

See [WORK_PLAN.md](docs/WORK_PLAN.md), [BASELINES.md](docs/BASELINES.md), and
[EXPERIMENTS.md](docs/EXPERIMENTS.md) for the maintained interpretation.

## Basic checks

```bash
python3 -m unittest discover -s tests
python3 -m acoustic.evaluation.verify_transfer_packages --project-root .
git diff --check
```

Method-specific setup and execution commands remain in each `baseline/<method>/README.md`.
