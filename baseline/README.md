# Baselines

This directory contains existing methods only. New project-authored model ideas
belong in `model/`; experiment-specific choices belong in `experiments/`.

## Controlled local comparisons

| Directory | Representation or role |
|---|---|
| `ast/`, `clap/`, `beats/` | Frozen foundation embeddings with LR/MLP downstream heads. |
| `hear/`, `opera/`, `simple_acoustic/` | Additional frozen or handcrafted representations. |
| `common/` | Shared controlled-baseline and official-reproduction infrastructure. |

The controlled comparisons cover flat-four and binary tasks, downstream head
capacity, imbalance losses, and a strict patient-grouped long-tail benchmark.
Their maintained results and interpretation are in `docs/BASELINES.md`.

## Strong-method reproduction packages

| Method | Executable interpretation |
|---|---|
| `patch_mix_cl/` | Official-like ICBHI reproduction plus verified author-checkpoint evaluation. |
| `pafa/` | Official-like, test-selected BEATs+PAFA reproduction. |
| `sg_scl/` | Metadata-aware, test-selected reproduction; not audio-only. |
| `mvst/` | Author-code random-file-split reproduction; not official ICBHI split. |
| `add_rsc/` | Two bounded tracks because paper and repository settings conflict. |

Each method owns its environment, bootstrap, smoke, full-run, and prediction
verification entry points. Paper facts are stored in `paper_contract.json` inside
the corresponding method directory. Generated source clones, weights, and runs
must not be added here; use `.cache/` and `result/`.

Cross-dataset transfer evaluators are shared through `acoustic/evaluation/`.
Their scientific status is tracked by `experiments/index.csv`, not by directory
names or a fixed project-wide protocol schema.
