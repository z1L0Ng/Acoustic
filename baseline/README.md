# Baselines

This directory contains existing methods only. New project-authored model ideas
belong in `model/`; experiment-specific choices belong in `experiments/`.

## Conda environments

Environment names are stable and never carry release suffixes such as `-r4`.
Dependency fixes update the YAML in the owning baseline directory in place.
Do not create a new environment name for a dependency patch.

| Baseline scope | Environment | Configuration |
|---|---|---|
| AST, BEATs, CLAP, HeAR, OPERA, simple acoustic downstream heads | `acoustic-baseline-downstream` | Each directory contains the same `environment.yml`; `baseline/environment.yml` is the shared source. |
| Patch-Mix CL | `acoustic-patchmix` | `patch_mix_cl/environment.yml`; CUDA: `patch_mix_cl/environment.linux-cu118.yml`. |
| PAFA | `acoustic-pafa` | `pafa/environment.yml`; CUDA: `pafa/environment.linux-cu118.yml`. |
| SG-SCL | `acoustic-sgscl` | `sg_scl/environment.yml`; CUDA: `sg_scl/environment.linux-cu118.yml`. |
| MVST | `acoustic-mvst` | `mvst/environment.yml`; CUDA: `mvst/environment.linux-cu118.yml`. |
| ADD-RSC | `acoustic-addrsc` | `add_rsc/environment.yml`; CUDA: `add_rsc/environment.linux-cu121.yml`. |

The shared downstream YAML covers the checked-in frozen-feature notebooks, not
feature extraction. A future extractor must add a separately named
`environment.feature.yml` before its environment is treated as maintained.
Temporary diagnostics and release-suffixed environments are disposable after
their owning process exits and their receipt is preserved.

On macOS, keep Numba and plotting caches outside read-only site-packages:

```bash
export NUMBA_CACHE_DIR="$PWD/.cache/runtime/numba"
export MPLCONFIGDIR="$PWD/.cache/runtime/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache/runtime/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
```

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
must not be added here; use `.cache/` and `result/`. If an accepted experiment's
best checkpoint fits the repository policy, publish only that file under
`checkpoints/<experiment_id>/best.*`, never inside a baseline package.

Cross-dataset transfer evaluators are shared through `acoustic/evaluation/`.
Their scientific status is tracked by `experiments/index.csv`, not by directory
names or a fixed project-wide protocol schema.
