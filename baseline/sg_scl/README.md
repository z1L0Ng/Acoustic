# SG-SCL metadata-aware official-like reproduction

This tracked package reconstructs the author protocol from a fresh checkout.
Large source trees, checkpoints, caches, logs, and predictions are created only
under a new gitignored `result/sg_scl_YYYYMMDD_HHMMSS/` root.

Classification: `metadata_aware_official_like_test_selected`. The author
official recording split and device-domain objective are preserved. This is not
an audio-only baseline, the repository has no license file, and the official
test is evaluated every epoch to select the reported checkpoint.

Minimum compatible release: Patch-Mix/PAFA Release 1 commit
`51626840f6ec325086f68bd88446ff956f7e0357` plus this SG-SCL package.

## Linux/CUDA environment

```bash
conda env create -f baseline/sg_scl/environment.linux-cu118.yml
conda activate acoustic-sgscl
```

## Fresh-checkout bootstrap

Use a new America/Chicago timestamp for every run:

```bash
RUN_ROOT="result/sg_scl_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
python -m baseline.sg_scl.run_reproduction bootstrap \
  --project-root . \
  --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" \
  --device cuda
python -m baseline.sg_scl.run_reproduction verify-bootstrap --result-root "$RUN_ROOT"
```

The bootstrap clones and pins the author source, downloads the current
author-hosted AST runtime checkpoint, verifies its SHA256, reconstructs the
6,898-cycle manifest from raw data, asserts all device labels, and creates a
read-only author-layout adapter. An explicit checkpoint can be supplied with
`--checkpoint-path`; it must match the pinned SHA256.

## Smoke, profile, full, and resume

```bash
python -m baseline.sg_scl.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
python -m baseline.sg_scl.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100
python -m baseline.sg_scl.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0

# Resume only from a checkpoint inside the same timestamped run root.
python -m baseline.sg_scl.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0 \
  --resume "$RUN_ROOT/full/da2/<run-name>/epoch_<N>.pth"
```

The full entry preserves seed 1, 50 epochs, test-per-epoch selection, and the
paper/repo hyperparameters. After training it exports exactly 2,756 official
test cycle IDs, logits, probabilities, predictions, confusion matrix, Sp, Se,
and ICBHI Score and verifies them from the exported predictions.

The compatibility patch is receipt-backed and limited to device allocation,
complete interruption-safe checkpoint state, and prediction export. It does
not alter model, loss, split, preprocessing, hyperparameters, selection, or
metrics. Historical byte identity of the 2023 AST checkpoint remains
unresolved; the pinned file is the current artifact used by the author runtime
URL.
