# MVST author-code fixed-random-file-split reproduction

This tracked package reconstructs the executable author protocol from a fresh
checkout. Large source trees, checkpoints, caches, features, logs, and
predictions are created only under a new gitignored
`result/mvst_YYYYMMDD_HHMMSS/` root.

This is **not** an official ICBHI-split reproduction. The author loader sorts
920 recording IDs, applies `random.Random(1).shuffle`, and uses the first 60%
as train. That gives 552/368 recordings and 4,213/2,685 cycles. The paper's
66.55 target is only compared against this author-code split and its
test-selected fusion result.

Minimum compatible snapshot: Release 4 based on
`4172524a0e5d7b792de248820439f30874e2ae6d`. Release 3 environments are
immutable and must not be updated or reused.
The exact dependency rationale and artifact hashes are pinned in
`baseline/common/official_environment_r4_contract.json`; the runtime verifier
requires strict `pip check`, imports, version pins, and an allocated CUDA kernel.
The Linux declaration intentionally pins `pip=24.1.2`; do not upgrade pip in
place because later pip releases reject the retained CMake 3.26.4 wheel metadata.

## Environment and bootstrap

```bash
conda env create -f baseline/mvst/environment.linux-cu118.yml
conda activate acoustic-mvst

RUN_ROOT="result/mvst_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/receipts"
python -m baseline.common.verify_official_environment_r4 \
  --method mvst --cuda-mode runtime \
  --output "$RUN_ROOT/receipts/environment_r4.json"
python -m baseline.mvst.run_reproduction bootstrap \
  --project-root . \
  --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" \
  --device cuda
python -m baseline.mvst.run_reproduction verify-bootstrap --result-root "$RUN_ROOT"
```

The bootstrap pins author commit
`51f93fa6ffa580d0819ccb59f861582927937264`, downloads the current
author-hosted `audioset_16_16_0.4422.pth`, verifies SHA256
`dc71a6d4d07aeb7e746547f72a141f404e4c167d660bf003179f3865e06a970c`,
rebuilds the raw manifest, records both official and author-random splits, and
creates read-only data/checkpoint adapters.

## Five-view smoke and profile

```bash
python -m baseline.mvst.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1

# The 16x16 view is the checkpoint-native representative for timing/VRAM.
python -m baseline.mvst.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100 --views 16
```

Smoke runs all views `16x16, 32x8, 64x4, 128x2, 256x1`, then asserts exact
cycle-ID and label order before one fusion step. Only 16x16 matches the source
checkpoint patch projection. The author shape filter leaves the other four
view-specific patch projections initialized; this behavior is preserved and
reported rather than silently converted.

## Full and resume

```bash
python -m baseline.mvst.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0

# Resume one or more interrupted encoders and/or fusion in the same run root.
python -m baseline.mvst.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0 \
  --encoder-resume "16=$RUN_ROOT/full/encoders/16/<run-name>/last.pth" \
  --fusion-resume "$RUN_ROOT/full/fusion/last.pth"
```

Each encoder writes one rolling `last.pth` with model, classifier, optimizer,
scaler, epoch, and best Score; fusion also writes a rolling resume checkpoint.
The maintained extractor adds cycle IDs and requires all five ordered ID/label
arrays to be identical before fusion. Final output contains 2,685 unique
author-test cycles, logits, predictions, confusion matrix, Sp, Se, and ICBHI
Score. The repository has no license file or author-trained task checkpoint.
