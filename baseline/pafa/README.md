# PAFA author-repository reproduction

This directory owns the maintainable PAFA execution surface. Every run creates a new `result/pafa_YYYYMMDD_HHMMSS/` root; no source, checkpoint, cache, log, or prediction is stored under `baseline/`.

The planned track is `official_like_test_selected`. Author-equivalent patient ID extraction is verified. The server package may use the recorded public compatibility checkpoint, but it must retain the mirror provenance/SHA and may not claim Microsoft artifact identity. The repo has no license and no task checkpoint, so sharing constraints remain explicit.

Minimum compatible snapshot: `official-reproduction-release-1`.

```bash
conda env create -f baseline/pafa/environment.linux-cu118.yml
RUN_ROOT="result/pafa_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction bootstrap \
  --dataset-root dataset/raw/icbhi_2017 --result-root "$RUN_ROOT" --device cuda
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction verify-bootstrap \
  --result-root "$RUN_ROOT"
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda
```

The independent contract is `codex/2026-07-21/paper_contracts/pafa.json`.

`bootstrap` clones and pins the author repository, rebuilds the 6,898-cycle manifest from raw data, downloads and SHA-verifies the public BEATs compatibility mirror, builds read-only data adapters, and installs a receipted save/resume-only patch. It may instead receive `--checkpoint-path`; the same expected SHA is enforced. A resumable checkpoint must have been created by this release:

```bash
conda run -n acoustic-pafa python -m baseline.pafa.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda \
  --resume "$RUN_ROOT/full/<experiment>/epoch_<N>.pth"
```

The full entry automatically exports and verifies all 2,756 official-test cycle IDs, logits, predictions, confusion matrix, Sp, Se, and Score. `cuda_full_command.sh` preserves the author seed-1 test-selected protocol. With the current public mirror checkpoint the result remains official-like until Microsoft identity is proven.
