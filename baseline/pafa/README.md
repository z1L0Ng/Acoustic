# PAFA author-repository reproduction

This directory owns the maintainable PAFA execution surface. Every run creates a new `result/pafa_YYYYMMDD_HHMMSS/` root; no source, checkpoint, cache, log, or prediction is stored under `baseline/`.

The planned track is `official_like_test_selected`. Author-equivalent patient ID extraction is verified. The server package may use the recorded public compatibility checkpoint, but it must retain the mirror provenance/SHA and may not claim Microsoft artifact identity. The repo has no license and no task checkpoint, so sharing constraints remain explicit.

Minimum compatible snapshot: Release 4 based on
`4172524a0e5d7b792de248820439f30874e2ae6d`. Release 3 environments are
immutable and must not be updated or reused.
The exact dependency rationale and artifact hashes are pinned in
`baseline/common/official_environment_r4_contract.json`; the runtime verifier
requires strict `pip check`, imports, version pins, and an allocated CUDA kernel.
The Linux declaration intentionally pins `pip=24.1.2`, and both declarations
pin `transformers==4.38.2` so Torch 2.0.1 remains an enabled backend. Do not
upgrade either package in place.

```bash
conda env create -f baseline/pafa/environment.linux-cu118.yml
RUN_ROOT="result/pafa_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/receipts"
conda run -n acoustic-pafa-r4 python -m baseline.common.verify_official_environment_r4 \
  --method pafa --cuda-mode runtime \
  --output "$RUN_ROOT/receipts/environment_r4.json"
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction bootstrap \
  --dataset-root dataset/raw/icbhi_2017 --result-root "$RUN_ROOT" --device cuda
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction verify-bootstrap \
  --result-root "$RUN_ROOT"
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda
```

The independent contract is `codex/2026-07-21/paper_contracts/pafa.json`.

`bootstrap` clones and pins the author repository, rebuilds the 6,898-cycle manifest from raw data, downloads and SHA-verifies the public BEATs compatibility mirror, builds read-only data adapters, and installs a receipted save/resume-only patch. It may instead receive `--checkpoint-path`; the same expected SHA is enforced. A resumable checkpoint must have been created by this release:

```bash
conda run -n acoustic-pafa-r4 python -m baseline.pafa.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda \
  --resume "$RUN_ROOT/full/<experiment>/epoch_<N>.pth"
```

The full entry automatically exports and verifies all 2,756 official-test cycle IDs, logits, predictions, confusion matrix, Sp, Se, and Score. `cuda_full_command.sh` preserves the author seed-1 test-selected protocol. With the current public mirror checkpoint the result remains official-like until Microsoft identity is proven.
