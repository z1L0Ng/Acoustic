# Patch-Mix CL author-repository reproduction

This directory owns the maintainable entry points and compatibility adapters for the pinned Patch-Mix Contrastive Learning reproduction. Every run creates a new `result/patch_mix_cl_YYYYMMDD_HHMMSS/` root; no source, checkpoint, cache, log, or prediction is stored under `baseline/`.

The primary track preserves the author's official-test-per-epoch checkpoint selection and must be labelled `faithful_test_selected`. It is a paper-replication track, not a clean transfer protocol. The server package uses the current author-hosted runtime artifact with its recorded SHA; historical 2023 serialized-byte identity remains unresolved and must not be claimed.

Minimum compatible snapshot: `official-reproduction-release-1`.

```bash
conda env create -f baseline/patch_mix_cl/environment.linux-cu118.yml
RUN_ROOT="result/patch_mix_cl_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction bootstrap \
  --dataset-root dataset/raw/icbhi_2017 --result-root "$RUN_ROOT" --device cuda
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction verify-bootstrap \
  --result-root "$RUN_ROOT"
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda
```

`bootstrap` clones and pins the author repository, rebuilds the 6,898-cycle manifest from raw data, downloads and verifies the current author-hosted AST artifact, builds read-only data adapters, and installs a receipted save/resume-only patch. A resumable checkpoint must have been created by this release:

```bash
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda \
  --resume "$RUN_ROOT/full/<experiment>/epoch_<N>.pth"
```

The full entry automatically exports and verifies all 2,756 official-test cycle IDs, logits, predictions, confusion matrix, Sp, Se, and Score. The independent paper contract is `codex/2026-07-21/paper_contracts/patch_mix_cl.json`.
