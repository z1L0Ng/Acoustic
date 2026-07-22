# Official reproduction Release 1 candidate

## Snapshot status

**SAFE_TO_SNAPSHOT.** Patch-Mix CL and PAFA now rebuild from a candidate clean checkout using only tracked code plus `dataset/raw/icbhi_2017`. The bootstrap creates a new timestamped run under canonical `result/`, clones and pins the author repository, rebuilds the 6,898-cycle manifest, downloads or accepts an explicit checkpoint, enforces SHA256, builds read-only adapters, and writes server-local receipts. No command depends on a historical `result/` artifact, the removed `results/` tree, a local manifest, or an absolute workstation path.

This is a release-candidate state, not a deployable server revision until management snapshots it and communicates the resulting commit SHA. The server must run from that detached commit.

## Release-candidate files

All regular files under:

- `baseline/patch_mix_cl/` excluding `__pycache__/`
- `baseline/pafa/` excluding `__pycache__/`

Shared runtime files:

- `baseline/common/export_official_predictions.py`
- `baseline/common/install_official_compatibility.py`
- `baseline/common/official_reproduction_bootstrap.py`
- `baseline/common/official_reproduction_cli.py`
- `baseline/common/verify_official_predictions.py`
- `baseline/common/verify_extension.py` (explicit six-notebook scope regression fix)

Contracts and receipts:

- `codex/2026-07-21/paper_contracts/patch_mix_cl.json`
- `codex/2026-07-21/paper_contracts/pafa.json`
- `codex/2026-07-21/survey_ast_checkpoint_identity_audit_2026-07-21.md`
- `codex/2026-07-21/patch_mix_cl_reproduction_gate_report_2026-07-21.md`
- `codex/2026-07-21/pafa_reproduction_gate_report_2026-07-21.md`
- `codex/2026-07-21/official_reproduction_release_1_clean_checkout_receipt_2026-07-21.json`
- `codex/2026-07-21/official_reproduction_release_1_candidate_2026-07-21.md`
- `codex/2026-07-21/official_reproduction_release_1_candidate_sha256_2026-07-21.txt`

The companion SHA manifest contains the exact file list and excludes itself. SG-SCL, MVST, ADD-RSC, and aggregate five-method material remain Release 2.

## Fresh-checkout inputs

Patch-Mix default checkpoint:

- URL: `https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1`
- Size: 352,587,836 bytes
- SHA256: `dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`
- Boundary: current author-hosted runtime artifact; historical 2023 serialized-byte identity remains unresolved.

PAFA default checkpoint:

- URL: `https://huggingface.co/mooneyko/BEATs/resolve/18cfdf9d43820b2db86c6dfde4ae2f7531d1f5ad/BEATs_iter3_plus_AS2M.pt`
- Size: 361,499,833 bytes
- SHA256: `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`
- Boundary: public compatibility mirror; Microsoft-hosted byte identity is unverified, so results remain `official_like_test_selected`.

Both bootstrap commands also accept `--checkpoint-path`; the same SHA is mandatory. Checkpoints are copied or hard-linked into the new run root and receipted. Neither checkpoint is committed.

## Server commands

Patch-Mix:

```bash
git switch --detach <release-1-commit-sha>
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

PAFA:

```bash
git switch --detach <release-1-commit-sha>
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

The full command preserves author test-per-epoch selection, then exports and verifies 2,756 test cycle IDs, labels, logits, probabilities, predictions, confusion matrix, Sp, Se, and Score. Runs must be marked `test-selected`; PAFA must additionally be marked checkpoint-mirror official-like.

Resume is supported only for checkpoints created after the bootstrap-installed save/resume compatibility patch. For example:

```bash
conda run -n acoustic-patchmix python -m baseline.patch_mix_cl.run_reproduction full \
  --result-root "$RUN_ROOT" --device cuda \
  --resume "$RUN_ROOT/full/<experiment>/epoch_<N>.pth"
```

The patch stores/restores backbone, classifier, projector, optimizer, epoch, and historical best Score/model state. Its upstream file diff and SHA are in each bootstrap receipt; uninterrupted training behavior is unchanged.

## Acceptance

- Bootstrap: pinned source HEAD, expected compatibility diff SHA, checkpoint SHA, 920 recording links, 6,898 unique manifest IDs, author train/test cycles 4,142/2,756.
- Smoke/profile: finite inputs, loss, gradients, and logits; requested CUDA device and peak memory receipted on server.
- Full: 2,756 unique official-test cycle IDs, confusion total 2,756, finite logits/probabilities, and exact author-vs-derived Sp/Se/Score agreement.
- Scientific label: a single seed can be aligned with a paper-reported distribution but cannot reproduce a five-seed aggregate.

## Verification receipt

Candidate clean checkout: `/private/tmp/acoustic-release1-candidate.j25dPZ`, reconstructed from committed HEAD `6400127d65287dd0bd7150042892769e48881976` plus only the listed candidate files. The dataset appeared only at the expected `dataset/raw/icbhi_2017` path. No prior output tree was mounted.

- Public-URL Patch-Mix run: `result/patch_mix_cl_20260721_194715`; bootstrap verification passed, 1-step smoke passed, eight-row prediction/confusion wiring passed, and complete resume-state serialization unit check passed.
- Public-URL PAFA run: `result/pafa_20260721_195158`; bootstrap verification passed, patient-ID 1-step smoke passed, eight-row prediction/confusion wiring passed, and complete resume-state serialization unit check passed.
- Both author adapters independently counted train/test cycles as 4,142/2,756.
- Python compilation and Bash syntax passed.
- Formal-v2 extension verifier now checks an explicit list of six notebooks; isolated notebook check passed.
- Official-reproduction notebooks are valid and clean; `result/` is ignored; `results/` remains absent and ignored.
- No CUDA full run was attempted on the Mac. Full export code is packaged but awaits server execution.
- No dataset, Notion, stage, commit, or push action was performed by this thread.

Machine-readable evidence: `codex/2026-07-21/official_reproduction_release_1_clean_checkout_receipt_2026-07-21.json`.
