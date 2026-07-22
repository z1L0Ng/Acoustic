# PAFA reproduction gate report (2026-07-21)

## Status

PAFA has passed source, environment, official-split data, patient-ID,
preprocessing/forward, one-step optimization, and 100-step CPU profile gates.
A faithful full run has not started. The exact Microsoft BEATs_iter3+ AS2M
checkpoint identity remains unresolved, and the current host has no CUDA or
usable MPS. The author repository also has no license.

This is an official-like compatibility gate, not a paper numerical result.

## Frozen contract

- Paper/method: Patient-Aware Feature Alignment for Robust Lung Sound
  Classification.
- Author repo: `https://github.com/wa976/PAFA` at
  `e49e294d0db0d6af10ac46290512b9c85d3f71e1`.
- License: none found; code redistribution/use constraints remain unresolved.
- Independent contract: `codex/2026-07-21/paper_contracts/pafa.json`.
- Primary task: ICBHI cycle-level normal/crackle/wheeze/both under the bundled
  official 60/40 recording split.
- Paper five-run target: Sp `82.05 +/- 1.95`, Se `47.63 +/- 2.23`, Score
  `64.84 +/- 0.60`.
- Critical selection: official test every epoch, best checkpoint selected by
  official-test Score; any full run must be labeled **test-selected**.

The paper contract and repo script align on BEATs_iter3+ AS2M, 16 kHz, fixed
5-second repeat/truncate, `--nospec`, Adam `5e-5`, weight decay `1e-6`, batch
32, 100 epochs, cosine, EMA `0.5`, PCSL `50.0`, GPAL `0.0005`, CE/PAFA weights
`1.0/1.0`, LayerNorm projection, and output dimension 768.

## Environment

- Conda env: `acoustic-pafa` at `/opt/anaconda3/envs/acoustic-pafa`.
- Python `3.10.20`; torch `2.0.1`; torchaudio `2.0.2`; torchvision
  `0.15.2`; timm `0.4.5`; NumPy `1.26.4`.
- `wget==3.2` was added after the first smoke import showed that the repo's
  `models/__init__.py` imports AST even for the BEATs route.
- Host: macOS arm64; CUDA unavailable; MPS unavailable.
- Full `pip freeze`, conda explicit lock, and machine-readable environment
  receipt are under `result/pafa_20260721_171405/receipts/`.

## BEATs checkpoint identity

The official Microsoft README names `BEATs_iter3+ (AS2M)` and links a OneDrive
artifact, but the local download probe returned HTTP 403 and no official
checksum was found. The available cached candidate is from the non-Microsoft
`mooneyko/BEATs` Hugging Face mirror:

- Size: 361,499,833 bytes.
- File SHA256:
  `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`.
- Safe structural load: top-level `cfg` and `model`, 250 model tensors.
- Canonical sorted tensor-state digest:
  `ea048bce4e819fc8d0619c1efc60fdcd7ddbfe6061b514b0060a3f0721b98631`.
- Config signature: 12 layers, 768 encoder dimension, 12 heads, FFN 3072,
  patch size 16, tokenizer embedding dimension 512.

This satisfies the official BEATs loading structure and is acceptable for the
server-runnable official-like track. It is not established as the official
Microsoft artifact and cannot support an unconditional faithful identity claim.

## Patient-ID and smoke result

The repo extracts patient ID as `int(filename.split('_')[0])`. The complete data
receipt verifies that rule over the ICBHI manifest; the 8-cycle smoke uses four
official-train patients with exactly two cycles each: 103, 105, 106, and 107.
No recording/cycle ID substitution was used.

- Input waveform: `8 x 80000`, 16 kHz mono, in-memory cycle crop, fixed 5 s,
  repeat+fade, no SpecAugment.
- BEATs frame features: `8 x 248 x 768`.
- PAFA projection: `8 x 768`; logits: `8 x 4`.
- One-step CE `1.6203741`, PAFA loss `4.3141704`, total `5.9345446`.
- Loss and gradients finite; EMA beta `0.5` applied after optimizer update.

## 100-step profile and full blocker

- Mean measured compute: `1.47618 s/step` on the 8-cycle CPU smoke batch.
- Wall time including state copy and EMA: `150.7934 s` for 100 steps.
- Final CE `0.3011833`, PAFA loss `0.0794559`, total `0.3806391`.
- Peak process RSS: 6,601,146,368 bytes.
- No numerical/convergence warning occurred. Warnings only reflected disabled
  CUDA autocast on CPU and the pinned librosa deprecation.

The faithful script requires batch 32, 100 epochs, full BEATs fine-tuning, and
hardcodes `.cuda()`. Even the impossible lower bound obtained by applying the
batch-8 timing to roughly 12,900 train steps exceeds five hours before larger
batch cost and 100 official-test evaluations; the real CPU cost is materially
higher. A Mac CPU full run is therefore not an acceptable reproduction path.

The exact author-aligned seed-1 command is packaged at
`baseline/pafa/cuda_full_command.sh`, and its `--help` entry imports successfully
inside the pinned env. A project-local adapter provides read-only symlinks to
920 WAVs/920 annotations, author official split files (539/381 recordings), and
keeps author cache files under the timestamped result root. With the recorded
mirror checkpoint, the command produces an official-like run and must retain
checkpoint provenance and SHA in the result title/receipt.

## Decision

Current classification: **server-runnable official-like / dependency, data,
patient-ID, model and profile gates passed / numerical result pending on CUDA**.

Required next resources/decisions:

1. Prefer verifying the Microsoft BEATs_iter3+ AS2M artifact against the cached
   candidate, but do not block the explicitly official-like server run on it.
2. Resolve the repository's absent LICENSE before redistribution or external use.
3. Run the packaged one-seed command on CUDA Linux, preserving test-selected
   behavior.
4. Export 2,756 unique official-test cycle IDs and require prediction-derived
   confusion/Sp/Se/Score to match repo metrics exactly before paper comparison.

## Artifacts

- `result/pafa_20260721_171405/receipts/beats_checkpoint_identity_receipt.json`
- `result/pafa_20260721_171405/receipts/environment_receipt.json`
- `result/pafa_20260721_171405/receipts/data_adapter_receipt.json`
- `result/pafa_20260721_171405/smoke/pafa_smoke_8.json`
- `result/pafa_20260721_171405/profile/pafa_profile_100.json`
- `result/pafa_20260721_171405/profile/pafa_smoke_8_outputs.npz`
