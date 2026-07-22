# Patch-Mix CL reproduction gate report (2026-07-21)

## Status

Patch-Mix CL has passed the source, environment, data, preprocessing/forward,
one-step optimization, and 100-step CPU profile gates. A faithful full run has
not started. The bounded checkpoint comparison verifies that both currently
served author Dropbox objects contain the same state dict, but it does not prove
their 2023 serialized-byte identity. This Apple arm64 host exposes neither CUDA
nor usable MPS, so local full training is intentionally deferred. The package
is server-runnable on CUDA Linux.

This status is an official-like compatibility reproduction gate, not a paper
result and not a numerical reproduction.

## Frozen source and paper contract

- Paper: *Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on
  Respiratory Sound Classification*, INTERSPEECH 2023.
- Author repository: `https://github.com/raymin0223/patch-mix_contrastive_learning`.
- Pinned commit: `836b09fea1b70eb29fe0b25afa481286b56f5104`.
- License: MIT.
- Independent contract: `codex/2026-07-21/paper_contracts/patch_mix_cl.json`.
- Task/split: cycle-level normal/crackle/wheeze/both, bundled official 60/40
  recording split.
- Selection: official test is evaluated every epoch; best checkpoint is selected
  by official-test Score. Any full result must be labeled **test-selected**.
- Paper five-run target: Sp `81.66 +/- 3.83`, Se `43.07 +/- 2.80`, Score
  `62.37 +/- 0.61`. A seed-1 run can only be located relative to this reported
  distribution; it cannot reproduce the aggregate.

The survey contract and executable repo script agree on 16 kHz, 8-second repeat
padding with fade, 128-bin fbank, SpecAugment, AST base384, Adam `5e-5`, weight
decay `1e-6`, batch 8, 50 epochs, cosine schedule, EMA `0.5`, temperature
`0.06`, projection dimension 768, alpha `1.0`, and mix beta `1.0`.

## Environment

- Conda env: `acoustic-patchmix` at `/opt/anaconda3/envs/acoustic-patchmix`.
- Python `3.10.20`; PyTorch `2.0.1`; torchaudio `2.0.2`; torchvision
  `0.15.2`; timm `0.4.5`; NumPy `1.26.4`.
- Host: macOS arm64; `torch.cuda.is_available() == False`;
  `torch.backends.mps.is_available() == False`.
- Full lock receipts: `result/patch_mix_cl_20260721_171405/receipts/`.

## Checkpoint boundary and bounded author-artifact comparison

The author Dropbox/AST original `.pth` could not be pinned to a public checksum.
For compatibility smoke only, the cached MIT/Hugging Face model
`MIT/ast-finetuned-audioset-10-10-0.4593` at revision
`f826b80d28226b62986cc218e5cec390b1096902` was reverse-mapped to the author's
legacy state-dict layout.

- Converted checkpoint SHA256:
  `e7efffdc9d7a63eab33b5919f578333a07c9873b9c93c6979e42ff54f24d4d4d`.
- All 155 converted tensors load exactly; only unused timm heads are missing.
- Converted author logits and HF logits match with maximum absolute difference
  `2.956390380859375e-05` under the recorded tolerance.
- This proves format/forward compatibility only. It does **not** prove identity
  with the original `audioset_10_10_0.4593.pth` expected by the paper repo.

After the survey provenance audit, both currently accessible author-hosted
Dropbox objects were downloaded with HTTP header receipts and loaded with
`torch.load(weights_only=True, map_location='cpu')`:

- README/script `ca0b` object: 352,587,700 bytes; SHA256
  `dd1042f2a5a283e6fbe5f914678e3ec3ffefe21235c8eec38d186bd7bd6aa995`.
- Runtime-loader/Patch-Mix `cv4` object: 352,587,836 bytes; SHA256
  `dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`.
- Both contain 159 tensors and have the same canonical sorted state-dict digest:
  `81bf63544bc93fb4d5c7ddafa7c3391cbafbe7d982ba429d5b0181b0e7ecc269`.
- All 159 common tensors are bitwise identical despite different serialization
  bytes. The HF reverse conversion's 155 tensors are also bitwise identical to
  the corresponding tensors in both author artifacts. The four author-only
  tensors are unused ImageNet `head`/`head_dist` parameters.

Therefore the bounded result is **verified current author-hosted state-dict
equivalence**, including the model-used HF conversion tensors. It is not proof
that either file's serialization bytes are unchanged from 2023. The portable
command now points to the current `cv4` artifact because that is the exact branch
referenced by pinned Patch-Mix executable code.

## Smoke and profile

The smoke selected eight cycles covering every official split x flat4 label
combination. Raw WAVs were read-only; cycles were cropped in memory.

- Fbank shape: `8 x 1 x 798 x 128`.
- Encoder/classifier output: embedding `8 x 768`, logits `8 x 4`.
- One-step smoke: CE `1.3972045`, contrastive loss `2.0908487`, total
  `3.4880533`; loss and gradients finite.
- 100-step profile: mean `11.1936 s/step`, measured step total about `1119.36 s`;
  final CE `0.0045905`, contrastive loss `0.6929616`; all losses and gradients
  finite.
- Peak process RSS: `21,109,587,968` bytes on this Mac.
- Compatibility changes: author-expected checkpoint path symlink, device-aware
  allocation for the author's CUDA-hardcoded PatchMixConLoss masks, and
  cycle-ID/output export. Model, split, preprocessing, loss algebra, optimizer,
  and EMA behavior were unchanged.

Warnings were limited to CUDA autocast disabling on CPU, the pinned librosa
`pkg_resources` deprecation, and torchvision's future antialias default. No
numerical or convergence warning occurred.

## Full-run feasibility and command

The author train loop hardcodes `.cuda()` and CUDA AMP. CPU projection is about
`100.5` hours for 25,850 train steps before full validation overhead, while the
100-step smoke already peaks near 21.1 GB RSS. A CPU full run would therefore
be an impractical and misleading substitute for the intended CUDA execution.

The exact author-aligned command is packaged at
`baseline/patch_mix_cl/cuda_full_command.sh`; its `--help` entry was validated in
the pinned environment. The timestamped `portable_run` directory contains a
read-only symlink adapter with 920 WAVs, 920 annotations, and author split counts
539/381 recordings. Its checkpoint symlink targets the currently served `cv4`
author artifact, not the HF conversion. A future result must still say “current
author-hosted checkpoint” unless the authors confirm historical identity.

## Decision

Current classification: **server-runnable / current author-hosted state dict
pinned / numerical reproduction pending on CUDA**.
Required external decisions/resources:

1. Supply a CUDA Linux host; the command package is ready for one seed.
2. Author confirmation remains necessary only for a historical 2023 byte-identity
   claim; it is not inferred from current Dropbox state-dict equivalence.
3. After training, use the shared prediction verifier and require 2,756 unique
   official-test cycle IDs, confusion total 2,756, and exact repo-vs-derived
   Sp/Se/Score agreement before comparing with the paper distribution.

## Artifacts

- `result/patch_mix_cl_20260721_171405/receipts/ast_checkpoint_conversion.json`
- `result/patch_mix_cl_20260721_171405/receipts/ast_author_artifact_comparison.json`
- `result/patch_mix_cl_20260721_171405/receipts/environment_receipt.json`
- `result/patch_mix_cl_20260721_171405/receipts/data_adapter_receipt.json`
- `result/patch_mix_cl_20260721_171405/smoke/patch_mix_cl_smoke_8.json`
- `result/patch_mix_cl_20260721_171405/profile/patch_mix_cl_profile_100.json`
- `result/patch_mix_cl_20260721_171405/profile/patch_mix_cl_smoke_8_outputs.npz`
