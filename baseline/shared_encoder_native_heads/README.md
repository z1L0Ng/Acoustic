# ICBHI + SPRSound Shared Encoder with Native Heads

This package is the minimal controlled multi-dataset reference. It uses one
AudioSet-initialized AST encoder and three independent heads:

- ICBHI cycle flat4: normal / crackle / wheeze / both.
- SPRSound BioCAS2022 event Task 1-1: normal / adventitious.
- SPRSound BioCAS2022 event Task 1-2: the official seven event classes.

The SPR binary head is trained independently; it is not a post-hoc collapse of
the seven-class head. Each homogeneous batch is routed only to its dataset's
head(s). Missing labels are never synthesized as negatives. On SPR batches the
two native CE losses are averaged, so adding the second native head does not
silently double the dataset's loss weight.

## Protocol boundaries

- Initialization is pinned
  `MIT/ast-finetuned-audioset-10-10-0.4593@f826b80...`, source safetensors
  SHA256 `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d`,
  converted into the Patch-Mix legacy key layout. The 155 model-used tensors
  were previously verified equivalent to the then-served author state dicts,
  but this does not establish original serialized-byte identity. No ICBHI
  task-selected checkpoint is allowed.
- ICBHI uses the official recording split and discloses patient 156/218
  overlap. Validation is patient-grouped inside official train.
- SPRSound is pinned to commit
  `874eeb8736ddb78937c2fb5332fc7e7293d0f0ca`. Inter is primary; intra is a
  separate repeated-subject diagnostic and is never pooled with inter.
- Primary training uses unweighted CE and source-proportional batches. The
  dataset-balanced interface is preregistered only; it is not run here.
- Results are reported per dataset/head. Raw challenge Scores are never mixed
  across datasets.

## Bounded local checks

Use the existing `acoustic-patchmix` environment:

```bash
conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.run \
  --phase audit --dataset-root dataset/raw

conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.run \
  --phase bootstrap

conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.run \
  --phase smoke --dataset-root dataset/raw --device cpu

conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.verify \
  --mode smoke

conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.run \
  --phase profile --dataset-root dataset/raw --device cpu --profile-steps 100

conda run -n acoustic-patchmix python -m baseline.shared_encoder_native_heads.verify \
  --mode profile
```

Generated outputs are restricted to
`result/icbhi_sprsound_shared_encoder_native_heads/`; source/checkpoint/runtime
cache is restricted to `.cache/icbhi_sprsound_shared_encoder_native_heads/`.
The profile decides whether the future full run is local-feasible. This package
does not authorize or launch that full run.
