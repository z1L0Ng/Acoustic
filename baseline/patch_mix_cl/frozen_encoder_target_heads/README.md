# Patch-Mix Frozen Encoder Target Heads

This bounded local pilot freezes the verified Patch-Mix ICBHI encoder and
trains three new SPRSound event heads. It never reuses the source classifier,
updates the encoder, uses intra-subject data, or selects anything on the
official inter-subject test.

The fixed tasks are official Task 1-1 binary, official Task 1-2 seven-class,
and a secondary narrow-four shared-ontology diagnostic. Each target head is
`LayerNorm(768)+Linear`, randomly initialized, trained for five epochs with
unweighted cross entropy, and selected only by the fixed patient-grouped
validation fold.

Run the mandatory CPU gate first:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.frozen_encoder_target_heads.verify --mode package

conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.frozen_encoder_target_heads.run \
  --phase profile \
  --dataset-root dataset/raw/sprsound \
  --checkpoint .cache/checkpoints/patch_mix_cl/author_seed1/best.pth \
  --result-root result/sprsound_patchmix_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_patchmix_frozen_encoder_target_heads \
  --device cpu --threads 8 --batch-size 8

conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.frozen_encoder_target_heads.verify \
  --mode profile \
  --result-root result/sprsound_patchmix_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_patchmix_frozen_encoder_target_heads
```

Full execution is mechanically blocked unless the profile projects at most 90
minutes and peak RSS is at most 24 GiB:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.frozen_encoder_target_heads.run \
  --phase full \
  --dataset-root dataset/raw/sprsound \
  --checkpoint .cache/checkpoints/patch_mix_cl/author_seed1/best.pth \
  --result-root result/sprsound_patchmix_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_patchmix_frozen_encoder_target_heads \
  --device cpu --threads 8 --batch-size 8

conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.frozen_encoder_target_heads.verify \
  --mode full \
  --result-root result/sprsound_patchmix_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_patchmix_frozen_encoder_target_heads
```

The label-free inter predictions are written before annotations are opened for
terminal scoring. Narrow-four results are never named an official SPRSound
Task 1-2 Score.
