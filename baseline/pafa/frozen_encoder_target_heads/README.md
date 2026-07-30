# PAFA Frozen Encoder Target Heads

This matched local pilot loads the accepted PAFA task checkpoint, retains only
the trained BEATs audio encoder, mean-pools its pre-classifier frame
representations to 768 dimensions, and trains three new SPRSound target heads.
The source classifier and PAFA projector are verified but discarded.

The data split, heads, optimization, validation selection, metrics, runtime
gate, and label-isolation policy match the accepted Patch-Mix pilot. Run
`profile` before `full`; the full phase refuses to start unless the 100-event
projection is at most 90 minutes and peak RSS is at most 24 GiB.

```bash
conda run -n acoustic-pafa python -m \
  baseline.pafa.frozen_encoder_target_heads.run \
  --phase profile \
  --dataset-root dataset/raw/sprsound \
  --source-repo result/pafa_sprsound_transfer_20260722_235659/source/repo \
  --checkpoint .cache/checkpoints/pafa/server_epoch27/best.pth \
  --backbone-checkpoint .cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt \
  --result-root result/sprsound_pafa_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_pafa_frozen_encoder_target_heads \
  --device cpu --threads 8 --batch-size 8
```

Replace `--phase profile` with `--phase full` only after the independent
profile verifier passes. Official inter labels are not opened until all
label-free predictions have been written.

Package and result verification:

```bash
conda run -n acoustic-pafa python -m \
  baseline.pafa.frozen_encoder_target_heads.verify --mode package
conda run -n acoustic-pafa python -m \
  baseline.pafa.frozen_encoder_target_heads.verify --mode full \
  --result-root result/sprsound_pafa_frozen_encoder_target_heads \
  --cache-root .cache/sprsound_pafa_frozen_encoder_target_heads
```

`profile_full_finetune.py` performs exactly one optimizer step for local
feasibility planning. It does not save a checkpoint or authorize full encoder
training.
