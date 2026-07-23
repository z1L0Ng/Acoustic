# SG-SCL B2: Frozen Checkpoint to SPRSound Inter

This package runs the accepted SG-SCL server checkpoint whose container was
saved at epoch 50 and whose embedded/top-level predictive state was selected at
best epoch 27. It uses the exact SPRSound BioCAS2022 inter-subject event set
from Patch-Mix B0. It is a published-model, ICBHI-test-selected,
zero-target-tuning exploratory transfer, not a clean source anchor or a
target-trained reference.

## Frozen Contract

- Target: 1,429 official inter-subject events, ordered event-ID SHA256
  `81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b`.
- Source classes: `normal`, `crackle`, `wheeze`, `both`.
- Binary result: normal iff four-class argmax is normal; all other predictions
  collapse to abnormal. No threshold is fitted.
- SG-SCL preprocessing: mono 16 kHz, full-recording fade, event-boundary crop,
  author 8 s repeat/truncate, 128-bin fbank, resize to `798 x 128`, augmentation
  off.
- SG-SCL source training is metadata/device-aware. The author `validate()` path
  uses only the audio encoder and class classifier. This package loads that
  predictive state and does not invent SPRSound device/domain labels.
- Labels are absent from smoke and model inference artifacts. Full mode joins
  labels only after all logits have been written for final scoring.
- No target calibration, preprocessing tuning, checkpoint selection, intra
  subset, target training, HF_Lung, or KAUH execution.

## Prerequisites

- Checkout containing this tracked package.
- Frozen environment `acoustic-sgscl-r4` from
  `baseline/sg_scl/environment.linux-cu118.yml`.
- Accepted SG-SCL task-checkpoint container: size `1,413,896,983` bytes, SHA256
  `8b3652d0dc82b9033251e3aab50ec1b51d328b6d3c836807b7c8de571581c256`.
  Its container epoch is 50; selected best epoch is 27. Bootstrap requires
  top-level `model` and `classifier` to be exactly tensor-equal to embedded
  `best_model[0:2]`. The binary is not stored in Git.
- SPRSound package at `dataset/raw/sprsound` or a path with the same official
  BioCAS2022 layout.

## Server Commands

Set the server-owned checkpoint path. The digest below is immutable; the CLI
requires the caller value, file size, and file digest all to match. Bootstrap
adds the pinned author repository to `sys.path` before `torch.load`, solely to
resolve classes referenced by the checkpoint container.

```bash
export SGSCL_TASK_CHECKPOINT=/absolute/server/path/to/sg_scl_accepted_container_epoch50.pth
export SGSCL_TASK_SHA256=8b3652d0dc82b9033251e3aab50ec1b51d328b6d3c836807b7c8de571581c256
export DATASET_ROOT=dataset/raw/sprsound
export RUN_ROOT="result/sg_scl_sprsound_transfer_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"

conda run -n acoustic-sgscl-r4 python -m acoustic.evaluation.verify_sprsound_inter_contract \
  --dataset-root "$DATASET_ROOT"

conda run -n acoustic-sgscl-r4 python -m baseline.sg_scl.checkpoint_eval.bootstrap \
  --result-root "$RUN_ROOT" \
  --checkpoint "$SGSCL_TASK_CHECKPOINT" \
  --checkpoint-sha256 "$SGSCL_TASK_SHA256"

conda run -n acoustic-sgscl-r4 python -m baseline.sg_scl.checkpoint_eval.run_sprsound_transfer \
  --mode smoke --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT" \
  --checkpoint "$SGSCL_TASK_CHECKPOINT" --checkpoint-sha256 "$SGSCL_TASK_SHA256" \
  --device cuda --batch-size 8 --max-events 8

conda run -n acoustic-sgscl-r4 python -m baseline.sg_scl.checkpoint_eval.verify_sprsound_transfer \
  --mode smoke --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT"
```

After a human or coordinator accepts the label-free smoke receipt, run the
separate full command and verifier:

```bash
conda run -n acoustic-sgscl-r4 python -m baseline.sg_scl.checkpoint_eval.run_sprsound_transfer \
  --mode full --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT" \
  --checkpoint "$SGSCL_TASK_CHECKPOINT" --checkpoint-sha256 "$SGSCL_TASK_SHA256" \
  --device cuda --batch-size 8

conda run -n acoustic-sgscl-r4 python -m baseline.sg_scl.checkpoint_eval.verify_sprsound_transfer \
  --mode full --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT"
```

The accepted server audit reports source Sp 74.9842, Se 46.9839, and Score
60.9840, with confusion
`[[1184,274,86,35],[217,400,11,21],[164,38,105,78],[46,22,27,48]]`.
Its source prediction CSV SHA256 is
`2ed6abb6fb0a05f97c2325881c2372b6d803bd5290bc2d3f212a0b44b0867b90`.
These values are provenance copied from the accepted server audit; the transfer
bootstrap does not recompute them. The source run achieved successful one-seed
numerical alignment, each component within about 0.6 SD of the paper's
five-seed values (79.87+/-8.89, 43.55+/-5.93, 61.71+/-1.61). It does not
reproduce the five-seed aggregate.
