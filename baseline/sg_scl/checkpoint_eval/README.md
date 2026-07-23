# SG-SCL B2: Frozen Checkpoint to SPRSound Inter

This package runs a verified server SG-SCL epoch-27 checkpoint on the exact
SPRSound BioCAS2022 inter-subject event set used by Patch-Mix B0. It is a
published-model, ICBHI-test-selected, zero-target-tuning exploratory transfer,
not a clean source anchor or a target-trained reference.

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
- Verified server SG-SCL task checkpoint at epoch 27 and its SHA256. The binary
  is not stored in Git.
- SPRSound package at `dataset/raw/sprsound` or a path with the same official
  BioCAS2022 layout.

## Server Commands

Set the server-owned checkpoint path and digest from the accepted source-run
receipt. Do not infer or substitute either value.

```bash
export SGSCL_TASK_CHECKPOINT=/absolute/server/path/to/sg_scl_epoch27_best.pth
export SGSCL_TASK_SHA256=<sha256-from-accepted-server-receipt>
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

The source run achieved successful one-seed numerical alignment
(Sp 74.984, Se 46.984, Score 60.984; each within about 0.6 SD of the paper's
five-seed 79.87+/-8.89, 43.55+/-5.93, 61.71+/-1.61). It does not reproduce the
five-seed aggregate.
