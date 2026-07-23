# PAFA B1: Frozen Checkpoint to SPRSound Inter

This package runs the accepted PAFA server checkpoint whose container was saved
at epoch 100 and whose embedded/top-level predictive state was selected at best
epoch 27. It uses the exact SPRSound BioCAS2022 inter-subject event set from
Patch-Mix B0. It is a published-model, ICBHI-test-selected,
zero-target-tuning exploratory transfer, not a clean source anchor or a
target-trained reference.

## Frozen Contract

- Target: 1,429 official inter-subject events, ordered event-ID SHA256
  `81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b`.
- Source classes: `normal`, `crackle`, `wheeze`, `both`.
- Binary result: normal iff four-class argmax is normal; all other predictions
  collapse to abnormal. No threshold is fitted.
- PAFA preprocessing: mono 16 kHz, full-recording fade, event-boundary crop,
  author 5 s repeat/truncate, raw waveform, `--nospec`, augmentation off.
- Labels are absent from smoke and model inference artifacts. Full mode joins
  labels only after all logits have been written for final scoring.
- No target calibration, preprocessing tuning, checkpoint selection, intra
  subset, target training, device metadata, HF_Lung, or KAUH execution.

## Prerequisites

- Checkout containing this tracked package.
- Frozen environment `acoustic-pafa` from
  `baseline/pafa/environment.linux-cu118.yml`.
- Accepted PAFA task-checkpoint container: size `1,464,382,039` bytes, SHA256
  `94afaed43a1546af26f9d8d99d2d27329cb8d348fd57cbe142d24310c68ca2b6`.
  Its container epoch is 100; selected best epoch is 27. Bootstrap requires
  top-level `model`, `classifier`, and `projector` to be exactly tensor-equal
  to embedded `best_model[0:3]`. The binary is not stored in Git.
- Audited BEATs_iter3+ AS2M backbone and SHA256
  `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`.
- SPRSound package at `dataset/raw/sprsound` or a path with the same official
  BioCAS2022 layout.

## Server Commands

Set the two server-owned checkpoint paths. The task digest below is immutable;
the CLI requires the caller value, file size, and file digest all to match.

```bash
export PAFA_TASK_CHECKPOINT=/absolute/server/path/to/pafa_accepted_container_epoch100.pth
export PAFA_TASK_SHA256=94afaed43a1546af26f9d8d99d2d27329cb8d348fd57cbe142d24310c68ca2b6
export PAFA_BEATS_CHECKPOINT=/absolute/server/path/to/BEATs_iter3_plus_AS2M.pt
export DATASET_ROOT=dataset/raw/sprsound
export RUN_ROOT="result/pafa_sprsound_transfer_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"

conda run -n acoustic-pafa python -m acoustic.evaluation.verify_sprsound_inter_contract \
  --dataset-root "$DATASET_ROOT"

conda run -n acoustic-pafa python -m baseline.pafa.checkpoint_eval.bootstrap \
  --result-root "$RUN_ROOT" \
  --checkpoint "$PAFA_TASK_CHECKPOINT" \
  --checkpoint-sha256 "$PAFA_TASK_SHA256" \
  --backbone-checkpoint "$PAFA_BEATS_CHECKPOINT" \
  --backbone-sha256 d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34

conda run -n acoustic-pafa python -m baseline.pafa.checkpoint_eval.run_sprsound_transfer \
  --mode smoke --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT" \
  --checkpoint "$PAFA_TASK_CHECKPOINT" --checkpoint-sha256 "$PAFA_TASK_SHA256" \
  --backbone-checkpoint "$PAFA_BEATS_CHECKPOINT" \
  --backbone-sha256 d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34 \
  --device cuda --batch-size 8 --max-events 8

conda run -n acoustic-pafa python -m baseline.pafa.checkpoint_eval.verify_sprsound_transfer \
  --mode smoke --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT"
```

After a human or coordinator accepts the label-free smoke receipt, run the
separate full command and verifier:

```bash
conda run -n acoustic-pafa python -m baseline.pafa.checkpoint_eval.run_sprsound_transfer \
  --mode full --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT" \
  --checkpoint "$PAFA_TASK_CHECKPOINT" --checkpoint-sha256 "$PAFA_TASK_SHA256" \
  --backbone-checkpoint "$PAFA_BEATS_CHECKPOINT" \
  --backbone-sha256 d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34 \
  --device cuda --batch-size 32

conda run -n acoustic-pafa python -m baseline.pafa.checkpoint_eval.verify_sprsound_transfer \
  --mode full --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT"
```

The accepted server audit reports source Sp 76.8841, Se 51.4019, and Score
64.1430, with confusion
`[[1214,251,87,27],[232,401,6,10],[127,31,159,68],[34,18,46,45]]`.
Its source prediction CSV SHA256 is
`b4102a572c5ba3a755958c238734837b182d65cb0e50981c5ffa6fafc00c6d4a`.
These values are provenance copied from the accepted server audit; the transfer
bootstrap does not recompute them. The source run was numerically aligned at
Score level for one seed versus the paper five-seed values
(82.05+/-1.95, 47.63+/-2.23, 64.84+/-0.60), but it is not an exact
component-wise or five-seed aggregate reproduction.
