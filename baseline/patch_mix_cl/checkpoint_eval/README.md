# Patch-Mix author-checkpoint evaluation

This isolated track evaluates only the ICBHI task checkpoint posted by the
Patch-Mix repository owner in
[`issue #3`](https://github.com/raymin0223/patch-mix_contrastive_learning/issues/3#issuecomment-1710443767).
It does not modify or replace the Release 4/server reproduction pipeline.

The checkpoint is test-selected: the author code evaluates the official test
split each epoch and retains the best test Score. Results from this directory
must be labelled exploratory author-checkpoint inference and must not be used as
the clean source checkpoint for cross-dataset transfer.

Audit the checkpoint before inference:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.inspect_checkpoint \
  --checkpoint result/patch_mix_cl_author_checkpoint_TIMESTAMP/checkpoint/best.pth \
  --output result/patch_mix_cl_author_checkpoint_TIMESTAMP/receipts/checkpoint_structure.json
```

Run ICBHI checkpoint-only inference:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.run_icbhi_checkpoint_eval \
  --manifest dataset/processed/manifests/icbhi_2017_cycles.csv \
  --author-repo result/patch_mix_cl_author_checkpoint_TIMESTAMP/source/repo \
  --checkpoint result/patch_mix_cl_author_checkpoint_TIMESTAMP/checkpoint/best.pth \
  --result-root result/patch_mix_cl_author_checkpoint_TIMESTAMP \
  --device cpu --batch-size 4
```

The evaluator uses the author validation preprocessing without SpecAugment and
exports 2,756 unique official-test cycle IDs, logits, probabilities,
predictions, labels, a confusion matrix, and independently recomputed
specificity, sensitivity, and ICBHI Score.

Verify the export independently:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.verify_icbhi_checkpoint_eval \
  --manifest dataset/processed/manifests/icbhi_2017_cycles.csv \
  --predictions result/patch_mix_cl_author_checkpoint_TIMESTAMP/icbhi/predictions.csv \
  --metrics result/patch_mix_cl_author_checkpoint_TIMESTAMP/icbhi/metrics.json \
  --output result/patch_mix_cl_author_checkpoint_TIMESTAMP/receipts/icbhi_verification.json
```

Run the frozen, test-selected author checkpoint as an explicitly exploratory
SPRSound BioCAS2022 inter-subject event-level transfer. This B0 entry point is
locked to inter only and also emits an all-normal B0-floor on exactly the same
1,429 target rows. It runs the binary-broad and narrow-four tasks without
target tuning. Intra, broad-CAS, and C0 are not executed by this command.

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.run_sprsound_checkpoint_transfer \
  --dataset-root dataset/raw/sprsound/source_original/SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca/BioCAS2022 \
  --author-repo result/patch_mix_cl_author_checkpoint_TIMESTAMP/source/repo \
  --checkpoint result/patch_mix_cl_author_checkpoint_TIMESTAMP/checkpoint/best.pth \
  --result-root result/patch_mix_cl_author_checkpoint_transfer_TIMESTAMP \
  --device cpu --batch-size 8 --threads 8

conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.verify_sprsound_checkpoint_transfer \
  --dataset-root dataset/raw/sprsound/source_original/SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca/BioCAS2022 \
  --result-root result/patch_mix_cl_author_checkpoint_transfer_TIMESTAMP
```

If inference predictions completed but receipt writing was interrupted, finalize
only those existing predictions without a second model pass, then run the same
independent verifier:

```bash
conda run -n acoustic-patchmix python -m \
  baseline.patch_mix_cl.checkpoint_eval.finalize_sprsound_checkpoint_transfer \
  --checkpoint result/patch_mix_cl_author_checkpoint_TIMESTAMP/checkpoint/best.pth \
  --result-root result/patch_mix_cl_author_checkpoint_transfer_TIMESTAMP
```
