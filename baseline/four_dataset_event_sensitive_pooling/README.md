# Four-Dataset Event-Sensitive Pooling

This package tests whether PAFA 5-second window-level pooling improves the
corrected four-dataset frozen-feature baseline after the shortcut diagnostic was
negative/inconclusive.

It is a local controlled diagnostic, not a PAFA paper reproduction and not a
clean source-generalization claim. The encoder is the corrected Gate-B-selected
R0 PAFA ICBHI task encoder, with the ICBHI official-test-selected caveat.

## Conditions

- `p0_r0_d2_pooled_reference`: read-only corrected R0+D2 pooled reference.
- `p1_event_sensitive_learned_pooling`: masked attention over PAFA 5 s windows.
- `p2_parameter_matched_pooled_control`: mean-pooled features plus a scalar gate
  with the same trainable parameter count as P1's pooling scorer.

The six dataset-native tasks, corrected HF date-proxy split, SPRSound label-free
terminal join, KAUH five-fold OOF policy, and D2 dataset-balanced update schedule
are inherited from the verified four-dataset baseline.

## Commands

```bash
conda run -n acoustic-patchmix python -m baseline.four_dataset_event_sensitive_pooling.run \
  --phase smoke-profile \
  --dataset-root dataset/raw \
  --source-repo result/pafa_sprsound_transfer_20260722_235659/source/repo \
  --checkpoint .cache/checkpoints/pafa/server_epoch27/best.pth \
  --backbone-checkpoint .cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt \
  --device cpu

conda run -n acoustic-patchmix python -m baseline.four_dataset_event_sensitive_pooling.run \
  --phase all \
  --dataset-root dataset/raw \
  --source-repo result/pafa_sprsound_transfer_20260722_235659/source/repo \
  --checkpoint .cache/checkpoints/pafa/server_epoch27/best.pth \
  --backbone-checkpoint .cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt \
  --device cpu

conda run -n acoustic-patchmix python -m baseline.four_dataset_event_sensitive_pooling.verify \
  --result-root result/four_dataset_event_sensitive_pooling \
  --cache-root .cache/four_dataset_event_sensitive_pooling \
  --dataset-root dataset/raw
```

Outputs are written only under `result/four_dataset_event_sensitive_pooling/`
and `.cache/four_dataset_event_sensitive_pooling/`.
