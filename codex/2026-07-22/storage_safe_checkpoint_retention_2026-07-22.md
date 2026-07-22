# Storage-safe checkpoint retention

This compatibility change bounds checkpoint storage without changing training
or selection semantics. Patch-Mix, PAFA, SG-SCL, ADD-RSC, each of MVST's five
encoders, and MVST fusion retain:

- `best.pth`, updated only under the author's existing best-selection condition;
- `last.pth`, a complete interruption-safe state atomically replaced each epoch;
- `retention_receipts.jsonl`, recording epoch, role, bytes, SHA256, and whether an
  existing target was replaced.

The replacement is accepted only after the temporary file SHA256 equals the
post-replacement SHA256. Periodic `epoch_N.pth` files are suppressed. Logs,
metrics, curves, predictions, confusion matrices, bootstrap receipts, and final
reports are unaffected.

## Semantic boundary

No model, optimizer, loss, data, split, preprocessing, hyperparameter, epoch,
metric, test-selection condition, seed, or numerical execution path is changed.
Only checkpoint naming, atomic replacement, and retention receipts are changed.

## Conservative storage envelope

The prelaunch envelope is 58 GiB: MVST 24 GiB; Patch-Mix, PAFA, and SG-SCL 6 GiB
each; ADD-RSC 8 GiB; and 8 GiB for temporary atomic writes, features, logs,
predictions, and receipts. Only one temporary checkpoint exists per active lane.
The launch gate recomputes free bytes and requires projected free space to remain
strictly above the user-approved 20 GiB floor.

## Validation

- Fresh author `HEAD` copies: compatibility installation and Python compilation
  passed for Patch-Mix, PAFA, SG-SCL, ADD-RSC, and all five MVST encoder trees.
- Static inventory: each maintained encoder training entry point contains one
  rolling `last.pth`, fixed `best.pth` writes, no `best_epoch_N.pth`, and no
  periodic `epoch_N.pth` write.
- Resume-state test: two atomic replacements restored the latest epoch and
  optimizer state; the directory contained only `best.pth`, `last.pth`, and the
  three-row retention ledger, with matching SHA256 values.
- Fresh Linux solver dry-run: all five unchanged Release 4 environment YAMLs
  resolved successfully with conda 25.1.1.
