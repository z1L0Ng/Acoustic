# Published checkpoints

This directory is the only location for model binaries committed to Git.

Use exactly this layout:

```text
checkpoints/<experiment_id>/best.<pt|pth|ckpt|safetensors|bin>
```

`<experiment_id>` must exist in `experiments/index.csv`. Each experiment may
publish at most one file, selected as its accepted best checkpoint, and that file
must be no larger than 104,857,600 bytes (100 MiB). GitHub warns for files above
50 MiB, so publication should still be limited to checkpoints that materially
improve reproducibility.

Rolling checkpoints, resume state, optimizer/scaler/RNG state, source archives,
and oversized checkpoints belong in `.cache/`, `result/`, or server storage.
Record their SHA256 and retrieval location in the relevant experiment evidence
instead of committing them here.
