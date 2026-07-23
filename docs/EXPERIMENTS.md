# Experiments

`experiments/index.csv` is a locator, not a scientific schema. It deliberately
contains only identity, ownership, status, code/definition/result pointers,
dependencies, and timestamps. New tasks can add arbitrary fields inside their
own YAML definition without changing the registry.

## Definition guidance

A definition should make the research question and comparison boundary clear.
Useful sections include data units, source and target roles, model state,
selection rules, evaluation, expected artifacts, and claim limitations. These
are guidance, not mandatory global fields.

Current labels such as source alignment, frozen transfer, target reference, or
pooled training are comparison roles for the present study. They are not fixed
project protocol types and should not be encoded into directory names as a
permanent taxonomy.

## Result contract

Use `result/<experiment_id>/` and keep only evidence needed to interpret or
recheck the run:

```text
run.json
metrics.json
predictions.csv.gz
confusion.csv                 # or clearly named task-specific confusion files
train.log                     # optional
```

Large source clones, downloaded archives, feature tensors, preprocess caches,
and temporary checkpoints go under `.cache/`. Result fragments from smoke,
profile, receipt, and full phases should be merged into `run.json` after a run is
accepted. Superseded runs are deleted after the replacement is independently
verified; Git history and run hashes provide provenance.

## Checkpoint publication

An experiment may publish one checkpoint at
`checkpoints/<experiment_id>/best.<supported-extension>` when all of the
following hold:

- `<experiment_id>` exists in `experiments/index.csv`;
- the directory contains exactly one checkpoint file and it is named `best.*`;
- the file is no larger than `104,857,600` bytes (100 MiB);
- it is the accepted best model, not a rolling, resume, optimizer, scaler, or RNG snapshot.

Supported extensions are `.pt`, `.pth`, `.ckpt`, `.safetensors`, and `.bin`.
Larger checkpoints remain in `.cache/`, `result/`, or server storage and are
represented by a checksum and retrieval location. No global experiment field is
required for this optional publication decision.

## Interpretation rule

Generated results do not become canonical because a process exited zero. The
experiment owner must verify row identity, label coverage, prediction-derived
metrics, checkpoint/source hashes, and selection boundaries before changing the
index status to `completed`.
