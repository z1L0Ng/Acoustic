# ADD-RSC bounded reproduction tracks

This directory is the tracked, fresh-checkout execution surface for ADD-RSC.
Source clones, checkpoints, caches, logs, and predictions are generated under a
new `result/add_rsc_YYYYMMDD_HHMMSS/` root. The minimum compatible project
snapshot is Release 1 commit `51626840f6ec325086f68bd88446ff956f7e0357` plus
this Release 2 directory.

ADD-RSC does not expose one unambiguous faithful executable protocol. The two
supported tracks must remain separate:

| Track | Split | Mel bins | Weight decay | Interpretation |
|---|---:|---:|---:|---|
| `paper_declared_reconstruction` | official 60/40 | 64 | 0.1 | Bounded reconstruction of the paper declarations; not author-repo faithful |
| `author_repo_default_official_like` | sorted files + `random.Random(1)` 60/40 | 128 | 1e-6 | Author-code default execution; paper target 65.53 is not directly comparable |

Both preserve the repo model, ADD loss, 16 kHz/8 s preprocessing, test-per-epoch
selection, and four-class metric implementation. The independent source
contract is `codex/2026-07-21/paper_contracts/add_rsc.json`.

## Environment

Linux/CUDA server:

```bash
conda env create -f baseline/add_rsc/environment.linux-cu121.yml
conda activate acoustic-addrsc
```

The local compatibility environment receipt remains in `environment.yml` and
`environment-lock.txt`. Full training requires CUDA; the CPU path is only for
bootstrap and bounded smoke verification.

## Fresh-checkout bootstrap

Run from the project root. The default public checkpoint is the current
AST-author runtime Dropbox artifact. It is a compatibility checkpoint with a
pinned SHA256, not a verified byte-identical copy of the original 2023 artifact.

```bash
RUN_ROOT="result/add_rsc_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
python -m baseline.add_rsc.run_reproduction bootstrap \
  --project-root . \
  --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" \
  --device cuda
python -m baseline.add_rsc.run_reproduction verify-bootstrap \
  --result-root "$RUN_ROOT"
```

For an externally supplied checkpoint, add `--checkpoint-path /path/to/file`.
Bootstrap clones and pins the author repo, verifies the checkpoint SHA, rebuilds
the cycle manifest from raw ICBHI files, and writes all runtime receipts under
the new result root.

## Smoke and profile

```bash
python -m baseline.add_rsc.run_reproduction smoke \
  --result-root "$RUN_ROOT" --track all --device cuda \
  --steps 1 --train-batch-size 1

python -m baseline.add_rsc.run_reproduction profile \
  --result-root "$RUN_ROOT" --track paper_declared_reconstruction \
  --device cuda --steps 100 --train-batch-size 8
```

Smoke selects eight training cycles, verifies preprocessing, forward execution,
finite loss/gradient, and sample-level prediction/metric wiring for both tracks.
Profile should be completed on the target L40 before a full run is queued.

## One-seed full commands

Use a separate timestamped result root for each full track. The author workflow
evaluates the test set each epoch and selects the best test Score; every result
must therefore be labelled `test-selected`.

```bash
python -m baseline.add_rsc.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" \
  --track paper_declared_reconstruction --device cuda:0

python -m baseline.add_rsc.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" \
  --track author_repo_default_official_like --device cuda:0
```

Resume only from a checkpoint inside the same result root:

```bash
python -m baseline.add_rsc.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" \
  --track paper_declared_reconstruction --device cuda:0 \
  --resume "$RUN_ROOT/full/paper_declared_reconstruction/models/last.pth"
```

After training, the CLI exports cycle IDs, logits, probabilities, predictions,
confusion matrix, specificity, sensitivity, and ICBHI Score, then checks the
expected test cardinality. A numerical match may only be described as an
aligned bounded reconstruction or aligned author-code execution, never an
official faithful reproduction.
