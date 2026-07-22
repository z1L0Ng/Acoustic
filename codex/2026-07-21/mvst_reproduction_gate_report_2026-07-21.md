# MVST server-runnable gate report

## Verdict

MVST is **server-runnable as an author-code fixed-random-file-split
reproduction**. It is not an official ICBHI-split reproduction, and no local
full numerical result was attempted.

## Source and protocol boundary

- Author repo: `https://github.com/wentaoheunnc/MVST`
- Commit: `51f93fa6ffa580d0819ccb59f861582927937264`
- Contract: `codex/2026-07-21/paper_contracts/mvst.json`
- Paper target: Sp 81.99, Se 51.10, Score 66.55

The repo sorts 920 recordings, applies `random.Random(1).shuffle`, and assigns
552/368 recordings, giving 4,213/2,685 train/test cycles. It does not read the
official split. Score 66.55 can only be compared against this exact author-code
split. The pipeline also selects the best fusion checkpoint by test Score.

## Checkpoint and five-view identity

The current author-hosted `audioset_16_16_0.4422.pth` is 350,437,422 bytes,
SHA256 `dc71a6d4d07aeb7e746547f72a141f404e4c167d660bf003179f3865e06a970c`.
Only the 16x16 patch projection matches this checkpoint. The author key/shape
filter leaves 32x8, 64x4, 128x2, and 256x1 patch projections initialized; this
behavior is preserved and disclosed.

Each view smoke consumed `8 x 1 x 1024 x 256`, produced `8 x 768` embeddings
and `8 x 4` logits, and completed a finite backward step. The five output
archives had byte-identical ordered cycle IDs and labels. Fusion consumed all
five `8 x 768` arrays, produced finite `8 x 4` logits/loss/gradients, and passed
eight-ID prediction and metric wiring. These are wiring checks only.

The source compatibility patch adds rolling encoder resume state without
changing model/data/loss behavior. Its clean diff SHA256 is
`b927df3ec979608dd047d040b3fb81073ceb359bdb131bc7676320db8937dcd9`.

## Fresh-checkout L40 commands

```bash
conda env create -f baseline/mvst/environment.linux-cu118.yml
conda activate acoustic-mvst
RUN_ROOT="result/mvst_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
python -m baseline.mvst.run_reproduction bootstrap \
  --project-root . --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" --device cuda
python -m baseline.mvst.run_reproduction verify-bootstrap --result-root "$RUN_ROOT"
python -m baseline.mvst.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
python -m baseline.mvst.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100 --views 16
python -m baseline.mvst.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0
```

Final verification requires exactly 2,685 unique author-test cycle IDs,
confusion total 2,685, and exact five-view ID/label order. A close result still
must not be described as official-split faithful reproduction. The repo has no
license file or author-trained task checkpoint.
