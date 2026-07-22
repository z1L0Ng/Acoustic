# ADD-RSC server-runnable gate report

## Verdict

ADD-RSC is **server-runnable through two bounded, explicitly separated
protocols**. The paper and executable repository do not define one unique
faithful protocol. Neither run may be called an official faithful reproduction.

## Source and unresolved contradictions

- Author repo: `https://github.com/deegy666/ADD-RSC`
- Commit: `e2b0f8213cb7ca451ef28757cb1329e17469fe72`
- Contract: `codex/2026-07-21/paper_contracts/add_rsc.json`
- Paper AST target: Sp 85.13, Se 45.94, Score 65.53
- License/task checkpoint/exact command: unavailable

History, tags, README, and executable defaults do not resolve the conflicts:
the paper states 64 mel bins, weight decay 0.1, and official protocol, whereas
the repo uses 128 mel bins, weight decay `1e-6`, and sorted recordings with a
fixed `random.Random(1)` 60/40 file split.

## Frozen tracks

| Track | Split | Test cycles | Mel | WD | Claim boundary |
|---|---:|---:|---:|---:|---|
| `paper_declared_reconstruction` | official 60/40 | 2,756 | 64 | 0.1 | bounded paper reconstruction; may be compared cautiously with 65.53 |
| `author_repo_default_official_like` | author random(1) file split | 2,685 | 128 | 1e-6 | author-code execution; 65.53 is not directly comparable |

Both preserve the repo AST+ADD architecture, hybrid loss beta 0.5, 16 kHz,
8 s repeat/cyclic padding, test-per-epoch selection, paper alpha 0.02 mapping
to `AFNO1D.scale`, and epsilon 0.2 mapping to label smoothing.

## Checkpoint, data, and smoke

- Audited raw cycles: 6,898; official train/test 4,142/2,756.
- Author random split: 552/368 recordings and 4,213/2,685 cycles.
- Compatibility AST checkpoint SHA256:
  `dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`.
- Source compatibility diff SHA256:
  `c7f2495ef1013d1ffcb5d098d9eea7be2727d77b516fef5d45956aafe2474663`.

Both eight-cycle smokes produced input `8 x 1 x 1024 x 256`, classifier and
denoised logits `8 x 4`, finite hybrid losses and finite gradients. The paper
track used 64 fbank bins; the author-code track used 128. Both exported eight
unique cycle IDs and passed confusion/Sp/Se/Score wiring. CPU batch-size-one
steps took about 1.12 s and 0.90 s respectively; this is not an L40 batch-8
profile.

The patch makes the hardcoded LayerNorm device portable, exposes both named
split protocols, and stores complete rolling resume state. The paper track's
split/mel/weight-decay changes are semantic by design and are prominently
recorded; they are not disguised as compatibility-only changes.

## Fresh-checkout L40 commands

```bash
conda env create -f baseline/add_rsc/environment.linux-cu121.yml
conda activate acoustic-addrsc
RUN_ROOT="result/add_rsc_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
python -m baseline.add_rsc.run_reproduction bootstrap \
  --project-root . --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" --device cuda
python -m baseline.add_rsc.run_reproduction verify-bootstrap --result-root "$RUN_ROOT"
python -m baseline.add_rsc.run_reproduction smoke \
  --result-root "$RUN_ROOT" --track all --device cuda \
  --steps 1 --train-batch-size 1
python -m baseline.add_rsc.run_reproduction profile \
  --result-root "$RUN_ROOT" --track paper_declared_reconstruction \
  --device cuda --steps 100 --train-batch-size 8
python -m baseline.add_rsc.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" \
  --track paper_declared_reconstruction --device cuda:0
```

Use a separate run root for the author-code full track. Acceptance requires the
track-specific exact test cardinality, unique cycle IDs, confusion total,
finite probabilities, and prediction-derived metric identity. Numerical
alignment cannot erase the protocol classification above.
