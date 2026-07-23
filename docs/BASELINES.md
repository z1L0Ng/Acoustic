# Baselines

## Controlled frozen representations

The local comparison uses identical ICBHI cycle IDs, fixed train/validation/test
assignments, subtrain-only normalization, and LR, one-layer MLP, or two-layer MLP
heads. These are controlled local benchmarks, not reproductions of encoder papers.

Best descriptive test macro F1 by representation:

| Representation | Flat four | Binary | Best head pattern |
|---|---:|---:|---|
| BEATs | 0.4366 | 0.6447 | MLP2 |
| AST CLS | 0.4232 | 0.6289 | MLP2 flat four, MLP1 binary |
| CLAP | 0.4161 | 0.6009 | MLP1 |
| HeAR | 0.3670 | 0.5835 | LR flat four, MLP1 binary |
| OPERA-CT official-like | 0.3123 | 0.5592 | LR |
| Simple acoustic | 0.3103 | 0.5510 | MLP2 flat four, LR binary |

Small nonlinear heads help AST, CLAP, and BEATs, but not every representation.
The result establishes downstream capacity as one limitation, not the full paper gap.

## Loss and strict-patient evidence

- Class-weighted CE usually raises UAR and `both` recall while lowering normal
  specificity. No loss is best across all representations.
- BEATs focal loss is the strongest balanced local point in the official-split
  comparison, but its advantage is small and not statistically established.
- In strict five-fold patient-grouped OOF evaluation, Logit Adjustment has the
  highest mean macro F1/UAR/`both` recall (`0.4585/0.4679/0.2134`), while focal
  loss is the strongest calibration/Score control. No policy improves minority
  recall, UAR, and specificity together.

## Project model ablations already completed

Soft joint heads and event fusion did not solve the minority class. Joint BCE
slightly improved macro F1, specificity, Score, and calibration but reduced
`both` recall. Event fusion moved the operating point further toward specificity
and lower sensitivity. A Mac-compatible selective-state pooling model was weaker
than simple BEATs frame max/CNN controls. These are negative or tradeoff results,
not method claims; see `docs/MODEL.md`.

## Strong paper methods

| Method | Reproduction boundary | Paper target or status |
|---|---|---|
| Patch-Mix CL | Official split, official-test-selected | Author checkpoint locally matches Score 62.17 exactly. |
| PAFA | Official-like, official-test-selected | Five-seed paper Score 64.84; server artifact must be imported and verified. |
| SG-SCL | Metadata/device-aware, official-test-selected | Paper Score 61.71; not an audio-only comparator. |
| MVST | Author random-file split, test-selected | Reported 66.55 is not an official-split target. |
| ADD-RSC | Bounded reconstruction only | Paper/repo split, mel-bin, and weight-decay settings conflict. |

Every imported run must retain sample IDs, logits or probabilities, predictions,
confusion, Sp, Se, Score, source commit, checkpoint hash, and selection caveat.
Paper mean alignment and cross-dataset transfer are separate claims.

Machine-readable reproduction and comparison tables are in `docs/tables/`.
