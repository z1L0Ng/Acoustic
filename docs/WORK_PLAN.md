# Work Plan

Updated: 2026-07-22

## Objective

Determine whether strong single-dataset respiratory-sound methods retain useful
performance across heterogeneous datasets, quantify the gap against matched
target-trained references, and then decide whether the paper contribution should
focus on cross-dataset learning, imbalance-aware modeling, or data curation.

## Current gates

| Gate | State | Evidence needed to close it |
|---|---|---|
| Strong source method alignment | partial | Import and independently verify all terminal server runs. Patch-Mix checkpoint alignment is already exact. |
| Frozen cross-dataset transfer | partial | Patch-Mix to SPRSound B0 is complete; PAFA and SG-SCL packages await accepted checkpoint inference. |
| Target-domain learnability | ready, not run | Train the matched Patch-Mix C0 reference on SPRSound train and evaluate inter once. |
| Clean source generalization | pending | Train/select a Patch-Mix source checkpoint without official-test model selection. |
| HF_Lung and KAUH task policy | blocked by decisions | Resolve negative intervals, identity proxies, recording labels, and patient-grouped split. |
| New model entry | hold | Require a stable benchmark gap that simple pooling, loss, or calibration controls do not explain. |

## Ordered next actions

1. Finish repository/server consolidation and import only terminal, verified server results.
2. Compare PAFA and SG-SCL source results with their paper distributions and record exact caveats.
3. Run their frozen SPRSound inter inference using the same target IDs and all-normal floor as B0.
4. Run C0, the target-trained same-architecture Patch-Mix reference, with patient-grouped validation.
5. Report target-reference gap, normalized retention, coverage, classwise recall, and grouped uncertainty. Do not subtract incomparable source and target Scores.
6. If the gap repeats across at least two strong source methods, add pooled/multi-dataset training and a clean-source checkpoint.
7. Only then choose the method branch: shared representation with dataset-specific heads, domain generalization, imbalance-aware training, or label expansion.

## Imbalance and labeling line

The current evidence shows a real minority/specificity tradeoff but not a solved
imbalance problem. A labeling effort becomes justified only after error analysis
shows that performance is limited by missing/ambiguous target labels rather than
preprocessing, domain shift, or a learnable target reference. The first labeling
task should therefore be a bounded audit of unsupported or ambiguous events and
metadata, with double review and disagreement reporting, not immediate bulk
pseudo-label generation.

## Decision rule

A cross-dataset failure claim requires a source-aligned model that is above a
trivial target floor yet materially below a target-trained reference on exactly
the same target units. The pattern should repeat for two strong methods or one
strong method plus a foundation-representation control. If it does not, report
retention or ontology-specific failure instead of forcing a degradation story.
