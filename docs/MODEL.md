# Project Models

`model/` is reserved for methods proposed by this project. Reproduction adapters,
paper baselines, generic evaluation helpers, and one-off diagnostics do not belong
there. A new subdirectory should be created only after the experiment definition
and entry gate are approved.

## Current hypothesis

The strongest current evidence supports studying frame-level temporal evidence,
heterogeneous labels, and cross-dataset robustness. It does not yet support a
Mamba-specific claim or a claim that a joint/event head solves class imbalance.

Completed exploratory findings:

- BEATs frame max pooling produced a strong temporal control, but `both` recall
  varied across seeds and threshold transfer was unstable.
- A PyTorch selective-state/Mamba-inspired fallback was feasible on Mac but
  weaker than max and CNN controls. Official Mamba was not evaluated locally.
- Joint flat-four, binary, and event heads mainly improved specificity,
  calibration, or Score while reducing `both` recall.
- Validation-only event fusion repeated that specificity/sensitivity shift.

## Entry gate for a new method

Before implementation, define a matched baseline, target unit, split, metric
set, expected failure addressed, and a stop rule. A method should become a main
line only if it improves a reproducible gap that simple head capacity, pooling,
loss weighting, calibration, or target-domain training does not already explain.

Likely future branches include a shared acoustic representation with
dataset-specific heads, domain-generalization training, and temporal/multilabel
models. Their implementations will be added here only after cross-dataset source,
frozen-transfer, and target-reference evidence is complete.
