# Work Plan

Updated: 2026-08-24

## Objective

Build a unified respiratory-sound system on the compatible ICBHI and SPRSound
label space, then determine whether input normalization, waveform augmentation,
or imbalance-aware loss can recover a reasonable and literature-comparable
performance level. BEATs is the current candidate backbone because it is the
strongest relevant local package reference, not because backbone superiority has
already been established.

## Reference protocol

This protocol keeps the next comparisons interpretable. It is a working
reference rather than a permanent restriction and may be revised when the Wade
window analysis or new meeting decisions provide better evidence.

| Item | Current reference |
|---|---|
| Datasets | ICBHI + SPRSound only for shared training and validation |
| Shared outputs | Level1 Normal/Abnormal, Crackle, Wheeze |
| Unknown labels | Masked; never converted to negative |
| Excluded node | `Other` is not trained in the shared atomic head |
| HF Lung | Excluded from this comparison; no positive-only auxiliary loss |
| KAUH | Excluded from training; remains a possible external evaluation set |
| Backbone | BEATs iter3+ AS2M candidate package |
| Encoder scope | Full fine-tuning for the main comparison; frozen mode only for attribution or resource fallback |
| Seed | 42 |
| Audio | Mono 16 kHz; provisional 2 s window / 1 s stride |
| Batch / epochs | Effective native-unit batch 8 via unit-wise gradient accumulation; 50 epochs |
| Optimizer | Adam; backbone LR 1e-5, head LR 5e-5, weight decay 1e-6 |
| Schedule / precision | Cosine per update; FP32 |
| Selection | Equal mean of ICBHI and SPRSound validation eligible-node loss |
| Attribute thresholds | One shared Crackle threshold and one shared Wheeze threshold, chosen on selected validation predictions by equal ICBHI/SPRSound F1 |
| Test | One separately approved terminal evaluation after selection |

## Ordered experiment sequence

Each stage changes one intervention axis while keeping the remaining reference
settings fixed.

1. **R0 — Full-fine-tuning reference:** raw waveform, no augmentation, CE+BCE.
2. **N1 — Waveform normalization:** change only native-waveform normalization;
   start with peak normalization. RMS normalization remains available only after
   an explicit training-derived target dBFS is selected.
3. **A1 — Waveform augmentation:** retain the selected normalization and add
   seeded gain plus additive-noise augmentation.
4. **L1/L2 — Loss:** retain the selected waveform policy and compare focal loss
   and effective-number class-balanced loss against CE+BCE.

The next stage should start only after reviewing the prior stage. A stage is not
promoted merely because one aggregate metric improves; ICBHI specificity,
sensitivity, official Score, classwise recall, and SPRSound official-inter
metrics must be read together.

## Output contract

Every validation and later terminal prediction artifact must retain sample ID,
group/patient ID, source filename, raw ground truth, unified target and
eligibility mask, logits, probabilities, and predictions. These fields allow
confusion matrices and alternative approved label groupings to be recomputed
without retraining.

For the ICBHI literature-facing readout, the two validation-selected atomic bits
reconstruct flat4 as `00 Normal`, `10 Crackle`, `01 Wheeze`, and `11 Both`.
Level1 Normal/Abnormal is reported separately and does not override this flat4
decoder. SPRSound Task1-1 uses the Level1 readout; Task1-2 raw7 remains a native
reference and is not presented as an output of the shared three-node head.

## Evidence and comparison boundaries

- ICBHI literature comparison uses the official recording split, 2,756-cycle
  test set, flat4 Sp/Se, and ICBHI Score. It is not described as strict
  patient-held-out.
- SPRSound literature comparison uses BioCAS2022 inter only. Task1-1 binary,
  Task1-2 raw7, and any mapped hierarchical readout remain separate metrics.
- The new run is a local single-seed controlled result, not a paper-faithful
  reproduction and not an absolute SOTA claim.
- A completed subtrain/validation run is not a terminal result. Test data does
  not affect normalization choice, checkpoint selection, thresholds, or loss
  selection.

## Current state

Implementation and a zero-output notebook are prepared locally. No waveform
decoding, feature/cache build, training, validation, test, or server execution
has started. Launch remains `READY_FOR_USER_START` and requires an explicit
resource and execution decision.
