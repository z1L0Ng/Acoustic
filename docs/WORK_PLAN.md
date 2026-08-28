# Work Plan

Updated: 2026-08-27

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
| Audio | Mono 16 kHz; current reference 4 s window / 2 s stride |
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

For the primary ICBHI literature-facing readout, the shared hierarchy is applied
directly. A Level1 Normal prediction produces the final Normal class. A Level1
Abnormal prediction is then refined by the validation-selected Crackle and
Wheeze attributes into Crackle, Wheeze, or Both. If Level1 is Abnormal while
neither attribute crosses its threshold, the attribute with the larger
probability-minus-threshold margin is selected, with Crackle winning an exact
tie. The former bits-only reconstruction (`00 Normal`, `10 Crackle`, `01
Wheeze`, `11 Both`) is retained as a diagnostic ablation rather than the primary
readout. SPRSound Task1-1 continues to use Level1; Task1-2 raw7 remains a native
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

The seed-42 R0/N1/A1/L1/L2 full-fine-tuning queue and its separately approved
terminal evaluation are complete. The terminal diagnostic showed that the
former bits-only ICBHI decoder converts frequent attribute false positives on
Normal cycles into low specificity. An independent saved-prediction posthoc
evaluation of the hierarchical decoder is also complete: it improves ICBHI
Score by 6.21 to 11.50 percentage points across the five runs, while reducing
sensitivity. R0 has the highest posthoc hierarchical ICBHI Score; N1 has the
highest SPRSound Score; L1 remains the ICBHI-recall-oriented ablation. The
posthoc result is test-informed and is not promoted to a new primary result.
Existing terminal results remain historical bits-only Local Results and are not
overwritten. No new training or terminal evaluation is authorized by this plan
update.

## ICASSP 2027 deadline plan

The official full-paper deadline is 2026-09-16. As of 2026-08-27, the project
has 20 calendar days remaining. The internal schedule preserves one day of
submission buffer.

## Two-day execution cadence after the 2026-08-27 meeting

The project no longer uses weekly progress as its control loop. Every two-day
checkpoint must produce a reviewable artifact, a numeric result where the task
is empirical, and a decision for the following checkpoint. Student work is a
supporting lane and cannot block the lead-owned paper path.

| Checkpoint | Lead-owned output | Wade output | Hanlin output | Decision gate |
|---|---|---|---|---|
| 2026-08-28 | Five-minute lab update using the frozen current tables; publish exact student task definitions | Confirm 1/2/3/4 s analysis contract | Confirm official-reproduction-first contract | No new run before the lab feedback is reviewed |
| 2026-08-29 to 2026-08-30 | Interpret R0/N1/A1/L1/L2 by dataset and metric; pre-register the next two normalization comparisons | First support/padding/truncation table plus one standardized PCA/separability result | AST paper score, local reproduction score and delta; BEATs checkpoint/status table | If either student artifact lacks numbers, the task is not complete and the lead activates the backup |
| 2026-08-31 to 2026-09-01 | Close the minimum waveform/spectrogram normalization implementation and paper skeleton | Final window recommendation with quantitative rationale | Close AST reproduction and provide the same evidence format for BEATs | Freeze which student evidence enters the paper |
| 2026-09-02 to 2026-09-03 | Freeze method, main condition, headline metrics and must-have experiment list | Supporting analysis only | Supporting reproduction table only | No broad new axis after this gate |
| 2026-09-04 to 2026-09-05 | Complete approved multi-seed main/reference evidence and core tables | No critical-path dependency | No critical-path dependency | Freeze headline numbers |
| 2026-09-06 to 2026-09-07 | Complete advisor-readable four-page draft | Respond only to targeted figure/data questions | Respond only to targeted baseline questions | Full-draft review |
| 2026-09-08 to 2026-09-09 | Resolve advisor comments and run only named repairs | As requested | As requested | Freeze contribution and claims |
| 2026-09-10 to 2026-09-11 | Final result/citation/table consistency pass | Closed | Closed | Result freeze |
| 2026-09-12 to 2026-09-13 | Content freeze and IEEE layout | Closed | Closed | No new scientific scope |
| 2026-09-14 to 2026-09-15 | Coauthor approval and submission | Closed | Closed | Submit by 2026-09-15 |

### Student reporting contract

- **Wade:** choose a window/preprocessing policy by class separability, not by
  feature magnitude alone. For 1/2/3/4 s at 16 kHz, report support,
  padding/truncation/silence coverage, RMS/power/SNR proxy/PSD/band energy/MFCC/
  spectral centroid/bandwidth, and a quantitative class-separability result.
  Standardized PCA is the primary visualization; UMAP/t-SNE may be secondary.
  Each update must state the observed result and conclusion. First numeric
  artifact is due by 2026-08-30; if it is unavailable, the lead takes over.
- **Hanlin:** first reproduce the original paper on its original benchmark with
  the official checkpoint/evaluation when available. Each row must contain the
  paper result, local reproduction result and delta. Cross-dataset adaptation
  begins only after reproduction is verified. AST is the first complete row,
  BEATs is next; “finished” without a metric is not completion.
- Both students report every two days and immediately when a small result is
  ready. Neither student is a dependency for lead-owned normalization,
  end-to-end window evidence, main experiments or paper writing.

### 2026-08-27 to 2026-08-28 — readout and result closure

- Complete the saved-prediction hierarchical ICBHI recomputation and retain it
  as a separate posthoc diagnostic artifact.
- Freeze the paper-facing output contract, main condition, primary metrics, and
  claim-evidence labels.
- Start the four-page paper skeleton immediately; writing does not wait for the
  final multi-seed runs.

### 2026-08-29 to 2026-08-30 — advisor decision gate

- Confirm that the Level1-gated hierarchical readout matches the intended
  unified-system claim.
- Confirm whether N1 is the main cross-dataset condition and L1 is retained as
  the ICBHI recall/loss ablation.
- Freeze the must-have experiment list. Every later run must have a named
  reviewer question and a destination in a paper table or figure.

### 2026-08-31 to 2026-09-03 — core evidence freeze

- After approval, complete multi-seed confirmation for the selected main method
  and minimum matched reference set.
- Close the ICBHI confusion/per-class table, SPRSound Task1-1 table, and any
  retained KAUH patient-level external descriptive result.
- Freeze headline numbers by 2026-09-03. Do not begin broad encoder, loss,
  augmentation, or window sweeps after this gate.

### 2026-09-04 to 2026-09-07 — complete first draft

- Finish Dataset/Ontology, Method, Experimental Setup, Results, Related Work,
  limitations, the pipeline figure, and the two core tables.
- Deliver a complete advisor-readable draft by 2026-09-07.

### 2026-09-08 to 2026-09-10 — review and targeted repair

- Resolve advisor comments on claims, metrics, comparisons, and minimum
  experiments.
- Run only targeted repairs tied to a named review comment.
- Freeze the title, abstract, contribution list, and conclusion.

### 2026-09-11 to 2026-09-12 — content freeze

- Finalize all numbers, citations, captions, and claim-evidence mappings.
- Independently check split descriptions, sample counts, metric formulas, and
  table consistency.
- Add no new scientific scope after 2026-09-12.

### 2026-09-13 to 2026-09-15 — submission buffer

- Complete IEEE formatting, the four-page technical-content limit, optional
  reference-only fifth page, PDF checks, metadata, and coauthor approval.
- Submit no later than 2026-09-15. The official 2026-09-16 deadline is an
  emergency buffer rather than the planned submission time.

## Deadline scope control

Must-have work is the frozen hierarchical output contract, one defensible main
configuration, a minimum matched reference, multi-seed headline evidence,
native-compatible ICBHI/SPRSound comparisons, and a complete draft by
2026-09-07. KAUH external evaluation, one frozen-vs-full or window analysis, and
Wade's acoustic mechanism analysis are optional only if closed by 2026-09-03.
New encoder sweeps, HF positive-only shared training, new dataset acquisition,
a project-management single-dataset matrix, and application extensions are out
of scope unless the advisor explicitly reopens them.
