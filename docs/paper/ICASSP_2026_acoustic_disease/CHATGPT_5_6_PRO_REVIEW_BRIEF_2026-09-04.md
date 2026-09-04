# Acoustic ICASSP Paper Review Brief

Date: 2026-09-04

Purpose: provide a compact, evidence-bounded overview for an independent
paper-story review. This document is context, not a manuscript draft.

## 1. Immediate advisor questions

The 2026-09-03 advisor meeting asked the project to answer:

1. What precise research question does the paper answer?
2. What limitation or gap is addressed beyond the broad word adaptability?
3. What two to four concrete contributions are actually supported?
4. How do heterogeneous prediction units, label availability, class balance,
   recording conditions, and acoustic characteristics affect cross-dataset
   learning?
5. Can acoustic feature analysis explain why some datasets benefit from joint
   training while ICBHI may decline, without turning association into causality?
6. Which results are essential after the main result, and which experiments
   should be removed under a four-page limit?
7. How should generic acoustic foundation-model baselines and strong
   respiratory task-specific baselines be positioned?
8. How should specificity, sensitivity, and per-class behavior be used to
   explain the main result and the HF-supervision trade-off?

The advisor explicitly rejected a chronological story of trying many
configurations and choosing the best. The paper must present a gap, a research
question, a logically motivated method, and experiments that directly answer
that question.

## 2. Candidate research problem

Respiratory-sound datasets are heterogeneous in acquisition, native prediction
unit, label ontology, annotation availability, class support, grouping, and
evaluation metric. A model optimized for one dataset cannot be assumed to
retain its behavior on another. Naive pooling may convert unavailable labels
into negatives or erase the meaning of native tasks.

The current candidate question is:

How can heterogeneous respiratory-sound datasets be aligned for shared
representation learning while preserving label availability and native-task
evaluation, and what dataset-dependent retention and trade-offs result?

This wording is not frozen.

## 3. Dataset roles

### ICBHI

- Core supervised dataset.
- Native unit: annotated respiratory cycle.
- Primary task: flat four classes, Normal, Crackle, Wheeze, Both.
- Literature-facing evaluation: official recording 60/40 split and 2,756 test
  cycles.
- Metrics: specificity, sensitivity, and ICBHI Score equal to their mean.
- The official split must not be described as strict patient-held-out.

### SPRSound

- Core supervised dataset.
- Native unit: annotated event.
- Primary paper readout: BioCAS2022 official inter-patient Task1-1.
- Test support: 1,429 events.
- Native official score must remain distinct from ICBHI Score.
- Inter, intra, raw-seven, and binary results must not be pooled.

### HF Lung

- Positive-only auxiliary source in the JH4 extension.
- D maps to observed Crackle supervision and Wheeze maps to observed Wheeze
  supervision.
- Empty files, annotation gaps, phase-only regions, and unsupported labels are
  not Normal or negative examples.
- HF is excluded from the core checkpoint-selection objective.

### KAUH

- External evaluation only.
- B, D, and E recordings are correlated views of the same patient.
- Current compatible-overlay evaluation excludes unresolved Crep, Bronchial,
  and I C B labels.
- It is not a reproduction of the OPERA disease task or a complete native
  raw-nine benchmark.

## 4. Main method: JH2 reader-facing core

JH2 currently defines the main method candidate:

- BEATs iter3+ AS2M full fine-tuning.
- Mono 16 kHz waveform input.
- Native units converted to 5 s by repeat-padding short units and
  front-truncating long units.
- ICBHI and SPRSound joint training.
- Native-unit homogeneous batch size 32 with equal dataset batch counts.
- Shared representation and hierarchical heads:
  - Level 1 Normal versus Abnormal;
  - Level 2 Crackle attribute;
  - Level 2 Wheeze attribute;
  - simultaneous positive attributes represent Both.
- Eligible-node loss prevents unavailable targets from being used as negative
  labels.
- Level 1 uses CE; Crackle and Wheeze use eligible BCE.
- Patient-aware PAFA objectives use PCSL weight 50 and GPAL weight 0.0005.
- Adam learning rate 5e-5, weight decay 1e-6, cosine schedule, EMA 0.5.
- Maximum 50 epochs with patience-10 early stopping.
- Attribute thresholds are fitted on validation predictions.
- Reader-facing terminology should use core model rather than JH2.

## 5. Main three-seed result

Formal seeds are 0, 1, and a fresh 42. Seed 2 was stopped and excluded.

### ICBHI mean plus or minus sample standard deviation

- Specificity: 0.749842 plus or minus 0.056896.
- Sensitivity: 0.473520 plus or minus 0.053528.
- Score: 0.611681 plus or minus 0.003139.
- Macro-F1: 0.497322 plus or minus 0.010593.
- UAR: 0.504049 plus or minus 0.030277.
- Per-class recall:
  - Normal: 0.749842 plus or minus 0.056896.
  - Crackle: 0.601952 plus or minus 0.084820.
  - Wheeze: 0.296104 plus or minus 0.020779.
  - Both: 0.368298 plus or minus 0.113695.

### SPRSound Task1-1 mean plus or minus sample standard deviation

- Sensitivity: 0.976007 plus or minus 0.012149.
- Specificity: 0.842949 plus or minus 0.013643.
- AS: 0.909478 plus or minus 0.003113.
- HS: 0.904495 plus or minus 0.003934.
- Official Score: 0.906986 plus or minus 0.003424.
- Macro-F1: 0.862568 plus or minus 0.006839.
- UAR: 0.909478 plus or minus 0.003113.

Per-seed ICBHI selected epochs and Scores:

- seed 0: epoch 15, Score 0.614814;
- seed 1: epoch 11, Score 0.608537;
- fresh seed 42: epoch 17, Score 0.611692.

## 6. Selection boundary

The three JH2 checkpoints were selected and early-stopped using ICBHI official
test Score. SPRSound official inter was accessed once after the selected
checkpoint for each seed.

Therefore:

- the main aggregate is test-selected and not a clean generalization estimate;
- this is comparable to the test-selection behavior of much ICBHI literature,
  including released PAFA code, only if disclosed explicitly;
- it must not be called clean, leakage-free, or unbiased;
- the paper must decide whether to use an author-faithful benchmark framing or
  add a prospective validation-selected companion experiment.

## 7. Strong primary-source references

Direct task-compatible ICBHI paper claims:

- PAFA BEATs plus CE:
  Sp 78.77 plus or minus 3.07,
  Se 48.21 plus or minus 2.32,
  Score 63.49 plus or minus 1.08.
- PAFA:
  Sp 82.05 plus or minus 1.95,
  Se 47.63 plus or minus 2.23,
  Score 64.84 plus or minus 0.60.

Both use BEATs iter3+ AS2M, full fine-tuning, 5 s repeat-pad/truncate,
batch 32, and official-test checkpoint selection. PAFA adds PCSL and GPAL.

The original BEATs paper reports no respiratory-dataset result.

No task-compatible BEATs result was found in the scoped primary-source audit
for:

- SPRSound BioCAS2022 official inter event Task1-1 or Task1-2;
- HF Lung native temporal positive-only evaluation;
- KAUH patient-grouped raw-nine evaluation.

Nearby published results using different units or disease-label tasks must be
kept as contextual background rather than direct comparisons.

## 8. HF auxiliary extension: JH4

JH4 changes JH2 only by adding an HF positive-only Crackle/Wheeze auxiliary
loss with fixed weight 0.25. HF does not affect thresholds, checkpoint
selection, or early stopping.

The completed comparison is currently single-seed and test-selected.

### Core changes relative to the historical JH2 checkpoint

- ICBHI:
  - Score 0.600502 to 0.597148;
  - specificity 0.747308 to 0.801773;
  - sensitivity 0.453696 to 0.392523;
  - Both recall 0.356643 to 0.188811.
- SPRSound official Score:
  0.892010 to 0.918086.

### HF changes

- D or Crackle recording AUROC:
  0.508124 to 0.698037.
- D or Crackle AUPRC:
  0.411840 to 0.514865.
- D or Crackle positive-interval recall:
  0.526490 to 0.931015.
- Wheeze AUROC:
  0.855048 to 0.898314.
- Wheeze AUPRC:
  0.818500 to 0.868533.
- Wheeze positive-interval recall:
  0.937762 to 0.769231.

Interpretation boundary:

- HF supervision improves selected HF measures and SPRSound;
- it moves ICBHI toward higher specificity and lower sensitivity;
- it is not a uniform improvement;
- a causal claim that missing HF Normal labels cause the shift is unsupported;
- the alternative explanation that source class proportions change the
  supervision distribution must also be considered.

## 9. Fixed-checkpoint external evaluation

JH2 seed 0, 1, and fresh 42 selected checkpoints were evaluated on HF Lung and
KAUH without tuning, retraining, threshold fitting, or epoch reselection.

### HF Lung three-seed summary

- Crackle AUROC: 0.678225 plus or minus 0.044249.
- Crackle AUPRC: 0.591083 plus or minus 0.029342.
- Wheeze AUROC: 0.863603 plus or minus 0.013765.
- Wheeze AUPRC: 0.843379 plus or minus 0.022611.

### KAUH three-seed summary

- recording Level-1 Score:
  0.709866 plus or minus 0.035324.
- patient Level-1 Score:
  0.721662 plus or minus 0.016411.
- recording flat-four Score:
  0.477840 plus or minus 0.091395.
- patient flat-four Score:
  0.470028 plus or minus 0.082964.

This suggests stronger coarse Normal/Abnormal transfer than fine-grained
Crackle/Wheeze/Both transfer. These are post-hoc external diagnostics, not
native benchmark reproductions.

## 10. Existing ablation history

Single-seed test-selected variants:

- Soft Confidence Bridge with equal ICBHI/SPR selection:
  ICBHI 57.21, SPRSound 90.82.
- Soft Confidence Bridge with 60/40 selection:
  ICBHI 58.69, SPRSound 89.91.
- Soft Confidence Bridge plus scalar unit-level fbank MVN:
  ICBHI 58.66, SPRSound 92.77.
- Hard Hierarchy plus scalar unit-level fbank MVN:
  ICBHI 58.43, SPRSound 89.08.

These did not surpass JH2 on ICBHI. Several variants change both method and
selection, so they should not automatically be presented as clean single-factor
causal ablations.

## 11. Current acoustic analysis and Figure 1

An existing draft uses:

- Panel A: native-unit counts and mapped class composition;
- Panel B: group-balanced acoustic-feature PCA using 112 groups per dataset;
- Panel C: standardized recording-level RMS, spectral centroid, bandwidth,
  energy below 500 Hz, and flatness summaries.

The source feature table contains 13,704 recordings. Current PCA is descriptive
and not proof of significance or causality.

Advisor requirements:

- connect acoustic analysis to observed model behavior;
- do not report PCA or outlier counts merely as standalone exploration;
- add a quantitative group-aware domain or class separability measure;
- check whether patterns remain on a shared Normal/Abnormal or class-matched
  surface;
- assess duration, padding or truncation, silence, level, spectrum, device, and
  filter as possible explanations;
- retain association language unless a causal experiment is performed.

## 12. Student work

### Wade

Wade owns descriptive acoustic analysis, not model training:

- complete four-dataset acoustic comparison;
- verify counts, native units, label mapping, group definitions, and PCA
  sampling;
- quantify 5 s repeat-padding, truncation, duration, and silence coverage;
- add group-aware quantitative separability;
- connect observations to the specific ICBHI/SPR/HF/KAUH performance pattern;
- provide paper-usable figures, tables, actual values, and short conclusions.

### Hanlin

Hanlin owns baseline consolidation:

- collect completed four-dataset single-dataset foundation-model results;
- separate paper reproduction from local adaptation;
- report method, dataset, task, split, unit, selection, paper score, local
  score, and delta;
- add one or two strong ICBHI task-specific baselines;
- assess whether an ICBHI-optimized method transfers to other datasets;
- do not force incompatible tasks into one table.

Student outputs must not block the lead-owned manuscript.

## 13. Current manuscript structure

The paper uses an IEEE conference four-page technical-content target:

1. Introduction, including compressed motivation and related work.
2. Data Preparation and Task Alignment.
3. Method.
4. Evaluation, including ablation and discussion.
5. Conclusion.

Current planned presentation elements:

- Figure 1: dataset scale, label composition, PCA, and acoustic features.
- Figure 2: high-level input to harmonization to model to output flow.
- Table I: three-seed ICBHI and SPRSound main results plus task-compatible
  published references.
- Table II: baselines and controlled ablations.

The manuscript is currently a compilable LaTeX skeleton with placeholders.
Figure 1 is inserted; full prose has not been written.

## 14. Scope and deadline

- Target: ICASSP 2027.
- Deadline: 2026-09-16.
- Current short sprint: Friday 2026-09-04 through Sunday 2026-09-06.
- No broad architecture redesign or hyperparameter sweep is planned.
- Data and Method should be drafted before Introduction.
- By Sunday the project should have a reviewable story, two figures, two table
  contracts, and a short list of submission-critical missing evidence.

## 15. Claim restrictions

Do not claim:

- SOTA;
- universal generalization;
- uniform improvement across datasets;
- clean test performance for the current JH2 aggregate;
- causality from PCA or feature association;
- HF gaps as Normal or negative;
- KAUH B, D, and E as independent patients;
- a pooled score across heterogeneous native tasks.

Potentially supportable statements must distinguish:

- published paper claims;
- local test-selected results;
- single-seed ablations;
- fixed-checkpoint external diagnostics;
- interpretations and hypotheses.
