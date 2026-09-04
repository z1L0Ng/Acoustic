# Work Plan

Updated: 2026-09-03

## Current short-cycle plan

The active plan is the meeting-specific Friday--Sunday sprint:

- Meeting record:
  docs/meeting_records/2026-09-03_project_meeting_record_zh.md
- Work plan:
  docs/work_plans/2026-09-04_to_2026-09-06_work_plan_zh.md

This short-cycle plan supersedes the dated execution priorities below for
2026-09-04 through 2026-09-06. Older sections remain as historical context and
do not authorize experiments, Notion writes, Git operations, or changes to the
scientific protocol.

## Objective

Build a unified respiratory-sound system on the compatible ICBHI and SPRSound
label space, then determine whether input normalization, waveform augmentation,
or imbalance-aware loss can recover a reasonable and literature-comparable
performance level. BEATs is the current candidate backbone because it is the
strongest relevant local package reference, not because backbone superiority has
already been established.

## 2026-09-01 Work Plan audit

The former clean Local Queue is archived and is not a reference. The current
training line is the explicitly test-selected JH series: JH2 is the baseline,
JH3/JH3.1 are completed Soft Bridge diagnostics, JH3.2 ended at a failed
validation-only feasibility audit, and JH3.3 unit-level fbank MVN is complete.
JH3.4 Hard Hierarchy + MVN is the active local experiment. JH4 HF auxiliary
supervision is authorized conditionally after JH3.4 under the base-selection
and core-retention rules below.

The paper line is active and independent of JH3.3. The current LaTeX directory
contains an assembly-only `main.tex`, an empty Introduction and an empty
bibliography. The prior Markdown skeleton is absent from the working tree and
must not be silently restored or treated as current manuscript content.

Writing tasks for 2026-09-01 to 2026-09-02 are frozen in
`docs/paper/ICASSP_2026_acoustic_disease/WRITING_TRACK_2026-09-01_TO_09-02.md`.
The immediate deliverables are the claim ledger, section topology, Introduction
v1, dataset/task table, two figure specifications, three result-table
placeholders and primary-source citation bootstrap. JH3.3 changes only the
pending result row; it does not block drafting.

Where older sections below refer to the clean Local Queue, clean main selection,
or Confidence Bridge as HOLD, they are historical and superseded by this audit.

## 2026-08-30 advisor update: new sprint objective

By 2026-09-02, produce a reviewable story supported by measurable dataset
differentiation and a clean joint-model result. The matched single-source
cross-domain matrix remains important but is deferred until the current main
method is acceptable. By 2026-09-06, freeze the main clean protocol and the
minimum paper experiment matrix. The paper does not need to claim native-task
SOTA; it must show why heterogeneous datasets are hard to combine and whether
the proposed eligibility-aware hierarchy offers useful cross-dataset retention.

The current candidate narrative is:

1. ICBHI and SPRSound share partial respiratory-event semantics but differ in
   level, spectrum, duration, class balance, recording setup, prediction unit,
   and annotation support.
2. Models optimized for one source may not retain performance on the other.
3. A shared BEATs encoder with eligibility-aware hierarchical supervision is
   evaluated by native metrics on both datasets, without pooling them into one
   headline score.

JH2 is supporting diagnostic evidence only. Its epoch-19 ICBHI Score 60.05 and
SPRSound Score 89.20 were selected after 29 epochwise official-test accesses and
cannot be used as the clean main result or to tune the successor protocol.

## Paper-critical workstreams

### A. Dataset differentiation and Figure 1

- Expand the existing 20-track descriptive panel to dataset-wide, group-aware
  statistics using training partitions for any analysis that informs model
  design.
- Report class counts, duration/padding/truncation, RMS/dBFS, silence proxies,
  band power, spectral centroid/bandwidth, and the teacher-requested MVN-relevant
  feature statistics.
- Produce one compact multi-panel figure combining class/duration distribution,
  standardized PCA, and a quantitative domain-separability result.
- Quantitative evidence must use patient/recording-group-aware resampling. A
  dataset-ID classifier and a patient-level permutation/bootstrap distance are
  preferred over interpreting PCA separation alone.
- Repeat the key comparison on a shared Normal/Abnormal surface or a class-matched
  subset so that dataset identity is not trivially explained by label composition.

**Gate:** the paper may claim measurable dataset differentiation only after a
dataset-wide statistic with group-aware uncertainty/significance is complete.

### B. Single-source cross-domain matrix (deferred)

- Train matched ICBHI-only, SPRSound-only, and joint references with the same
  BEATs frontend and Level1 Normal/Abnormal readout.
- Evaluate the shared binary surface as a source-by-target matrix. Keep ICBHI
  flat4 and SPRSound Task1-1 native tables separate from this transfer analysis.
- Select each checkpoint only from its source-valid group-safe validation data;
  official tests are accessed once after the checkpoint and thresholds are frozen.

This matrix is not part of the current local queue. It will be reconsidered
after the joint main method, HF supervision decision, and KAUH evaluation plan
are reviewed. Until then, the manuscript must not claim superiority over
single-source transfer.

### C. Clean joint hierarchy successor

The clean successor starts from the JH1/JH2 architecture and PAFA training path,
but never uses official test for selection or early stopping.

| Item | Prospective clean contract |
|---|---|
| Data | ICBHI official-train + SPRSound train only |
| Split | Group-safe subtrain / calibration / selection partitions inside each training source |
| Input | 16 kHz, 5 s repeat-pad/front-truncate; no SpecAugment |
| Encoder/head | BEATs iter3+ AS2M full fine-tuning; Level1 CE + eligible Crackle/Wheeze BCE; PAFA PCSL/GPAL path |
| Batch | Native-unit batch 32; one-factor sampler ablation may replace equal batches with source-proportional 99:163 batches |
| Early stop | Patience 10 on the preregistered validation-only checkpoint objective; earliest epoch wins exact ties |
| Thresholds | Fit on calibration groups only; evaluate checkpoint candidates on disjoint selection groups |
| Test | One ICBHI and one SPRSound official-test access after checkpoint and thresholds are frozen |

The checkpoint objective must consider both datasets without an unqualified raw
mean. The existing JH1 epoch-6 validation predictions are split into fixed
calibration and selection groups to define a test-free SPRSound retention
guardrail. Candidate epochs that satisfy the guardrail are ranked by ICBHI
native validation Score and then SPRSound Score. If no candidate satisfies it,
rank by SPRSound retention and then ICBHI Score. Exact ties retain the earlier
epoch.

### D. One-factor ablations

Run only after the clean baseline and its selection rule are frozen.

1. **MVN:** unit-level scalar mean-variance normalization on the 5 s log-fbank,
   applied before patch embedding and replacing rather than stacking on top of
   the fixed BEATs affine normalization. Use the same rule for both datasets.
2. **Source-proportional sampling:** change only equal source batches to the
   natural 99 ICBHI / 163 SPRSound batch ratio; do not also multiply the dataset
   loss by sample count.
3. **Balanced hierarchical loss:** change only the classification objective.
   Balance positive/negative loss inside each eligible node, assign half of the
   classification weight to Level1 and half to Level2, and distribute Level2
   weight between Crackle/Wheeze using subtrain-only inverse effective positive
   support. Do not tune weights on test.
4. **PAFA loss ablation:** deferred until the three main interventions above are
   reviewed.

The confidence bridge between Level1 and Crackle/Wheeze is optional and remains
HOLD until the sampler, weighting, and MVN evidence is reviewed. It must not be
combined with another intervention in its first run.

### E. HF_Lung and external evaluation

- HF_Lung enters only if the validation-frozen main variant reaches clean
  terminal ICBHI Score at least 59, SPRSound Task1-1 Score at least 90, and no
  class collapse. HF uses a separate dataset-native/observed-positive auxiliary
  head and loss; empty files and annotation gaps remain unknown/not annotated.
- KAUH is a separate patient-grouped external evaluation with no training or
  checkpoint selection. B/D/E siblings stay grouped and unresolved labels HOLD.
- Neither HF nor KAUH may delay the ICBHI+SPRSound paper core or be described as
  four-dataset generalization before compatible metrics are complete.

## Two-day delivery schedule after the 2026-08-30 update

| Due | Reviewable artifact | Decision enabled |
|---|---|---|
| 2026-09-01 | Dataset-wide statistics table, Figure 1 draft, clean split/selection formula, refreshed result ledger | Is dataset differentiation quantitatively supported, and is the clean protocol runnable? |
| 2026-09-02 | Introduction/story v1, Figure 1/2 layout, dataset table, experiment-table placeholders | Advisor story checkpoint; freeze at most three contributions |
| 2026-09-03 | J-Clean/J-MVN validation status or completed rows | Confirm main-method runtime and MVN behavior |
| 2026-09-05 | J-Clean/J-MVN/J-SourceProp/J-BalancedLoss validation rows; freeze main role before terminal access | Select candidate main configuration without test-based reselection |
| 2026-09-07 | Complete terminal matrix; HF supervision and KAUH evaluation go/no-go | Freeze method and external scope |
| 2026-09-09 | Chosen main configuration multi-seed status; final Figure 1 and method diagram | Freeze headline evidence and figures |
| 2026-09-11 | Complete four-page draft with all claim labels and citations | Coauthor review and targeted repair only |
| 2026-09-13 | Revised compressed draft, final tables, limitations, reproducibility details | Content freeze |
| 2026-09-15 | Submission-ready package with one-day buffer | Final submission decision |

## Ownership and backup

- **Zilong:** clean protocol, model runs, story, paper skeleton, figures, and final
  evidence ledger.
- **Wade:** dataset-wide window/acoustic separability table and plots. If a
  numerical artifact is not available at the 2026-09-01 checkpoint, Zilong uses
  the existing panel pipeline to close the ICBHI+SPRSound critical subset.
- **Hanlin:** paper/local/delta rows for original-benchmark baselines. Missing
  student rows do not block the already completed local PAFA receipts or paper core.
- **Advisor/coauthors:** story/contribution review beginning 2026-09-02 rather
  than after all experiments finish.

## Approval boundary for the new sprint

On 2026-08-31 the user paused and archived the four-run Local Clean Queue.
J-Clean and J-MVN validation artifacts, the partial J-SourceProp run, and the
queue reference are `ARCHIVED_BY_USER_NOT_A_REFERENCE`; J-BalancedLoss was not
started. No archived artifact may be used for future checkpoint selection,
guardrails, protocol tuning, paper evidence, or HF/KAUH decisions. Official
tests for that archived queue remain closed.

The user subsequently authorized one new, separate local experiment:
`JH3 Soft Confidence Bridge`. JH3 reuses the JH2 training recipe, replaces the
hard main readout with a differentiable Level1/Crackle/Wheeze probability
bridge, and adds an ICBHI soft-flat4 loss while retaining the original node
loss and PAFA auxiliary objective. Checkpoint selection is preregistered as
50% ICBHI official-test Score plus 50% SPRSound validation Task1-1 Score;
SPRSound test is accessed only after selection. JH3 is explicitly test-selected
and is not a clean estimate.

After JH3 completes, the user authorized an isolated `JH3.1` rerun with the
same model, data, loss, seed, and optimization. Its only change is checkpoint
selection and early stopping:

\[
S_{JH3.1}=0.6\,S_{\mathrm{ICBHI\ test}}+0.4\,S_{\mathrm{SPR\ validation}}.
\]

This weighting gives approximately equal baseline contributions when ICBHI is
near 60 and SPRSound is near 90, while making a one-point ICBHI improvement
more influential. JH3.1 must run in a separate output directory, access SPRSound
test only after selection, and remain explicitly test-selected/non-clean.

The user authorized `JH3.2 Specificity-Calibrated Soft Bridge` after JH3.1.
Phase 0 is a validation-only feasibility audit that adds one shared Normal-logit
margin and maximizes ICBHI validation specificity subject to ICBHI validation
sensitivity dropping no more than one point and SPRSound validation Score
dropping no more than half a point. Margin candidates are derived from exact
validation decision boundaries. A formal JH3.2 run starts only if specificity
can improve by at least two points under those constraints. If started, JH3.2
keeps the JH3.1 60/40 checkpoint criterion, fits and freezes the margin on
validation each epoch, and remains test-selected/non-clean.

The JH3.2 validation-only feasibility gate failed, so no formal JH3.2 training
was started. The user then authorized `JH3.3 Unit-Level Scalar Log-Fbank MVN`.
JH3.3 reuses the complete JH3.1 model, loss, data, seed, optimization, and 60/40
selection contract. Its only change replaces the fixed BEATs fbank affine with
per-unit scalar mean-variance normalization over all time-frequency bins. It is
not per-frequency CMVN, dataset-specific normalization, or waveform RMS/peak
normalization. JH3.3 remains test-selected/non-clean and must run in an isolated
output directory.

After JH3.3 completed without recovering ICBHI specificity, the user authorized
`JH3.4 Hard Hierarchy + Unit-Level Scalar Fbank MVN` as the direct one-factor
comparison to JH2. JH3.4 restores the JH2 hard hierarchy, eligible-node loss,
PAFA objective, equal-source batching, and ICBHI-test checkpoint selection; its
only method change is the same per-unit scalar fbank MVN used in JH3.3.
SPRSound test is accessed only once after selection. JH3.4 is local,
test-selected/non-clean, and must run in an isolated output directory. Its main
gate is ICBHI Score at least 60.05, specificity at least 74, SPRSound test Score
at least 89, and no class collapse.

`JH4 HF auxiliary supervision` is now conditionally authorized after JH3.4.
JH4 uses JH3.4 as its base only if JH3.4 reaches ICBHI Score at least 60.05,
specificity at least 74, SPRSound test Score at least 89, and no class collapse;
otherwise it falls back to JH2. JH4 keeps the selected base's ICBHI-test
checkpoint criterion unchanged and adds HF through a shared Crackle/Wheeze
auxiliary loss with fixed weight 0.25. HF gaps, empty labels, phase-only windows,
and Level1 are masked; HF validation is logged but excluded from selection.
KAUH remains evaluation-only and is not rerun without separate approval.

On 2026-09-02 the user authorized the paper-facing JH2 three-seed confirmation
locally. The final seed set is 0, 1, and a fresh 42, run sequentially under the
JH2 Hard Hierarchy contract with ICBHI official-test checkpoint selection and
patience 10. Seed 1 is already complete. Seed 2 was stopped by the user after
an early all-Normal collapse and is excluded from the formal set; seed 3 is no
longer scheduled. SPRSound official inter is accessed only once after selection
for each formal seed. The fresh seed-42 run must not reuse the historical
seed-42 checkpoint because its SPRSound test-access history differs. The final
aggregate reports mean and sample standard deviation across seeds 0, 1, and the
fresh 42; no HF, KAUH, MVN, Soft Bridge, or additional experiment is included.

## Historical Core-2 reference protocol (audit trail)

The sections below preserve the completed 2026-08-27 to 2026-08-29 plan and
result lineage. Where they conflict with the 2026-08-30 sprint above, the new
sprint controls future work. Historical selection rules and proposed runs are
not automatically carried into the clean successor.

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

## Completed Core-2 experiment sequence (historical)

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

## Historical Core-2 state

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
overwritten. No additional Core-2 training or terminal evaluation is authorized
by this plan update. The subsequent PAFA BEATs+CE line is now complete: the
clean track reached terminal Score 54.83 after validation selection, and the
author-faithful test-selected receipt reached 59.86. These remain separate
evidence rows.

## 2026-08-29 overnight queue outcome (historical)

The overnight queue completed the preparation and result lines below without
retrospectively changing the accepted five-run Core-2 results.

| Lane | Overnight output | Evidence/decision boundary |
|---|---|---|
| PAFA BEATs+CE | ICBHI direct flat4 reproduction entry and notebook; 16 kHz, 5 s repeat-pad/front-truncate, full BEATs iter3+ AS2M fine-tuning, batch 32, Adam 5e-5, weight decay 1e-6, 100 epochs, EMA 0.5, no SpecAugment | Run the clean patient-grouped validation-selected track as the main local comparator. Keep the author-faithful official-test-selected track as a separately labeled reproduction receipt. Never merge the two result directories or evidence labels. |
| Checkpoint selection | Read-only R0/N1/A1/L1/L2 loss-criterion sensitivity plus a prospective native-composite design | Existing selected checkpoints remain unchanged. Epoch-1-normalized and worst-dataset views are design diagnostics only; missing alternative checkpoints remain `HOLD`. |
| Audacity/acoustic evidence | Approved 20-selection panel, native-rate RMS/power/dBFS, Welch PSD, band power, spectral centroid/bandwidth, silence and SNR proxies | Representative visual/mechanism evidence only. HF empty annotation is `not_annotated`, never Normal/Negative. `.aup3` GUI projects remain separate from the quantitative script artifact until the interactive import is completed. |
| Paper preparation | Four-page English skeleton and primary-source BEATs/PAFA/follow-up comparison audit | Keep Paper Claim, Verified Local Result, Posthoc Diagnostic/Test-Informed, Proposed Method and HOLD separate. The current story is cross-dataset inconsistency and ontology/readout analysis, not uniform improvement or SOTA. |
| Student coordination | Natural Chinese clarification drafts for Wade and Hanlin | Draft-only and unsent. Wade must return numeric window/separability evidence; Hanlin must return paper score, local score and delta for an original-benchmark reproduction. |

PAFA ultimately ran locally with best-checkpoint-only storage. The clean track
kept official test lazy until epoch selection; the author track remained
explicitly test-selected. Future multi-seed expansion is a separate decision.

## 2026-08-29 PAFA-JH1/JH2 outcome (historical)

The earlier full PAFA single-seed reproduction reached Sp 76.88, Se 51.40 and
ICBHI Score 64.14 at its official-test-selected epoch 27. This verifies the
local PAFA execution path within its stated test-selection caveat. The newer
BEATs+CE receipts remain separate: the clean validation-selected terminal Score
is 54.83, while the author-faithful test-selected receipt is 59.86. These rows
motivate the following joint experiment but do not constitute evidence for it.

`PAFA-JH1` was a proposed method experiment, not a PAFA paper reproduction. It
kept the PAFA input, optimization and patient-aware representation recipe while
replacing the direct flat-four classifier with the project's shared hierarchy.

| Field | Frozen PAFA-JH1 contract |
|---|---|
| Training datasets | ICBHI official-train respiratory cycles and SPRSound BioCAS2022 training events only; HF and KAUH excluded |
| Input | Mono 16 kHz; one 5 s input per native unit; short units repeat-padded and long units front-truncated using the PAFA waveform path; no SpecAugment |
| Encoder | BEATs iter3+ AS2M, full end-to-end fine-tuning |
| Prediction head | Shared `Level1` Normal/Abnormal softmax plus Abnormal attributes `Crackle` and `Wheeze`; no `Other`; ICBHI flat four reconstructed as Normal/Crackle/Wheeze/Both |
| PAFA auxiliary path | Author projection head plus PCSL/GPAL on dataset-valid patient IDs; patient identifiers remain dataset-scoped and are never shared across ICBHI and SPRSound |
| Batching | Native-unit batch 32, dataset-homogeneous batches, equal dataset contribution per epoch; no cross-dataset patient centroid |
| Optimization | 50 epochs; Adam, learning rate 5e-5, weight decay 1e-6, cosine schedule and EMA beta 0.5 |
| Loss | Classification weight 1.0 with equal-node eligible loss over Level1, Crackle and Wheeze; PAFA auxiliary weight 1.0 with lambda-PCSL 50 and lambda-GPAL 0.0005; unavailable SPRSound attributes remain unknown/masked, never negative |
| Checkpoint selection | Equal mean of ICBHI and SPRSound validation eligible-node loss; official/outer test absent during training and selection |
| Thresholds/readout | After selecting the epoch, fit one shared Crackle threshold and one shared Wheeze threshold on core validation only; Level1 Normal overrides attributes; freeze before terminal evaluation |
| Terminal evaluation | One access after selection: ICBHI official 2,756-cycle Sp/Se/Score plus Macro-F1/UAR/per-class recall; SPRSound BioCAS2022 inter Task1-1 native Score plus Macro-F1/UAR/support |
| Storage | Best checkpoint only, plus validation/terminal predictions and summaries; no per-epoch checkpoint archive |

The single-seed go/no-go gate is ICBHI Score at least 59.0 and SPRSound
Task1-1 Score at least 90.0, with no unexplained class collapse. Passing this
gate promotes PAFA-JH1 to the candidate paper mainline and authorizes a separate
multi-seed confirmation decision. Failing either threshold keeps it as a
diagnostic and does not replace the current Core-2 evidence.

JH1 stopped abnormally during epoch 26 after 25 complete epochs. Its validation-
selected diagnostic produced ICBHI Score 54.29 and SPRSound Score 90.00, but it
is permanently labeled incomplete. JH2 reused the same scientific recipe and
changed selection to epochwise ICBHI official-test Score. It selected epoch 19
with ICBHI Score 60.05 and SPRSound Score 89.20, then stopped at epoch 29 after
ten non-improving epochs. Because official tests were accessed every epoch, JH2
is test-exposed diagnostic evidence and does not replace a clean JH1 successor.

### Prospective early-stopping policy (2026-08-30)

All newly started formal training runs use epoch-boundary early stopping with
patience 10. A run stops after ten consecutive completed epochs without a
strict improvement in its preregistered checkpoint-selection objective; ties
retain the earlier epoch, and any strict improvement resets the counter. The
maximum epoch count remains an upper bound. The monitor must be identical to
the run's checkpoint-selection objective: validation-only experiments use only
their frozen validation criterion, while explicitly test-selected diagnostics
may use their declared test criterion and must remain permanently labeled
test-exposed. Early stopping never authorizes test access for a clean run or a
change of metric after training begins.

## Superseded 2026-08-27 deadline plan (audit trail)

The official full-paper deadline is 2026-09-16. As of 2026-08-27, the project
has 20 calendar days remaining. The internal schedule preserves one day of
submission buffer.

## Superseded two-day cadence after the 2026-08-27 meeting

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
