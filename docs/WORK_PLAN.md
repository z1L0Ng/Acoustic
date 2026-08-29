# Work Plan

Updated: 2026-08-29

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
overwritten. No additional Core-2 training or terminal evaluation is authorized
by this plan update. A separate PAFA BEATs+CE reproduction line was authorized
on 2026-08-29, subject to code review, server resource checks, and the evidence
separation below.

## 2026-08-29 overnight execution queue

The overnight queue advances paper-critical work without retrospectively
changing the accepted five-run Core-2 results.

| Lane | Overnight output | Evidence/decision boundary |
|---|---|---|
| PAFA BEATs+CE | ICBHI direct flat4 reproduction entry and notebook; 16 kHz, 5 s repeat-pad/front-truncate, full BEATs iter3+ AS2M fine-tuning, batch 32, Adam 5e-5, weight decay 1e-6, 100 epochs, EMA 0.5, no SpecAugment | Run the clean patient-grouped validation-selected track as the main local comparator. Keep the author-faithful official-test-selected track as a separately labeled reproduction receipt. Never merge the two result directories or evidence labels. |
| Checkpoint selection | Read-only R0/N1/A1/L1/L2 loss-criterion sensitivity plus a prospective native-composite design | Existing selected checkpoints remain unchanged. Epoch-1-normalized and worst-dataset views are design diagnostics only; missing alternative checkpoints remain `HOLD`. |
| Audacity/acoustic evidence | Approved 20-selection panel, native-rate RMS/power/dBFS, Welch PSD, band power, spectral centroid/bandwidth, silence and SNR proxies | Representative visual/mechanism evidence only. HF empty annotation is `not_annotated`, never Normal/Negative. `.aup3` GUI projects remain separate from the quantitative script artifact until the interactive import is completed. |
| Paper preparation | Four-page English skeleton and primary-source BEATs/PAFA/follow-up comparison audit | Keep Paper Claim, Verified Local Result, Posthoc Diagnostic/Test-Informed, Proposed Method and HOLD separate. The current story is cross-dataset inconsistency and ontology/readout analysis, not uniform improvement or SOTA. |
| Student coordination | Natural Chinese clarification drafts for Wade and Hanlin | Draft-only and unsent. Wade must return numeric window/separability evidence; Hanlin must return paper score, local score and delta for an original-benchmark reproduction. |

PAFA server execution starts only after the clean track proves that official-test
annotations and audio are not read before validation checkpoint selection. The
server must also confirm sufficient free storage for 100 complete epoch
checkpoints and select a fully available GPU without interfering with other
users. The first local seed is 42; future multi-seed expansion is a separate
decision after the single-seed paper/local/delta row is reviewed.

## 2026-08-29 proposed mainline: PAFA-JH1 joint hierarchy

The earlier full PAFA single-seed reproduction reached Sp 76.88, Se 51.40 and
ICBHI Score 64.14 at its official-test-selected epoch 27. This verifies the
local PAFA execution path within its stated test-selection caveat. The newer
BEATs+CE receipts remain separate: the clean validation-selected terminal Score
is 54.83, while the author-faithful test-selected receipt is 59.86. These rows
motivate the following joint experiment but do not constitute evidence for it.

`PAFA-JH1` is a proposed method experiment, not a PAFA paper reproduction. It
keeps the PAFA input, optimization and patient-aware representation recipe while
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
| Loss | Classification weight 1.0 with equal-node eligible loss over Level1, Crackle and Wheeze; PAFA auxiliary weight 0.5 with lambda-PCSL 50 and lambda-GPAL 0.0005; unavailable SPRSound attributes remain unknown/masked, never negative |
| Checkpoint selection | Equal mean of ICBHI and SPRSound validation eligible-node loss; official/outer test absent during training and selection |
| Thresholds/readout | After selecting the epoch, fit one shared Crackle threshold and one shared Wheeze threshold on core validation only; Level1 Normal overrides attributes; freeze before terminal evaluation |
| Terminal evaluation | One access after selection: ICBHI official 2,756-cycle Sp/Se/Score plus Macro-F1/UAR/per-class recall; SPRSound BioCAS2022 inter Task1-1 native Score plus Macro-F1/UAR/support |
| Storage | Best checkpoint only, plus validation/terminal predictions and summaries; no per-epoch checkpoint archive |

The single-seed go/no-go gate is ICBHI Score at least 59.0 and SPRSound
Task1-1 Score at least 90.0, with no unexplained class collapse. Passing this
gate promotes PAFA-JH1 to the candidate paper mainline and authorizes a separate
multi-seed confirmation decision. Failing either threshold keeps it as a
diagnostic and does not replace the current Core-2 evidence.

Implementation and execution remain `READY_FOR_USER_START`. The runner must be
prepared and directly checked before a formal run. Server execution requires a
fresh all-GPU occupancy read and a completely available card; otherwise the
experiment remains on HOLD rather than sharing or preempting another user's
GPU.

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
