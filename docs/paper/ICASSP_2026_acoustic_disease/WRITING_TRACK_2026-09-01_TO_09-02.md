# Writing Track｜2026-09-01 to 2026-09-02

Status: `READY_FOR_EXECUTION / PAPER ONLY`

## Goal

By 2026-09-02, produce an advisor-reviewable story and manuscript skeleton that
can absorb the final JH3.3 row without restructuring the paper. Writing proceeds
independently from the active local training task.

## Current evidence boundary

- JH2, JH3 and JH3.1 are test-selected diagnostics, not clean estimates.
- JH3.2 stopped at a validation-only feasibility audit; no formal run exists.
- JH3.3 is running and remains a placeholder until its final handoff.
- HF_Lung and KAUH results are fixed-JH2 external-transfer diagnostics, not
  native task reproductions.
- The Local Clean Queue is `ARCHIVED_BY_USER_NOT_A_REFERENCE` and must not enter
  manuscript text, tables, figures, method selection or citations.
- Paper claims must separate Paper Claim, Verified Local/Test-Selected Result,
  External Diagnostic, Interpretation, Proposed Method and HOLD.

## P0 tasks

### W1 — Freeze story and claim ledger

**Objective:** define what the paper says before filling sections.

**Deliverable:** `story_and_claim_ledger.md`.

**Required content:**

- one-sentence problem statement;
- one-sentence method statement;
- at most three contribution bullets;
- accepted evidence for each contribution;
- forbidden wording and unresolved HOLDs;
- role of JH2/JH3/JH3.1/JH3.3, HF_Lung and KAUH.

**Acceptance:** no `SOTA`, `robust`, `generalizable`, `first` or clean-test claim
without explicit evidence.

### W2 — Freeze manuscript topology

**Objective:** define a compact four-page section layout.

**Proposed topology:**

1. Introduction and compressed Related Work;
2. Datasets, task alignment and eligibility;
3. Method: BEATs + PAFA + hierarchical/soft-bridge core;
4. Experimental protocol;
5. Results and cross-dataset diagnostics;
6. Discussion, limitations and conclusion.

**Deliverable:** section-file inventory and `main.tex` input plan. Do not rename
the current `ICASSP_2026_acoustic_disease` directory without a user decision,
even though the target venue is ICASSP 2027.

### W3 — Draft Introduction v1

**Objective:** fill `Section/1introduction.tex` without depending on JH3.3.

**Paragraph contract:**

1. respiratory-sound models are usually optimized within one dataset/task;
2. acquisition, unit, label support and benchmark differences limit naive joint
   training;
3. existing shared-label approaches risk manufacturing supervision or losing
   native-task meaning;
4. our eligibility-aware, patient-aware hierarchical BEATs approach and native
   reporting contract;
5. at most three contribution bullets.

**Acceptance:** no final headline number until JH3.3 closes; citations inserted
through `citation.bib`, not raw URLs in prose.

### W4 — Prepare Dataset/Task table

**Objective:** create a compact table source covering ICBHI, SPRSound,
HF_Lung and KAUH.

**Columns:** dataset; prediction unit; usable labels; supervision role; split;
native/external metric; hard boundary.

**Mandatory boundaries:**

- ICBHI official recording split is not described as strict patient-held-out;
- SPRSound inter and intra are never pooled;
- HF gap/empty annotation is not Normal/Negative;
- KAUH B/D/E are same-patient filters; unresolved strings remain excluded.

### W5 — Prepare Figure 1 and Figure 2 specifications

**Figure 1:** dataset heterogeneity and label/support differences. Use only
accepted quantitative evidence; the 20-track panel is illustrative, not
dataset-wide significance.

**Figure 2:** pipeline from native units to 16-kHz/5-s input, BEATs, PAFA
patient objective, Level1 + Crackle/Wheeze heads, Soft Bridge, native readouts,
selection boundary and HF/KAUH external lanes.

**Deliverable:** figure captions, required source fields and explicit missing
panels; no final graphical rendering is required in this task.

### W6 — Prepare Results tables and placeholders

**Table A — ICBHI:** paper references; JH2; JH3; JH3.1; JH3.3 placeholder.
Report Sp/Se/Score/Macro-F1/UAR and evidence/selection label.

**Table B — SPRSound:** corresponding Task1-1 rows and exact inter protocol.

**Table C — External diagnostics:** HF D/Wheeze and KAUH recording/patient rows
with explicit incompatibility notes.

**Ablation row:** hard hierarchy, bits-only and Soft Bridge only when derived
from the same checkpoint and clearly labeled.

### W7 — Bootstrap citations

Add verified BibTeX entries for BEATs, PAFA, Patch-Mix, SG-SCL, SPRSound,
HF_Lung_V1, KAUH/OPERA and the primary dataset papers. Do not use secondary
review numbers where a primary source is available.

## Schedule

### 2026-09-01

- Complete W1, W2 and W4.
- Draft W3 paragraphs 1–4.
- Prepare W5/W6 structures with HOLD placeholders.

### 2026-09-02

- Complete Introduction v1 and contribution bullets.
- Complete table/caption specifications and citation bootstrap.
- Insert JH3.3 only if the final handoff is complete and evidence-labeled.
- Deliver one advisor-reviewable manuscript skeleton and a list of missing
  experiments/figures.

## Dependencies and HOLD

- JH3.3 final row: pending local training; does not block W1–W7 structure.
- Dataset-wide separability/PCA: pending; Figure 1 must retain placeholders if
  unavailable.
- Title, authors, affiliations, abstract and keywords: HOLD pending user input.
- Final directory naming (`ICASSP_2026_acoustic_disease` versus ICASSP 2027):
  HOLD; do not rename automatically.
- No experiment, Notion mutation, Git commit/push, paper submission or external
  communication is authorized by this writing task.

## Review gate

- Story, figures and tables form one coherent argument.
- Every numeric row has a task/split/unit/selection/evidence label.
- No archived Local Clean result appears.
- JH3.3 remains a placeholder until completion.
- The manuscript can be understood from Introduction + Figure 1/2 + Tables A–C.
