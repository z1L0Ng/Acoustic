# Acoustic End-to-End Pipeline — Frozen Textual Specification v1

- **Status:** P0-2 content freeze for visual v1
- **Date:** 2026-08-01
- **Scope:** ICBHI 2017, SPRSound BioCAS2022, HF_Lung_V1, KAUH/Fraiwan v3
- **Authoritative upstream:** accepted Notion [`Pipeline Module SOTA Map`](https://app.notion.com/p/3b0309efda2981899476c6fcaca45d47) and the reviewed [`End-to-End Pipeline — Phase 0`](https://app.notion.com/p/3ab309efda2981f7a7ceeed39832700c) box-by-box specification
- **Local evidence boundary:** read-only task contracts, executable-baseline receipts, and completed frozen-feature diagnostic decisions
- **Out of scope:** new/private/self-collected data onboarding, experiments, training, server/cache work, model selection, and architecture promotion

## 1. Evidence grammar used by the figure

The visual must never collapse the following states:

| State | Meaning in this project | Allowed visual treatment |
|---|---|---|
| **Fixed Contract** | Audited data/task/interface/evaluation rule. It may be implementable without being empirically superior. | Solid navy outline. |
| **Executable Reference** | Code path with protocol/smoke/profile receipts. This is engineering evidence, not a performance result. | Solid green fill/outline and an explicit `engineering only` label. |
| **Verified Result** | A completed local controlled result within its stated protocol. Negative and inconclusive outcomes remain results. | Evidence-registry callout, never a forward audio module. |
| **Interpretation** | A bounded reading of verified facts/results. It is not causal proof. | Small amber note attached to the relevant evidence rail. |
| **Candidate / Proposed Method** | An architecture or method that remains to be compared. External-paper support is traceability, not local validation. | Dashed blue/purple outline plus SOTA work-card IDs. |
| **HOLD** | Blocked by missing semantics, capability, parity, provenance, or a failed/inconclusive prior gate. | Orange/red label; no solid forward-flow arrow. |
| **Future Plan** | Work requiring a new approval or preregistered comparison. | Dotted control arrow only. |

Explicit prohibitions:

- A smoke/profile receipt is not a performance result.
- Frozen encoder + target-head adaptation uses target labels and is **not zero-shot**.
- Dataset-ID information being probe-accessible does **not** prove that a pathology shortcut is used.
- HF_Lung unlabeled gaps are `not_annotated`, not verified normal/negative.
- KAUH `B/D/E` filter replicas stay patient-grouped; 336 recordings are not 336 independent patients.
- SPRSound `inter` and `intra` remain separate protocols.
- Shared encoder, multiple encoders, general/specific factorization, router, and MoE are candidates, not the selected method.
- Phase 1A is not shown as a successful method-selection result.
- An external paper claim is never relabeled as a local result.

## 2. Master-figure topology

The v1 master figure is one auditable composition with four visually separated regions:

1. **Main audio / inference path:** four audited inputs → dataset adapters → native prediction units → proposed mask-aware preprocessing → declared backbone interface → eligible native/shared heads → predictions.
2. **Executable reference inset:** only ICBHI cycles + SPRSound events → legacy fixed-8-s AST path → pooled AST representation → active native heads. It is not merged into the four-dataset target path.
3. **Training and evaluation control plane:** eligibility, source sampling/weighting, masked losses, evidence-gated tail controls, evaluation, and claim ledger act laterally; they are not sequential audio transforms.
4. **Candidate / HOLD rail:** architecture alternatives and SOTA-traceable candidates remain outside the committed main path until a controlled gate promotes them.

The master figure therefore represents a **target system contract plus bounded references**, not a claim that a four-dataset end-to-end model has already been trained.

## 3. Box contracts

### Box 01 — Four audited dataset inputs

**State:** Fixed source/task facts. Full audit detail lives in the textual specification; the figure uses compact cards.

| Dataset | Input structure and scale | Native unit / output contract | Grouping and split boundary |
|---|---|---|---|
| **ICBHI 2017** | 920 recordings, 5.492508 h; heterogeneous 4/10/44.1 kHz WAV; 6,898 annotated respiratory cycles; 126 patients | Primary acoustic unit: cycle. Native flat four: normal, crackle, wheeze, both. | Official recording split: 4,142/2,756 cycles; patients 156 and 218 overlap, so it is not patient-independent. Internal validation must be patient-grouped. |
| **SPRSound BioCAS2022** | 2,683 recordings, 8.162338 h; 8 kHz mono 16-bit; 9,089 events | Event binary and event seven-class are active native tasks. Recording task is a later native extension. | Official `inter` is unseen-patient primary; `intra` is repeated-subject diagnostic. Never pool them. |
| **HF_Lung_V1** | 9,765 15-s recordings, 40.6875 h; 4 kHz mono 16-bit; 81,933 positive interval labels | Native temporal multi-label: phase I/E and adventitious D/Wheeze/Rhonchi/Stridor. | Train/test folders use deidentified-date grouping proxy; no patient ID. Overlap and unlabeled gaps are preserved. Shared binary/narrow-four targets are blocked. |
| **KAUH / Fraiwan v3** | 336 recordings, 112 patients, 1.623918 h; 4 kHz mono 16-bit; three filter replicas per patient | Native recording sound-type head over raw strings; diagnosis is separate patient-level secondary/HOLD. | No official split. P-number is the grouping key; B/D/E replicas stay together. `Crep`, `Bronchial`, and `I C B` are not silently harmonized. |

**Figure output:** four compact source cards with dataset, count, native unit, and one critical warning.

### Box 02 — Deleted

Dataset background/detail is supplementary content inside Box 01 and the audit documents. Box 02 does not appear in the roadmap or figure.

### Box 03 — Unified dataset adapter layer

**State:** Proposed software contract for the current four audited datasets; non-trainable by default.

```text
CanonicalRecord
  dataset_id, release_id, adapter_version, schema_version
  recording_id, audio_ref, source_sample_rate, duration
  native_annotation_list/table, native_record_label, native_diagnosis
  group_id + group_status, source_split + split_status
  device/site/quality/demographic metadata where provided
  annotation-state codes, lineage, source receipt
```

The adapter preserves native values. `not_applicable`, `not_provided`, and `unknown` are distinct. It does not fabricate missing metadata, convert unlabeled time to negative, perform target harmonization, or encode trainable dataset identity.

### Box 04 — Prediction unit builder

**State:** Fixed Contract.

```text
recording/source object
  → ICBHI cycle
  → SPRSound event
  → HF_Lung 15-s temporal sequence
  → KAUH recording
```

Output is a variable-length waveform `x_i [T_i]`, native targets/support, interval annotations where available, and immutable source-time lineage. Unit mismatch is a hard error.

### Box 05 — Dataset-aware preprocessing and collation

Two contracts remain separate.

**Figure A — Executable legacy AST reference:**

```text
annotated cycle/event crop → mono → 16 kHz
fixed 8 s repeat/truncate → 128-bin fbank
F_ast [B,1,798,128] | no waveform/token padding mask
```

This path is implemented only for the current ICBHI+SPRSound reference.

**Figure B — Four-dataset target contract:**

```text
unit-aware extraction → resample to f_enc
variable or windowed waveform → pad/collate
X [B,T] + M_pad [B,T]
aligned targets + token_mask + time_map
```

The target contract freezes mask-awareness and source-time alignment, but not the final sampling rate, window length, overlap, frontend, or aggregation rule. Spectrogram backbones may receive `F [B,L,F_bin]` or `[B,1,L,F_bin]`; `L` cannot be declared constant before the backbone/window decision.

### Box 06 — Backbone slot and representation interface

**Target interface — Fixed Contract:**

```text
BackboneOutput
  pooled    [B,D]                  required for pooled tasks
  tokens    [B,L,D]                required only if temporal_capable
  token_mask[B,L]                  required only if temporal_capable
  time_map  [B,L,2]                required only if temporal_capable
  capability = pooled_only | temporal_capable
  backbone/init/checkpoint/input/output receipts
```

Special tokens are excluded from `L` unless explicitly declared. A temporal-capable path must map each acoustic token to source-time support.

**Current executable reference — AST base384:**

- Input `F_ast [B,1,798,128]`, internally transposed to `[B,1,128,798]`.
- Overlapping Conv2d patches: kernel `16×16`, stride `10×10`, grid `12×79`, 948 acoustic patch tokens, width 768.
- Add CLS and distilled tokens: sequence length 950.
- AudioSet source positional grid `12×101` is center-cropped on the time dimension to 79 under the current 798-frame input; this path does not use interpolation for time.
- 12 DeiT-Base Transformer blocks, 12 attention heads, width 768, MLP width 3,072.
- Public output is `(CLS + DIST)/2 = pooled z [B,768]` only.
- No public acoustic tokens, padding mask, token mask, or time map.
- Receipts establish protocol/smoke/profile executability, not completed joint-training performance.

**SOTA-traceable encoder candidates:** AST `W09`, BEATs `W10`, OPERA `W12`; PANNs `W11` is a lower-complexity control and HeAR `W13` is an input-contract-sensitive alternative. None is promoted by this specification.

### Boxes 07–12 — Architecture alternatives, not the main path

#### Box 07 — Dataset-specific encoder bank

- **State:** Candidate / HOLD.
- **Interface:** one declared backbone output per dataset or unit, followed by explicit projection to a common width.
- **Controls:** equal parameter/compute budget and identical splits, heads, optimization, and update counts.
- **SOTA trace:** multi-view/specific branches `W07`; native dataset branches `W08`, `W25`, `W29`, `W32`.

#### Box 08 — General respiratory representation

- **State:** Proposed Method; no local empirical evidence.
- **Interface:** `pooled/tokens → g [B,Dg]` while preserving capability and masks.
- **Purpose:** expose cross-dataset-compatible acoustic evidence without forcing native labels to be identical.
- **SOTA trace:** conceptual multi-dataset/representation candidates `W12`, `W19` (the latter remains HOLD due missing code/checkpoint).

#### Box 09 — Dataset/domain-specific residual

- **State:** Proposed Method.
- **Interface:** `pooled/tokens → s [B,Ds]`; it may serve only eligible native tasks after controlled fusion.
- **Risk:** dataset-ID lookup or pathology/acquisition shortcut; dataset-ID predictability alone is not causal proof.
- **SOTA trace:** metadata/domain adapters `W15`, `W16`, patient-aware adapter `W18`; each requires metadata/patient parity and native-task controls.

#### Box 10 — Soft router

- **State:** Candidate / HOLD until dense baselines pass.
- **Interface:** acoustic representation `→ α [B,E]`, `α_e≥0`, `Σα_e=1`; dataset ID is not a main-method input.
- **Required audit:** usage/load balance, collapse, counterfactual dataset-ID control, equal-parameter dense baseline.
- **SOTA trace:** CNN-MoE/router reference `W23` only; no four-dataset local empirical support.

#### Box 11 — Expert bank / MoE

- **State:** Candidate / HOLD.
- **Interface:** experts preserve pooled/temporal capability and expose utilization receipts.
- **SOTA trace:** `W23`; multimodal expert/side-channel idea `W24` is separate HOLD and must not be inserted into the acoustic-only main path.

#### Box 12 — Representation fusion

- **State:** Candidate.
- **Interface:** general/specific, multi-view, or expert outputs `→ fused representation` with mask/time alignment preserved.
- **SOTA trace:** MVST `W07`; EZhouNet temporal fusion `W08`; BTS `W24` is multimodal HOLD.

### Box 13 — Shared-compatible heads

**State:** Proposed interface / HOLD as a claimed harmonization solution.

- Shared binary and narrow-four outputs are legally open only for ICBHI and the approved SPRSound subset.
- HF_Lung and KAUH are blocked from those shared losses under the current contract.
- The corrected frozen-feature comparison found **0 material improvements** for a parameter-matched shared head versus independent heads. This result belongs in the evidence registry and does not prove that all future shared heads are impossible.
- Any re-entry requires a new preregistered comparison with native-task side-effect and worst-dataset guardrails.

### Box 14 — ICBHI native head

**State:** Fixed Contract; active in the executable reference.

Input `pooled [B,D*]`; output cycle flat-four logits `[B,4]`. Report sensitivity, specificity, ICBHI Score, macro-F1, UAR, per-class recall, and support with the official-split patient-overlap caveat.

### Box 15 — SPRSound native heads

**State:** Fixed Contract; active in the executable reference.

Input event embedding `[B,D*]`; output event binary `[B,2]` and seven-class `[B,7]`. `inter` is primary and `intra` is a separate repeated-subject diagnostic. Target-supervised head adaptation is not zero-shot.

### Box 16 — HF_Lung native temporal heads

**State:** Fixed safe task contract; implementation HOLD until a temporal-capable backbone exists.

Input `tokens [B,L,D] + token_mask [B,L] + time_map [B,L,2]`; output phase logits `[B,L,2]` and adventitious multilabel logits `[B,L,4]`, plus temporal validity masks. The current pooled-only AST cannot drive this branch.

**SOTA trace:** source temporal benchmark `W02`; temporal head/fusion candidates `W08`, `W29` remain compatibility-gated.

### Box 17 — KAUH native recording head

**State:** Fixed safe task contract; shared mapping remains blocked.

Input recording embedding `[B,D*]`; output raw sound-type logits `[B,C_kauh]` with the current raw ontology around nine strings. P-number grouped evaluation and B/D/E replica integrity are mandatory. Disease prediction is a separate secondary/HOLD branch.

**SOTA trace:** dataset/source card `W04`; recording/disease references `W21`, `W22` do not change the native contract.

### Box 18 — Task eligibility control

**State:** Fixed Control-Plane Contract.

```text
M_head [B,H] | M_class [B,C_h] | M_time [B,L,K]
```

Task capability and annotation-state codes laterally control allowed heads, loss denominators, and evaluation rows. At inference, declared task/capability opens legal outputs without reading target labels. Missing/unknown/not-annotated never becomes negative by default.

### Box 19 — Source sampling / weighting control

**State:** Mixed evidence; training-only.

- Executable two-dataset reference: naive source-proportional homogeneous batches and unweighted CE.
- Four-dataset `D2` exists only as a target-supervised cached frozen-feature diagnostic; it does not establish full-encoder source-balancing benefit.
- Future comparison must hold model, initialization, optimizer, update budget, validation, and seed fixed.

Outputs are batch composition, source weight `w_d`, and effective update receipts. The sampler acts on batch construction; `w_d` acts on Box 21.

### Box 20 — Tail research control and evidence pointer

**State:** Research control, not a selected solution.

- Support-aware cRT: two local material improvements, but `all_regression_guardrails_pass=false`; decision is HOLD/negative.
- Learned event-sensitive pooling: zero material votes; inconclusive/not supported.
- A future class-balanced loss, margin, logit adjustment, or pooling mechanism needs a new preregistration and connects to exactly one intervention point.

**SOTA trace for future controlled candidates:** focal/imbalance `W06`; Patch-Mix contrastive `W14`; KD `W27`; SPRSound focal/SupCon references `W32`, `W33`. None is drawn as verified.

### Box 21 — Training-only masked loss aggregator

**State:** Fixed control interface; exact future loss composition not selected.

```text
L = Σ_h λ_h · masked_loss_h
effective mask = eligibility × valid annotation × approved class control
optional source weight = w_d
```

The executable reference uses ICBHI CE plus the mean of SPRSound binary/seven CE, sample-count weighted. Boxes 18 and 19 feed the loss laterally. Box 20 can connect only after a new approval. Inference bypasses Box 21 entirely.

### Box 22 — Evaluation and claim ledger

**State:** Fixed Evaluation Contract.

Inputs are predictions, targets, valid masks, support, split/lineage receipts, checkpoint identity, target-label use, and selection caveats. Outputs are per-dataset/per-unit/per-class metrics, worst-dataset results, LODO within the current four datasets, attribution/shortcut diagnostics, and a claim-evidence map.

The ledger must separate:

- zero-target transfer;
- frozen encoder + target-supervised target head;
- full target training;
- joint multi-dataset training;
- engineering smoke/profile;
- external paper claims.

Current corrected evidence pointers:

1. Shared-compatible head harmonization: 0 material improvements; HOLD/negative within frozen-feature scope.
2. Support-aware cRT: local gains but global regression guardrail failure.
3. Shortcut diagnostic: 0/2 evidence votes; not supported/inconclusive. Dataset-ID access alone is insufficient.
4. Event-sensitive pooling: 0 material votes; not supported/inconclusive.
5. Four-dataset representation attribution: target-supervised, single-seed frozen-feature evidence with selection caveats; no full-encoder conclusion.

## 4. Data, training, inference, and evaluation paths

### 4.1 Target four-dataset training path

```text
datasets
  → adapter → prediction-unit builder → mask-aware preprocessing
  → declared backbone interface → eligible native/shared heads → logits

task/capability contract → eligibility masks → heads + loss + evaluation
dataset membership/update budget → sampler → batches
source weights ───────────────────────────────→ masked loss
targets + logits + valid masks ──────────────→ masked loss → update
approved tail experiment --dotted, one point→ sampler OR loss OR pooling
```

No arrow may imply that audio sequentially passes through Boxes 18–21.

### 4.2 Known-dataset inference path

```text
raw audio + declared dataset/task
  → same adapter/unit/preprocessing contract
  → selected backbone
  → eligibility opens only legal heads
  → prediction + provenance/context
```

Boxes 19–21 are bypassed. Box 18 does not consume target labels at inference.

### 4.3 Executable reference path

```text
ICBHI cycles + SPRSound events
  → legacy 8-s repeat/truncate fbank
  → pooled-only AudioSet-initialized AST
  → ICBHI [B,4] + SPRSound [B,2]/[B,7]
```

Its receipts prove executable data flow, finite loss/gradients, routing, resume, and CPU profiling. They do not authorize performance or generalization claims.

### 4.4 Evaluation path

```text
predictions + targets + masks + supports + receipts
  → per-dataset/native metrics
  → protocol-separated comparisons
  → claim ledger with explicit evidence level
```

LODO is limited to the current four audited datasets. This version makes no generic unseen/private-data claim.

## 5. Visual freeze for SVG v1

- Landscape page with a left-to-right main path and a distinct bottom control plane.
- Four source cards stay visible, but detailed counts remain in this specification/matrix.
- Solid navy: fixed contract; solid green: executable reference; dashed blue/purple: candidate; orange/red: HOLD; dotted grey/amber: control or evidence pointer.
- Each candidate shown in the figure carries one or more `Wxx` SOTA work-card IDs.
- The executable reference is a self-contained inset and does not visually feed HF_Lung or KAUH.
- HF_Lung temporal branch is visibly capability-gated by `tokens + token_mask + time_map`.
- P-Shared is labeled `proposed interface — no local empirical evidence`.
- Corrected negative/inconclusive evidence stays in a registry/rail, not the forward audio flow.

## 6. Acceptance criteria

The v1 package passes only if:

1. Four dataset units, counts, and critical split/label constraints match the audited task contract.
2. Figure A contains only ICBHI and SPRSound.
3. Current AST dimensions, center-crop behavior, pooled-only output, and no-mask limitation are exact.
4. HF temporal heads cannot be reached from a pooled-only capability.
5. Training controls are lateral and explicitly bypassed at inference.
6. Shared/general/specific/router/MoE paths are visibly candidate/HOLD, not selected.
7. Every figure candidate has a SOTA work-card ID that resolves in the compatibility matrix.
8. Negative/inconclusive diagnostics are not rendered as solution modules.
9. No Phase 1A success, generic unseen-data, pathology-shortcut, or zero-shot adaptation claim appears.
10. SVG validates, renders without clipping/overlap, and remains editable in Inkscape-compatible SVG.
