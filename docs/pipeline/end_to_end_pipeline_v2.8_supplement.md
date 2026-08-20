# Acoustic End-to-End Pipeline v2.8 — Companion Specification

## 0. Purpose and scope

This document is the audit-oriented companion to:

- `figures/pipeline/acoustic_end_to_end_pipeline_v2.8.svg`
- `figures/pipeline/acoustic_end_to_end_pipeline_v2.8.png`

The SVG is intentionally limited to the scientific story:

```text
four native datasets
→ native prediction units
→ provisional shared windows
→ one selected encoder
→ shared representation + eligibility
→ unified and native heads
→ unified/native evaluation
```

This Markdown contains information deliberately removed from the main figure: complete dataset facts, support counts, tensor contracts, architecture limits, training/comparison controls, the S/JN/JU/LODO generalization comparison, evidence boundaries, operational status and open decisions.

The v2.8 SVG keeps the ontology crosswalk as a compact supervision-side contract, while the complete mapping and eligibility semantics remain here. The generalization comparison is retained in §9.3 as a separate scientific evaluation contract and is intentionally omitted from the main SVG.

The figure is a system and method specification. It does not claim that the proposed unified hierarchy has been implemented or validated.

### 0.1 Shared-scope decision status

The P0 shared-scope decision remains open. The SVG shows the **maximum candidate supervision surface**, not an approved four-dataset closed-set mapping.

Current boundary:

- **Approved shared core:** ICBHI plus the SPRSound compatible event subset.
- **HF candidate contribution:** explicit positive temporal supervision only; HF provides no shared Normal target and its gaps are not shared negatives.
- **KAUH candidate contribution:** source-defined eligible subset only; unresolved sound strings remain outside shared supervision.
- **Fallback scope:** ICBHI + SPRSound remain the shared core, while HF and KAUH remain native/auxiliary or external-generalization lanes.

No unified implementation or training is authorized before ontology review and acceptance of the support ledger.

The support ledger is a separate ontology/sample-ledger artifact. This revision does not add a `PENDING support` row to the SVG and does not create the ledger.

## 1. Evidence vocabulary

| State | Meaning | Prohibited interpretation |
|---|---|---|
| Fixed contract | Source-grounded dataset, unit, label, split or interface fact | Not a model result |
| Executable/reference | A current reference path or interface exists | Not superiority or harmonization gain |
| Verified result | A completed result within its declared native task and protocol | Not automatically cross-dataset generalization |
| Interpretation | A bounded explanation of observed evidence | Not a causal result |
| Candidate/proposed | A method or evaluation design awaiting validation | Not completed work |
| HOLD/paused | A scientific, ontology, asset, license or interface gate remains | Not a negative model result |
| Future comparison | A preregistered comparison after earlier gates close | Not an active result |

Required claim boundaries:

- P1 AST and P2 BEATs are verified joint-native references; they do not validate the unified hierarchy.
- Small differences between encoder references do not establish an encoder ranking.
- Shared-encoder executability does not prove harmonization gain.
- Target-head adaptation uses target labels and is not zero-shot.
- Dataset-ID accessibility does not prove that a pathology model uses a shortcut.
- HF annotation gaps are `not_annotated`, not raw Normal or Negative.
- KAUH Bell/Diaphragm/Extended recordings are same-patient filter replicas.
- No pooled global or cross-task score is allowed.

## 2. Shape notation

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Waveform samples |
| `Nw` | Windows per native prediction unit |
| `τ` | Time frames in a spectrogram/fbank representation |
| `L0` | Encoder token length before or within a transformer |
| `D_enc` | Encoder-native representation width |
| `D_shared` | Shared representation width after alignment/projector |
| `K` | Task class/channel count |
| `Ni` | Interval rows in one HF recording |

Parser outputs are single native units and may have variable `T`. Batch and padding dimensions are introduced by shared windowing.

## 3. Layer 1 — Native dataset inputs

### 3.1 ICBHI 2017

- Official 2017 challenge package.
- 920 recordings, 126 patients, 6,898 respiratory cycles, 5.492508 h.
- Mono WAV; 4/10/44.1 kHz; 16/24 bit in the audited package.
- Cycle row: `start_s`, `end_s`, `crackle_flag`, `wheeze_flag`.
- Native prediction unit: respiratory cycle.

| Flags | Native state | Cycles |
|---|---|---:|
| `00` | Normal | 3,642 |
| `10` | Crackle | 1,864 |
| `01` | Wheeze | 886 |
| `11` | Both | 506 |

Official recording split:

- Train: 539 recordings / 4,142 cycles.
- Test: 381 recordings / 2,756 cycles.
- Patient IDs 156 and 218 occur in both partitions; the official split is not patient-independent.
- Patient diagnosis is a separate patient-level target and does not enter the cycle event ontology.

### 3.2 SPRSound BioCAS2022

- BioCAS2022 classification release.
- 2,683 recordings, 292 patients, 9,089 events, 8.162338 h.
- Mono WAV; 8 kHz; 16 bit.
- Two distinct native units: respiratory event and whole recording.

Event support:

| Raw event label | Events |
|---|---:|
| Normal | 6,887 |
| Fine Crackle | 1,167 |
| Wheeze | 865 |
| Coarse Crackle | 66 |
| Rhonchi | 53 |
| Wheeze+Crackle | 34 |
| Stridor | 17 |

Recording labels:

```text
Normal · CAS · DAS · CAS & DAS · Poor Quality
```

`Poor Quality` is a recording-quality class, not an acoustic pathology/event class.

Source partitions:

- Train: 1,949 recordings / 251 patients.
- Inter test: 355 recordings / 41 patients; zero training-patient overlap.
- Intra test: 379 recordings / 162 patients; all 162 also occur in training.
- Inter and intra are evaluated separately and never pooled.
- Event and recording outputs remain separate tasks.

### 3.3 HF_Lung_V1

- 9,765 recordings, exactly 15 s each, total 40.6875 h.
- Mono WAV; 4 kHz; 16 bit.
- Native input unit: 15-second recording.
- Native prediction object: overlapping temporal interval/frame sequence.

Positive interval rows:

| Raw token | Source meaning | Rows |
|---|---|---:|
| `I` | Inhalation | 34,095 |
| `E` | Exhalation | 18,349 |
| `D` | Discontinuous adventitious sound; described by the source paper as crackles | 15,606 |
| `Wheeze` | Wheeze | 8,457 |
| `Rhonchi` | Rhonchi | 4,740 |
| `Stridor` | Stridor | 686 |

Annotation coverage:

- Total recording time: 146,475.000 s.
- Any annotation union: 52,051.639 s / 35.536%.
- Unannotated gap: 94,423.361 s / 64.464%.
- All 9,765 recordings contain a gap.
- 58 label files are empty.
- Explicit raw negative intervals: 0.

Native paper tasks are four one-vs-rest temporal detections:

```text
Inhalation
Exhalation
CAS = Wheeze / Rhonchi / Stridor
DAS = Discontinuous sound
```

Intervals can overlap. Respiratory-phase and sound-event labels are not mutually exclusive. Gap or absence of a token is not raw Normal/Negative. Patient identity is not provided; deidentified date is at most a grouping proxy.

### 3.4 KAUH / Fraiwan v3

- Mendeley Data v3.
- 112 patients, 336 filtered recordings, 1.623918 h.
- Mono WAV; 4 kHz; 16 bit.
- Exactly three recordings per patient: Bell, Diaphragm and Extended filter modes.
- Native prediction unit: whole recording.
- Native target: raw sound string.
- P-number is the group/patient identity; three filter siblings remain in one partition.
- No official split is provided.
- Diagnosis is a separate patient-level target.

Raw support is reported as patients / filtered recordings:

| Raw string | Patients / recordings | Source-grounded reading |
|---|---:|---|
| `N` | 35 / 105 | `N = Normal` |
| `E W` | 39 / 117 | `E = Expiratory`, `W = Wheezes` |
| `I E W` | 2 / 6 | `I = Inspiratory`, `E = Expiratory`, `W = Wheezes` |
| `C` | 7 / 21 | `C = Crackles` |
| `I C` | 1 / 3 | Inspiratory + Crackles |
| `I C E W` | 2 / 6 | Inspiratory + Crackles + Expiratory + Wheezes |
| `Crep` | 23 / 69 | Crepitations; source lists it separately from `C` |
| `Bronchial` | 1 / 3 | Bronchial sound |
| `I C B` | 2 / 6 | `I` and `C` are defined; `B` is not defined inside the source sound ontology |

Important distinction:

- Filename prefix `B` explicitly means Bell filter.
- This does not establish the meaning of `B` inside the sound string `I C B`.
- `I C B` remains unresolved until an approved source/clinical interpretation exists.
- `Crep` is not silently normalized to Crackle.

## 4. Layer 2 — Native parser/unit contracts

| Dataset/unit | Input waveform | Native target/annotation |
|---|---|---|
| ICBHI cycle | `x_cycle [T_cycle]` | Raw crackle/wheeze flags `[2]` |
| SPRSound event | `x_event [T_event]` | Binary class index `[1]`; seven-class index `[1]` |
| SPRSound recording | `x_record [T_record]` | Five-class index `[1]` |
| HF 15-s recording | `x_record [60,000]` | Interval table `A [Ni,3] = [label,start,end]` |
| KAUH recording | `x_record [T_record]` | Raw sound class `[1]`; P-number `[1]`; filter mode `[1]` |

The parser does not resample, window, harmonize labels or impute missing targets.

## 5. Layer 3 — Shared waveform and provisional windows

Current reference policy:

```text
all native waveforms → mono → 16 kHz
2.0 s window = 32,000 samples
1.0 s stride
short native unit → zero-pad to one 2.0 s window
long native unit → regular sliding windows
when the regular grid does not cover the end, add one unique end-aligned tail window
never emit a duplicate tail window
```

Output:

```text
X_win [B,Nw,32,000]
M_win [B,Nw]
time_map [B,Nw,2]
```

Preserved state:

- Dataset and native-unit identity.
- Source-time boundaries.
- Annotation state.
- Patient/group identity.
- Ordered HF windows.

This is a provisional reference, not an optimality conclusion. Final window length, stride, tail handling and short-unit policy depend on window-level acoustic/event-duration analysis.

## 6. Layer 4 — Encoder alternatives

Exactly one encoder package is selected. The candidates are alternatives, not an ensemble.

### 6.1 AST base384

```text
X_win [B,Nw,32,000]
→ fbank image [B,Nw,1,798,128]
→ transpose [B,Nw,1,128,798]
→ Conv2d patch16×16 / stride10×10
→ grid 12×79 = 948 acoustic tokens
→ prepend CLS + DIST
→ [B,Nw,950,768]
→ DeiT-Base block ×12
   12 heads; MLP 768→3072→768
→ LayerNorm; (CLS+DIST)/2
→ z_AST [B,Nw,768]
```

Current wrapper returns pooled output and exposes no token mask. The 798-frame frontend is a reference package behavior; it is not the natural frame count of a 2-second waveform.

### 6.2 BEATs iter3+ AS2M

```text
X_win [B,Nw,32,000]
→ 128-bin fbank [B,Nw,τ,128]
→ patch Conv2d p×p / stride p
→ [B,Nw,L0,E]
→ LN + optional projection
→ [B,Nw,L0,768]
→ grouped position Conv1d kernel128 / groups16
→ Transformer block ×12
   12 heads; FFN 768→3072→768
→ masked token mean
→ z_BEATs [B,Nw,768]
```

`p`, `E` and therefore `L0` remain checkpoint-configured/symbolic where the accepted evidence does not freeze them. Token output and a padding mask exist in code, but a generic source-time token map is not currently part of the main path.

### 6.3 PANNs Cnn14

```text
waveform
→ 64-bin log-mel [B,Nw,1,τ,64]
→ ConvBlock widths:
   64 → 128 → 256 → 512 → 1024 → 2048
→ [B,Nw,2048,floor(τ/32),2]
→ frequency mean + temporal max/mean
→ FC 2048→2048
→ z_PANNs [B,Nw,2048]
```

### 6.4 HeAR 1.0.0

```text
[B,Nw,32,000]
→ official ViT-L masked-autoencoder service
→ z_HeAR [B,Nw,512]
```

The accepted serving contract does not expose internal patch counts, block counts, heads or intermediate shapes. Those values remain unknown rather than inferred from a generic ViT-L.

### 6.5 OPERA-CT

Official reference path:

```text
8-s waveform
→ model image [B,1,256,256]
→ patch4 / stride4
→ [B,64×64,96]
→ Swin stage ×2; merge → [B,32×32,192]
→ Swin stage ×2; merge → [B,16×16,384]
→ Swin stage ×6; merge → [B,8×8,768]
→ Swin stage ×2 → [B,8×8,768]
→ pooling → [B,768]
```

The official reference uses 8 s while the shared-window reference is 2 s. Input adaptation and pretraining-overlap/provenance interpretation remain separate gates.

### 6.6 Encoder evidence grouping

- AST and BEATs: verified joint-native reference packages.
- PANNs, HeAR and OPERA: paused engineering assets for the current unified-method critical path.
- Encoder cards describe architecture only; result status is carried by the surrounding visual group.

Common output:

```text
z_enc [B,Nw,D_enc]
M_win [B,Nw]
time_map [B,Nw,2]

D_enc ∈ {768,512,2048}
```

Masks and time maps are passed through from shared windowing; they are not generated by every encoder.

## 7. Layer 5 — Shared representation and eligibility

### 7.1 Representation alignment

```text
AST / BEATs / OPERA: 768→768
HeAR: 512→768
PANNs: 2048→768
shared projector: 768→D_shared
current reference: D_shared=256
```

Output:

```text
z_shared [B,Nw,D_shared]
window mask [B,Nw]
time_map [B,Nw,2]
```

Width compatibility is not evidence of domain generalization. `D_shared=256` is a current reference, not a proven optimum.

### 7.2 Unit-aware aggregation

```text
cycle / event / recording tasks:
z_shared [B,Nw,D_shared]
→ masked valid-window aggregation
→ z_unit [B,D_shared]

HF temporal task:
z_shared [B,Nw,D_shared]
→ retain ordered Nw
→ z_time [B,Nw,D_shared]
```

HF receives no early temporal mean pooling.

### 7.3 Raw-label crosswalk and eligibility

The raw target is retained for native evaluation. A separately mapped target is opened only for an eligible unified node.

Eligibility states:

```text
positive
explicit_negative
unknown
not_annotated
not_applicable
unresolved / HOLD
```

`unknown`, `not_annotated`, `not_applicable` and unresolved mappings are mask states. Unknown is not a learned output class.

Dataset mapping boundary:

| Dataset | Safest unified contribution | Excluded/masked boundary |
|---|---|---|
| ICBHI | Exact Normal/Crackle/Wheeze/Both anchor | Official split caveat remains |
| SPRSound | Compatible event subset for Normal, Crackle, Wheeze, Both | Rhonchi/Stridor require an explicit Other/exclusion decision; record task remains native |
| HF | Explicit positive temporal evidence for compatible attributes | No shared Normal/Negative from gaps; gaps remain masked |
| KAUH | Source-defined N/C/W-containing subset only after policy approval | `Crep`, `Bronchial`, `I C B` remain unresolved/HOLD |

Outputs are symbolic until the ontology and sample ledger are frozen:

```text
y_unified
M_unified

unit-level: [B,Ku]
temporal: [B,Nw,Ku]
```

Every unified class must later report eligible support by dataset and split. Cycles, events, intervals, recordings and patients are never added into a false common sample total.

## 8. Layer 6 — Unified and native heads

### 8.1 Proposed unified hierarchical head

Inputs:

```text
z_unit [B,D_shared]
or z_time [B,Nw,D_shared]
+ M_unified
```

Proposed hierarchy:

```text
Level 1 — event state
Linear D_shared→2
→ Normal / Abnormal

Level 2 — abnormal attributes
Linear D_shared→2 + sigmoid
→ Crackle attribute
→ Wheeze attribute

Both = Crackle and Wheeze active
Other = optional output only for explicitly eligible labels
Unknown = mask state; no Unknown logit
```

Output shapes:

```text
unit Level 1: [B,2]
unit attributes: [B,2]
optional unit Other: [B,1]

temporal Level 1: [B,Nw,2]
temporal attributes: [B,Nw,2]
optional temporal Other: [B,Nw,1]
```

This hierarchy is Proposed Method. It is not a verified result.

### 8.2 Native benchmark/retention lane

| Dataset/task | Input | Native logits |
|---|---|---|
| ICBHI cycle | `z_unit [B,D_shared]` | `[B,4]` |
| SPRSound event binary | `z_unit` | `[B,2]` |
| SPRSound event seven-class | `z_unit` | `[B,7]` |
| SPRSound recording five-class | `z_unit` | `[B,5]` |
| HF temporal I/E/CAS/DAS | `z_time [B,Nw,D_shared]` | `[B,Nw,4]` |
| KAUH raw sound | `z_unit` | `[B,9]` |

Native outputs preserve source semantics and literature comparability. They do not, by themselves, establish a unified label space.

## 9. Layer 7 — Evaluation

### 9.1 Unified hierarchy evaluation

- Metric denominators are eligibility-masked.
- Unknown and `not_annotated` rows do not enter ordinary class metrics.
- Level 1: macro-F1, UAR, per-class precision/recall/F1/support for Normal and Abnormal.
- Level 2: per-attribute precision/recall/F1/support for Crackle and Wheeze; Both recall/support; Other only where eligible.
- Hierarchy-consistency reporting links Normal/Abnormal to attributes and Both to Crackle/Wheeze activation.
- Unit-level and temporal outputs are reported separately.

### 9.2 Native evaluation and retention

| Dataset | Required native reporting |
|---|---|
| ICBHI | Sensitivity, specificity, ICBHI Score, macro-F1, UAR, per-class recall/support; official split caveat attached |
| SPRSound | SE/SP/AS/HS; event and recording separate; inter and intra separate; per-class support |
| HF | Segment F1/AUROC/AUPRC; event F1/Jaccard > 0.5/MAPE; annotated-duration and valid-mask support |
| KAUH | Patient-grouped macro-F1/UAR/balanced accuracy/per-class recall and patient support |

KAUH native-evaluation boundary:

- **Primary evaluable:** `N`, `E W`, `Crep`.
- **Diagnostic only:** `C`.
- **Currently not evaluable:** `I E W`, `I C`, `I C E W`, `Bronchial`, `I C B`.
- Current P1/P2 KAUH evidence is fold0 terminal only, `n=69`; it is not 5-fold out-of-fold evaluation.
- Native evaluation of `Crep` does not approve or imply a unified `Crep → Crackle` mapping.

Native retention is compared per dataset and per class. It is never collapsed into one native score.

### 9.3 Cross-dataset comparison

| Comparator | Scientific question |
|---|---|
| `S` — Single-source native | What can each dataset achieve when trained independently under its native task? |
| `JN` — Joint encoder + native heads | Does shared training without unified output help or hurt native tasks? |
| `JU` — Joint encoder + eligibility-aware hierarchy | Does unified supervision improve generalization while retaining native performance? |
| `LODO` — Leave one dataset out | Does the unified system learn transferable event structure rather than dataset-mixture identity? |

LODO begins only after JU produces an interpretable signal. Reporting includes held-out-dataset metrics, per-dataset native retention, worst-dataset behavior and per-class support. Accuracy is not the only criterion.

## 10. Training and comparison controls omitted from the SVG

These controls are required scientifically but are not forward audio modules.

### 10.1 Source composition

- `S`: single-source batches.
- `JN/JU`: declared multi-dataset sampler and declared source weights.
- Report sample counts and effective optimization updates per dataset.
- Avoid implicit domination by dataset size, particularly the larger HF lane.
- Dataset-balanced versus source-proportional sampling remains a declared comparison choice, not a hidden default.

### 10.2 Eligibility-masked objectives

Symbolic objective:

```text
L = L_native
  + λu L_unified
  + λh L_hierarchy
```

- `L_native` uses observed native targets.
- `L_unified` uses eligible mapped nodes only.
- `L_hierarchy` uses samples/nodes for which the relevant hierarchy relation is eligible.
- Unknown, `not_annotated`, `not_applicable` and unresolved/HOLD states are masked.
- `λu` and `λh` are proposed/TBD until the method contract is frozen.
- Auxiliary loss, pooling, fusion and sampler studies change one axis at a time only after the minimal JU reference exists.

### 10.3 Matched comparison contract

Hold fixed where the scientific comparison requires attribution:

```text
dataset release
native prediction unit
split and patient/group rule
window policy
encoder package
training budget
seed
checkpoint-selection rule
metric denominator
```

Validation-only selection is distinct from terminal outer/test evaluation. Results from different units, splits or preprocessing are not directly compared.

## 11. Operational/evidence status omitted from the SVG

As reflected in the 2026-08-13 meeting records and the 2026-08-18 Working Plan:

- AST and BEATs joint-native references completed their declared native-task evaluation and verifier path.
- These references demonstrate four-lane executability and native evaluation, not a unified-method result or a consistent encoder ranking.
- PANNs completed substantial engineering/full-run work but remains outside the current unified-method critical path.
- HeAR remains gated by access/terms.
- OPERA remains an overlap-aware reference with input/provenance questions.
- The previous encoder-screening queue is paused while ontology, eligibility, support and unified evaluation are defined.
- The proposed hierarchy, `JU` comparison and `LODO` are not completed results.

Operational timestamps, process IDs, update counts and server receipts are intentionally excluded from the scientific figure. They belong in project-status records.

## 12. Deferred components and future ablations

The following are not part of the v2.8 main forward path:

- PAFA/patient-aware objectives.
- Metadata or device/stethoscope supervised contrastive objectives.
- MVST multi-view fusion.
- BTS or Resp-Agent multimodal fusion.
- Router/MoE or dataset-ID routing.
- Token-level HF refinement without a closed token mask and token-level time map.
- Fine-tuning, pooling, loss, sampler and fusion sweeps before a minimal JU signal exists.

Dataset-ID routing must not be presented as domain generalization. Side-channel methods require comparable, observed metadata and shortcut-aware controls.

## 13. Open gates

1. **Ontology gate:** Freeze the raw-to-unified crosswalk, eligibility state and whether Rhonchi/Stridor enter Other or remain excluded.
2. **Support gate:** Produce and accept a separate ontology/sample-ledger artifact with per-dataset/per-split eligible counts for every unified node and mask state; do not begin unified implementation/training before acceptance.
3. **KAUH gate:** Resolve or retain HOLD for `Crep`, `Bronchial` and `I C B`.
4. **Window gate:** Use window-level acoustic/event-duration analysis to retain or revise 2 s / 1 s.
5. **Other-node gate:** Open an Other output only when source and clinical semantics justify it.
6. **Baseline-comparability gate:** Match `S`, `JN` and `JU` contracts before interpreting changes.
7. **Unified-signal gate:** Enter LODO or module ablations only after JU produces a prespecified interpretable signal.

If a scientifically defensible four-dataset shared surface cannot be formed, reduce the shared core rather than force every dataset into a closed label table.

## 14. Primary project references

- Dataset/task contract: `docs/datasets/four_dataset_task_contract_review_2026-07-28.md`
- Machine-readable dataset contract: `docs/datasets/four_dataset_task_contract_draft_2026-07-28.json`
- Candidate layer shapes: `docs/pipeline/pipeline_candidate_layer_shapes_v2.md`
- Interface matrix: `docs/pipeline/module_interface_compatibility_matrix_v2.md`
- Prior textual specification: `docs/pipeline/end_to_end_pipeline_textual_spec_v2.md`
- [2026-08-13 Lab Meeting Record](https://app.notion.com/p/3c0309efda2981d5a08be80152d27ebd)
- [2026-08-13 1v1 Meeting Record](https://app.notion.com/p/3c0309efda2981b3bd3bf2c72debe801)
- [2026-08-18 Working Plan](https://app.notion.com/p/3c0309efda298132a9cecab23028ad44)
