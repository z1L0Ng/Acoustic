# Module interface and compatibility matrix v2

**Status:** canonical shared-window compatibility specification, 2026-08-12
**Purpose:** close every legal connection in the master figure and surface incompatible paths as explicit gates. A check mark means shape/interface compatibility only; it does not mean empirical validation or selection. All P1–P13 performance results remain `Not run`.

> **Alignment rule.** The current four-dataset path calls one candidate encoder independently on source-time windows and stacks its pooled outputs. Therefore “encoder does not expose tokens” does **not** mean “encoder cannot serve HF.” It means only that the encoder cannot enter the deferred P6 token-level refinement without an additional audited token mask/time map. The current 2.0-s window / 1.0-s stride is an adjustable Proposed Benchmark Policy.

## 1. Semantic module contracts

| Component ID | Module | Required input | Output | State |
|---|---|---|---|---|
| `UNIT-ICBHI-CYCLE` | ICBHI unit builder | official WAV + cycle row | waveform `[T_i]`, flat4 target, patient/split lineage | Fixed Contract |
| `UNIT-SPR-EVENT-RECORD` | SPR event/record builders | WAV + event/record rows | event or recording waveform, native targets, split lineage | Fixed Contract |
| `UNIT-HF-15S` | HF sequence builder | 15-s WAV + interval rows | waveform, interval table, annotation states | Fixed Contract |
| `UNIT-KAUH-RECORD` | KAUH record builder | WAV + workbook row | recording, raw-9 target, P-number/filter lineage | Fixed Contract |
| `PREP-SHARED-WINDOW` | shared-window collator | one native waveform + lineage | `X_win [B,N_w,32000]`, `M_win [B,N_w]`, `Q_win [B,N_w,2]` | P1–P5 current contract |
| `PREP-HF-PAPER` | HF paper frontend | 15-s 4-kHz waveform | `[B,938,193]` | External paper-faithful reference |
| `ENC-AST` | AST base384 `W09` | one 2-s source window; internal audited AST frontend | pooled `[B,768]` per window | P1 local code/asset/CPU ready; performance not run |
| `ENC-BEATS` | BEATs iter3+ `W10` | one 16-kHz source window + mask | pooled `[B,768]`; optional tokens + mask | P2 local code/asset/CPU ready; performance not run |
| `ENC-PANNS` | PANNs Cnn14 `W11` | one source window | pooled `[B,2048]` | P3 asset/dependency HOLD |
| `ENC-HEAR` | HeAR 1.0.0 `W13` | one 2-s 16-kHz window | pooled `[B,512]` | P4 gated asset/license HOLD |
| `ENC-OPERA` | OPERA-CT `W12` | declared OPERA input adapter | pooled `[B,768]` | P5 provenance/overlap/input-adapter HOLD |
| `PROJ-SHARED-768-256` | shared projector | `[B,N_w,768]` | `[B,N_w,256]` | P1/P2 matched trainable block |
| `PROJ-PANNS-2048-768` | PANNs adapter | `[B,N_w,2048]` | `[B,N_w,768]` | P3 package component |
| `PROJ-HEAR-512-768` | HeAR adapter | `[B,N_w,512]` | `[B,N_w,768]` | P4 package component |
| `HEAD-ICBHI-FLAT4` | ICBHI native head | masked unit representation `[B,256]` | `[B,4]` | Fixed Contract |
| `HEAD-SPR-EVENT` | SPR event heads | event representation `[B,256]` | `[B,2]`, `[B,7]` | Fixed Contract |
| `HEAD-SPR-RECORD5` | SPR recording head | recording representation `[B,256]` | `[B,5]` | Fixed Contract |
| `HEAD-HF-WINDOW4` | HF window-level temporal head | `[B,N_w,256] + M_win + Q_win` | `[B,N_w,4]` + valid mask | P1–P5 current contract |
| `HEAD-HF-TOKEN4` | HF token-level temporal head | tokens + token mask + token time map | `[B,L,4]` + valid mask | P6 deferred/HOLD |
| `HEAD-HF-PAPER` | HF paper RNN head `W02` | `[B,938,193]` | four independent `[B,938,1]` | External paper-faithful reference |
| `HEAD-KAUH-RAW9` | KAUH raw sound head | masked recording representation `[B,256]` | `[B,9]` | Fixed Contract |
| `CTRL-ELIGIBILITY` | eligibility/annotation masks | task/source/annotation capability | head/class/time masks | Fixed Control Contract |
| `CTRL-SAMPLER` | sampler/source weights | train rows + source ID | batch composition receipt | Training only |
| `CTRL-MASKED-LOSS` | masked native/objective loss | logits, targets, eligibility, valid masks | scalar loss + denominators | Training only |
| `EVAL-NATIVE` | native terminal evaluation | predictions, targets, support, split/checkpoint receipts | per-native-task metrics + claim level | Scorer/verifier HOLD before execution |

## 2. Dataset lane → shared-window compatibility

| Lane | `PREP-SHARED-WINDOW` | Output semantics | Non-negotiable gate |
|---|---:|---|---|
| ICBHI cycle | ✓ current | one cycle → one or more windows | cycle boundary and patient/split lineage retained |
| SPR event | ✓ current | one event → one or more windows | inter/intra never pooled |
| SPR recording | ✓ current | recording windows + masked aggregation | recording task remains separate from event task |
| HF 15-s sequence | ✓ current | ordered windows + mask + source-time map | gap/missing/unknown/not_annotated never becomes raw/shared negative |
| KAUH recording | ✓ current | recording windows + masked mean | P-number grouping; B/D/E replicas stay together |

Current policy: mono 16 kHz, 2.0-s window, 1.0-s stride, short-unit zero-pad, and one unique end-aligned tail window. This geometry is fixed only for the first matched P1/P2 comparison and may be changed in a later dedicated comparison.

## 3. Shared-window → encoder compatibility

| Encoder package | Window input adapter | Per-window output | Four-lane shape status | Scientific/execution gate |
|---|---|---|---|---|
| `ENC-AST` / P1 | internal AST fbank; each 2-s source window tail-padded to audited 798-frame grid | `D=768` | ✓ | package-level reference; CUDA/full/terminal result not run |
| `ENC-BEATS` / P2 | official 16-kHz waveform frontend | `D=768` | ✓ | matched P1 replacement; CUDA/full/terminal result not run |
| `ENC-PANNS` / P3 | official Cnn14 waveform frontend | `D=2048` → trainable 2048→768 | shape-legal | official asset/dependency HOLD; package comparison only |
| `ENC-HEAR` / P4 | native 2-s serving input | `D=512` → trainable 512→768 | shape-legal | gated asset/license HOLD; package comparison only |
| `ENC-OPERA` / P5 | 2-s source window → declared 8-s zero-pad frontend | `D=768` | shape-legal | provenance/pretraining-overlap/input-adapter HOLD |

No “universal frontend” is asserted. Source-time window geometry is shared; each encoder package may still contain a deterministic, declared internal frontend. P1↔P2 is the cleanest matched replacement because both produce `D=768` and use the identity width adapter. P3/P4 include additional trainable width adapters and therefore are package-level comparisons.

## 4. Encoder/window output → native head compatibility

Legend: `✓` shape-legal after the declared shared projector/aggregation; `GATE` needs a missing interface or scientific approval; `—` unsupported.

| Encoder package | ICBHI flat4 | SPR event/record | HF window4 | KAUH raw9 | P6 token4 |
|---|---:|---:|---:|---:|---:|
| AST P1 | ✓ | ✓ | **✓ through window sequence** | ✓ | — |
| BEATs P2 | ✓ | ✓ | **✓ through window sequence** | ✓ | **GATE:** audited token-level time map |
| PANNs P3 | ✓ | ✓ | **✓ through window sequence** | ✓ | — |
| HeAR P4 | ✓ | ✓ | **✓ through window sequence** | ✓ | — |
| OPERA-CT P5 | ✓ | ✓ | **✓ through window sequence** | ✓ | — |
| HF paper frontend + RNN | — | — | external fixed-grid alternative | — | not the generic P6 path |

The bold HF cells are the correction to the stale token-only interpretation. They rely on `PREP-SHARED-WINDOW` to provide temporal order, masks, and source-time support outside the encoder. They support window-level temporal classification only; they do not prove token/frame localization or comparability with the 938-frame HF paper grid.

## 5. Optional adapter/fusion gates

| Candidate | ICBHI | SPRSound | HF | KAUH | Gate |
|---|---:|---:|---:|---:|---|
| SG-SCL/stethoscope | eligible if device labels match source protocol | HOLD | HOLD | HOLD | side-channel semantics and coverage |
| PAFA/patient | eligible with true patient IDs | eligible with true patient IDs under new protocol | blocked: patient IDs absent | potentially eligible; replicas are not patients | patient identity and patient-grouped batching |
| Metadata-SCL | partial | partial | HOLD/partial | partial | metadata availability and comparable bins |
| MVST five-view fusion | paper ICBHI reference | new candidate only | incompatible with current native HF route | new candidate only | same-width five-view embeddings; P12 eligible non-HF only |
| BTS / Resp-Agent | paper/reference only | HOLD | HOLD | HOLD | multimodal inputs, label provenance, shortcut/governance audit |

## 6. Loss/supervision eligibility

| Work/loss | Required evidence | Legal task scope | Prohibited inference |
|---|---|---|---|
| native CE/BCE | observed native label + valid mask | P1/P2 native tasks | missing/unknown/not_annotated is not negative |
| eligibility-masked compatible objective / P8 | explicitly compatible target + eligibility denominator receipt | only compatible rows/classes | not universal label harmonization |
| Patch-Mix CE/CL | patch-compatible encoder + observed native class | paper ICBHI/future bounded ablation | does not legalize HF gaps or KAUH mapping |
| Metadata-SCL / SG-SCL | observed comparable metadata/device IDs | eligible rows only | absent metadata cannot be imputed |
| PAFA | true patient IDs and eligible batching | patient-identified lanes only | HF date proxy is not patient ID; B/D/E replicas are not patients |

## 7. Training, inference, and evaluation separation

| Module | Training | Inference | Notes |
|---|---:|---:|---|
| unit builders + shared windows | ✓ | ✓ | same release/schema/window identity and lineage |
| selected frozen encoder + shared projector | ✓ | ✓ | P1/P2 encoder frozen; projector trainable; deterministic cache may be used only with complete identity key |
| native heads | ✓ | ✓ | only declared native heads open |
| eligibility | ✓ uses declared target support | ✓ uses task capability only | inference cannot read target labels |
| sampler/source weights | ✓ | ✗ | batch composition receipt required |
| auxiliary losses | ✓ if separately approved and eligible | ✗ | no forward audio arrow |
| terminal evaluator | terminal only | prediction audit | rejects pooled/global score and binds checkpoint/prediction/split identity |

## 8. Pipeline experiment mapping

| Pipeline | Compared block | Current status | Allowed interpretation |
|---|---|---|---|
| P1 | AST four-dataset shared-window package | local engineering ready; performance `Not run` | first-batch reference |
| P2 | BEATs matched package replacement | local engineering ready; performance `Not run` | AST↔BEATs package screening |
| P3 | PANNs + 2048→768 adapter | asset HOLD | package-level encoder candidate |
| P4 | HeAR + 512→768 adapter | gated asset/license HOLD | package-level encoder candidate |
| P5 | OPERA-CT + input adapter | scientific/provenance HOLD | overlap-aware reference only |
| P6 | BEATs token-level HF temporal refinement | deferred/interface HOLD | token route vs P2 window route; not first-batch blocker |
| P7 | pooling comparator | deferred | only after encoder shortlist |
| P8 | eligibility-masked objective | deferred | harmonization evidence only with matched native reference |
| P9 | dataset-balanced sampler | deferred | sampler effect only |
| P10/P11 | PAFA projector-only / projector+loss pair | patient-ID eligible lanes only | matched PAFA component attribution |
| P12 | MVST eligible non-HF fusion | deferred | not universal four-dataset fusion |
| P13 | target-supervised LODO adaptation | deferred | not zero-shot; not pooled four-dataset evidence |

Legacy R0–R3 are historical evidence labels only and are not current pipeline IDs.

## 9. Cross-pipeline execution gates

1. Per-native-task terminal scorer exists and rejects pooled/global scores.
2. Independent verifier is a different execution identity from the scorer implementer.
3. P1/P2 asset manifest freezes source URL/revision, checkpoint SHA256/size/license, and server path; P2 does not depend on an orphan result directory.
4. Frozen embedding cache, if used, binds dataset/split, preprocessing, window policy, encoder/checkpoint, dtype, and code/config identity; any mismatch fails closed.
5. GPU 2/3 real-asset zero-update CUDA preflight passes before smoke/full.
6. Single seed, update budget, selection rule, output directory, and checkpoint identity are frozen before launch.
7. Sequential versus one-pipeline-per-GPU execution requires explicit user approval; this document does not authorize a run.

## 10. Explicit HOLD register

1. P6 token-level HF path until `tokens + token_mask + token-level time_map` closes.
2. Terminal scorer/verifier, asset manifest, cache identity, and CUDA preflight before P1/P2 full execution.
3. KAUH shared ontology mapping and diagnosis normalization.
4. Any one-head/four-dataset label collapse or pooled global ranking.
5. SG-SCL/Metadata-SCL/PAFA outside rows with real comparable side channels.
6. OPERA clean-generalization claims because pretraining overlap/provenance remain gated.
7. Router/MoE or dataset-ID routing as a selected architecture.
8. Any claim that smoke/profile, legacy frozen-feature diagnostics, or target-supervised adaptation establishes full-encoder harmonization/generalization.
9. Any claim that the current 2.0-s/1.0-s geometry is optimal or supports fine event localization.
