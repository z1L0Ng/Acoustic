# Module interface and compatibility matrix v2

**Purpose:** close every legal connection in the master figure and surface incompatible paths as explicit gates. A check mark means shape/interface compatibility only; it does not mean empirical validation or selection.

## 1. Module contracts

| ID | Module | Required input | Output | State |
|---|---|---|---|---|
| `D1` | ICBHI adapter/unit builder | official WAV + cycle row | waveform `[T_i]`, flat4 target, patient/split lineage | Fixed Contract |
| `D2` | SPR event/record unit builder | BioCAS2022 WAV + event/record rows | event `[T_i]` or recording `[T_i]`, native targets, split lineage | Fixed Contract |
| `D3` | HF sequence unit builder | 15-s WAV + interval rows | waveform `[60000]`, interval table, annotation states | Fixed Contract |
| `D4` | KAUH record unit builder | WAV + workbook row | recording `[T_i]`, raw-9 target, P-number/filter lineage | Fixed Contract |
| `P0` | R0 legacy AST frontend | ICBHI cycle or SPR event | fbank `[B,1,798,128]` | Executable Reference; two datasets only |
| `P1` | v2 waveform collator | native unit waveforms | `[B,T]`, `M_wav [B,T]`, source-time lineage | Proposed interface |
| `P2` | HF paper frontend | 15-s 4-kHz waveform | `[B,938,193]` | Paper-faithful Reference |
| `E1` | current AST `W09` | `[B,1,798,128]` | pooled `[B,768]` | Executable Reference; pooled only |
| `E2` | BEATs `W10` | 16-kHz `[B,T]` + optional waveform mask | tokens `[B,L,768]` + downsampled mask; pooled by adapter | Candidate; `time_map` missing |
| `E3` | OPERA-CT `W12` | OPERA-preprocessed window | pooled `[B,768]` in official example | Candidate; local adapter absent |
| `E4` | HeAR `W13` | `[B,32000]` at 16 kHz | pooled `[B,512]` | Candidate; 2-s pooled only |
| `E5` | PANNs Cnn14 `W11` | waveform/front-end config | pooled `[B,2048]` | Candidate; pooled only |
| `E6` | RespireNet `W05` | ICBHI image `[B,3,H,W]` | `[B,128]` then logits | Paper-faithful ICBHI reference |
| `E7` | MVST `W07` | ICBHI spec `[B,1,256,1024]` | five/fused `[B,768]` | Paper-faithful ICBHI reference |
| `A1` | task-width projector | pooled `[B,D_e]` | pooled `[B,D*]` | Proposed interface |
| `A2` | SG-SCL `W16` | pooled `[B,768]` + device ID | projected `[B,D_p]` + training loss | Candidate/HOLD by metadata |
| `A3` | PAFA `W18` | BEATs tokens/pooled + patient ID | projected `[B,D_p]` + PCSL/GPAL | Candidate/HOLD by patient ID |
| `A4` | Metadata-SCL `W15` | pooled + age/sex groups | projected representation + SupCon | Candidate/HOLD by parity |
| `F1` | MVST fusion `W07` | 5× pooled `[B,768]` | pooled `[B,768]` | Paper-faithful, fixed five-view only |
| `F2` | BTS `W24` | CLAP audio 512 + optional text 512 | pooled 512 or 1024 | Multimodal HOLD |
| `F3` | Resp-Agent `W30` | BEATs 496×768 + text | Longformer sequence/logits | Multimodal preprint HOLD |
| `H1` | ICBHI native head | pooled `[B,D*]` | `[B,4]` | Fixed Contract |
| `H2` | SPR event binary/seven | pooled event `[B,D*]` | `[B,2]`, `[B,7]` | Fixed Contract |
| `H3` | SPR recording five | pooled recording `[B,D*]` | `[B,5]` | Fixed Contract; not active in R0 |
| `H4` | HF generic temporal head | tokens + mask + time map | `[B,L,4]` + valid mask | Fixed safe target interface; implementation HOLD |
| `H5` | HF paper RNN head `W02` | `[B,938,193]` | four independent `[B,938,1]` | Paper-faithful Reference |
| `H6` | KAUH raw sound head | pooled recording `[B,D*]` | `[B,9]` | Fixed safe native head; implementation HOLD |
| `H7` | DeepBreath head `W21` | 5-s 32-mel segments | disease clip/frame outputs | External disease reference only |
| `C1` | eligibility masks | task/source/annotation capability | `M_head`, `M_class`, `M_time` | Fixed Control Contract |
| `C2` | sampler/source weights | dataset membership + train rows | batch composition + `w_d` receipt | Training only |
| `C3` | masked loss | logits, targets, eligibility and valid masks | scalar loss + denominators | Training only |
| `C4` | evaluation/claim ledger | predictions, targets, masks, support, receipts | native metrics + claim level | Fixed Evaluation Contract |

## 2. Dataset lane → preprocessing compatibility

| Lane | `P0` legacy AST | `P1` v2 collator | `P2` HF paper frontend | Gate/comment |
|---|---:|---:|---:|---|
| ICBHI cycle `D1` | ✓ active | ✓ proposed | — | preserve cycle boundary and patient lineage |
| SPR event `D2` | ✓ active | ✓ proposed | — | inter/intra separate |
| SPR recording `D2` | — | ✓ proposed | — | requires recording aggregation/head `[B,5]` |
| HF 15-s sequence `D3` | ✗ | ✓ required | ✓ paper reference | legacy repeat/truncate/no-mask is not temporal-safe |
| KAUH recording `D4` | ✗ | ✓ proposed | — | no implicit crop to event/cycle; patient-group replicas |

## 3. Preprocessing → encoder compatibility

| Frontend | AST `E1` | BEATs `E2` | OPERA-CT `E3` | HeAR `E4` | PANNs `E5` | RespireNet `E6` | MVST `E7` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P0 [B,1,798,128]` | ✓ exact | ✗ expects waveform | ✗ different frontend | ✗ waveform 2 s | ✗ waveform/frontend | ✗ 3-channel image path | ✗ `[256,1024]` view grid |
| `P1 [B,T]+M_wav` | adapter needed | ✓ code accepts | adapter needed | 2-s window adapter | ✓ code frontend, mask absent | image adapter; ICBHI only | fixed spec adapter; ICBHI only |
| `P2 [B,938,193]` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

No “universal adapter” is asserted. A common software record schema can dispatch to encoder-specific deterministic frontends, but preprocessing parameters remain backbone-specific.

## 4. Encoder → native head compatibility

Legend: `✓` shape-legal after a declared projection; `GATE` needs a missing interface; `—` not a supported use.

| Encoder/output | ICBHI `H1` | SPR event `H2` | SPR record `H3` | HF generic `H4` | KAUH raw9 `H6` | Primary gate |
|---|---:|---:|---:|---:|---:|---|
| AST pooled 768 `E1` | ✓ active | ✓ active | ✓ | **GATE** | ✓ | no tokens/mask/time map; no HF |
| BEATs tokens/mask 768 `E2` | ✓ after pooling | ✓ after pooling | ✓ after aggregation | **GATE** | ✓ after aggregation | construct and verify `time_map`; do not reuse frozen means for temporal head |
| OPERA-CT pooled 768 `E3` | ✓ candidate | ✓ candidate | ✓ candidate | **GATE** | ✓ candidate | public v2 adapter/receipt absent; no exported temporal contract |
| HeAR pooled 512 `E4` | ✓ after projection | ✓ after projection/window aggregation | ✓ after aggregation | **GATE** | ✓ after aggregation | 2-s windows and aggregation must be declared; pooled only |
| PANNs pooled 2048 `E5` | ✓ after projection | ✓ after projection | ✓ after aggregation | **GATE** | ✓ after aggregation | Cnn14 public output is pooled; no token/time interface |
| RespireNet 128 `E6` | ✓ paper reference | — | — | — | — | ICBHI-specific preprocessing/task |
| MVST fused 768 `E7` | ✓ paper reference | candidate only after new five-view adapter | candidate only after aggregation | **GATE** | candidate only after aggregation | fixed five-view pooled fusion, no temporal output |
| HF paper frontend `P2` + RNN `H5` | — | — | — | paper-faithful alternative | — | separate fixed-grid baseline, not generic backbone slot |

## 5. Optional adapter/fusion gates

| Candidate | ICBHI | SPRSound | HF | KAUH | Gate |
|---|---:|---:|---:|---:|---|
| `A2` SG-SCL/stethoscope | eligible if device labels match source protocol | HOLD | HOLD | HOLD | side-channel semantics and coverage |
| `A3` PAFA/patient | eligible with patient IDs | eligible with patient IDs, new protocol | **blocked** | potentially eligible but replicas are not independent samples | patient identity availability and patient-grouped batching; no extrapolation from frozen diagnostics |
| `A4` Metadata-SCL | partial | partial | HOLD/partial | partial | age/sex availability and comparable bins |
| `F1` MVST fusion | exact paper reference | new candidate only | incompatible temporal | new candidate only | five view embeddings, same width, native protocol |
| `F2` BTS | paper metadata reference | HOLD | HOLD | HOLD | text/device/location parity; shortcut and governance audit |
| `F3` Resp-Agent | HOLD | HOLD | HOLD | HOLD | heterogeneous labels, derived clinical text, preprint/overlap; not acoustic-only |

## 6. Loss/supervision eligibility

| Work/loss | Required evidence | Legal task scope | Prohibited inference |
|---|---|---|---|
| `W14` Patch-Mix CE/CL | patch-compatible encoder + observed native class | paper ICBHI; future task-specific ablation | mixed patches do not legalize HF gaps or KAUH mapping |
| `W15` Metadata-SCL | observed comparable metadata | rows with the side-channel | metadata absence cannot be imputed as a group |
| `W16` SG-SCL | device/stethoscope identity | source-compatible device experiment | device supervision is not domain-free generalization |
| `W18` PAFA | patient IDs and ≥2 patients per eligible batch | patient-identified datasets only | HF date proxy is not patient ID; B/D/E replicas are not patients |
| `W26` ADD-RSC | exact repo/paper variant + event-retention gate | ICBHI paper-faithful ablation | denoising gain is not multi-dataset or event-preservation evidence |
| shared/native CE/BCE | `M_head × M_class × M_time` | only observed/approved outputs | unknown/not-annotated never becomes a negative |

## 7. Training and inference separation

| Module | Training | Inference | Notes |
|---|---:|---:|---|
| adapters/unit builders | ✓ | ✓ | same release/schema and lineage |
| selected frontend/backbone | ✓ | ✓ | same deterministic contract, augmentation off at inference |
| native heads | ✓ | ✓ | only declared legal heads open |
| eligibility | ✓ uses target support | ✓ uses declared task capability only | inference cannot read labels |
| sampler/source weights | ✓ | ✗ | receipt required |
| contrastive/alignment losses | ✓ if eligible | ✗ | no forward audio arrow |
| masked loss aggregator | ✓ | ✗ | effective denominators recorded |
| evaluation/claim ledger | validation/test | prediction audit | separates paper/local/engineering evidence |

## 8. Operational recipes R0–R3 versus four lanes

| Recipe | Datasets/tasks | Architecture/evidence | Relation to v2 lanes |
|---|---|---|---|
| `R0` | ICBHI cycles + SPR events | current AST/native heads; smoke/profile engineering receipts | executable inset touches only lanes 1–2 |
| `R1` | ICBHI | RespireNet paper-faithful draft | candidate/reference for lane 1 |
| `R2` | ICBHI | MVST identity/smoke; paper claim under its protocol | candidate/reference for lane 1 |
| `R3` | ICBHI | PAFA identity/smoke/profile; test-selected checkpoint caveat | adapter/loss reference; frozen-feature use spans four lanes but is not end-to-end |

The four recipe rows are not renamed as ICBHI/SPRSound/HF/KAUH baselines.

## 9. Explicit HOLD register

1. Generic HF temporal path until `tokens + token_mask + time_map` closes.
2. KAUH shared ontology mapping and diagnosis normalization.
3. Any one-head/four-dataset label collapse.
4. SG-SCL/Metadata-SCL/PAFA outside rows with real comparable side channels.
5. BTS/Resp-Agent in the acoustic-only main path.
6. Router/MoE or dataset-ID routing as a selected architecture.
7. Any claim that four-dataset frozen-feature findings establish full-encoder effects.
8. Any promotion of corrected negative/inconclusive evidence into a solution module.
