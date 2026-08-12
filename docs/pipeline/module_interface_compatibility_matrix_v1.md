# Acoustic Pipeline — Module Interface & Compatibility Matrix v1

- **Status:** Frozen companion to `end_to_end_pipeline_textual_spec_v1.md`
- **Date:** 2026-08-01
- **Traceability source:** accepted Notion [`Pipeline Module SOTA Map`](https://app.notion.com/p/3b0309efda2981899476c6fcaca45d47) (`W01`–`W33`)
- **Rule:** a work card documents a primary-source fact, paper claim, interpretation, candidate use, or HOLD. It does not imply local reproduction.

## 1. Symbols

| Symbol | Meaning |
|---|---|
| `PASS` | Legal under the frozen task/interface contract. |
| `ACTIVE-REF` | Implemented in the current ICBHI+SPRSound executable reference. |
| `CAPABILITY-GATE` | Legal only if the declared output capability is present. |
| `CANDIDATE` | Proposed comparison; no promotion to selected method. |
| `BLOCKED` | Current source semantics prohibit the mapping or loss. |
| `HOLD` | Missing parity/provenance/control or negative/inconclusive prior gate. |
| `N/A` | Not applicable to the native task. |

## 2. Dataset × task/interface compatibility

| Dataset | Native unit | Adapter | Legacy AST 8 s | Target mask-aware path | Pooled task | Temporal task | Shared binary | Shared narrow-four | Native head | Split/group guardrail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ICBHI | cycle | PASS | ACTIVE-REF | PASS | PASS | N/A | PASS primary | PASS secondary | `[B,4]` | Official split has patient overlap; grouped validation required. |
| SPRSound | event | PASS | ACTIVE-REF | PASS | PASS | event-classification only | PASS primary | PASS for approved subset; Rhonchi/Stridor excluded | `[B,2]`, `[B,7]` | `inter` primary; `intra` separate diagnostic. |
| HF_Lung | 15-s temporal sequence | PASS | N/A | PASS | diagnostic summaries only | CAPABILITY-GATE | BLOCKED | BLOCKED | `[B,L,2]` + `[B,L,4]` | Gaps remain not-annotated; date is proxy, not patient ID. |
| KAUH | recording | PASS | N/A | PASS | PASS | N/A | BLOCKED except audited partial overlay | BLOCKED except audited partial overlay | `[B,C_kauh]` | P-number grouped; B/D/E replicas together; no official split. |

## 3. Interface contracts

| Interface | Producer | Shape / fields | Consumer | Current implementation status | Fail-closed rule |
|---|---|---|---|---|---|
| `CanonicalRecord` | Box 03 adapters | lineage, audio ref, native annotations, group/split/metadata status codes | Box 04 | Proposed common interface; four source adapters are the scope | Unknown/missing values are explicit, never defaulted. |
| `UnitSample` | Box 04 | `x_i[T_i]`, native target/support, interval list/frame target, lineage | Box 05 | Unit contracts audited | Unit mismatch or lost lineage is an error. |
| `LegacyASTInput` | Figure A Box 05 | `[B,1,798,128]`, fixed 8 s, no pad mask | current AST | ACTIVE-REF | Must not connect to HF temporal head. |
| `WaveformBatch` | Figure B Box 05 | `X[B,T]`, `M_pad[B,T]`, aligned targets | target backbone slot | Proposed Contract | No all-ones fake mask for padded/repeated support. |
| `SpectrogramBatch` | Figure B Box 05 | `F[B,L,F_bin]` or `[B,1,L,F_bin]`, derived mask/time alignment | spectrogram backbone | Proposed Contract | `L` is not frozen before frontend/window choice. |
| `PooledBackboneOutput` | Box 06 | `pooled[B,D]`, receipts, `capability=pooled_only` | pooled native/shared heads | Current AST returns `[B,768]` | No temporal consumer. |
| `TemporalBackboneOutput` | Box 06 candidate | `tokens[B,L,D]`, `token_mask[B,L]`, `pooled[B,D]`, `time_map[B,L,2]` | HF temporal head, optional token pooling/router | Contract only | Tokens without valid support/time mapping are incompatible. |
| `GeneralRepresentation` | Box 08 | `g[B,Dg]` or aligned token form | fusion/shared/native heads | CANDIDATE; no local empirical evidence | Cannot be labeled domain-general from name alone. |
| `SpecificResidual` | Box 09 | `s[B,Ds]` | controlled fusion/native heads | CANDIDATE | Dataset-ID lookup and unapproved metadata side channels are prohibited. |
| `RouterWeights` | Box 10 | `α[B,E]` or future `[B,L,E]` | Box 11 experts | HOLD | Dataset ID is not the main-method router input. |
| `EligibilityMasks` | Box 18 | `M_head[B,H]`, `M_class[B,C]`, `M_time[B,L,K]` | heads, Box 21, Box 22 | Fixed Control Contract | Missing/not-annotated does not become negative. |
| `SourceControl` | Box 19 | batch composition, `w_d`, effective-update receipt | sampler, Box 21 | current 2-dataset source-proportional; four-dataset D2 frozen diagnostic only | Do not infer full-encoder benefit from D2. |
| `MaskedLoss` | Box 21 | scalar plus per-head denominators and gradient contribution | optimizer/update | executable simple CE reference; target interface fixed | All-masked tasks are skipped safely and logged. |
| `ClaimLedger` | Box 22 | protocol, split, label use, support, metric, checkpoint/selection caveat | paper/meeting/project reporting | Fixed Evaluation Contract | No pooled score across incompatible tasks/protocols. |

## 4. Module × dataset compatibility

| Module | ICBHI | SPRSound | HF_Lung | KAUH | Status / compatibility note |
|---|---|---|---|---|---|
| Box 03 adapter | PASS | PASS | PASS | PASS | Four dataset-specific implementations behind one source-preserving interface. |
| Box 04 unit builder | cycle | event | temporal sequence | recording | Units remain distinct. |
| Box 05 mask-aware preprocessing | PASS | PASS | PASS | PASS | Exact frontend/window not selected. |
| Box 06 pooled-only backbone | PASS | PASS | pooled diagnostic only | PASS | Cannot supply temporal logits. |
| Box 06 temporal-capable backbone | optional | optional | REQUIRED | optional | Contract only; no current implementation claim. |
| Box 08 general representation | CANDIDATE | CANDIDATE | CANDIDATE | CANDIDATE | P-Shared interface only; no local empirical evidence. |
| Box 09 specific residual | CANDIDATE | CANDIDATE | metadata parity gate | patient/filter guardrail | Must not collapse to dataset-ID lookup. |
| Boxes 10–11 router/MoE | HOLD | HOLD | HOLD | HOLD | Dense/equal-parameter controls and usage/shortcut audits required. |
| Box 13 shared binary/narrow-four heads | PASS | PASS/partial | BLOCKED | BLOCKED/partial | Corrected shared-head result is 0 material improvements within frozen-feature scope. |
| Boxes 14–17 native heads | `[B,4]` | `[B,2]`,`[B,7]` | `[B,L,2]`,`[B,L,4]` | `[B,C_kauh]` | Native heads are the safe default. |
| Box 18 eligibility | PASS | PASS | REQUIRED | REQUIRED | Controls heads/loss/evaluation; not audio. |
| Box 19 source control | ACTIVE-REF | ACTIVE-REF | diagnostic only | diagnostic only | Figure A and four-dataset D2 remain separate. |
| Box 20 tail research | HOLD | HOLD | HOLD | HOLD | cRT guardrail fail; learned pooling 0 material votes. |
| Box 21 masked loss | PASS | PASS | CAPABILITY-GATE | PASS | Future composition remains unselected. |
| Box 22 evaluation | PASS | PASS | PASS | PASS | Per-dataset/unit/class reporting only. |

## 5. Candidate-module SOTA traceability

Every candidate printed in the SVG resolves here. `Local evidence` is intentionally conservative.

| Candidate module in v1 | Work cards | What the cards can support | Required compatibility/gate | Local evidence status |
|---|---|---|---|---|
| AST encoder reference | `W09` | General-audio AST architecture/checkpoint reference | Fixed fbank/input audit; pooled vs temporal capability explicit | Current ICBHI+SPRSound engineering path only; no completed performance result. |
| BEATs encoder candidate | `W10` | Public pretrained encoder candidate | Checkpoint, layer, pooling, input/pretraining-overlap audit | Used in frozen-feature diagnostics; not current joint encoder. |
| OPERA encoder/benchmark candidate | `W12` | External respiratory foundation-model benchmark/encoder candidate | Public checkpoint and task/input parity | External candidate; no local reproduction in accepted SOTA map. |
| PANNs lower-complexity control | `W11` | Public audio encoder/control | Equal-input/equal-update comparison | Candidate only. |
| HeAR encoder candidate | `W13` | 2-s/512-d representation candidate | Window/input mismatch and aggregation gate | Candidate only. |
| Dataset-specific encoder bank | `W07`, `W08`, `W25`, `W29`, `W32` | Multi-view, native temporal/recording, and dataset-specific branch references | Equal-parameter/compute and native protocol preservation | Candidate/HOLD; no four-dataset end-to-end evidence. |
| General representation | `W12`, `W19` | Multi-dataset representation idea; conceptual disentanglement | Dense baseline, attribution, LODO; `W19` lacks code/checkpoint | Proposed/HOLD; no local empirical evidence. |
| Domain-specific residual/adapter | `W15`, `W16`, `W18` | Metadata-aware, domain, or patient-aware adaptation references | Metadata/patient parity; no dataset-ID/pathology shortcut claim | `W18` has bounded frozen-feature diagnostics; full-encoder effect unverified. |
| Input/domain augmentation | `W20`, `W26` | Cross-domain mixing/denoising candidate use | HF gap semantics; KAUH mapping; transient-event preservation | Candidate/HOLD. |
| HF temporal native branch | `W02`, `W08`, `W29` | Source temporal benchmark and temporal aggregation/head references | Temporal-capable interface, overlap masks, source-time map | Source facts/paper claims; no current full target branch. |
| KAUH recording/disease reference | `W04`, `W21`, `W22` | Native source facts and recording/disease-decoder precedents | Patient grouping, raw sound strings, disease separate | Native contract fixed; disease extension HOLD. |
| MVST-style fusion | `W07` | Multi-view fusion candidate | Equal compute/parameter, same units/splits | Candidate only. |
| EZhouNet temporal fusion | `W08` | Variable event/interval branch reference | Dataset/protocol/input parity | HOLD for direct transplant. |
| Soft router / CNN-MoE | `W23` | Expert-routing precedent | Acoustic-only routing, no dataset-ID main signal, collapse/usage audits | Candidate/HOLD; no four-dataset local evidence. |
| Multimodal side channel | `W24` | External multimodal fusion precedent | Metadata availability and shortcut audit | HOLD; excluded from acoustic-only main path. |
| Focal / class-imbalance control | `W06`, `W32` | Native-task focal-loss references | One-axis preregistration and specificity/native guardrails | Future controlled candidate; cRT result does not validate it. |
| Contrastive / Patch-Mix control | `W14`, `W33` | Representation/supervision references | Protocol and challenge-variant parity | Candidate only; not a four-dataset solution. |
| Knowledge distillation control | `W27` | Student regularization candidate | Teacher provenance and capacity controls | Candidate only. |

## 6. Evidence-registry placement

| Evidence item | Correct status | Permitted implication | Prohibited implication |
|---|---|---|---|
| Current ICBHI+SPRSound AST receipts | Executable Reference | Data, fbank, routing, finite loss/gradients, resume/profile path work. | Accuracy, generalization, or superiority. |
| Four-dataset representation attribution | Verified Result, frozen-feature scope | Controlled target-supervised representation differences were measured. | Full-encoder selection, source generalization, or zero-shot. |
| Shared-compatible head harmonization | Verified negative/HOLD | 0 material improvements in the corrected controlled comparison. | Shared heads are universally impossible. |
| Support-aware cRT | Verified negative/HOLD | Local gains occurred but global regression guardrails failed. | Dual imbalance is solved or no task improved. |
| Shortcut diagnostic | Verified inconclusive | 0/2 votes; acquisition-correlated error not supported/inconclusive. | Dataset-ID accessibility proves pathology shortcut use. |
| Event-sensitive pooling | Verified inconclusive | 0 material votes in the preregistered comparison. | All temporal/event-sensitive methods are impossible. |
| SOTA work cards | Verified Source Fact / Paper Claim / Interpretation / Candidate Use / HOLD | Trace candidate provenance and compatibility concerns. | Local reproduction or local empirical result. |

## 7. Promotion gates

| Gate | Promotion question | Minimum receipt required |
|---|---|---|
| Backbone | Does a candidate beat the simple AST/control under the same legal tasks? | Same inputs, splits, masks, heads, optimizer, updates, seeds, capacity/compute report, per-dataset/worst-dataset metrics. |
| Temporal capability | Can HF native labels be aligned without converting gaps to negative? | Real token mask and source-time map, overlap-aware targets, temporal unit tests. |
| General/specific factorization | Does decomposition add value beyond parameter-matched dense capacity? | Dense/no-residual controls, attribution, rare-event and native-task guardrails, LODO. |
| Router/MoE | Is routing acoustic and non-collapsed? | Equal-parameter dense control, expert-use entropy/load, counterfactual dataset-ID audit, per-dataset/native results. |
| Tail control | Does one intervention improve tails without unacceptable native/specificity regression? | Preregistered material band, support-aware metrics, global regression guardrails. |
| Shared head | Does a new formulation survive corrected independent-head control? | Parameter-matched heads, native side effects, worst-dataset and support guardrails. |

## 8. Complete W01–W33 disposition

This table closes the traceability loop for all accepted work cards. `Not printed` means the card remains available in the SOTA registry but is intentionally omitted from the master figure to avoid implying selection.

| Card | Short role | v1 disposition | Figure/module link |
|---|---|---|---|
| `W01` | ICBHI source/task facts | Fixed source anchor | Box 01 / input card |
| `W02` | HF_Lung source temporal benchmark | Fixed source anchor + compatibility reference | Box 01; Box 16 |
| `W03` | SPRSound source/task facts | Fixed source anchor | Box 01 / input card |
| `W04` | KAUH source/task facts | Fixed source anchor | Box 01; Box 17 |
| `W05` | RespireNet | Candidate P-Ref preprocessing/end-to-end reference; not selected | Not printed; retained under Module A/G SOTA registry |
| `W06` | CNN-LSTM + focal loss | Candidate class-imbalance/native baseline | Box 20 candidate trace |
| `W07` | MVST | Candidate multi-view/fusion and end-to-end reference | Boxes 07/12 candidate rail |
| `W08` | EZhouNet | Dataset/protocol-gated temporal/fusion candidate | Boxes 07/12/16 HOLD |
| `W09` | AST | Current executable encoder reference and open target candidate | Box 06; executable inset |
| `W10` | BEATs | Encoder candidate | Box 06 candidate rail |
| `W11` | PANNs | Lower-complexity encoder control | Box 06 candidate rail |
| `W12` | OPERA | External respiratory encoder/benchmark candidate | Box 06; Box 08 candidate rail |
| `W13` | HeAR | Input-contract-sensitive encoder candidate | Box 06 candidate rail |
| `W14` | Patch-Mix contrastive learning | Candidate supervision/representation control | Box 20 candidate trace |
| `W15` | Metadata supervised contrastive | Candidate specific adapter, metadata parity gate | Box 09 candidate rail |
| `W16` | Stethoscope-SCL | Candidate domain adapter, domain-metadata gate | Box 09 candidate rail |
| `W17` | CEDANN | Private/controlled domain-adaptation reference only | Not printed; HOLD in Module C/F/G registry |
| `W18` | PAFA | Patient-aware adapter candidate with bounded frozen diagnostics | Box 09 candidate rail |
| `W19` | DDE-MAE | Conceptual general/specific factorization; no code/checkpoint | Box 08 HOLD |
| `W20` | Lungmix | Cross-dataset augmentation clue; HF/KAUH semantics gate | Box 20 candidate trace |
| `W21` | DeepBreath | Recording/disease-task precedent; dataset/task mismatch | Box 17 reference trace |
| `W22` | Deep auscultation | Historical decoder/task-separation precedent | Box 17 reference trace |
| `W23` | CNN-MoE | Future soft-router/expert candidate | Boxes 10–11 HOLD |
| `W24` | BTS multimodal | Metadata side-channel candidate with shortcut risk | Box 11/12 multimodal HOLD |
| `W25` | SPRSound multi-spectrogram | Dataset-specific native reference | Box 07/native candidate rail |
| `W26` | ADD-RSC | Denoising candidate; transient-event preservation gate | Box 20 candidate trace |
| `W27` | Ensemble knowledge distillation | Candidate student regularization | Box 20 candidate trace |
| `W28` | CycleGuardian | Representative regularization reference only | Not printed; retained under Module B/F/G SOTA registry |
| `W29` | Multi-stage respiratory analysis | Temporal aggregation/P-Specific reference | Box 07/16 candidate rail |
| `W30` | Resp-Agent | Multimodal preprint watch item | Not printed; HOLD in Module D/E/G registry |
| `W31` | Patient-consistent multi-cycle model | Different prediction unit; HF patient ID unavailable | Not printed; HOLD in Module D/F/G registry |
| `W32` | SPRSound ResNet + focal | Native SPRSound/focal reference only | Box 07/20 candidate trace |
| `W33` | SPRSound SupCon + MixUp | Challenge-variant-sensitive supervision reference | Box 20 candidate trace |
