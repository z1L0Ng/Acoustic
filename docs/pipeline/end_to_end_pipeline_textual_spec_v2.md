# Acoustic End-to-End Pipeline v2 — complete textual specification

**Status:** canonical shared-window execution contract; performance remains `Not run`
**Date:** 2026-08-12
**Scope:** four audited datasets and their dataset-native tasks
**Visual package:** one master topology plus three readable detail panels

> **2026-08-12 alignment note.** P1/P2 now use one shared-window route for all four datasets, including HF. The older token-only rule “no encoder tokens/time map ⇒ no HF” is no longer a pipeline-level gate: a pooled encoder is called independently on source-time windows to produce a window sequence. Token-level capability remains a separate encoder fact and is needed only for the deferred P6 refinement. The current 2.0-s window / 1.0-s stride is an adjustable **Proposed Benchmark Policy**, not an optimality claim.

## 0. What v2 claims—and does not claim

v2 is a complete **system contract and candidate map**. It traces four native data lanes from raw release objects through legal prediction units, candidate encoders/adapters/fusion modules, eligible native heads, training controls, and native metrics. It does **not** claim that a four-dataset end-to-end model has been trained.

The phrase “four baselines” is ambiguous in older project materials. This specification resolves it as follows:

- **Four dataset-native input/baseline lanes:** ICBHI 2017, SPRSound BioCAS2022, HF_Lung_V1, and KAUH/Fraiwan v3. This is the requested v2 input topology.
- **Current pipeline IDs P1–P13:** experiment combinations in the canonical Notion Pipeline 组合实验主表. P1/P2 are the first-batch four-dataset shared-window references; P3–P5 are encoder-package candidates; P6–P13 are deferred block studies. Legacy R0–R3 remain historical evidence only and are not execution keys.

Evidence grammar:

| State | Meaning | Allowed claim |
|---|---|---|
| **Fixed Contract** | Accepted dataset, task, interface, split, or evaluation rule | May be implemented; not evidence of superiority |
| **Executable Reference** | Local code path with protocol/smoke/profile receipts | Engineering executability only |
| **Verified Result** | Completed local controlled result in its exact protocol | Report with scope, including negative/inconclusive results |
| **Paper-faithful Reference** | Architecture/protocol recovered from primary paper/repository | External paper/code fact, not local result |
| **Candidate / Proposed** | Alternative to compare | No selection or performance implication |
| **HOLD** | Missing capability, semantics, parity, provenance, or approved test | No forward solid arrow |

Hard claim boundaries:

- smoke/profile is not performance;
- target-head adaptation uses target labels and is not zero-shot;
- dataset-ID accessibility does not prove that a pathology shortcut is used;
- HF gaps are `not_annotated`, not normal/negative;
- KAUH B/D/E files are filter replicas grouped by patient;
- SPRSound `train`, `inter`, and `intra` remain separate;
- Phase 1A is not a successful architecture-selection result;
- shared encoder, specific encoders, residual branches, router, and MoE remain alternatives.

## 1. Canonical tensors and module contract

```text
waveform       X      [B,T]            mono, padded only at collation
waveform mask  M_wav  [B,T]            true/1 = valid source samples
windows        X_win  [B,N_w,32000]    current 2-s windows at 16 kHz
window mask    M_win  [B,N_w]          true/1 = valid source window
window map     Q_win  [B,N_w,2]        source-time [start_s,end_s] per window
spectrogram    F      [B,C,F_bin,tau]  C=1 unless a paper-faithful RGB/image path says otherwise
tokens         Z      [B,L,D]          acoustic tokens only unless declared
token mask     M_tok  [B,L]            true/1 = valid token
time map       Q      [B,L,2]          source-time [start_s,end_s] for every token
pooled         z      [B,D]
window embed   Z_win  [B,N_w,D]        one pooled encoder output per source-time window
flat logits           [B,K]
temporal logits       [B,N_w,K] or token/paper-faithful fixed-grid output
```

Every selected backbone must return:

```text
BackboneOutput(
  pooled: [B,D],
  tokens: [B,L,D] | null,
  token_mask: [B,L] | null,
  time_map: [B,L,2] | null,
  capability: pooled_only | temporal_capable,
  source/checkpoint/input/output receipts
)
```

For the current P1–P5 route, the wrapper must additionally return `WindowEncoderOutput(embeddings=[B,N_w,D], window_mask=[B,N_w], time_map=[B,N_w,2])`. This window-level contract is sufficient for the HF native temporal head even when the underlying encoder is `pooled_only` per window.

`temporal_capable` remains legal only when all of `tokens`, `token_mask`, and token-level `time_map` are present. This describes the **encoder's token-level capability**, not whether the full shared-window pipeline can serve HF. A tensor of frame embeddings alone is insufficient for P6 token-level refinement until its valid mask and source-time mapping are declared.

## 2. Four dataset-native lanes

### Lane 1 — ICBHI 2017 (`W01`)

- **Release:** official 2017 challenge package.
- **Raw object:** heterogeneous mono WAV plus cycle TXT rows `(start_s,end_s,crackle_flag,wheeze_flag)`; diagnosis is a separate patient file.
- **Audio:** 4/10/44.1 kHz; 16/24 bit in the audited materialization.
- **Scale:** 920 recordings, 5.492508 h, 6,898 cycles, 126 patients.
- **Prediction unit:** respiratory cycle; recording and patient are not silently substituted.
- **Native labels:** `normal`, `crackle`, `wheeze`, `both`; output `[B,4]`.
- **Grouping/split:** official recording split 539/381 recordings and 4,142/2,756 cycles; patients 156 and 218 overlap. It is literature-comparable but not patient-independent.
- **Native parser/head:** cycle-boundary crop → flat-four head.
- **Metric:** sensitivity, specificity, ICBHI Score; also macro-F1, UAR, class recall and support with split caveat.
- **Gate:** any strict robustness result needs an additional patient-grouped protocol.

### Lane 2 — SPRSound BioCAS2022 (`W03`)

- **Release:** commit `874eeb8736ddb78937c2fb5332fc7e7293d0f0ca`, BioCAS 2022 classification release.
- **Raw object:** 8 kHz mono 16-bit recording WAV plus event start/end and recording labels.
- **Scale:** 2,683 recordings, 8.162338 h, 9,089 events; audited split contains 251 train, 41 inter-test and 162 intra-test patient identities.
- **Prediction units:** (a) respiratory event; (b) recording. They use different heads.
- **Event labels:** seven classes `Normal`, `Rhonchi`, `Wheeze`, `Stridor`, `Coarse Crackle`, `Fine Crackle`, `Wheeze+Crackle`; outputs event binary `[B,2]` and event seven `[B,7]`.
- **Recording labels:** five raw classes `Normal`, `CAS`, `DAS`, `CAS & DAS`, `Poor Quality`; output `[B,5]`. A ternary variant is a separate declared task, not an implicit collapse.
- **Grouping/split:** train 1,949 recordings/251 patients; inter 355/41 with zero train-patient overlap; intra 379/162 with all 162 present in train. Never pool inter and intra.
- **Native parser/head:** event crop → event heads; whole recording → recording head.
- **Metric:** SE/SP/AS/HS family, plus macro-F1/UAR/per-class support, reported separately for inter and intra.

### Lane 3 — HF_Lung_V1 (`W02`)

- **Release:** V1, README update 2022-01-18.
- **Raw object:** exactly 15-s mono WAV plus overlapping interval rows.
- **Audio:** 4 kHz, 16 bit.
- **Scale:** 9,765 recordings, 40.6875 h, 81,933 positive interval labels.
- **Raw interval tokens:** `I`, `E`, `D`, `Wheeze`, `Rhonchi`, `Stridor`; there is no raw Normal/Negative token.
- **Paper task semantics:** four separate one-vs-rest detection tasks: inhalation, exhalation, CAS=`Wheeze|Rhonchi|Stridor`, DAS=`D`.
- **Paper-faithful preprocessing:** 10th-order 80-Hz high-pass; STFT Hanning 256 samples, hop 64, no extra zero padding; spectrogram `[938,129]`; concatenate 129 log-magnitude + 60 MFCC/delta/delta2 + 4 band-energy features → `[938,193]`; per-feature min-max normalization.
- **Paper outputs:** RNN variants output `[B,938,1]` per task; CNN-RNN variants output `[B,469,1]` per task. Postprocessing maps frames to events. The current shared-window route instead produces per-window embeddings `[B,N_w,D]` and native temporal logits `[B,N_w,4]` with `M_win/Q_win`. P6 optionally refines this to token-level `[B,L,4]` after a valid `token_mask/time_map` contract is built.
- **Grouping/split:** source folders 7,809/1,956 recordings; deidentified-date proxy has no folder overlap, but this does not prove patient independence.
- **Metric:** segment accuracy/PPV/SE/SP/F1/AUROC; event F1 with Jaccard > 0.5 and event MAPE; proposed native reporting adds per-label AUPRC and annotated-duration coverage.
- **Gate:** a pooled-only encoder may connect through the audited shared-window wrapper; it may not claim token-level localization. Unlabeled or unclear gaps remain `not_annotated`, not negative. Any constructed negatives belong only to the declared paper-native rasterized one-vs-rest condition and are never reinterpreted as raw normal intervals or shared-label negatives.

### Lane 4 — KAUH/Fraiwan v3 (`W04`)

- **Release:** Mendeley `jwyy9np4gv_v3`.
- **Raw object:** recording WAV plus one patient workbook row repeated across B/D/E filtered files.
- **Audio:** mono, 4 kHz, 16 bit.
- **Scale:** 336 recordings, 112 patients, 3 filter replicas per patient, 1.623918 h.
- **Prediction unit:** recording for sound type; patient for diagnosis. Diagnosis is a separate secondary/HOLD task.
- **Raw recording ontology (`K=9`):** `N`, `E W`, `I E W`, `C`, `I C`, `I C E W`, `Crep`, `Bronchial`, `I C B`; output `[B,9]`.
- **Counts, patients/recordings:** `N 35/105`, `E W 39/117`, `I E W 2/6`, `C 7/21`, `I C 1/3`, `I C E W 2/6`, `Crep 23/69`, `Bronchial 1/3`, `I C B 2/6`.
- **Grouping/split:** P-number is the grouping key; B/D/E mean Bell/Diaphragm/Extended filters and stay in one partition; no official split.
- **Metric:** proposed patient-grouped macro-F1, UAR, balanced accuracy, per-class recall and patient support.
- **HOLD:** `Crep`, `Bronchial`, and `I C B` are not silently mapped to a shared crackle/wheeze ontology; disease strings are not the same target as sound type.

## 3. Dataset-native adapters and preprocessing

Each adapter emits a non-trainable `CanonicalRecord` preserving source semantics:

```text
dataset_id, release_id, adapter_version, schema_version
recording_id, audio_ref, source_sample_rate, duration
prediction_unit_id, crop/interval lineage
native targets + annotation-state codes
group_id + group_status, source_split + split_status
device/site/filter/quality/demographic metadata when provided
```

`not_applicable`, `not_provided`, `unknown`, and `not_annotated` remain distinct. Adapter outputs do not fabricate metadata, harmonize labels, or encode trainable dataset identity.

Three preprocessing paths must stay visibly separate:

1. **Current P1–P5 shared-window contract:** native unit extraction → mono 16 kHz → source-time 2.0-s windows / 1.0-s stride → `X_win`, `M_win`, `Q_win` → one frozen candidate encoder per window → shared projector → native head. Short units zero-pad; long units receive one unique end-aligned tail window. This geometry is adjustable after first-batch feasibility.
2. **Deferred P6 token-level refinement:** BEATs tokens + `M_tok` + audited token-level `time_map`; it must preserve exact pooled parity with P2 on non-HF lanes or be downgraded to a package comparison.
3. **Legacy R0 AST evidence:** ICBHI cycle/SPRSound event crop → mono → 16 kHz → fixed 8-s repeat/truncate → 128-bin fbank → resize `[B,1,798,128]`; historical engineering evidence only, not the current four-dataset path.

## 4. Parallel candidate architecture slots

The following are alternatives. Their placement in one panel does not mean they are simultaneously composed.

### 4.1 Input and preprocessing candidates

- `W05 RespireNet`: blank-region clipping, smart padding, concatenation-based augmentation; ImageNet ResNet34 path. Paper-faithful ICBHI reference only.
- `W07 MVST`: one spectrogram with five patch-view AST branches; fixed multi-view design.
- `W26 ADD-RSC`: adaptive differential denoising intervention before its ICBHI backbone; event-preservation and protocol consistency remain gates.

### 4.2 Encoder/backbone candidates

- `W09 AST base384`: P1 first-batch package; each source-time window is encoded to pooled `D=768`, then stacked as `[B,N_w,768]`. The internal AST frontend pads each 2-s window to the audited 798-frame grid, so P1 is a package-level reference.
- `W10 BEATs iter3+ AS2M`: P2 matched replacement; 16-kHz waveform → 128-bin fbank → patch Conv2d → 12× Transformer, `D=768`. P2 uses one pooled output per source-time window; the official token output and mask are reserved for P6, which still needs an audited token-level `time_map`.
- `W12 OPERA-CT`: HTS-AT/Swin respiratory foundation candidate; `256×256` model grid, patch 4/stride 4, stages `[2,2,6,2]`, widths `96/192/384/768`, heads `4/8/16/32`, MLP ratio 4. Official feature example uses 8 s and 768-d OPERA-CT output.
- `W13 HeAR 1.0.0`: 2-s 16-kHz waveform `[B,32000]` → gated ViT-L masked-autoencoder service → embedding `[B,512]`. The public serving/model card does not expose a token mask or time map; pooled-only in v2.
- `W11 PANNs Cnn14`: waveform → 64-bin log-mel → six 2×Conv blocks `64/128/256/512/1024/2048` → global max+mean → FC 2048 → embedding `[B,2048]`; pooled-only Cnn14 contract.

### 4.3 Adapter/domain candidates

- `W16 Stethoscope-SCL`: AST representation `[B,768]`; device/domain projector `Linear 768→768, BN, ReLU, Linear 768→D_p`; contrastive/domain supervision is legal only when the side-channel exists with comparable meaning.
- `W18 PAFA`: BEATs tokens `[B,L,768]`; optional temporal attention → patient projection `768→H→D_p`; PCSL/GPAL operate on patient groups. HF patient IDs are not provided; ICBHI checkpoint selection is official-test-selected. Four-dataset D2 evidence is frozen-feature only.
- `W15 Metadata-SCL`: representation plus declared sex/age metadata → supervised-contrastive projector. Metadata parity is absent across the four lanes; remains HOLD outside eligible sources.

### 4.4 Fusion/aggregation candidates

- `W07 MVST`: five `[B,768]` view embeddings; five learned sigmoid gates `768×768`; weighted sum `[B,768]`; Linear `768→4` in the paper ICBHI head.
- `W24 BTS`: CLAP audio embedding `[B,512]`; optional text embedding `[B,512]`; `concat→[B,1024]` or weighted add `→[B,512]`; classifier. Text/device/location parity and shortcut controls are mandatory; multimodal HOLD for P-Shared.
- `W30 Resp-Agent`: 16-kHz, max 10-s waveform; BEATs feature sequence fixed to 496×768 in the released diagnosis config; Linear `768→H_longformer`, audio/text token composition, Longformer classifier. It is a multimodal preprint reference, not an acoustic-only native-head solution.

No extra fusion work is added merely to fill the category; the repo-backed operational shortlist contains only these three.

### 4.5 Decoder/prediction candidates

- Dataset-native heads in Section 2 are the primary legal outputs.
- `W21 DeepBreath`: 4-kHz recording → 5-s, 32-bin log-mel segments; five ConvBlocks with channels `64/128/256/512/1024`; attention pooling; four separate one-vs-rest disease models. It is a recording aggregation reference, not a native acoustic-event target for the current four datasets.
- `W02 HF RNN/CNN-RNN`: exact fixed-grid temporal reference described in Lane 3; no located official code/checkpoint.

### 4.6 Loss and supervision candidates

- `W14 Patch-Mix CL`: AST patch sequence and mixed labels; CE mixture plus contrastive projector `768→768→768`. ICBHI-only paper-faithful candidate.
- `W15 Metadata-SCL`, `W16 SG-SCL`, `W18 PAFA`: side-channel-dependent supervised contrastive/alignment objectives; eligibility-masked by metadata/patient availability.
- `W26 ADD-RSC`: classification CE plus label-smoothing denoise loss; released default mixes them by `beta=0.5`. Repo/paper configuration discrepancies remain recorded in the reproduction matrix.

All task losses are multiplied by eligibility and valid-annotation masks. No shared label is synthesized to increase supervision.

### 4.7 Complete-pipeline references

The operational entry points remain `W05 RespireNet`, `W07 MVST`, `W12 OPERA`, `W02 HF native temporal benchmark`, and `W03 SPRSound native task repository`. They are references, not a single composable four-dataset recipe.

## 5. Legal connection rules

1. `pooled_only per window → ICBHI/SPRSound-event/SPRSound-recording/KAUH-recording` is structurally legal after the declared shared projector, masked aggregation, and protocol approval.
2. `pooled_only per window → HF window-level temporal head` is legal when `M_win` and `Q_win` are preserved; it supports window-level classification, not token/frame localization.
3. `tokens without token_mask/time_map → P6 token-level HF head` remains HOLD. This does not block the P1–P5 window-level HF route.
4. MVST view fusion requires five same-width pooled embeddings; it is not a generic temporal fusion.
5. SG-SCL, PAFA, Metadata-SCL, BTS and Resp-Agent require explicit side-channel capability declarations.
6. KAUH raw-9 is legal; a shared KAUH acoustic mapping remains HOLD.
7. SPRSound record-5 and event-7/event-2 are distinct native tasks.
8. Dataset-ID routing is not domain generalization. Router/MoE remains outside the v2 selected path.

## 6. Training-only control plane

Training controls are lateral, never sequential audio boxes:

```text
M_head [B,H]
M_class[B,K_h]
M_time [B,L,K]
effective mask = task eligibility × valid annotation × approved class control
L = sum_h lambda_h * masked_loss_h
```

- source sampler defines update composition and produces a receipt;
- optional source weight `w_d` affects only eligible loss terms;
- unknown/not-provided/not-annotated rows are excluded, not treated as negative;
- inference bypasses sampling and loss controls;
- task declaration can open a legal inference head but cannot read target labels.

P1/P2 use frozen encoders, source-proportional homogeneous batches, the same shared projector/native heads, and native objectives. The frozen training/selection/seed/update-budget contract must remain matched. Four-dataset D2 remains a target-supervised frozen-feature diagnostic, not evidence that the new shared-window joint training works.

## 7. Current execution and historical evidence

| Pipeline/result | Exact status | Allowed placement |
|---|---|---|
| `P1` AST four-dataset shared-window | local code/assets/real-data CPU ready; CUDA/full/terminal score not run | first-batch reference; no performance claim |
| `P2` BEATs four-dataset shared-window | local code/assets/real-data CPU ready; CUDA/full/terminal score not run | first-batch matched package replacement |
| `P3/P4` PANNs/HeAR | code/synthetic CPU path only; official/gated assets HOLD | encoder-package candidates after asset closure |
| `P5` OPERA-CT | provenance/pretraining-overlap and input-adapter HOLD | overlap-aware reference only |
| `P6–P13` | deferred block studies; all results `Not run` | require shortlist and separate approval |
| legacy `R0–R3` | historical protocol/smoke/profile or paper-faithful evidence | registry only; never current execution IDs |
| shared-compatible head | 0 material improvements in corrected frozen-feature comparison | negative within scope; registry only |
| support-aware cRT | 2 improvements but global regression guardrail failed | HOLD/negative; registry only |
| shortcut diagnostic | 0/2 votes | not supported/inconclusive; no causal shortcut claim |
| event-sensitive pooling | 0 material votes | not supported/inconclusive |
| four-dataset attribution/D2 | target-supervised, single-seed frozen-feature evidence | no full-encoder conclusion |

## 8. Training, inference and evaluation paths

### Four-dataset target training

```text
raw native releases
  → dataset adapter → prediction-unit builder
  → shared 16-kHz source-time window collator
  → one selected frozen candidate encoder per window
  → shared projector → optional eligible block
  → dataset-native head → logits

task/source capability ─→ eligibility masks ─→ head + loss + evaluation
source membership ──────→ sampler/weights ───→ masked loss
native targets + masks ──────────────────────→ masked loss → update
```

### Known-dataset inference

```text
raw audio + declared dataset/task
  → same adapter/unit/preprocessing
  → selected backbone
  → only legal native head(s)
  → prediction + lineage + capability receipt
```

### Evaluation

Predictions, targets, support masks, split/group receipts, checkpoint identity, target-label use and selection caveats flow to per-dataset/per-unit metrics. The claim ledger keeps zero-target transfer, target-supervised head adaptation, full target training, joint training, engineering receipts, and paper claims separate.

## 9. Sources and traceability

- Accepted project registries: [Pipeline 组合实验主表](https://app.notion.com/p/3ba309efda29816a8fe0d56424a18897), [Pipeline Module SOTA Map](https://app.notion.com/p/3b0309efda2981899476c6fcaca45d47).
- Dataset anchors: [ICBHI](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge), [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound), [HF_Lung_V1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254134), [KAUH/Fraiwan v3](https://data.mendeley.com/datasets/jwyy9np4gv/3).
- Encoder sources: [AST](https://github.com/YuanGongND/ast), [BEATs](https://github.com/microsoft/unilm/tree/master/beats), [OPERA](https://github.com/evelyn0414/OPERA), [HeAR](https://huggingface.co/google/hear), [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn).
- Candidate sources: [RespireNet](https://github.com/microsoft/RespireNet), [MVST](https://github.com/wentaoheunnc/MVST), [SG-SCL](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning), [PAFA](https://github.com/wa976/pafaofficialpytorch), [BTS](https://github.com/kaen2891/bts), [DeepBreath](https://github.com/epfl-iglobalhealth/DeepBreath-NatMed23), [Patch-Mix](https://github.com/raymin0223/patch_mix_contrastive_learning), [ADD-RSC](https://github.com/deegy666/ADD-RSC), [Resp-Agent](https://github.com/zpforlove/Resp-Agent).

Exact architecture rows, symbolic dimensions and unresolved gates are in `pipeline_candidate_layer_shapes_v2.md`; legal module connections are in `module_interface_compatibility_matrix_v2.md`.
