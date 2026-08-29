# ICASSP 2027 Four-Page Paper Skeleton

Status: `DRAFT SKELETON / NOT CLAIM-FROZEN`

Prepared from: Pipeline v2.8, the 2026-08-27 Meeting Record/Report, and the currently accepted result ledger

Technical-page target: 4 pages, with references on the optional fifth page

Evidence vocabulary: `Paper Claim`, `Verified Local Result`, `Posthoc Diagnostic / Test-Informed`, `Proposed Method`, `TODO`, `HOLD`

> This file is a writing scaffold, not an experiment or claim receipt. Values are included only when they already appear in the accepted meeting/result materials. A `TODO` or `HOLD` cell must not be silently replaced from memory, a paper table, or an unmatched protocol.

## Working Title

**Eligibility-Aware Hierarchical Learning for Cross-Dataset Respiratory Sound Recognition**

- `TODO — title decision:` decide whether the title should foreground cross-dataset consistency, shared respiratory attributes, or native-task retention.
- `HOLD — claim wording:` do not use “state of the art,” “generalizable,” “robust,” or “first” until the final evidence and literature audits support the exact wording.

## Abstract

Single-dataset respiratory-sound systems can obtain strong results under their native benchmarks, yet their labels, prediction units, annotation completeness, and operating points differ enough that joint training can yield inconsistent behavior across datasets. We study an eligibility-aware shared hierarchy over two compatible datasets, ICBHI 2017 and SPRSound BioCAS2022, while retaining the native meaning of each benchmark. The system fine-tunes a shared BEATs encoder on 16-kHz audio using 4-s windows with a 2-s stride, predicts a Level-1 Normal/Abnormal state and Crackle/Wheeze attributes, and masks supervision where a node is not supported by the source annotation. Current single-seed evidence shows strong SPRSound Task1-1 performance but weaker ICBHI flat-four retention, exposing a decoder- and dataset-dependent conflict rather than a uniform cross-dataset gain. `TODO — after evidence freeze:` insert only claim-approved headline results and the final main/ablation configuration. `HOLD:` multi-seed evidence, paper-facing ICBHI decoder, PAFA direct reproduction, and KAUH patient-level external evaluation.

## 1. Introduction

### 1.1 Problem

Respiratory-sound classification is commonly evaluated within one dataset and one native task. These results are not automatically comparable across datasets because the prediction unit, label space, split, and metric denominator change:

- ICBHI evaluates respiratory cycles with a four-state Normal/Crackle/Wheeze/Both target.
- SPRSound evaluates respiratory events and recordings under distinct tasks and inter-/intra-subject protocols.
- HF_Lung supplies positive temporal intervals but no explicit shared Normal/Negative interval.
- KAUH supplies patient-linked filtered recordings and raw sound strings, including unresolved mappings.

`Paper Claim / background:` direct BEATs-based ICBHI systems report substantially stronger native ICBHI Scores than the present shared-core runs under their declared protocols.

`Verified Local Result:` the present shared-core runs perform consistently above 90 on SPRSound Task1-1 but do not show a corresponding ICBHI improvement.

`Interpretation:` single-dataset performance and joint executability do not imply consistent cross-dataset behavior.

### 1.2 Gap

Existing pipelines often flatten labels into a nominal common taxonomy or treat missing labels as negatives. That can create supervision that the source dataset never provided. We instead ask:

> Can a shared encoder and a small respiratory-event hierarchy use only source-supported supervision while preserving each dataset's native evaluation boundary?

`HOLD — novelty sentence:` replace this question with a bounded novelty statement only after the final primary-source comparison is approved. Do not claim the first multi-label respiratory model, the first cross-dataset training method, or the first masked partial-label method.

### 1.3 Intended contributions

1. **Dataset/ontology contract (`Proposed Method`):** a unit-preserving shared core for ICBHI and the compatible SPRSound event surface, with explicit eligibility masks for unsupported nodes.
2. **Hierarchical readout (`Proposed Method`; paper-facing decoder `HOLD`):** Level-1 Normal/Abnormal plus Crackle/Wheeze attributes, retaining Both as attribute co-occurrence rather than an unrelated class.
3. **Controlled cross-dataset analysis (`Verified Local Result` + `Posthoc Diagnostic`):** matched 4-s/2-s BEATs runs that expose different normalization/loss behavior across ICBHI and SPRSound and isolate a decoder-induced ICBHI specificity failure.
4. **Native-boundary reporting (`Proposed reporting contract`):** results are separated by dataset, task, prediction unit, protocol, and evidence status; no pooled global score is used.

`TODO — contribution freeze:` reduce to at most three contributions after the main result and external-evaluation decision.

## 2. Dataset and Ontology

### 2.1 Dataset roles

| Dataset | Native prediction unit | Native target used in this paper | Paper role | Hard boundary |
|---|---|---|---|---|
| ICBHI 2017 | Respiratory cycle | Normal / Crackle / Wheeze / Both | Shared-core training and native evaluation | Official recording 60/40 split is not strictly patient-independent; retain the split caveat |
| SPRSound BioCAS2022 | Respiratory event for Task1-1 | Normal / Adventitious for terminal Task1-1; compatible event evidence for shared nodes | Shared-core training and official inter-subject evaluation | Do not pool inter and intra; event Task1-2/raw7 and recording-level tasks are outside the current three-node result |
| HF_Lung_V1 | 15-s recording with temporal intervals | None in the current Core-2 result | Positive-only/native auxiliary, currently excluded | Gaps are `not_annotated`, never shared Normal/Negative; no current paper-primary training result |
| KAUH / Fraiwan | Whole recording; three filter replicas per patient | Native raw-sound/external target `TODO` | Patient-level external evaluation only | No training or model selection; B/D/E filter replicas stay grouped and aggregate at patient level; unresolved mappings remain `HOLD` |

### 2.2 Shared ontology and eligibility

For sample/unit $i$, the shared target surface is

\[
y_i = (y_i^{L1}, y_i^{C}, y_i^{W}), \qquad
m_i = (m_i^{L1}, m_i^{C}, m_i^{W}),
\]

where $L1$ is Normal/Abnormal, $C$ is Crackle, $W$ is Wheeze, and $m$ marks whether the source annotation makes a node eligible. The model never receives an “Unknown” output logit.

| State | Loss/evaluation treatment |
|---|---|
| Positive or explicit negative | Eligible for the corresponding node |
| Unknown | Masked |
| Not annotated | Masked |
| Not applicable | Masked |
| Unresolved mapping | `HOLD`; masked and excluded from the shared claim |

The ICBHI four-state target is exactly reconstructable from Crackle/Wheeze presence when the readout contract is fixed. SPRSound contributes only compatible event evidence. Rhonchi/Stridor handling and any Other node remain `HOLD` unless an explicit ontology decision is approved. HF and KAUH are not forced into this closed shared table.

### 2.3 Dataset/ontology paragraph draft

We preserve each source's native prediction unit and construct shared supervision only when the source annotation supports the requested node. ICBHI cycles provide complete Normal/Crackle/Wheeze/Both evidence. SPRSound events provide the compatible shared-core surface while retaining the official inter-subject Task1-1 evaluation. HF_Lung is not treated as a source of shared negatives because its unannotated gaps are not verified normal intervals. KAUH is reserved for grouped external evaluation, with the three filter variants of each patient kept together. This design avoids manufacturing cross-dataset labels merely to increase the number of supervised samples.

`TODO:` insert exact dataset/sample counts only after the final table is checked against the accepted dataset ledger.

`HOLD:` KAUH external target/mapping and aggregation metric; SPRSound Rhonchi/Stridor policy.

## 3. Method

### 3.1 Shared BEATs representation

Each native audio unit is converted to mono 16 kHz and segmented into 4-s windows with a 2-s stride. Short units are zero-padded; longer units use overlapping windows with source-time lineage retained. A BEATs iter3+ AS2M encoder is fine-tuned jointly on ICBHI and SPRSound. Valid window representations are aggregated by a mask-aware mean to one native-unit representation before the shared head.

Current implemented head:

\[
h_i = W_p z_i + b_p, \quad h_i \in \mathbb{R}^{256},
\]

\[
\ell_i^{L1} = W_{L1}h_i+b_{L1}, \qquad
\ell_i^C = W_Ch_i+b_C, \qquad
\ell_i^W = W_Wh_i+b_W.
\]

- Encoder output width: 768.
- Shared projector: Linear $768\rightarrow256$.
- Level-1 head: Linear $256\rightarrow2$.
- Crackle/Wheeze heads: two Linear $256\rightarrow1$ sigmoid logits.
- No Other node exists in the current Core-2 head.

`HOLD:` do not describe token-level temporal modeling, router/MoE, PAFA, MVST, OPERA, or HF temporal refinement as part of the implemented method.

### 3.2 Eligibility-masked objective

For node $k$, only eligible samples contribute:

\[
\mathcal{L}_k =
\frac{\sum_i m_{ik}\,\ell(\hat{y}_{ik},y_{ik})}
     {\sum_i m_{ik}}.
\]

The reference objective averages the eligible Level-1 cross-entropy and Crackle/Wheeze binary cross-entropies. Ablations replace this reference with focal or class-balanced losses while retaining the same eligibility boundary.

Checkpoint selection uses the equal mean of ICBHI and SPRSound validation eligible-node loss. Terminal/test data are evaluated only after selection. This selection rule must be stated exactly; it is not “the loss of whichever dataset converged first.”

### 3.3 Native readout and decoder gate

SPRSound Task1-1 reads the Level-1 Normal/Abnormal output directly. ICBHI requires a frozen conversion from the hierarchy to flat four classes.

- **Historical bits-only decoder (`Verified Local Result`):** Crackle/Wheeze threshold bits map to Normal/Crackle/Wheeze/Both and ignore Level-1.
- **Level-1-gated decoder (`Posthoc Diagnostic / Test-Informed`):** Level-1 Normal maps to Normal; Level-1 Abnormal is refined with validation-selected Crackle/Wheeze thresholds; if neither attribute crosses its threshold, the larger probability-minus-threshold margin selects Crackle or Wheeze.
- **Paper-facing decoder (`HOLD`):** advisor/management must decide whether the Level-1-gated rule is scientifically accepted and must define a non-test-informed evaluation path before it becomes a primary paper result.

## 4. Experimental Setup

### 4.1 Current Core-2 protocol (`Verified execution/result protocol`)

| Item | Frozen current contract |
|---|---|
| Shared training data | ICBHI + SPRSound only |
| Audio/window | Mono 16 kHz; 4-s window / 2-s stride |
| Backbone | BEATs iter3+ AS2M, full fine-tuning |
| Shared outputs | Level-1, Crackle, Wheeze |
| Batch/budget | True native-unit batch 8; 50 epochs; 70,200 updates |
| Optimizer | Adam; backbone LR $1\times10^{-5}$, head LR $5\times10^{-5}$, weight decay $1\times10^{-6}$; FP32 |
| Seed | 42 |
| Selection | Equal mean of ICBHI and SPRSound validation eligible-node loss |
| ICBHI evaluation | Official recording 60/40 test; 2,756 cycles; Sp, Se, Score, macro-F1, UAR |
| SPRSound evaluation | BioCAS2022 official inter-subject Task1-1; 1,429 events; Sp, Se, Official Score, macro-F1, UAR |

The 4-s/2-s setting is the only matched end-to-end full-fine-tuning window condition currently closed. It is a current reference, not an optimality claim. Earlier 2-s/1-s cached results are not an apples-to-apples window comparison because encoder scope, decoder, and other components changed.

### 4.2 Controlled conditions

| ID | Normalization | Train augmentation | Loss | Selected epoch | Evidence status |
|---|---|---|---|---:|---|
| R0 | None | None | CE + BCE | 2 | `Verified Local Result` protocol |
| N1 | Peak | None | CE + BCE | 2 | `Verified Local Result` protocol |
| A1 | Peak | Gain + noise | CE + BCE | 2 | `Verified Local Result` protocol |
| L1 | Peak | Gain + noise | Focal, gamma 2.0 | 6 | `Verified Local Result` protocol |
| L2 | Peak | Gain + noise | Class-balanced, beta 0.9999 | 1 | `Verified Local Result` protocol |

### 4.3 PAFA direct baseline reproduction (`TODO / HOLD`)

The PAFA row must be a direct, protocol-labeled reproduction rather than a paper number copied into the local-result column.

Required baseline receipt for paper inclusion:

1. Use the ICBHI official recording 60/40 split and report all 2,756 official-test cycles.
2. Use the author-declared PAFA preprocessing and checkpoint/evaluation path; do not relabel a 4-s/2-s Core-2 adapter as paper-faithful PAFA.
3. Report Sp, Se, Score, seed(s), checkpoint-selection rule, and the difference from the paper claim.
4. Label an official-test-selected checkpoint explicitly; do not describe it as clean patient-held-out or source-validation-selected evidence.
5. Keep the direct reproduction separate from any matched 4-s/2-s architectural control. If both are retained, name them “paper-faithful reproduction” and “matched control,” respectively.

| PAFA evidence layer | Sp | Se | ICBHI Score | Status |
|---|---:|---:|---:|---|
| Published BEATs + PAFA | 82.05 | 47.63 | 64.84 | `Paper Claim`, five-run mean |
| Direct local paper-faithful reproduction | `HOLD` | `HOLD` | `HOLD` | Awaiting accepted reproduction row and paper/local/delta audit |
| Matched 4-s/2-s PAFA control | `HOLD` | `HOLD` | `HOLD` | Optional; requires a separately frozen attribution question |

`HOLD:` no PAFA value may be placed in the primary Local Result table until the reproduction owner provides the full protocol row and management accepts it.

### 4.4 Remaining experimental decisions

- `HOLD:` main configuration, currently R0 versus N1.
- `HOLD:` paper-facing ICBHI decoder and primary metric.
- `HOLD:` multi-seed confirmation for headline claims.
- `HOLD:` KAUH patient-level external evaluation.
- `TODO:` exact final window/padding statistics and the decision on whether a matched shorter-window condition is reviewer-critical.
- `TODO:` whether L1 remains the recall-oriented loss ablation.

## 5. Results

### 5.1 ICBHI official flat-four

**Table placeholder — do not merge evidence layers.**

| System/readout | Sp | Se | Score | Macro-F1 | UAR | Evidence layer | Paper use |
|---|---:|---:|---:|---:|---:|---|---|
| Bae23 AST + CE | 77.14 | 41.97 | 59.55 | `TODO` | `TODO` | `Paper Claim`, five-run mean | Related baseline only |
| Bae23 Patch-Mix CL | 81.66 | 43.07 | 62.37 | `TODO` | `TODO` | `Paper Claim`, five-run mean | Related baseline only |
| Jeong25 BEATs + CE | 78.77 | 48.21 | 63.49 | `TODO` | `TODO` | `Paper Claim`, five-run mean | Direct BEATs reference |
| Jeong25 BEATs + PAFA | 82.05 | 47.63 | 64.84 | `TODO` | `TODO` | `Paper Claim`, five-run mean | Direct PAFA reference; reproduction `HOLD` |
| Ours N1, historical bits-only readout | 28.37 | 61.68 | 45.03 | `TODO` | `TODO` | `Verified Local Result`, single seed | Diagnostic/current terminal row; not final main |
| Ours L1, historical bits-only readout | 29.13 | 66.86 | 48.00 | `TODO` | `TODO` | `Verified Local Result`, single seed | Recall-oriented diagnostic |
| Ours R0, Level-1-gated readout | 67.89 | 42.91 | 55.40 | 45.35 | 52.01 | `Posthoc Diagnostic / Test-Informed` | Not primary unless protocol is prospectively frozen and rerun |
| Ours N1, Level-1-gated readout | 63.52 | 44.44 | 53.98 | 44.23 | 49.62 | `Posthoc Diagnostic / Test-Informed` | Same boundary |
| Ours A1, Level-1-gated readout | 65.17 | 42.14 | 53.65 | 43.63 | 49.45 | `Posthoc Diagnostic / Test-Informed` | Same boundary |
| Ours L1, Level-1-gated readout | 49.97 | 58.45 | 54.21 | 46.59 | 51.42 | `Posthoc Diagnostic / Test-Informed` | Same boundary |
| Ours L2, Level-1-gated readout | 58.83 | 42.31 | 50.57 | 41.62 | 49.06 | `Posthoc Diagnostic / Test-Informed` | Same boundary |
| Final paper main, multi-seed | `HOLD` | `HOLD` | `HOLD` | `HOLD` | `HOLD` | `HOLD` | Headline row after evidence freeze |

Draft observation: the Level-1-gated recomputation raises ICBHI Score by 6.21–11.50 percentage points across the five saved runs, primarily through specificity recovery, but it lowers sensitivity in every condition. Because this decoder was developed after test behavior was examined, it is mechanism evidence rather than a primary generalization result.

### 5.2 SPRSound BioCAS2022 inter Task1-1

| System | Sp | Se | Official Score | Macro-F1 | UAR | Evidence layer |
|---|---:|---:|---:|---:|---:|---|
| Zhang22 MFCC + Naive Bayes official benchmark | 79.04 | 75.83 | 77.42 | `TODO` | `TODO` | `Paper Claim` |
| Ours R0 | 91.44 | 92.80 | 92.12 | 90.13 | 92.12 | `Verified Local Result`, single seed |
| Ours N1 | 91.83 | 95.63 | 93.71 | 91.44 | 93.73 | `Verified Local Result`, single seed |
| Ours A1 | 91.83 | 92.80 | 92.31 | 90.44 | 92.31 | `Verified Local Result`, single seed |
| Ours L1 | 84.62 | 97.17 | 90.68 | 86.35 | 90.89 | `Verified Local Result`, single seed |
| Ours L2 | 92.79 | 93.32 | 93.05 | 91.41 | 93.05 | `Verified Local Result`, single seed |
| Final paper main, multi-seed | `HOLD` | `HOLD` | `HOLD` | `HOLD` | `HOLD` | `HOLD` |

Draft observation: all five conditions exceed 90 in Official Score, but this does not justify an absolute SPRSound SOTA claim because published systems may use different releases, inter/intra pooling, mappings, or target-adaptation protocols.

### 5.3 Compact ablation table

| Condition | Changed axis vs preceding reference | ICBHI primary terminal Score | ICBHI Level-1-gated Score | SPRSound Score | Evidence-safe reading |
|---|---|---:|---:|---:|---|
| R0 | Base: no normalization/augmentation; CE+BCE | `TODO — retrieve accepted bits-only row` | 55.40 (`Posthoc`) | 92.12 (`Verified`) | Highest posthoc ICBHI Score; not a unified winner |
| N1 | Peak normalization | 45.03 (`Verified`) | 53.98 (`Posthoc`) | 93.71 (`Verified`) | Best SPRSound; not best posthoc ICBHI |
| A1 | Add gain+noise | `TODO — retrieve accepted bits-only row` | 53.65 (`Posthoc`) | 92.31 (`Verified`) | No consistent gain over N1 |
| L1 | Replace loss with focal | 48.00 (`Verified`) | 54.21 (`Posthoc`) | 90.68 (`Verified`) | Shifts operating point toward ICBHI abnormal sensitivity |
| L2 | Replace loss with class-balanced | `TODO — retrieve accepted bits-only row` | 50.57 (`Posthoc`) | 93.05 (`Verified`) | No metric-independent improvement |

`TODO — table compression:` after the decoder decision, retain only the rows/columns necessary for the four-page story and move the evidence ledger outside the technical pages.

### 5.4 External evaluation

`HOLD — KAUH patient-level table:` report only after the target, split/grouping, B/D/E aggregation, metrics, and selection independence are frozen. KAUH must not be used for training, threshold selection, or checkpoint selection in this submission.

## 6. Discussion

### 6.1 Decoder/ontology mismatch

The shared model learns a Level-1 Normal/Abnormal node, and SPRSound Task1-1 reads it directly. The historical ICBHI decoder ignored that node and reconstructed flat four classes only from low-threshold Crackle/Wheeze attributes. For N1, this sent many Normal cycles to abnormal classes and strongly reduced specificity. The Level-1-gated diagnostic restores much of this specificity but reveals a sensitivity trade-off. The result supports a decoder failure analysis; it does not yet validate the posthoc decoder as a paper-primary method.

### 6.2 Window and padding

Only the 4-s/2-s end-to-end full-fine-tuning condition is matched and closed. Short ICBHI cycles may therefore receive substantial zero-padding, while longer SPRSound events/records may benefit differently from contextual coverage. Earlier 2-s/1-s cached evidence changes multiple factors and cannot identify a window effect. `TODO:` insert Wade/lead-owned duration, padding, effective-signal, silence, and separability statistics if they are completed and accepted. `HOLD:` do not state that 4 s is optimal or that 1/2/3/4 s were all matched.

### 6.3 Joint-training conflict

Peak normalization, gain/noise augmentation, focal loss, and class-balanced loss do not improve both datasets under every readout and metric. Focal loss increases ICBHI abnormal sensitivity but reduces specificity and SPRSound performance. This suggests an operating-point and supervision conflict across datasets rather than a simple lack of encoder capacity. `TODO:` decide whether the paper's central contribution is a method improvement or an evidence-bounded analysis of eligibility-aware joint learning.

### 6.4 Checkpoint selection and early overfit

The selected epochs are 1, 2, 2, 2, and 6 even though all runs continue for 50 epochs. This is consistent with early validation deterioration under the current joint objective, but it is not by itself proof of causal “overfitting.” Report the selection rule and validation curves before using stronger language. `TODO:` add compact validation-loss/metric evidence or replace “early overfit” with “early checkpoint selection.”

### 6.5 What the present evidence supports

- `Verified Local Result:` joint BEATs Core-2 is executable and yields strong SPRSound Task1-1 single-seed results under the declared protocol.
- `Verified Local Result:` the historical ICBHI bits-only readout remains substantially below direct BEATs paper claims.
- `Posthoc Diagnostic:` a Level-1 gate removes a major decoder-induced specificity artifact but is test-informed.
- `Interpretation:` the main scientific obstacle is ICBHI native retention and a prospectively valid hierarchical decoder, not SPRSound learnability.
- `HOLD:` universal cross-dataset improvement, native-task SOTA, multi-seed robustness, PAFA superiority/delta, KAUH generalization, and any causal statement about padding or normalization.

## 7. Limitations and Deadline-Bounded Scope

The paper is developed within the approximately 20-day preparation window from the 2026-08-27 meeting to the 2026-09-16 ICASSP deadline. The scope is therefore deliberately narrow:

1. Shared training currently covers two datasets, ICBHI and SPRSound; HF positive-only shared training remains excluded and KAUH is external-only.
2. Current headline candidates are single-seed. Multi-seed confirmation is required before a robustness or stable-improvement claim.
3. The strongest ICBHI hierarchical numbers are posthoc/test-informed and cannot be presented as prospective primary results.
4. The 4-s/2-s window is not established as optimal, and shorter-window comparisons are unmatched.
5. The direct PAFA baseline reproduction and paper/local/delta row are `HOLD`.
6. KAUH label mapping and patient-level external reporting are `HOLD`.
7. The official ICBHI recording split contains patient overlap and must not be described as strict patient-held-out evaluation.

Deadline rule: after the core evidence freeze, add only experiments that close a named reviewer-facing gap in a designated table or figure. New encoders, HF shared training, routers/MoE, broad loss/augmentation sweeps, and new datasets are out of scope unless management explicitly reopens them.

## 8. Conclusion

We formulate cross-dataset respiratory-sound learning as a unit-preserving, eligibility-aware problem rather than a forced label merge. The current BEATs hierarchy produces strong SPRSound results but inconsistent ICBHI retention, and the analysis identifies the flat-four decoder as a major source of specificity loss. `TODO — after evidence freeze:` state the final prospective result and bounded contribution. `HOLD:` do not conclude that the method improves generalization until the paper-facing decoder, multi-seed comparison, and required baselines are accepted.

## Figure and Table Inventory

This inventory maps the paper artifacts to the existing Pipeline Module Map without modifying the SVG/PNG.

| ID | Four-page role | Existing source/module-map correspondence | Planned content | Status |
|---|---|---|---|---|
| Fig. 1 | Main method figure | `figures/pipeline/acoustic_end_to_end_pipeline_v2.8.svg/.png`; Pipeline v2.8 Layers 1–7 | Crop/redraw only the ICBHI+SPRSound lanes, 4-s/2-s windowing, BEATs, 768→256 projector, Level-1 + attributes, eligibility masks, and native evaluation | `HOLD — no figure edit authorized in this task`; decide whether a paper-specific derivative is needed |
| Tbl. 1 | Dataset/ontology contract | Pipeline v2.8 native inputs, crosswalk, eligibility, and evaluation layers | Four dataset roles, native units, shared eligibility, and exclusion boundaries | Skeleton complete; counts `TODO` |
| Tbl. 2 | Main ICBHI result | Pipeline native evaluation/claim ledger | Paper claims, verified local primary rows, posthoc rows, final multi-seed row | Evidence layers separated; final rows `HOLD` |
| Tbl. 3 | SPRSound + compact ablation | Pipeline native evaluation + loss controls | Task1-1 result and R0/N1/A1/L1/L2 comparison | Existing single-seed values inserted; final main `HOLD` |
| Tbl. 4 optional | KAUH external | Pipeline native KAUH lane | Patient-grouped external metric with B/D/E aggregation | `HOLD`; omit if not closed by evidence freeze |
| Fig. 2 optional | Decoder/window analysis | Pipeline hierarchy/readout and shared-window modules | ICBHI confusion/readout change or padding-vs-duration summary | `TODO`; choose one only if it directly supports the final claim |

### Module-to-paper trace

| Pipeline module | Paper destination | Evidence status |
|---|---|---|
| Native ICBHI/SPRSound parsers and units | Sec. 2, Tbl. 1 | Fixed contract |
| Shared 16-kHz window layer | Sec. 3.1, Sec. 4.1, Sec. 6.2 | 4-s/2-s current verified protocol; optimality `HOLD` |
| BEATs encoder | Sec. 3.1 | Implemented/current backbone; universal superiority `HOLD` |
| Shared 768→256 projector | Sec. 3.1 | Implemented/current head |
| Eligibility controller and masked losses | Sec. 2.2, Sec. 3.2 | Implemented method boundary |
| Level-1/Crackle/Wheeze head | Sec. 3 | Implemented; paper-facing ICBHI decoder `HOLD` |
| Native ICBHI and SPRSound evaluation | Sec. 4–5 | Mixed `Verified Local Result` and `Posthoc Diagnostic`; labeled per row |
| HF temporal lane | Limitations/future work only | Positive-only auxiliary; not in current Core-2 paper result |
| KAUH native lane | Optional external table | `HOLD` |
| PAFA module | Direct baseline protocol | `Paper Claim`; local paper-faithful reproduction `HOLD` |
| PANNs/HeAR/OPERA/MVST/router/token refinement | Excluded from four-page method | Candidate/reference/HOLD; no present method claim |

## Claim and Decision Ledger

| Decision/claim | Current status | Owner/gate |
|---|---|---|
| Main condition: R0 or N1 | `HOLD` | Advisor/management |
| Retain L1 as recall-oriented ablation | `TODO` | Paper-scope decision |
| Accept Level-1-gated ICBHI decoder | `HOLD` | Scientific decision plus prospective evaluation contract |
| Claim cross-dataset improvement | `HOLD` | Requires matched main/control and multi-seed evidence |
| Include PAFA direct baseline | `HOLD` | Paper-faithful local reproduction with paper/local/delta row |
| Include KAUH external evaluation | `HOLD` | Patient-level target/grouping/metric closure |
| Include HF | No in current main method | Reopen only by explicit approval |
| Window claim | `HOLD` | Quantitative padding/separability plus matched end-to-end evidence |
| Absolute SOTA claim | Prohibited under current evidence | Would require complete comparable literature and final result proof |
| Submission framing | `TODO` | Method contribution vs bounded cross-dataset analysis |

## References Placeholder

- `[REF-ICBHI]` ICBHI 2017 challenge/dataset.
- `[REF-SPR]` SPRSound BioCAS2022 dataset/benchmark.
- `[REF-BEATS]` BEATs pretraining/model paper.
- `[REF-AST-CE]` Bae23 AST + CE.
- `[REF-PATCHMIX]` Bae23 Patch-Mix CL.
- `[REF-BEATS-CE]` Jeong25 BEATs + CE.
- `[REF-PAFA]` Jeong25 BEATs + PAFA.
- `[REF-PARTIAL-LABEL]` `TODO — select only from the approved primary-source audit.`

`TODO:` build the BibTeX file from verified primary sources; do not infer titles, venues, author lists, years, or identifiers from shorthand names in this skeleton.
