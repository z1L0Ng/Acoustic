# 2026-08-27 Acoustic Meeting Report（中文审阅稿 v1）

## 1. Executive Summary

本周已经完成 BEATs unified-core 的五组 matched full-fine-tuning 实验：
R0、N1、A1、L1 和 L2。五组实验均使用 ICBHI + SPRSound、16 kHz、
4 s window / 2 s stride、seed 42、true native-unit batch 8、50 epochs，
并仅通过 ICBHI 与 SPRSound validation 选择 checkpoint 和 Crackle/Wheeze
threshold。

本周最重要的结果不是某一种 preprocessing 或 loss 已经解决 ICBHI，而是：

1. Peak normalization 是目前唯一同时改善 ICBHI 和 SPRSound terminal score
   的单一干预；因此 N1 是当前最合理的 cross-dataset reference。
2. Focal loss 明显提高 ICBHI sensitivity，但降低 SPRSound specificity；因此
   L1 更适合作为 ICBHI recall-oriented ablation，而不是 unified default。
3. SPRSound Task1-1 已达到稳定的 90% 以上表现；当前主要问题集中在 ICBHI。
4. ICBHI 的低 specificity 主要不是 BEATs 无法识别 Normal，而是旧 terminal
   flat4 decoder 完全忽略 Level1 Normal/Abnormal，将低阈值属性 false positive
   直接转成 Crackle/Wheeze/Both。
5. 新的 hierarchical readout 已完成独立重算。它使五组实验的 ICBHI Score
   提高 6.21–11.50 pp，主要来自 specificity 恢复，但 sensitivity 同时下降。
   该结果只能标记为 posthoc diagnostic / test-informed，不能覆盖原有 terminal
   result，也不能直接升级为 paper primary result。

<div>

**Project status：At risk but recoverable.** 模型与训练基础已经闭合，
但 paper-facing readout、最终主配置、多种子证据和论文文本必须在 9 月上旬前冻结。
ICASSP 2027 full paper deadline 为 **2026-09-16**。

</div>

## 2. Protocol Closed This Week

| Item | Current contract |
|---|---|
| Shared training core | ICBHI + SPRSound |
| Shared outputs | Level1 Normal/Abnormal, Crackle, Wheeze |
| Unknown labels | Masked; never converted to negative |
| HF Lung | Not used in the present comparison; positive-only auxiliary remains NO-GO |
| KAUH | Not used for training/selection; reserved for patient-level external evaluation |
| Backbone | BEATs iter3+ AS2M, full fine-tuning |
| Input | Mono 16 kHz, 4 s window / 2 s stride |
| Optimization | Seed 42, true native-unit batch 8, 50 epochs, Adam, FP32 |
| Selection | Equal mean of ICBHI and SPRSound validation eligible-node loss |
| Terminal evaluation | Separately executed after validation selection |

The 4 s / 2 s policy replaces the earlier provisional 2 s / 1 s reference. It
is a current engineering/scientific reference, not an optimality claim.

## 3. Completed Full-Fine-Tuning Matrix

| ID | Normalization | Augmentation | Loss | Selected epoch |
|---|---|---|---|---:|
| R0 | None | None | CE + BCE | 2 |
| N1 | Peak | None | CE + BCE | 2 |
| A1 | Peak | Gain + noise | CE + BCE | 2 |
| L1 | Peak | Gain + noise | Focal, gamma = 2.0 | 6 |
| L2 | Peak | Gain + noise | Class-balanced, beta = 0.9999 | 1 |

All five runs completed 50 epochs / 70,200 updates and produced validation
selection, run summary, terminal predictions and native metrics. They are local
single-seed controlled results rather than paper-faithful reproductions or
multi-seed claims.

## 4. Current Terminal Results: Hierarchical Readout（Posthoc Diagnostic）

### 4.1 ICBHI official flat4

| ID | Sp | Se | ICBHI Score | Macro-F1 | UAR |
|---|---:|---:|---:|---:|---:|
| R0 | **67.89** | 42.91 | **55.40** | 45.35 | **52.01** |
| N1 | 63.52 | 44.44 | 53.98 | 44.23 | 49.62 |
| A1 | 65.17 | 42.14 | 53.65 | 43.63 | 49.45 |
| L1 | 49.97 | **58.45** | 54.21 | **46.59** | 51.42 |
| L2 | 58.83 | 42.31 | 50.57 | 41.62 | 49.06 |

### 4.2 SPRSound BioCAS2022 inter Task1-1

| ID | Sp | Se | Official Score | Macro-F1 | UAR |
|---|---:|---:|---:|---:|---:|
| R0 | 91.44 | 92.80 | 92.12 | 90.13 | 92.12 |
| N1 | 91.83 | **95.63** | **93.71** | **91.44** | **93.73** |
| A1 | 91.83 | 92.80 | 92.31 | 90.44 | 92.31 |
| L1 | 84.62 | 97.17 | 90.68 | 86.35 | 90.89 |
| L2 | **92.79** | 93.32 | 93.05 | 91.41 | 93.05 |

## 5. Comparison With Last Week and Literature

### 5.1 Against last week's BEATs HF-off reference

| Readout | ICBHI Score | ICBHI Sp | ICBHI Se | SPRSound Score |
|---|---:|---:|---:|---:|
| Last week BEATs HF-off, 2 s / 1 s cached hierarchy | 49.03 | 52.69 | 45.37 | 91.07 |
| This week N1, 4 s / 2 s full fine-tuning, bits-only terminal decoder | 45.03 | 28.37 | 61.68 | 93.71 |
| Difference | -4.00 | -24.32 | +16.31 | +2.64 |

This is not an apples-to-apples model comparison because encoder scope, window
geometry and flat4 decoder changed together. The table is useful for locating the
failure mode: the new training preserves high abnormal sensitivity and improves
SPRSound, but the bits-only terminal decoder destroys Normal specificity.

### 5.2 ICBHI literature-facing comparison

Comparison contract: official recording split 60/40, 2,756-cycle test, flat4,
and ICBHI Score = (Sp + Se) / 2.

| Work | Sp | Se | ICBHI Score | Evidence boundary |
|---|---:|---:|---:|---|
| Bae23 AST + CE | 77.14 | 41.97 | 59.55 | Paper Claim, five-run mean |
| Bae23 Patch-Mix CL | 81.66 | 43.07 | 62.37 | Paper Claim, five-run mean |
| Jeong25 BEATs + CE | 78.77 | 48.21 | 63.49 | Paper Claim, five-run mean |
| Jeong25 BEATs + PAFA | 82.05 | 47.63 | 64.84 | Paper Claim, five-run mean |
| Ours N1, bits-only decoder | 28.37 | 61.68 | 45.03 | Local single-seed result |
| Ours L1, bits-only decoder | 29.13 | 66.86 | 48.00 | Local single-seed result |
| Ours R0, hierarchical decoder | 67.89 | 42.91 | 55.40 | Posthoc diagnostic / test-informed |
| Ours N1, hierarchical decoder | 63.52 | 44.44 | 53.98 | Posthoc diagnostic / test-informed |
| Ours L1, hierarchical decoder | 49.97 | 58.45 | 54.21 | Posthoc diagnostic / test-informed |

The hierarchical diagnostic narrows the gap to the directly relevant BEATs + CE
reference from 15.49–18.46 points under the bits-only decoder to 8.09 points for
R0, 9.28 points for L1 and 9.51 points for N1. It therefore removes a major
decoder-induced artifact but does not close the native-task performance gap.
The posthoc result must be reviewed with the advisor before the paper-facing
readout is frozen.

### 5.3 SPRSound literature-facing comparison

Comparison contract: BioCAS2022 official inter, 41 unseen patients, 355
recordings and 1,429 events; Task1-1 Normal/Adventitious.

| Work | Sp | Se | Official Score | Evidence boundary |
|---|---:|---:|---:|---|
| Zhang22 MFCC + Naive Bayes official benchmark | 79.04 | 75.83 | 77.42 | Paper Claim |
| Ours N1 | 91.83 | 95.63 | 93.71 | Local single-seed result |

N1 is 16.29 points above the official benchmark under the exact Task1-1 inter
contract. This is not an absolute SPRSound SOTA claim because later work often
mixes inter/intra/total test, different label mappings or target-supervised
adaptation.

## 6. Why ICBHI Specificity Is Low

N1 contains 1,579 Normal cycles in the official test. The bits-only decoder
classifies only 448 as Normal and sends 1,131 Normal cycles to abnormal classes:
795 Crackle, 241 Wheeze and 95 Both. The validation-selected Crackle threshold is
0.1487, so a weak Crackle activation is sufficient to override an otherwise
correct Level1 Normal decision.

This behavior exposes a decoder/ontology mismatch:

- The model is trained with a Level1 Normal/Abnormal node.
- SPRSound Task1-1 uses that Level1 node directly and performs well.
- The historical ICBHI terminal decoder ignores Level1 entirely and reconstructs
  flat4 only from Crackle/Wheeze bits.

An earlier same-prediction diagnostic using a Level1 gate raised N1 specificity
from 28.37 to 63.52 and Score from 45.03 to 53.98, while sensitivity decreased
to 44.44. This confirms that readout design controls a large part of the apparent
ICBHI failure, but the result is test-informed and remains diagnostic only.

## 7. Completed Hierarchical Readout Diagnostic

The frozen contract is:

1. Level1 Normal produces final Normal.
2. Level1 Abnormal is refined by validation-selected Crackle and Wheeze
   thresholds into Crackle, Wheeze or Both.
3. If neither attribute crosses its threshold, choose the larger
   probability-minus-threshold margin; Crackle wins an exact tie.

The local evaluation task recomputed all five ICBHI conditions from saved
prediction artifacts without model loading, retraining or new test inference.

| ID | Hierarchical Sp | Hierarchical Se | Hierarchical Score | Score change vs bits-only |
|---|---:|---:|---:|---:|
| R0 | **67.89** | 42.91 | **55.40** | **+11.50** |
| N1 | 63.52 | 44.44 | 53.98 | +8.95 |
| A1 | 65.17 | 42.14 | 53.65 | +7.87 |
| L1 | 49.97 | **58.45** | 54.21 | +6.21 |
| L2 | 58.83 | 42.31 | 50.57 | +8.37 |

The decoder increases specificity, Score, Macro-F1 and UAR for all five runs,
while decreasing sensitivity for all five. R0 has the highest hierarchical
Score; L1 retains the highest hierarchical sensitivity and macro-F1. N1 remains
the best SPRSound condition, but it is no longer a metric-independent unified
winner once the hierarchical ICBHI readout is considered.

The independent artifacts are:

- `HIERARCHICAL_READOUT_POSTHOC_4s2s_repaired_seed42.md`
- `HIERARCHICAL_READOUT_POSTHOC_4s2s_repaired_seed42.json`

They remain separate posthoc diagnostic artifacts and do not overwrite the
historical bits-only results.

## 8. Current Scientific Interpretation

1. Normalization, augmentation and loss are not interchangeable improvements.
   Peak normalization is the most stable intervention under the historical
   bits-only readout, but the hierarchical diagnostic does not identify N1 as a
   metric-independent winner.
2. Focal loss shifts the operating point toward abnormal sensitivity and is
   useful as an ablation, but it is not a unified default.
3. SPRSound is no longer the blocking dataset. The paper risk is ICBHI native
   retention and the interpretation of the unified hierarchy.
4. The current evidence supports BEATs as the working backbone package; it does
   not prove that BEATs is universally superior to AST/PANNs.
5. No further broad encoder, augmentation or loss sweep should begin before the
   readout and paper claim are frozen.

## 9. Decisions Required

- [ ] Does the hierarchical Level1-gated ICBHI readout match the intended
  unified-system story and advisor expectation?
- [ ] Select R0 or N1 as the paper main configuration? R0 has the highest
  hierarchical ICBHI Score; N1 has the best SPRSound Score and the strongest
  earlier cross-dataset validation profile. Retain L1 as the ICBHI recall/loss
  ablation?
- [ ] Freeze the method after the readout decision and run only multi-seed
  confirmation, external evaluation and reviewer-critical ablations?
- [ ] Keep HF positive-only auxiliary as NO-GO and KAUH as descriptive external
  evaluation for this submission?
- [ ] If the hierarchical readout remains substantially below the direct BEATs
  literature reference, narrow the claim to unified cross-dataset modeling and
  analysis rather than native-task SOTA?

## 10. ICASSP Deadline Plan

Official full-paper deadline: **2026-09-16**. The official call permits four
technical pages plus an optional fifth page containing references only.

### 8/28 大组组会：5 分钟汇报结构

1. **0:00–0:30｜研究问题与当前协议**：ICBHI + SPRSound Core-2、BEATs
   full fine-tuning、shared hierarchy、16 kHz、4 s / 2 s。
2. **0:30–1:20｜已完成与尚未完成**：展示 R0/N1/A1/L1/L2 的单变量
   对照；明确 RMS、waveform z-score、spectrogram mean-variance
   normalization 与 matched 1/2/3/4 s end-to-end comparison 尚未完成。
3. **1:20–2:20｜当前结果**：使用本报告 Section 4 的 ICBHI hierarchical
   diagnostic 与 SPRSound Task1-1 表，说明各自 evidence boundary。
4. **2:20–3:10｜结果解释**：Peak normalization 没有形成跨数据集一致
   改善；focal loss 提高 ICBHI sensitivity 但牺牲 specificity；旧 bits-only
   decoder 放大了 ICBHI Normal false positive。
5. **3:10–4:00｜Window 风险**：4 s 输入对短 ICBHI cycle 可能产生较多
   padding；需要 Wade 的 feature separability 与 lead-owned end-to-end evidence
   共同决定最终 window policy。
6. **4:00–5:00｜未来 48 小时**：关闭 Wade/Hanlin 的第一份数值结果，冻结
   下一组 normalization 比较和论文必须实验；向组内征求对 dataset conflict、
   normalization 与 window policy 的建议。

组会上不得把 student-reported “AST finished”、未闭合的 1/2/3/4 s 实验，
或 posthoc hierarchical readout 描述成正式论文结果。

### 8/27–8/28: Result and method closure

- Complete and audit the hierarchical readout posthoc table.
- Freeze the paper-facing output contract, primary metric and main condition.
- Update the result matrix with exact evidence labels: Paper Claim, Local Result,
  posthoc diagnostic and HOLD.
- Start the paper skeleton immediately: problem, dataset/ontology, method,
  experiment protocol, result-table placeholders and limitations.

### 8/29–8/30: Advisor decision gate

- Confirm whether the hierarchy/readout is scientifically acceptable.
- Confirm N1 main / L1 ablation roles and the minimum paper contribution.
- Freeze the must-have experiment list. Any new axis after this date requires a
  direct reviewer-facing reason and a named table/figure destination.

### 8/31–9/3: Core evidence freeze

- Run multi-seed confirmation for the selected main method and the minimum
  matched reference set after approval.
- Produce ICBHI confusion/per-class table and SPRSound Task1-1 table.
- Close KAUH patient-level external descriptive evaluation if it is retained in
  the paper.
- Freeze headline numbers by 9/3. Do not start new encoder sweeps after this gate.

### 9/4–9/7: Full first draft

- Complete Method, Experimental Setup and Results from frozen artifacts.
- Prepare one pipeline figure, one main comparison table and one compact ablation
  table; move secondary diagnostics to supplementary material if permitted.
- Complete Related Work and write limitations around two-dataset shared training,
  posthoc development history and single-/multi-seed evidence.
- Deliver a complete advisor-readable draft by 9/7.

### 9/8–9/10: Advisor review and targeted repair

- Resolve claim, metric and experiment objections.
- Run only targeted repairs that directly close a named review comment.
- Freeze title, abstract, contribution bullets and conclusion.

### 9/11–9/12: Content freeze

- Finalize all numbers, captions, citations and claim-evidence mapping.
- Independently check split descriptions, sample counts, metric formulas and
  table consistency.
- No new scientific scope after 9/12.

### 9/13–9/15: Submission buffer

- IEEE formatting, four-page technical-content limit, reference page, PDF and
  submission-system metadata.
- Final advisor/coauthor approval and submission no later than 9/15, preserving
  one calendar day for upload or formatting failures.

### 9/16: Official deadline

- Emergency buffer only; no planned model changes or new experiments.

## 11. Scope Control

### Must-have

- Frozen hierarchical output contract and independently recomputed metrics.
- One defensible main configuration, one matched reference and compact ablations.
- Multi-seed evidence for headline claims.
- Exact ICBHI and SPRSound native-compatible comparisons.
- Paper skeleton now, full draft by 9/7.

### Optional if completed by 9/3

- KAUH patient-level external descriptive table.
- One frozen-vs-full or window-policy analysis that directly explains ICBHI.
- Wade's acoustic analysis as mechanism evidence.

### Out of scope for this submission unless the advisor reopens it

- New encoder family sweep.
- HF positive-only shared training.
- New dataset acquisition as a critical dependency.
- Large single-dataset baseline matrix by project management.
- Natural speech, tele-doctor or personalized healthcare assistant extensions.

## 12. Owners and Dependencies

| Owner | Scope | Critical-path status |
|---|---|---|
| Zilong / project management | Readout decision, main experiments, result audit, paper writing | Critical |
| Local evaluation task | Saved-prediction hierarchical readout recomputation | Critical until closed |
| Wade | Window-level energy/PSD/silence/event-duration analysis | Supporting; must not block paper core |
| Hanlin | Single-dataset native baselines | Independent student lane; not current critical path |
| Jingping | Method/readout and paper-scope decision | Critical decision gate |

## 13. Evidence Boundary

- Section 4.1 的 ICBHI 数字来自已保存 prediction 的 hierarchical posthoc
  diagnostic / test-informed readout，不是新的 primary terminal result。
- Section 4.2 的 SPRSound Task1-1 数字是已经完成的 local single-seed terminal
  result；历史 bits-only ICBHI terminal numbers 仅保留在诊断和对照段落中。
- The hierarchical recomputation remains posthoc diagnostic / test-informed
  until a new formal evaluation protocol is approved and completed.
- Paper numbers from prior work remain Paper Claims.
- No result in this report constitutes an absolute SOTA claim.

## 14. 2026-08-29 Overnight Preparation Addendum

### Completed preparation

- 建立 ICASSP 2027 四页英文 paper skeleton；当前主叙事是 cross-dataset
  inconsistency 与 ontology/readout analysis，不是 uniform improvement 或 SOTA。
- 完成 BEATs、PAFA BEATs+CE、PAFA、fixed-BEATs、SPA、Resp-Agent 与 OPERA
  的 primary-source comparison/crosswalk。PAFA BEATs+CE 是 deadline-critical
  direct comparator。
- 完成 PAFA BEATs+CE 双证据入口：`clean_validation_only` 只用 official-train
  内 patient-grouped validation 选 checkpoint，selected checkpoint 后才首次读
  official test；`author_test_selected` 保留作者每 epoch official-test Score 选模，
  永久标为 test-selected reproduction receipt。
- 完成 Core-2 checkpoint-selection sensitivity。现有五条均保留原 selected
  checkpoint；epoch-1-normalized mean/worst 仅为 design diagnostic，缺失
  alternative checkpoint 时保持 `HOLD`。
- 批准并完成 Audacity 20-selection panel 的 native-rate quantitative script
  measurement 与 import/label plan；`.aup3` GUI projects 仍为 `HOLD`。HF empty
  annotation 只表示 `not_annotated`，不表示 Normal/Negative。
- Wade/Hanlin 澄清消息已形成草稿但未发送。两人的状态仍是
  `PARTIALLY_ALIGNED`，不能把计划或澄清问题记为数值交付。

### Authorized server queue

1. 先运行 PAFA BEATs+CE `clean_validation_only`, seed 42，形成主要
   paper/local/delta comparator。
2. 再运行 `author_test_selected`, seed 42，只形成作者忠实复现收据。
3. 两个 mode 使用独立目录；每 epoch 保存 model+classifier checkpoint 与
   prediction；完整 BEATs 权重仅在 selection Score 改善时覆盖保存为
   `best_checkpoint.pt`，不保存 Adam state。每个 mode 建议预留 2 GB，
   代码、数据和 pretrained checkpoint 均直接复用。
4. 单 seed 结果审阅后，才决定是否扩展作者 seeds 1–5 或进入 PAFA loss。

### Immediate decision after overnight results

- 如果 clean BEATs+CE 接近论文 63.49，则当前 Core-2 ICBHI gap 主要优先归因于
  direct flat4 objective、5 s cycle geometry 与 joint hierarchy/readout contract；
  不需要先加入 PAFA loss。
- 如果 author-faithful receipt 接近论文而 clean track 明显更低，则必须把
  test-selection optimism 与 inner-validation sample reduction 作为主要 protocol
  gap 单列，不能把差值归因于模型方法。
- 如果两条都明显偏低，先检查 paper/local protocol parity；不启动新的 broad
  encoder/loss/window sweep。

Official deadline source: [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/).
