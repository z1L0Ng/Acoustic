# 2026-07-24 respiratory acoustic primary-source novelty audit

日期：2026-07-24

范围：本报告服务当前 Phase-1 方向，即 multi-dataset respiratory acoustic learning、long-tail/imbalance、domain shift，以及 ICBHI、SPRSound BioCAS2022、HF_Lung_V1、KAUH/Fraiwan v3 四数据集下的可辩护 protocol。报告只做论文与协议审计，不运行训练，不修改 baseline/model/dataset/raw。

## 0. 必须纳入的当前项目证据

### 0.1 Source facts 与本地实验事实

| 证据 | 当前状态 | 对结论的约束 |
|---|---:|---|
| Patch-Mix 作者 checkpoint 在 ICBHI official split 上精确对齐作者结果：Score 62.1708，Sp 86.1938，Se 38.1478。 | 已由 `result/icbhi_patchmix_author_eval/metrics.json` 验证；该 checkpoint 是 official-test-selected。 | 只能作为 published-model exploratory evidence，不能作为 clean source generalization checkpoint。 |
| ICBHI -> SPRSound BioCAS2022 inter，zero target tuning：Patch-Mix 59.38、PAFA 55.82、SG-SCL 59.98 binary Score。 | 已由 `result/sprsound_patchmix_frozen_transfer/`、`result/pafa_sprsound_transfer_20260722_235659/`、`result/sg_scl_sprsound_transfer_20260722_235659/` 验证。 | 只比 all-normal/trivial Score 50 高约 5.8-10.0 pp；不能用 ICBHI source Score 减 SPRSound target Score 直接定义 degradation。 |
| 同一 SPRSound patient-disjoint inter split，冻结 encoder + 5 epochs target heads：Patch-Mix 87.56、PAFA 84.25、SG-SCL 90.58 binary Score。 | 已由 `result/sprsound_*_frozen_encoder_target_heads/metrics.json` 验证。 | 这是 target adaptation/reference，不是 zero-shot；说明 target split 在少量 target supervision 下可学习。 |
| Seven-class target heads：Score 82.37/75.77/84.41，但 macro-F1 只有约 0.36-0.39；narrow4 Score 76.94-85.26。 | 已由本地 metrics 验证。 | Aggregate Score 主要受 Normal/Wheeze 支撑；SPRSound inter split 的 Both support=1，不能做 Both/minority 结论。 |
| 四数据集声学分布审计：dataset-ID linear probe balanced accuracy 0.935；SPRSound 行为 336/336 完全识别。 | 已由 `result/acoustic_distribution/analysis_manifest.json` 与 `dataset_id_probe_confusion.csv` 验证。 | target-head/adaptation 高分可能利用 dataset/device/acquisition shortcut，必须正面讨论。 |

### 0.2 Proposed benchmark policy，不是官方 source facts

四数据集 benchmark 必须保留 raw labels 和 source metadata，不把我们提出的 harmonized labels 写成官方事实。ICBHI、SPRSound BioCAS2022、HF_Lung_V1、KAUH/Fraiwan v3 是当前 Phase-1 数据集；共享任务、排除标签、partial-label masks、multi-head 设计都属于 proposed policy，而不是 source datasets 自带定义。

## 1. 总体结论

- 没有发现一篇已审计论文完整覆盖我们的 first-stage 方向，但前提是我们的 framing 必须收窄为：audio-first auscultation benchmark/training framework，明确区分 source-only transfer、target-head adaptation、target-native reference，并显式处理 heterogeneous labels、long-tail 与 domain shift。
- 不能再声称“第一个 respiratory acoustic benchmark”“第一个 cross-dataset respiratory sound study”“第一个 ICBHI->SPRSound OOD evaluation”。这些都已有近邻工作。
- 最强近邻威胁是 LungMix、QLung、BTS-CARD、Resp-Agent/Resp229k。它们分别覆盖 cross-dataset augmentation、quality/imbalance margin、metadata shortcut debiasing、大规模 multimodal disease-level benchmark。
- OPERA 是最接近的 respiratory foundation model benchmark，但它主要是 frozen linear probe 的 19 个 dataset-specific downstream tasks；它没有覆盖 ICBHI event head 到 SPRSound event labels 的统一 source-only transfer protocol。
- LungMix 是最接近的 audio-only cross-dataset respiratory sound generalization 工作，但它主要报告 aggregate Score，缺少明确的 missing-label policy、target-native reference、clean source selection audit 和 per-class/tail evidence。
- QLung 与 BTS-CARD 都直接使用 ICBHI IND / SPRSound OOD setting。QLung 会限制我们对 quality + imbalance + OOD margin 的 novelty claim；BTS-CARD 会限制任何 metadata/device/dataset-ID adapter 方向的 novelty claim。
- Resp-Agent/Resp229k 不能被低估，但它是 disease-level multimodal generation/diagnosis，不是 auscultation event/cycle heterogeneous-label protocol。它限制 broad aggregation claim，不直接覆盖我们的 event-level audio-first protocol。
- 最安全的 contribution 叙事是：我们提供一个 source-grounded、audio-first、可复现实验矩阵，用统一但不伪装为官方的 label policy，把 source-only、target adaptation、target-native reference、partial-label objective、domain shortcut audit 放在同一个 respiratory auscultation framework 中比较。

## 2. 论文身份与 primary-source gate

| Work | 审计状态 | Primary source | Code/data 状态 | 与我们关系 |
|---|---|---|---|---|
| OPERA: Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking | NeurIPS 2024 Datasets and Benchmarks Track | NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f803abdcad9de35b45d5a656dade45c-Abstract-Datasets_and_Benchmarks_Track.html；official repo: https://github.com/evelyn0414/OPERA | 官方 MIT repo，repo 内链接模型与 benchmark。 | foundation benchmark / component threat。 |
| Lungmix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification | ICASSP 2025 / arXiv 2501.00064 | https://arxiv.org/abs/2501.00064 | 本轮未找到完整官方 train/eval repo。 | 最接近的 audio-only cross-dataset DG/mixup 工作。 |
| Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis | arXiv 2602.15909；arXiv 标注 published as ICLR 2026 conference paper；OpenReview 页面存在但浏览器验证阻塞直接读取 | https://arxiv.org/abs/2602.15909；GitHub https://github.com/zpforlove/Resp-Agent；HF models https://huggingface.co/AustinZhang/resp-agent-models | code、dataset、weights 均有链接；HF model repo 标 MIT，但权重页说明偏 academic research use。 | 大系统近邻；不是 event-label 直接覆盖。 |
| QLung: Quality Adaptive Angular Margin Learning for Respiratory Sound Classification | arXiv 2606.11915；arXiv 标注 accepted to Interspeech 2026 | https://arxiv.org/abs/2606.11915 | 论文列出 https://github.com/RSC-Toolkit/QLung，但本轮 live GitHub 返回 404。 | quality + imbalance + OOD 的 near-direct method threat。 |
| BTS-CARD: Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial Debiasing for Out-of-Distribution Robustness | arXiv 2510.22263 v2；ICASSP 2026 program 可核验 | https://arxiv.org/abs/2510.22263；ICASSP program https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4546；official repo https://github.com/RSC-Toolkit/BTS-CARD | 官方 Apache-2.0 repo，包含脚本、preprocessed link、checkpoint link。 | metadata/device shortcut 与 OOD robustness 的 near-direct threat。 |
| PC-MCL: Patient-Consistent Multi-Cycle Learning with multi-label bias correction for respiratory sound classification | arXiv 2601.17080 preprint | https://arxiv.org/abs/2601.17080 | 本轮未找到官方 repo。 | ICBHI 内部 label-formulation reference。 |
| Schutera et al., Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels | PLOS ONE 2022 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263656；OSF https://osf.io/uyk79/ | OSF 提供 code/data。 | heterogeneous/partial labels 方法参考。 |
| Sanap et al., Beyond Classification: A Cough Regression Benchmark for Respiratory Acoustic Foundation Models | arXiv 2606.15436；accepted ICML 2026 Workshop on Structured Data for Health | https://arxiv.org/abs/2606.15436；Centific page https://www.centific.com/publications/beyond-classification-a-cough-regression-benchmark-for-respiratory-acoustic-foundation-models | 本轮未找到 code repo。 | cough regression adjacent benchmark，不是 auscultation event classification。 |

纠错：此前候选 `2606.15427` PDF 是 spacecraft VLM 论文，不是 Sanap cough regression。正确 arXiv ID 是 `2606.15436`。

## 3. 逐篇深读与对我们主线的影响

### A. OPERA

准确身份：Yuwei Zhang, Tong Xia, Jing Han, Yu Yvonne Wu, Georgios Rizos, Yang Liu, Mohammed Mosuily, Jagmohan Chauhan, Cecilia Mascolo。NeurIPS 2024 Datasets and Benchmarks Track。

| 问题 | Primary-source evidence |
|---|---|
| 使用哪些 respiratory datasets？ | 论文 Section 4 与 Appendix dataset table：pretraining 使用 5 个开放 respiratory audio resources；benchmark 涵盖 COVID-19 Sounds、UK COVID-19、COUGHVID、ICBHI、HF Lung、Coswara、KAUH、Respiratory@TR、SSBPR、MMlung、NoseMic 等。 |
| 如何统一 preprocessing？ | Section 4：所有 recordings resample 到 16 kHz mono；转 64-bin Mel spectrogram；64 ms Hann window，32 ms shift。 |
| 是否统一 task/head？ | Section 5.1：不是统一一个 shared auscultation label head，而是 19 个 downstream tasks，每个 task 训练 frozen encoder 上的 single fully connected linear probe。 |
| 19 tasks 如何组织？ | Table 2：T1-T10 是 binary classification，T11-T12 是 5-class classification，T13-T19 是 regression；分类用 AUROC，回归用 MAE，总体用 MRR。 |
| split 怎么做？ | 有官方 split 的 Tasks 1-4 和 12-18 使用官方 split；Tasks 5-11 和 19 用 random participant-independent split；T13-T19 因 subject 数小使用 LOSO。 |
| cross-dataset/generalization 如何评估？ | Appendix A.4.3 Table 13：cross-domain zero-shot 是 train linear probe on source Task A, test target Task B；示例 T6->T9、T7->T10。OPERA-CT AUROC 分别为 0.600 与 0.823。 |
| 是否有 minority/tail 证据？ | 主要是 AUROC/MAE/MRR；本轮未找到 ICBHI/SPRSound event-level per-class tail recall 表。 |

对我们主线的影响：

OPERA 已经覆盖“open respiratory acoustic foundation model + benchmark”这一大方向，所以我们不能把 novelty 写成 broad FM benchmark。它没有解决的问题是：auscultation event/cycle label 的跨数据集统一协议、source-only ICBHI->SPRSound event transfer、target-native reference、partial-label/missing-label objective、per-class long-tail diagnostics。因此 OPERA 应该作为 foundation baseline 和 benchmark 近邻，而不是直接否定我们方向。

### B. LungMix

准确身份：Shijia Ge, Weixiang Zhang, Shuzhao Xie, Baixu Yan, Zhi Wang。ICASSP 2025 / arXiv 2501.00064。

| 问题 | Primary-source evidence |
|---|---|
| 使用哪些数据集？ | Section III-A：ICBHI、SPRSound、HF。ICBHI 是 respiratory cycle；SPRSound 使用 event level；HF 有 inhale/exhale 和 abnormal labels，但缺少 both annotations。 |
| label mapping 怎么做？ | Section III-A：统一成 ICBHI 四类 normal/crackle/wheeze/both；coarse crackle 与 fine crackle 归为 crackle；stridor 与 rhonchus 归为 wheeze。 |
| fine/coarse crackle 如何处理？ | 明确合并为 crackle。 |
| mixup label interpolation 是什么？ | Section II：LungMix 对 waveform 做 random/loudness-mask mixing；final label 通过 bitwise OR 合并，不是普通 fractional interpolation。 |
| 是否兼容 heterogeneous/missing labels？ | 论文没有完整说明。HF 缺少 both annotations，但论文没有明确 missing both 是 masked/unsupported，还是被当成 negative。必须标 unknown。 |
| unseen dataset 是否不参与训练/selection？ | 论文设定为 single source domain 与 unseen target domains；但 Section III-C 说明他们在 ICBHI train/test 时没有选 best ICBHI，而是选择 best COMB result。这说明 COMB-best 的 selection boundary 需要谨慎，不能直接当作 pure source-only selection。 |
| 报告数字是什么？ | Table I：source ICBHI w/o Mixup 在 ICBHI/SPR/HF/COMB 上为 52.88/65.48/61.71/60.79；source ICBHI Lungmix+Patchmix 为 52.82/66.65/63.52/63.06；source HF Lungmix+Patchmix 为 55.75/69.37/77.76/72.08。 |
| 是否有 tail evidence？ | Table II 只给 COMB 上 Sp/Se；未找到 crackle/wheeze/both support/recall/confusion 的完整表。 |

对我们主线的影响：

LungMix 是真正的 near-direct application reference。它已经展示 ICBHI/SPR/HF 间跨数据集 generalization 与 aggregate Score 改善。它没有覆盖我们想做的完整 protocol：明确 source-only selection、clean-source checkpoint、target-native reference、partial-label mask、missing-label 不当 negative 的风险、per-class tail evidence。因此我们不能把“跨数据集 RSC”当 novelty；可以把 novelty 收到“可审计 protocol + heterogeneous-label/tail diagnostics”上。

### C. Resp-Agent / Resp229k

准确身份：Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu。arXiv 2602.15909；论文文本标注 Published as a conference paper at ICLR 2026。OpenReview 页面存在但本轮直接读取被浏览器验证阻塞；arXiv 与 HF model card 可作为当前证据。

| 问题 | Primary-source evidence |
|---|---|
| 数据集与模态 | Table 1 / Appendix：Resp-229k 来自 UK COVID-19、ICBHI、SPRSound、COUGHVID、KAUH；HF Lung V1 只用于 initialization，不包含在 Resp-229k 中。 |
| 标签空间 | Appendix Table 5：原始 20-class disease labels 合并为 16-class taxonomy，包括 Control Group、COVID-19、Pneumonia、COPD、Asthma 等；238,074 raw clips 过滤后 229,101 quality-controlled clips。 |
| 任务 | ICBHI official 4-class classification；Resp-229k 16-class disease diagnosis；controllable respiratory sound generation。 |
| cross-domain split | Section 3 与 Ethics/Data Provenance：train/validation 使用 ICBHI、SPRSound、UK COVID-19；Test-CD 只用 KAUH 与 COUGHVID。另有 LoSO across five sources。 |
| long-tail evidence | Table 3：Test-CD no-synthesis Macro-F1 0.212 / Macro-F1tail 0.074；class-prior 0.512 / 0.349；Thinker-A2 CA at B=50k 达到 0.598 / 0.421；LoSO Thinker-A2 CA 为 0.532 / 0.383。 |
| ICBHI result | Table 2：Resp-Agent 在 ICBHI official 4-class 上 Score 72.70，Sp 79.29，Se 66.10；pretraining data 标 HF+SPR。 |
| code/data | Reproducibility statement：GitHub、HF dataset、HF model weights 均公开链接。 |

对我们主线的影响：

Resp-Agent 是 broad system threat，尤其会限制我们对“聚合 public respiratory datasets”“long-tail respiratory disease benchmark”“generation-based rebalancing”的任何宽泛 claim。边界也很清楚：它是 disease-level multimodal generation/diagnosis，依赖 metadata-derived clinical narratives，不是 audio-only auscultation event/cycle label transfer。我们的论文如果定位为 respiratory event/cycle heterogeneous-label benchmark 和 protocol，则不是直接重叠；如果写成“我们构建第一个 multi-dataset respiratory benchmark”，就会被 Resp-Agent 和 OPERA 直接挑战。

### D. QLung

准确身份：Yoon Tae Kim, Heejoon Koo, Miika Toikkanen, June-Woo Kim。arXiv 2606.11915；arXiv 标注 accepted to Interspeech 2026。

| 问题 | Primary-source evidence |
|---|---|
| severe imbalance 如何建模？ | Section 2.3：用 log-scaled inverse class-frequency angular margin，避免 tail margin 爆炸。 |
| quality signal 如何建模？ | Section 2.2：Audio Quality Score 由 spectral entropy 与 RMS energy 组成；高质量样本更大 margin，低质量样本更小 margin。 |
| angular margin 如何联合？ | Section 2.4：dual-factor angular margin，`m_d = gamma m_q + (1 - gamma)m_c`，配合 angular classifier。 |
| 数据集与 mapping | Section 3.1：ICBHI official 60/40 split；SPRSound seven labels 映射到 ICBHI 四类。coarse/fine crackle -> crackle；stridor/rhonchi -> wheeze。SPRSound 总分布 normal 6199、crackle 1044、wheeze 811、both 31。 |
| training details | Section 3.1：LR 5e-5，batch 8，50 epochs，five seeds；lambda=0.4、gamma=0.5、m_target=0.2、s_a=37、s_d=15、kappa=0.5。 |
| ICBHI result | Table 1/3：AST CE 59.55 -> QLung AST 62.01±1.18；Audio-CLAP 62.56 -> QLung Audio-CLAP 63.39±0.40。 |
| ICBHI->SPRSound OOD result | Table 2：Patch-Mix OOD 51.01，SG-SCL 51.84，Audio-CLAP 56.29，BTS 53.42，QLung AST 58.23±3.83，QLung Audio-CLAP 59.80±3.51。 |
| per-class/tail evidence | Figure 3 文字说明：normal +1%、wheeze +6%、both +5%、crackle -8%；减少 normal 被误判为 crackle（-6%）和 both 被混淆为 crackle（-8%）。未找到完整 numeric confusion table。 |
| repo | 论文列出 `https://github.com/RSC-Toolkit/QLung`，但本轮 live URL 返回 404。 |

对我们主线的影响：

QLung 是最直接的 quality + imbalance + OOD method threat。它已经把 ICBHI IND / SPRSound OOD、class imbalance、quality-aware margin 放在同一篇里。我们不能再把“用 margin 处理 imbalance 并提升 OOD”当作宽泛 novelty。仍可贡献的空间在：clean source selection、target-head/target-native references、公平 comparator、partial-label objective、per-class support-aware tail evidence，以及四数据集声学 shortcut analysis。

### E. BTS-CARD

准确身份：Heejoon Koo, Miika Toikkanen, Yoon Tae Kim, Soo Yong Kim, June-Woo Kim。arXiv 2510.22263 v2；ICASSP 2026 program 可核验。

| 问题 | Primary-source evidence |
|---|---|
| metadata shortcut 问题 | Abstract/Intro：age、sex、acquisition device 会产生 spurious correlations；不同 clinical sites、stethoscopes、protocols 下 generalization 下降。 |
| causal/counterfactual 机制 | Section 2.2：将 metadata 影响拆成 spurious direct path `t -> Y` 与 informative indirect path `(a,t) -> m -> Y`。 |
| adversarial debiasing | Section 2.3：对 device/location 使用 adversarial discriminator + GRL，目标是 metadata-insensitive representation。 |
| counterfactual metadata augmentation | Section 2.3：把敏感 metadata 替换为 neutral placeholders，例如 age unknown，而不是简单删除。 |
| 数据与 mapping | Section 3.1 / Table 1：ICBHI 为 IND，SPRSound 为 OOD；SPRSound seven labels 合并到四类；inter-patient validation set only for OOD test。 |
| training details | Section 3.1：8 s cycles，48 kHz，metadata text 最多 64 tokens，AdamW 5e-5，cosine，30 epochs，batch 8，five runs；location/device losses 分别 weighted 0.01/0.1。 |
| main result | Table 2：Fine-tuning OOD 51.13，Patch-Mix 51.01，SG-SCL 51.84，Audio-CLAP 56.29，BTS 53.42，BTS-CARD 61.96±1.50；IND 64.63±0.57。 |
| ablation | Table 3/4：去掉 counterfactual、adversarial 或 metadata augmentation 都降低 OOD；location+device debiasing 是主要选择。 |
| per-class/tail evidence | 本轮只找到 aggregate Sp/Se/Score；没有完整 crackle/wheeze/both tail table。 |

对我们主线的影响：

BTS-CARD 对我们的 shortcut concern 非常关键。我们的 acoustic distribution audit 已经显示 dataset-ID balanced accuracy=0.935，SPRSound 几乎完全可分；BTS-CARD 则提供了 respiratory sound 领域内 metadata/device/location shortcut 的直接论文证据。如果我们后续使用 dataset IDs、device metadata、adapters、metadata prompts，必须把 BTS-CARD 作为直接近邻并解释区别。最安全策略是 first-stage 先做 audio-only / audio-first，并把 metadata held out as analysis；metadata-aware robustness 作为 later branch 或 controlled ablation。

### F. PC-MCL

准确身份：Seung Gyu Jeong, Seong-Eun Kim。arXiv 2601.17080 preprint。

| 问题 | Primary-source evidence |
|---|---|
| multi-cycle aggregation | Section 2：两个 respiratory cycles concat，每个先 normalize 到 T/2；repeat padding 或 center cropping；构造 same-class 与 cross-class、intra/cross-patient pairs。 |
| 为什么需要 explicit normal label？ | Section 2.1：传统 two-label `[crackle, wheeze]` 把 normal 定义为 `[0,0]`，当 normal 与 abnormal concat 时 normal information 被擦掉。 |
| 3-label formulation | Section 2.2：用 `[normal, crackle, wheeze]`，concat 后 label 用 element-wise OR；主损失用 BCEWithLogits。 |
| 如何转回 ICBHI four-class？ | Section 2.2：inference 后 deterministic priority rule；crackle+wheeze -> both，其次 crackle/wheeze，若无 abnormal 则 normal。 |
| patient consistency | Section 2.3：auxiliary task 判断两个 cycles 是否来自同一 patient；hard negatives 采样不同 patient 但相同 pathology profile。 |
| protocol | Section 3.1：ICBHI official 60/40 split，16 kHz，128 Mel，25 ms window，10 ms shift，10 s fixed input，five seeds。 |
| result | Table 1：BEATs+PC-MCL Score 65.37±0.73，Sp 79.04±1.90，Se 51.71±2.98；AST+PC-MCL Score 62.30±0.50。 |
| per-class evidence | Figure 3 文字：CE baseline Crackle AP 0.209、Wheeze AP 0.116；multi-label formulation 变为 0.600/0.650；full PC-MCL 0.642/0.663。 |

对我们主线的影响：

PC-MCL 不是 cross-dataset paper，也没有解决 missing-label heterogeneity。它的价值在 label semantics：normal 不是 crackle/wheeze 的简单 complement。我们做 heterogeneous-label objective 时，不能把 normal/background 简化成“所有未标注异常的负例”。如果我们提出 multi-head 或 partial-label loss，PC-MCL 应作为 normal-label formulation 的直接参考。

### G. Schutera et al. 2022 heterogeneous labels

准确身份：Mark Schutera, Luca Rettenberger, Christian Pylatiuk, Markus Reischl。PLOS ONE 2022，DOI 10.1371/journal.pone.0263656。

| 问题 | Primary-source evidence |
|---|---|
| heterogeneous labels 定义 | Introduction：heterogeneous labels 是 partially labeled data；样本可能缺少某个实际存在类别的标签，或只包含部分 class labels。 |
| missing label 风险 | Introduction：把 missing class labels 当作 background 会假设所有未标注内容都是 background，通常不成立。 |
| masking 机制 | Section 2.4：binary label mask vector `m` 表示每个 class 是否有 ground truth mask；Dice loss 只计算 present masks，并按 present mask 数归一化。 |
| objective | Section 2.4.1-2.4.3：combined objective = modified Dice loss + class-asymmetric loss；利用 segmentation 中互斥类别的隐含信息，同时避免直接监督 missing masks。 |
| code/data | PLOS data availability 与 OSF page 提供。 |

对我们主线的影响：

这不是 respiratory paper，但给我们一个关键原则：missing/unobserved label 不应自动当 negative。可迁移的是 label availability mask 与 partial supervision 思路；不能直接照搬的是 segmentation 中 pixel-level mutually exclusive 假设。Respiratory event 里 crackle/wheeze 可以共现，normal 的语义也不同，因此需要 respiratory-specific objective。

### H. Sanap et al. cough regression benchmark

准确身份：Mayur Sanap, Prasanna Desikan, Edgar Lobaton。arXiv 2606.15436；accepted ICML 2026 Workshop on Structured Data for Health。

| 问题 | Primary-source evidence |
|---|---|
| 数据与任务 | Section 2 / Table 1：CIDRZ N=1049，targets 为 age、BMI、X-ray abnormality probability、TB probability；Coswara N=2560，age；CoughVID N=6858，age。 |
| preprocessing | Figure 1 / Section 2：audio resample 到 16 kHz mono，pad/trim 到 2 s。 |
| 模型 | OPERA-CT、OPERA-CE、OPERA-GT、HeAR、M2D+Resp frozen encoders；比较 linear、MLP-small、full MLP heads。 |
| split | CIDRZ 与 Coswara 用 subject-disjoint 64/16/20；CoughVID 用 official UUID-level split。 |
| metric | MAE + MAD mean-predictor baseline；best/MAD 用于判断是否超过 chance/mean predictor。 |
| within-dataset result | Table 3：HeAR 在 Coswara age MAE 9.12 yr；HeAR CIDRZ 结果因可能 pretraining overlap 被排除在 headline claims 外；CIDRZ 多个 target 接近 chance floor。 |
| cross-dataset transfer matrix | Table 5：CoughVID->CIDRZ cross 10.34 vs within 10.51，gap -0.17；Coswara->CIDRZ +0.03；CIDRZ->Coswara +2.43（+26.6%）；CIDRZ->CoughVID +0.94。 |
| low-data | Figure 2 / Section 3.5：HeAR/M2D+Resp 在 N=50/100 附近 plateau；OPERA models 到 N=400 更稳定。 |

对我们主线的影响：

这篇不能作为 lung-sound event classification comparator，因为它是 cough regression。它有用的是评估逻辑：cross-dataset transfer 具有方向性；大而多样的 source 可能迁移到小 clinical target，反向不成立；应把 cross-domain gap 与 target in-domain reference 或 naive baseline 对齐，而不是直接做 raw metric subtraction。这正好支持我们不能用 ICBHI source Score 减 SPRSound target Score 的判断。

## 4. 已有工作覆盖 vs 我们仍可贡献

| 维度 | 已有工作覆盖 | 我们仍可贡献 | Novelty 风险 |
|---|---|---|---|
| Multi-dataset representation | OPERA 建 respiratory FMs 与 19 downstream tasks；Resp-Agent/Resp229k 聚合五类 public corpora。 | ICBHI/SPRSound/HF_Lung/KAUH 的 auscultation event/cycle-level benchmark，强调 unit、label availability、source-only/target-reference protocol。 | 如果说 broad benchmark，风险高；如果限定 event-level auscultation，风险中等。 |
| Cross-dataset protocol | LungMix、QLung、BTS-CARD、Resp-Agent 都有 OOD/cross-dataset setting。 | 一个可复现 comparison matrix，严格区分 source-only、clean-source、frozen target head、target-native、pooled training、adaptation。 | 不能说首次观察 cross-dataset drop；可以说 fair protocol 和 comparator discipline。 |
| Heterogeneous-label objective | Schutera 给 partial-label/masked objective 思路；PC-MCL 给 explicit-normal multi-label 思路。 | 针对 respiratory event labels 的 label-availability masks、shared normal/abnormal head + dataset-specific heads、unsupported-label policy。 | 中等；必须靠实验而不是概念。 |
| Tail learning / imbalance | QLung 做 quality + class-frequency angular margin；Resp-Agent 做 generation-based balancing 并报告 Macro-F1tail。 | support-aware per-class recall/macro-F1；在 heterogeneous labels 下比较 margin/focal/sampling/masked loss；不只报 aggregate Score。 | generic imbalance claim 风险高；heterogeneous-label tail diagnostics 风险较低。 |
| Quality robustness | QLung 明确建模 no-reference quality。 | 把 quality 作为跨数据集 audit/control variable，检验 quality normalization 是否解释 transfer gap。 | quality margin 本身 novelty 高风险。 |
| Metadata shortcut control | BTS-CARD 直接处理 metadata/device/location shortcut；我们的 dataset-ID probe 说明该 concern 真实存在。 | audio-only first-stage；metadata held out as analysis；若做 metadata branch，需 controlled comparison to BTS-CARD。 | 使用 metadata/device adapters 而不 debias 风险高。 |
| Annotation/data curation | Resp-Agent/Resp229k 做 disease taxonomy aggregation 与 clinical narratives；OPERA 做 respiratory data curation。 | 较小但可审计的 raw-vs-harmonized overlay、label mask、event/recording/patient unit contracts。 | 不能把 aggregation 本身当 novelty。 |

## 5. 当前实验能说明什么，不能说明什么

### 能说明

- Strong ICBHI source methods 可以被对齐到 published checkpoint，并在 SPRSound inter 上做 zero-target-tuning 推理。
- Published-model frozen transfer 在 SPRSound binary 上只比 trivial floor 略高。
- 同样 encoders 在 5-epoch target-head adaptation 下能取得很高 SPRSound binary Score，说明 target split 本身可学习。
- 四数据集的 acoustic/device/domain 差异非常强，dataset-ID probe balanced accuracy 0.935，shortcut 不是假设，而是必须解释的现象。

### 不能说明

- 不能证明 clean source-domain generalization failure，因为 Patch-Mix source checkpoint 是 official-test-selected。
- 不能证明 Patch-Mix/PAFA/SG-SCL 表征本身弱，因为 target-head adaptation 很强。
- 不能证明 minority/Both 行为，因为 SPRSound inter Both support=1。
- 不能证明一个新模型解决了 long-tail 或 missing-label semantics。
- 不能用 source Score 减 target Score 作为 degradation。

## 6. 下一步实验，按信息增益排序

| 排名 | 实验 | baseline/control | 变量 | metrics | go/no-go |
|---:|---|---|---|---|---|
| 1 | Clean-source ICBHI -> SPRSound transfer for Patch-Mix/PAFA/SG-SCL。 | 当前 published-model transfer rows 与 all-normal floor。 | 用 source-validation-selected checkpoint 替换 official-test-selected checkpoint。 | SPRSound inter binary Score/UAR、narrow4 macro-F1、per-class recall/support、confusion/calibration。 | 如果 clean-source 仍接近 floor 而 target-head 高，cross-domain degradation 证据成立；若 clean-source 接近 target-head，不能讲 failure story。 |
| 2 | Same-architecture target-native SPRSound references。 | 当前 frozen encoder + 5-epoch target heads。 | 在同一 preprocessing/unit/split 下训练 target-native 或 controlled full reference。 | binary Score、macro-F1、seven-class macro-F1、narrow4、per-class recall、CI。 | 如果 target-native 明显高于 source-only，可解释 target-domain learnability；若差距消失，则 degradation claim 变弱。 |
| 3 | Pooled ICBHI+SPRSound(+HF) partial-label/missing-label objective ablation。 | missing-as-negative 与 simple four-class remap。 | label-availability masks、shared normal/abnormal head + dataset-specific heads、PC-MCL-style explicit normal。 | supported labels 的 macro-F1/UAR、coverage、unsupported-label false-positive audit。 | 只有 supported tail labels 改善且无 unsupported-label leakage 时才 go。 |
| 4 | Target-head shortcut-control ablation。 | 当前 frozen target heads。 | loudness/sampling/device controls；stratified split；dataset-ID residual/probe as analysis。 | dataset-ID probe accuracy、target Score、per-class recall；shortcut suppression 后 Score 变化。 | 如果抑制 shortcut 后仍保留 pathology signal，说明表征有效；如果高分塌陷，必须承认 shortcut。 |
| 5 | Tail/quality baseline against QLung-style method。 | CE/focal/class-weighted baselines。 | 加 quality score/margin 与 class-frequency angular margin；不使用 metadata。 | crackle/wheeze/both tail recall、macro-F1、support-aware CI，不只看 Score。 | 只有在足够 support 下 tail recall 改善才 go；aggregate Score 提升不足以支持 imbalance claim。 |

## 7. 推荐写法

推荐 contribution claim：

> 我们提出并审计一个 audio-first、source-grounded 的 respiratory auscultation cross-dataset evaluation and learning framework，在统一但显式标记为 proposed policy 的 label space 下，区分 source-only transfer、target-head adaptation、target-native references，并用 label-availability masks 与 support-aware metrics 分析 heterogeneous labels、long-tail 和 domain shift。

该写法避开的重叠：

- 避开 OPERA：不声称 broad respiratory FM benchmark，而是 auscultation event/cycle protocol。
- 避开 LungMix：不声称首次 cross-dataset DG，而是强调 fair comparator、partial-label semantics 与 target references。
- 避开 QLung：不把 quality/angular margin 本身当 novelty，而是作为 baseline/component。
- 避开 BTS-CARD：first-stage 做 audio-only/audio-first；metadata 先作为 analysis 或 later controlled branch。
- 避开 Resp-Agent：不声称大规模 multimodal respiratory disease benchmark，而是做 event-level auscultation protocol。

不安全 claim：

- first respiratory acoustic benchmark。
- first cross-dataset respiratory sound generalization study。
- first ICBHI->SPRSound OOD evaluation。
- aggregate Score improvement solves imbalance。
- target-head high Score proves pathology transfer。
- missing/unobserved label can be treated as negative。

下一轮数据出来后可能安全的 claim：

- 在固定 label/unit policy 下，提供 source-only、target adaptation、target-native references 的可审计 comparison matrix。
- 若 clean-source rows 复现当前趋势，可以说 published strong ICBHI methods 在 zero-target-tuning SPRSound inter 上迁移有限。
- 证明 domain/device/acoustic shortcuts 可测量，且必须从 pathology transfer 中分离。
- 如果 partial-label/multi-head objective 对 supported labels 有稳定增益，可以主张 respiratory-specific heterogeneous-label learning。

## 8. Submission 前仍缺的证据

- QLung repo availability 未解决：论文列出的 `RSC-Toolkit/QLung` 本轮 live 访问为 404。
- LungMix 官方完整 train/eval repo 未找到。
- Resp-Agent OpenReview 直接页面被浏览器验证阻塞；当前使用 arXiv、GitHub、HF 作为 primary/official evidence。
- Patch-Mix/PAFA/SG-SCL 需要 clean-source checkpoints 才能支撑 publishable transfer claim。
- SPRSound inter Both support=1，必须换 split/fold 或限制 minority claim。
- HF_Lung 和 KAUH 的 source facts 与 proposed harmonization policy 仍需 frozen contract，才能开始 pooled training。
