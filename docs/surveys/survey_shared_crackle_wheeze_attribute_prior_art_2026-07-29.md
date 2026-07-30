# Shared Crackle/Wheeze Attributes Modeling: Primary-Source Novelty / Prior-Art Audit

**日期：** 2026-07-29

**状态：** decision-ready survey；不构成方法批准或训练授权

**内部提案来源：** Notion `Notes Sync / 2026-07-30`（只读核验）

**审计边界：** respiratory-sound 多标签、跨数据集、domain generalization、异构/缺失标签、universal taxonomy、positive-unlabeled learning；技术结论优先依据论文正文、正式 proceedings/publisher 和作者仓库。

## 1. Executive verdict

1. **在本轮已核验的 primary-source 范围内，没有发现完整 direct overlap。** 未找到已有 respiratory-acoustic 工作同时完成：跨 ICBHI/SPRSound/HF_Lung/KAUH 联合训练、0/1/unknown 原子属性、unknown eligibility mask、独立 crackle/wheeze 属性头、joint normal/crackle/wheeze/both 头、两者 marginal consistency、以及 dataset-native heads。
2. **方法的新颖性不能建立在“首次把 flat4 分解成 crackle/wheeze”上。** Chua and Cheng (2024) 已在 ICBHI 上用 `[crackle,wheeze]` 二维多标签并重建 flat4；PC-MCL (ICASSP 2026) 已用显式 `[normal,crackle,wheeze]` 多标签并重建 flat4。
3. **数据融合的新颖性也不能表述为“首次统一 ICBHI 与 SPRSound”。** SPRSound/TBioCAS 已给出 ICBHI+SPRSound data-fusion benchmark；LungMix 已在 ICBHI、SPRSound、HF 上做 single-source domain generalization 和语义 OR 标签混合。
4. **当前最可辩护的贡献点是组合及其严格边界：** 保留 dataset-native units/heads，在 shared encoder 上以 eligibility-aware 0/1/unknown 属性监督共享 crackle/wheeze；仅在完整兼容样本上联合学习 joint four-state 和 marginal consistency；HF 只提供 observed-positive evidence；KAUH 暂不进入 shared supervision。
5. **missing-label policy 是关键但并非一般方法上的首创。** 异构语义分割和 partial-label/PU 文献已有 masked loss、universal taxonomy、probability marginalization 和 unknown-aware objectives；可主张的是这些原则在 respiratory event/cycle/recording 异构监督中的受控实例化与实证价值，不能主张 masked partial-label learning 本身的新颖。
6. **提案必须修改两个风险点。** `other_abnormal` 不是跨数据集稳定的原子属性，应先移出 shared atomic head 或严格 dataset-gated；HF positive-only BCE 不能单独辨识分类器，必须作为有界辅助项并设置 on/off 消融，不能把未标注区间当 negative。
7. **此前 equal-budget shared-compatible-head 结果构成强制对照。** 新方法必须在参数量、encoder、采样、优化和模型选择相同的条件下，与 native-only、shared flat4、attribute-only、joint-only 和 attribute+joint 模型比较；否则无法证明它不是已有 shared head 的重新参数化。

## 2. Source facts 与 proposed policy 的边界

### 2.1 已核验 source facts

| 数据集 | 原生 prediction unit / label | 可确认监督 | 不可自动推导 |
|---|---|---|---|
| ICBHI 2017 | respiratory cycle；Normal/Crackle/Wheeze/Both | `has_crackle`、`has_wheeze`、`explicit_normal` 和完整 flat4 | patient-overlap official split 不能代表 clean patient generalization |
| SPRSound BioCAS2022 | event 七类及 binary；recording 另有任务 | Normal、Wheeze、Coarse/Fine Crackle、Wheeze+Crackle 可映射到 narrow4 属性 | Rhonchi/Stridor 不能默认等同 Wheeze；inter split 的 Both support=1 |
| HF_Lung_V1 | 15 s recording 上 I/E/D/Wheeze/Rhonchi/Stridor 正向 intervals | D 可作为 crackle positive；Wheeze interval 可作为 wheeze positive | 无显式 Normal/Negative；未标注 gap、未出现某标签均不能当 negative |
| KAUH/Fraiwan v3 | recording-level raw9 | 原生 recording labels | Crep 不得静默并入 C；I/C/B 等映射尚未批准 |

以上事实来自本地冻结合同及官方数据源：[ICBHI challenge](https://bhichallenge.med.auth.gr/), [SPRSound paper](https://doi.org/10.1109/TBCAS.2022.3204910), [SPRSound repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound), [HF_Lung_V1 paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254134), [HF_Lung_V1 repository](https://gitlab.com/techsupportHF/HF_Lung_V1), [KAUH/Fraiwan v3](https://data.mendeley.com/datasets/jwyy9np4gv/3)。

### 2.2 当前 proposal，而非已批准事实

- shared attributes：`has_crackle`、`has_wheeze`、`explicit_normal`、候选 `other_abnormal`，状态为 0/1/unknown；
- 两个属性 sigmoid heads + joint four-state head；
- 在两种监督都 eligible 的样本上加入
  `p(crackle) ≈ p_joint(crackle)+p_joint(both)` 和
  `p(wheeze) ≈ p_joint(wheeze)+p_joint(both)`；
- HF positive-only evidence 作为辅助监督；
- dataset-native heads 和 pooling 保留。

这些候选项**不能自动覆盖**已冻结的 Phase-1 合同。当前合同只批准 ICBHI cycles 与兼容 SPRSound events 进入 shared surface；HF/KAUH 仍为 dataset-native/blocked，需单独 gate。

## 3. 核心问题逐条回答

### Q1. 是否已有跨多个 respiratory datasets 的共享 crackle/wheeze 二元属性，并重建 flat4？

**没有找到 exact match，但存在两个强方法先例和两个强应用先例。**

- Chua and Cheng 只用 ICBHI，将四类编码成 `[crackle,wheeze]`，BCE+sigmoid 后按阈值重建 flat4；这已覆盖“二属性替代 flat4”本身。
- PC-MCL 只用 ICBHI，将 normal 也显式编码为 `[normal,crackle,wheeze]`，对多 cycle 逻辑 OR 后再重建 flat4；这已覆盖“显式 normal 防止 `[0,0]` 信息丢失”。
- SPRSound data fusion 合并 ICBHI/SPRSound 并统一部分标签，但使用单一融合 label space，不是 0/1/unknown 属性监督。
- LungMix 在 ICBHI/SPRSound/HF 间做 single-source DG，并用语义 OR 构造混合标签，但没有跨源联合 masked attribute learning。

因此，“跨异构数据集、保留 native heads/units 的 eligibility-aware atomic supervision”仍是可辩护区别；“首次分解 crackle/wheeze”不安全。

### Q2. 是否已有 attribute heads + joint four-state head + consistency constraint？

**在审计范围内未找到 respiratory paper 同时具备三者。**

- Chua and Cheng：一个二属性输出；“multi-head”指 class-specific attention branches，不是 attribute head 与 joint head。
- PC-MCL：主 pathology multi-label head + patient-matching auxiliary head；不是 attribute+joint。
- Dietrich et al.：同一 ICBHI 提示实验同时报告 flat4 与两个独立 binary 分析，但没有训练出的双分支架构或 consistency loss。
- 通用 partial-label segmentation 的 marginal/exclusion loss提供“粗类概率等于细类概率和”的直接数学先例，因此 marginal consistency 不能描述为全新 loss 原理。

安全说法是：**把已有概率边缘化原则用于 respiratory crackle/wheeze 属性与 flat4 co-occurrence 的一致性约束，并限制在双重 eligible 样本上。**

### Q3. 是否已有 ICBHI、SPRSound、HF_Lung 或 KAUH 的 masked attribute learning？

**未找到。** 已发现的组合方式是：

| 工作 | 多数据集关系 | 标签处理 | missing policy |
|---|---|---|---|
| SPRSound/TBioCAS data fusion | ICBHI+SPRSound pooled train/test | coarse/fine crackle 合并，形成融合六类 | 未定义 unknown mask |
| LungMix | 每次单一 source 训练，在另外两个 unseen target 测试 | 映射到 flat4；Rhonchi/Stridor 并入 Wheeze；语义 OR | HF 缺 Both 的处理未充分说明；无 mask |
| Hsu et al. mixed-set training | HF_Lung_V2+HF_Tracheal_V1 pooled / target fine-tune | 两域采用相同 I/E/CAS event tasks | 不是 heterogeneous-label missing policy |
| OPERA | 多数据集预训练，19 个 dataset-specific downstream tasks | downstream 保持任务特定输出 | 不做 event-level shared attributes |
| Resp-Agent | 多来源聚合到 disease-level 16 类及文本 | disease taxonomy | 不处理 event-level unknown attributes |

SPRSound 与 ICBHI prediction unit 接近但仍非完全同一分布；HF 是 15 s interval detection；KAUH 是 recording label。合理最小实现是 shared encoder + unit-specific pooling/native heads，而不是把它们强制打包成 flat4 样本。

### Q4. 指定近邻工作覆盖与缺口

| 工作 | 已覆盖 | 未覆盖 | 判定 |
|---|---|---|---|
| LungMix | ICBHI/SPRSound/HF；single-source DG；语义 OR mix；flat4 映射 | pooled masked heterogeneous supervision；native heads；unknown≠negative；attr+joint consistency | **near-direct application threat** |
| OPERA | 10 respiratory datasets、19 task-specific probes；多数据集预训练 | event attribute harmonization；joint flat4；missing masks；tail/co-occurrence evidence | partial overlap |
| QLung | ICBHI quality-aware angular margin；SPRSound OOD 报告 | shared attributes；异构联合训练；unknown mask；native heads | partial overlap |
| BTS-CARD | ICBHI→SPRSound OOD；metadata counterfactual/adversarial debiasing | audio-only atomic attributes；masked heterogeneous labels；joint consistency | partial overlap |
| PC-MCL | `[normal,crackle,wheeze]`；逻辑 OR；flat4 reconstruction；per-class PR | 多数据集；unknown mask；joint head；consistency；native heads | **near-direct method threat** |
| Resp-Agent | 多数据源聚合、生成和 disease diagnosis | lung-event attributes；native units；partial labels；tail protocol | low/partial |
| SPRSound/TBioCAS | ICBHI+SPRSound data fusion；event/recording tasks | 0/1/unknown；dataset-native heads；consistency | **near-direct data threat** |
| Metadata-SCL | ICBHI 与 SPRSound 分别训练；class/metadata contrastive heads | 不是联合跨数据集学习；不是 attribute heads | partial overlap |
| RSC-FTF | 同文报告 ICBHI/SPRSound | 作者仓库当前只含 ICBHI loader；无共享标签目标 | low/partial |
| SG-SCL | ICBHI device-aware domain adaptation | 多 respiratory datasets；label harmonization；masked attributes | component reference |

### Q5. 通用方法先例如何影响 novelty？

- **Universal taxonomy / overlapping labels：** Bevandić et al. (WACV 2022) 将各数据集标签映射到 universal labels 的集合，并以细类概率和计算粗类 likelihood。这是 label marginalization 的强方法先例。
- **Heterogeneous labels / missing mask：** Schutera et al. (PLOS ONE 2022) 明确指出把未标注类当 background 会产生错误监督，并通过可用标签掩码训练。这是 eligibility mask 的强方法先例。
- **Partial supervision probability constraints：** Shi et al. (Medical Image Analysis 2021) 的 marginal/exclusion losses将粗粒度/未标注区域映射为细类概率边缘和，是 consistency/marginal loss 的组件先例。
- **Unknown-aware multi-label：** “Acknowledging the Unknown” (ECCV 2022) 在 single-positive multi-label setting 中把未标注标签视为 unknown；Yu et al. (ICML 2014) 研究 missing-label ERM。这些是 unknown≠negative 的方法先例。
- **Long-tail multi-label：** Distribution-Balanced Loss (CVPR 2021) 处理 label co-occurrence 与 negative dominance，是 tail loss 参考，但不解决 respiratory ontology/unit mismatch。

这些工作构成**方法组件 prior art**，而非 respiratory application exact overlap。我们的贡献需要落在 task formulation、eligibility contract、unit-preserving architecture、受控跨数据集验证及 tail/co-occurrence evidence，而不是宣称发明 masked loss 或 marginalization。

### Q6. 最安全 novelty framing 与不安全 “first” claims

**建议 framing：**

> 我们提出一个 unit-preserving、eligibility-aware 的多数据集 respiratory learning framework：在保留各数据集原生 prediction unit 和 native task head 的同时，将明确可观察的 crackle/wheeze/normal evidence 映射为 0/1/unknown 原子监督；仅在标签完备且兼容的样本上联合优化属性与 joint co-occurrence，并显式测量其对跨数据集迁移、minority co-occurrence 和 dataset/device shortcut 的影响。

**可安全保留的方法点：**

1. 0/1/unknown eligibility contract 是针对四数据集 source semantics 明确审计后定义的；
2. native unit/head 与 shared atomic supervision 并存；
3. attr↔joint consistency 只作用于 complete-compatible samples；
4. HF observed-positive evidence 被隔离成可消融辅助通道；
5. 评价同时覆盖 per-class/tail、跨域、coverage、calibration 和 shortcut controls。

**不安全 claims：**

- “first crackle/wheeze multi-label model”；
- “first to reconstruct normal/crackle/wheeze/both from attributes”；
- “first explicit normal label”；
- “first ICBHI+SPRSound/HF multi-dataset respiratory model/benchmark”；
- “first cross-dataset respiratory generalization method”；
- “first masked/partial-label or universal-taxonomy learning”；
- 在未完成更广检索和作者确认前使用无条件 “the first”；
- 把 HF unannotated gaps 称为 normal，或把 KAUH candidate mapping 写成 verified shared labels。

### Q7. 如何证明不是 shared-compatible head 的重新包装？

本地既有 equal-budget 结果表明：shared rank-96 compatible residual 与六个 task-specific rank-16 residual 都使用 98,304 residual parameters；task-specific 方案局部改善 SPRSound/KAUH，但 ICBHI `both_recall` 从 0.4336 降至 0.3413，未形成普遍优势。另有 joint flat4/binary/event heads 提高 specificity/calibration/Score、同时损害 Both recall 的已知 trade-off。

因此，新方法必须预注册以下 parameter-matched rows：

| Row | 模型 | 回答的问题 |
|---|---|---|
| A | shared encoder + native heads only | 多任务共享 encoder 的基础收益 |
| B | 当前 shared-compatible flat4 head | 新方案是否优于已有 shared head |
| C | attribute-only `[crackle,wheeze,explicit_normal]` + deterministic reconstruction | 收益是否只来自 label re-encoding |
| D | joint flat4-only | 属性支路是否必要 |
| E | attribute + joint，无 consistency | 收益是否只来自额外容量/多任务 |
| F | attribute + joint + consistency | consistency 的净增益 |
| G | F 的正确 unknown mask vs unknown-as-negative stress control | missing≠negative 是否实质影响结果 |
| H | F 中 HF positive-only evidence off/on | HF 正证据是否提供迁移信息而非全正捷径 |
| I | `other_abnormal` removed vs dataset-gated | 该非原子标签是否造成负迁移 |

**控制要求：** 相同 encoder/checkpoint、训练步数、dataset-balanced schedule、optimizer、model-selection rule 和随机种子；用投影宽度或 inert capacity 精确匹配 trainable parameters；不允许为每个 row 单独调 test threshold。

**必须报告：**

- ICBHI：Sp/Se/Score、macro-F1/UAR、四类 recall、Both support；
- SPRSound：seven-class 与 narrow4 分开，per-class support/recall，显式标注 inter Both support=1；
- shared attributes：AUROC/AUPRC、recall、calibration、attr↔joint disagreement；
- HF：只有在 negative eligibility 成立时才报告 specificity/precision；否则仅报告 observed-positive coverage/recall，并标注不可辨识性；
- KAUH：在 ontology gate 前只报告 native task，不作 shared-attribute claim；
- 每数据集/设备 shortcut probes，避免把 target head 的 dataset-ID 利用误写成 pathology transfer。

**建议 go/no-go：** 相比 A/B，F 必须在至少 ICBHI 与 SPRSound 的受支持属性上稳定改善，且不造成 ICBHI Both recall 的实质下降或 native-head collapse；F 必须优于 E 才能支持 consistency 贡献；H 必须优于 HF-off 且不能表现为 trivial all-positive，才允许保留 HF positive-only branch。

## 4. 直接与近直接 prior-art evidence

### 4.1 Chua and Cheng (2024): two-attribute reconstruction

- **论文：** [Towards Enhanced Classification of Abnormal Lung Sound in Multi-breath: A Light Weight Multi-label and Multi-head Attention Classification Method](https://arxiv.org/abs/2407.10828)
- **证据：** Sec. 3.4.1 将 ICBHI 四类编码为二维 binary label；sigmoid+BCE，阈值 0.5；Sec. 3.4.2 的 multi-head 是 class-specific attention；Table 5 报告 ICBHI Score 59.2。
- **威胁：** 高度覆盖“crackle/wheeze 二属性+flat4 reconstruction”。
- **缺口：** ICBHI-only；normal 仍隐式 `[0,0]`；无 unknown mask、joint flat4 head、consistency、native heads、跨数据集协议；未找到作者官方代码。

### 4.2 PC-MCL (ICASSP 2026): explicit normal and logical-OR composition

- **论文：** [PC-MCL: Patient-Consistent Multi-Cycle Learning with Multi-Label Bias Correction for Respiratory Sound Classification](https://arxiv.org/abs/2601.17080)
- **正式状态：** [ICASSP 2026 program entry](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=17214&bare=1)
- **证据：** Eq. (1) 对 `[normal,crackle,wheeze]` 逻辑 OR；Sec. 2.3 以 0.5 阈值及优先级重建 flat4；Table 1 报告 BEATs+PC-MCL Sp 79.04±1.90、Se 51.71±2.98、Score 65.37±0.73；Fig. 3 报告 crackle AP 0.642、wheeze AP 0.663。
- **威胁：** 最强方法级威胁，已明确指出 `[crackle,wheeze]` 中 normal=`[0,0]` 在多 cycle 组合时丢失 normal 信息。
- **缺口：** ICBHI-only；无 heterogeneous missing labels；两个 heads 是 pathology 与 patient matching，而非 attribute+joint；无 consistency/native heads；未找到官方代码。

### 4.3 LungMix (ICASSP 2025): three-dataset semantics and source-only DG

- **论文：** [arXiv](https://arxiv.org/abs/2501.00064), [ICASSP DOI](https://doi.org/10.1109/ICASSP49660.2025.10888016)
- **证据：** Sec. II 定义单源 domain generalization；Eq. (4) 以 bitwise OR 生成语义标签；Sec. III-A 映射 ICBHI/SPRSound/HF；Table I 给出每个 source 对三个 test domains 的结果。
- **协议：** 每次只用一个 source 训练，在另外两个 unseen datasets 上测试；不是多数据集 pooled/masked training。COMB 是 test aggregation，不是 pooled training。
- **威胁：** 最强 application-level threat；覆盖三数据集、flat4 harmonization、co-occurrence OR 与 unseen-domain evaluation。
- **缺口：** 无 attrs+joint heads、unknown mask、native tasks、marginal consistency；Rhonchi/Stridor→Wheeze 是作者的 flat4 policy，不应直接继承；HF missing Both/negative policy未充分定义；未发现足以完整重建论文 pipeline 的作者 repo。

### 4.4 SPRSound/TBioCAS (2022): ICBHI+SPRSound fusion precedent

- **论文/代码：** [TBioCAS DOI](https://doi.org/10.1109/TBCAS.2022.3204910), [official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
- **证据：** event-level seven-class/binary 与 recording-level tasks 分列；data-fusion experiment 合并 SPRSound/ICBHI train/test，并把 coarse/fine crackle合并到融合 label space。
- **威胁：** 直接阻止“首次 ICBHI+SPRSound label fusion”。
- **缺口：** 单一融合 taxonomy；无 unknown mask、attribute+joint consistency 或 dataset-native heads；fusion protocol 不能解决 HF/KAUH unit mismatch。

### 4.5 OPERA (NeurIPS 2024): multi-dataset representation and native tasks

- **论文/代码：** [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f803abdcad9de35b45d5a656dade45c-Abstract-Datasets_and_Benchmarks_Track.html), [official repository](https://github.com/evelyn0414/OPERA)
- **证据：** 10 个 respiratory datasets 预训练，19 个 downstream tasks；probe/finetune 按 dataset-task 单独组织。ICBHI downstream 主要是 recording-level disease task，而不是 cycle flat4。
- **威胁：** “first respiratory multi-dataset representation/framework”不安全。
- **缺口：** 不统一 heterogeneous event labels，不做 missing mask、flat4 co-occurrence 或 minority-tail attribution。

### 4.6 Hsu et al. (2023): mixed-set and adaptation precedent

- **论文：** [A dual-purpose deep learning model for auscultated lung and tracheal sound analysis based on mixed set training](https://www.sciencedirect.com/science/article/pii/S1746809423006559)
- **证据：** HF_Lung_V2/HF_Tracheal_V1 上比较 separate、mixed-set、cross-test 与 target fine-tuning；任务是共同定义的 I/E/CAS events。
- **威胁：** pooled respiratory datasets 和 target adaptation 的应用先例。
- **缺口：** 标签并非异构 crackle/wheeze ontology；无 0/1/unknown、attrs+joint 或 native task preservation。

## 5. 其他 respiratory 邻近工作

| 工作 | Primary source / repo | 与当前方案关系 |
|---|---|---|
| Dietrich et al. (2026), few-shot MLLM | [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001179) | 同时报告 ICBHI flat4 与 crackle/wheeze 两个 binary 分析；无训练架构/一致性约束 |
| Multi-Stage Respiratory Sound Analysis | [IEEE TBME / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13184872/) | 私有 PERCH；独立 normal-vs-wheeze、normal-vs-crackle classifiers，再做 recording aggregation；无 Both |
| Robust Interpretable TCN | [arXiv](https://arxiv.org/abs/2106.15835), [JBHI DOI](https://doi.org/10.1109/JBHI.2022.3144314) | HF/ICBHI/private data 上分别训练相同架构；不是 multi-dataset learning |
| Metadata-SCL | [arXiv](https://arxiv.org/abs/2210.16192), [repo](https://github.com/ilyassmoummad/scl_icbhi2017) | ICBHI/SPRSound 分别训练；class/metadata contrastive，不是 attribute+joint |
| Respiratory sounds classification by fusing the time-domain and 2D spectral features (RSC-FTF) | [publisher](https://www.sciencedirect.com/science/article/pii/S1746809425003015), [repo](https://github.com/deegy666/RSC-FTF) | 同文多数据集但作者 repo 当前仅有 ICBHI loader；不能视为联合训练 |
| SG-SCL | [arXiv](https://arxiv.org/abs/2312.09603), [repo](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning) | ICBHI device-guided adaptation，提供 shortcut/domain 组件 |
| QLung | [arXiv](https://arxiv.org/abs/2606.11915), [repo](https://github.com/RSC-Toolkit/QLung) | quality+frequency angular margin；ICBHI train、SPRSound OOD；无 heterogeneous attributes |
| BTS-CARD | [arXiv](https://arxiv.org/abs/2510.22263), [repo](https://github.com/RSC-Toolkit/BTS-CARD) | metadata counterfactual/adversarial OOD；直接影响 dataset/device shortcut controls |
| Resp-Agent | [arXiv](https://arxiv.org/abs/2602.15909), [repo](https://github.com/zpforlove/Resp-Agent) | 多来源 disease-level 16-class 与生成任务；不是 event attribute learning |
| Crackle transfer to IPF | [arXiv](https://arxiv.org/abs/2104.14921) | ICBHI 预训练后用私有 target labels fine-tune；属于 adaptation，不是 source-only 或 masked joint training |
| Asthma monitoring indices | [Frontiers in Physiology](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.745635/full) | 私有数据上预测 wheeze/rhonchi/fine/coarse crackle并聚合指数；多标签组件先例 |

## 6. 通用方法 prior art

| 方法 | Primary source / code | 可迁移思想 | 不可直接照搬 |
|---|---|---|---|
| Multi-class Semantic Segmentation on Heterogeneous Labels | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263656), [OSF](https://osf.io/uyk79/) | eligibility mask；未标注类不作 background | pixel-exclusive segmentation，不含 respiratory unit/presence labels |
| Multi-Domain Semantic Segmentation With Overlapping Labels | [WACV paper](https://openaccess.thecvf.com/content/WACV2022/papers/Bevandic_Multi-Domain_Semantic_Segmentation_With_Overlapping_Labels_WACV_2022_paper.pdf) | universal taxonomy；dataset label 对 universal labels 的集合映射；概率求和 | 每 pixel 通常单一 latent class；respiratory attributes可共现 |
| Marginal and Exclusion Loss | [arXiv](https://arxiv.org/abs/2007.03868), [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841521000256) | coarse label probability marginalization；partial supervision | organ segmentation 与音频多标签不同 |
| Acknowledging the Unknown for Multi-Label Learning | [arXiv](https://arxiv.org/abs/2203.16219), [repo](https://github.com/Correr-Zhou/SPML-AckTheUnknown) | unannotated=unknown；unknown-aware regularization | single-positive assumption不等同 HF interval annotation |
| Large-scale Multi-label Learning with Missing Labels | [ICML/PMLR](https://proceedings.mlr.press/v32/yu14.html) | missing-label ERM/low-rank先例 | 不给出 respiratory ontology或unit policy |
| Distribution-Balanced Loss | [arXiv](https://arxiv.org/abs/2007.09654) | label co-occurrence、rebalancing、negative dominance | 不能解决 unknown eligibility 或跨数据集语义冲突 |

## 7. Novelty map

| Candidate claim | Closest prior art | Risk | 可安全表述 |
|---|---|---:|---|
| crackle/wheeze 属性化并重建 flat4 | Chua and Cheng; PC-MCL | 高 | 作为已知表述基础，不作 novelty claim |
| 显式 normal 属性 | PC-MCL | 高 | 仅强调跨数据集 eligibility 约束 |
| ICBHI+SPRSound/HF 融合 | SPRSound data fusion; LungMix; OPERA | 高 | unit-preserving heterogeneous supervision，而非 first fusion |
| unknown 不作 negative | Schutera; Ack Unknown; Yu et al. | 高（一般方法） | respiratory source-semantics-driven eligibility contract |
| attr+joint marginal consistency | marginal/exclusion loss；本文未找到 respiratory exact match | 中 | respiratory co-occurrence 的受限实例化 |
| dataset-native heads + shared atomic attributes | OPERA native tasks + universal taxonomy methods | 中低 | 当前最强组合贡献候选 |
| HF positive-only attribute evidence | PU/missing-label literature；未找到 respiratory exact match | 中 | 谨慎称 auxiliary observed-positive supervision，并验证可辨识性 |
| tail/co-occurrence improvement | PC-MCL、QLung、DB Loss | 中高 | 必须用 Both/per-class/support证据，而非 aggregate Score |

## 8. 必须修改的设计

1. **将 `other_abnormal` 移出初始 shared atomic head。** Rhonchi、Stridor、CAS/DAS、KAUH disease labels不是同一原子事件。首轮应删除，或设为 dataset-gated native attribute，不能跨数据集共享 negative/positive。
2. **`explicit_normal` 只在显式 normal 标注源上 eligible。** ICBHI Normal、SPRSound Normal 可监督；HF gaps/absence 和 KAUH 未批准 mapping 必须为 unknown。
3. **HF positive-only 只作可关闭的辅助 loss。** 普通 masked BCE 若只有 positive 项会允许 all-positive trivial optimum。首轮要有 HF-off/HF-on；若未来需要完整 HF classifier，应采用经验证的 PU/negative sampling policy并单独论证假设。
4. **consistency 只在完整 joint+attribute eligible 样本上施加。** 首轮限 ICBHI 与兼容 SPRSound narrow4 events；不得将 HF/KAUH 强塞入 joint flat4。
5. **保留 unit-specific pooling 与 native heads。** cycle/event/15s interval/recording 的 pooling、metric 和 split分开；shared encoder 不等于 shared prediction unit。
6. **加入 shortcut controls。** 四数据集声学分布高度可分；必须报告 dataset/device probe、leave-dataset/device analyses，防止 target head 利用采集域而非病理属性。

## 9. 搜索边界、未核验项与证据等级

- 检索覆盖：Google/Scholar-style title/keyword search、arXiv、IEEE/ACM/Elsevier/PLOS/NeurIPS proceedings、作者 GitHub/GitLab/OSF，以及当前项目已审计文献链。
- 关键词覆盖 `respiratory sound multi-label crackle wheeze normal both`, `cross-dataset`, `heterogeneous labels`, `partial labels`, `universal taxonomy`, `co-occurrence`, `missing labels`, `positive unlabeled` 等组合。
- “未找到 exact overlap”只适用于上述检索边界；不能据此断言文献中绝对不存在。
- LungMix 未找到完整官方 train/eval repo；其 code completeness 仍为不完整/未确认。
- PC-MCL 和 Chua and Cheng 未找到作者官方 repo。
- QLung 当前官方仓库可访问；这更新了旧 survey 中 repo 404 的时点性结论。
- RSC-FTF 仓库当前只核验到 ICBHI loader；论文中的 SPRSound pipeline不能据 README 自动视为可复现。
- KAUH `Crep/I/C/B` 到 shared attributes 的语义仍需临床/数据字典 gate。

### 9.1 作者仓库审计快照

以下 commit 只用于固定本轮代码证据，不代表论文结果已复现：

| Repository | Audited commit |
|---|---|
| RSC-FTF | `fdd607ca658992f20648c5dc5ee33ed0ecf49a89` |
| Metadata-SCL | `935d5092a0b3e289b818ae643536024bc349a4ea` |
| OPERA | `3622310e667afb8aa40169050b4dd45de75946a2` |
| BTS-CARD | `a9982a0e658ee9bb1c5656470e18728b00685a08` |
| QLung | `46408ddf645a0a1600c528c145b42976d187bbf2` |
| Resp-Agent | `45b9c37d8210339d47172ab87bbd22a4d3acdb6a` |
| SG-SCL | `66564609595090b61540595d3d27764c00553086` |

## 10. Management handoff

**Decision：** 该方向可以继续作为候选方法，但应定位为“unit-preserving eligibility-aware shared respiratory attributes + joint co-occurrence consistency”，而不是“首次 crackle/wheeze multi-label”。目前没有发现完整 direct threat；最强威胁依次是 PC-MCL、Chua and Cheng、LungMix、SPRSound data fusion、OPERA、通用 universal-taxonomy/masked-loss 文献。

**批准前必须完成：**

1. 从 shared atomic set 移除或 dataset-gate `other_abnormal`；
2. 将 HF 限定为 positive-only auxiliary evidence，并加入 off/on 与 all-positive safeguard；
3. consistency 仅覆盖 ICBHI/兼容 SPRSound；
4. 保持 KAUH shared mapping 为 HOLD；
5. 预注册 A-I parameter-matched matrix，尤其保留 B（既有 shared-compatible head）和 E/F（无/有 consistency）；
6. 以 per-class/support、Both recall、attribute AUPRC/calibration、native-task retention 和 shortcut probes作为 go/no-go，而非 aggregate Score 单指标。

**最小可发表证据链：** 先证明 exact parameter-matched shared-head baseline；再证明属性分解有独立收益；再证明 consistency 超过额外容量；最后证明 HF positive-only evidence 和跨数据集训练没有通过 missing-as-negative 或 dataset shortcut 获得虚假收益。
