# 现有资产接收与论文缺项

更新：2026-09-11。四个 seed42 benchmark controls 的最终结果均已接收并写入 Table 2。Coarse 只采用 coarse_spr/seed_42_attempt2（完整30 epochs、selected20），Native 采用 native_attributes/seed_42（完整27 epochs、selected17）；两者的 run_summary、selection 与 terminal metrics 一致。原 coarse_spr/seed_42 是 partial，不取其过程分数，也不与 attempt2 拼接。Full 继续复用正式 JH2 fresh42，未替换为 Native 或三seed均值。

§4.1 与 §4.3 已改为实际采用的原 subtrain/validation 两分、official-test-based selection 和每 epoch cosine。旧 calibration/selection 三分及 14/18-run 预算不再作为本文现行协议。Table 1 保留三 seed benchmark；其同模型读出/属性/外部结果与独立历史 HF-off/on 案例继续留在 §4.2，不与 Table 2 的单 seed 数字混配。

## 当前 Table 2：单 seed benchmark

| 条件 | Checkpoint 选择 | ICBHI Score | SPR official Score | SPR C/W macro AUROC |
| --- | --- | ---: | ---: | ---: |
| Full fresh42，epoch17 | ICBHI test Score | 61.17 | 90.37 | 97.17 |
| ICBHI-only，epoch18 | ICBHI test Score | 56.17 | 56.86 | 81.23 |
| SPRSound-only，epoch24 | SPR inter Score | 47.93 | 92.32 | 97.86 |
| Coarse SPR attempt2，epoch20 | ICBHI test Score | 56.71 | 90.22 | 93.67 |
| Native + attributes，epoch17 | ICBHI test Score | 62.30 | 91.94 | 97.41 |

以上均为百分数，单次运行，不写 mean ± SD。Full 的 SPR 分数来自其 seed42 的 terminal/selected_native_metrics.json（90.37%），不是 Table 1 的三 seed 均值（90.70%）。两个单源在另一数据集上的结果，是选好 source checkpoint 与阈值后的固定模型评测。

当前结果不对称：Full 的 ICBHI Score 比 ICBHI-only 高 5.00 pp，但 SPR Score 比 SPRSound-only 低 1.95 pp，SPR C/W macro AUROC 低 0.69 pp。相对 ICBHI-only，Full 的 Sp 从 67.13% 到 80.37%，Se 从 45.20% 到 41.97%。已在正文同时报告收益、代价与解释边界，不将这些差值全归因于来源共享。

最后两组差值均从未舍入 JSON 数值计算，方向为该条件减 Full：

| 条件 − Full | ICBHI Score | SPR official Score | SPR C/W macro AUROC |
| --- | ---: | ---: | ---: |
| Coarse SPR | −4.46 pp | −0.15 pp | −3.50 pp |
| Native + attributes | +1.13 pp | +1.58 pp | +0.24 pp |

这次 benchmark 中，保留 fine attribute supervision 的 Full 比 Coarse 的属性识别与 ICBHI Score 更高；Native + attributes 三个指标均略高于 Full。因此，不能据此声称 hierarchy 优于 Native。Native 同时改变 native objective 和 head capacity，仍只解释为接口比较，不是纯 head 形状效应。本轮没有改换主方法、Table 1 Ours、最终 contribution、Abstract 或 Conclusion。

## 1. 已核对数字及其可用位置

以下均为百分数；三 seed 项报告 mean ± sample SD，成对变化的单位为 pp。

| 已有证据 | 核对后的关键数字 | 对应正文与呈现方式 |
| --- | --- | --- |
| 当前主 benchmark，seeds 0/1/fresh42 | ICBHI Score 61.17 ± 0.31；SPR official Score 90.70 ± 0.34 | 与现有 Table 1 的六项原生指标一致，无需改数。三个 checkpoint 由 ICBHI test Score 选择；SPR inter 在选定 checkpoint 后评测，不能写成 SPR inter 自身用于选模。 |
| 同一组三 checkpoint 的 Direct C/W readout | C/W-only Score 60.29 ± 1.81；相对 Full 的 paired Score 变化 −0.88 ± 1.60；Sp −6.80 ± 3.94，Se +5.04 ± 0.79；Both recall 变化为 0 | 已写入 §4.2。seed42 分析与当前 Table 2 复用同一 Full；三 seed 汇总仍独立报告，Table 2 不重复列读出行。 |
| SPR inter 属性识别 | C/W macro AUROC 97.32 ± 0.15、macro AUPRC 77.42 ± 2.03；Crackle AUROC 96.08 ± 0.41、AUPRC 60.32 ± 3.86；Wheeze AUROC 98.56 ± 0.11、AUPRC 94.51 ± 0.56 | 是已选 benchmark 模型的属性评测。每 seed 为 1,429 events；C 正/负为 84/1,345，W 为 306/1,123。macro 在每个 seed 内先平均 C/W，再跨 seed 计算 mean/sample SD；已用交付的三个 seed 值核对该汇总。 |
| 当前主 benchmark 的 HF 外部评测 | D/W recording AUROC 67.82 ± 4.42 / 86.36 ± 1.38；positive-interval recall 58.28 ± 12.63 / 85.43 ± 1.07 | 已有三模型结果。AUROC 对应 957 个含 D/W/R/S 标注的 eligible recordings；interval recall 的 D/W 正例分母为 1,812/1,430。不能将这两个指标改称 HF 原生时间定位 F1。 |
| 当前主 benchmark 的 KAUH 外部评测 | patient binary BA 72.17 ± 1.64；flat4 Macro-F1 33.83 ± 3.17；UAR 50.24 ± 6.10 | 正文所需指标均已存在于原始 external aggregate，简表没有列出不等于尚未计算。评测为 B/D/E 聚合后的 compatible overlay，不是 raw9 复现。 |
| 历史独立 seed42 的 HF-off/on 比较 | ICBHI Score 60.05 → 59.71；SPR Score 89.20 → 91.81；完整 HF/KAUH 前后值见下一节 | 可以单独报告一个 HF 辅助监督案例，注明独立的一组 HF-off/on 参照。不能把其 HF-on 与当前三 seed benchmark 的 HF-off 均值配成一对。 |
| JH3.3 的独立 seed42 外部评测，epoch 21 | HF D/W AUROC 57.16 / 86.57；positive-interval recall 19.54 / 85.94；KAUH patient BA 71.99、flat4 Macro-F1 37.29、UAR 49.56 | 本地任务已完成固定 checkpoint 推理。来源为 soft-bridge/MVN 变体，且采用 composite selection；不是 JH2 Full 或 HF-on。仅在讨论该方法变体时考虑使用，不并入现有配对。 |

Direct C/W 相对 Full 的三个 Score 差值分别为 −0.60、−2.60、+0.56 pp。因此，现有证据支持讨论这些 checkpoint 上的 specificity/sensitivity 取舍，不能写成两层读出在所有 seed 上一致更优。ICBHI 的 Se 是异常细类正确分类率，不能换称二分类异常检出率。

SPR 属性评测支持模型能够预测 C/W，但其绝对 AUROC 本身不能证明细标签增益或来源共享效应。当前来源和 Coarse 对照均已齐：隐藏 SPR fine annotations 后，SPR C/W macro AUROC 与 ICBHI Score 分别下降3.50和4.46 pp；该结论限于本次 single-seed、test-selected benchmark。

### 最终 metadata 的适用分母

| 旧 JH2 实际 subtraining | 独立 native units / groups | C eligible / masked | W eligible / masked |
| --- | ---: | ---: | ---: |
| ICBHI seed0 | 3,636 cycles / 63 patients | 3,636 / 0 | 3,636 / 0 |
| ICBHI seed1 | 2,880 cycles / 63 patients | 2,880 / 0 | 2,880 / 0 |
| ICBHI fresh42 | 3,174 cycles / 63 patients | 3,174 / 0 | 3,174 / 0 |
| SPR，各 seed 相同 | 5,219 events / 194 patients | 5,170 / 49 | 5,170 / 49 |

这组统计由管理按各 run 保存的 config 和原 metadata loader 重建；交付记录三个 seed 的重建 validation IDs 均与保存的 NPZ 相符。SPR 每个属性屏蔽 49/5,219，即约 0.94%；C/W 屏蔽的是同一批 49 个 event，不是 98 个不同 event。

| 已交付的旧 JH2 validation | 分母（rows） | C eligible | W eligible |
| --- | ---: | ---: | ---: |
| ICBHI seed0 / epoch15 | 506 | 506 | 506 |
| ICBHI seed1 / epoch11 | 1,262 | 1,262 | 1,262 |
| ICBHI fresh42 / epoch17 | 968 | 968 | 968 |
| SPR，各 seed 相同 | 1,437 | 1,432 | 1,432 |

后一张表是选中 checkpoint 对应的内部 validation NPZ；当前 Table 2 复用其中 seed42 的 subtrain/validation 划分。Validation 拟合属性阈值，checkpoint 则由指定 official-test Score 选择。SPR validation 中每个 C/W 节点有5行被mask；这些 native-unit 计数不能称为重复抽样后的 batch exposure 比例。

Figure 1 所用 feature table 的 SPR 2,683 行现已明确为 recordings；模型所用 events 是另一种单位。该图方案对 13 个特征先取 group median，再平衡到每数据集 112 groups；ICBHI/SPR/KAUH 按 patient 分组，HF 按 recording-date proxy 分组。它可以描述 corpus/recording 条件，不能直接作为 cycle/event 级声学差异的证据。

## 2. HF-on/off 已有全四数据集结果

以下属于同一历史单 seed 比较；HF-on 为 JH4，HF-off 为其配套旧 JH2 参照，不是当前 fresh42 或三 seed 均值。

| 端点 | HF-off | HF-on |
| --- | ---: | ---: |
| ICBHI Score | 60.05 | 59.71 |
| ICBHI 异常细类 Se | 45.37 | 39.25 |
| SPR official Score | 89.20 | 91.81 |
| HF D recording AUROC | 50.81 | 69.80 |
| HF W recording AUROC | 85.50 | 89.83 |
| HF D positive-interval recall | 52.65 | 93.10 |
| HF W positive-interval recall | 93.78 | 76.92 |
| KAUH patient binary BA | 74.76 | 70.14 |
| KAUH patient flat4 Macro-F1 | 51.47 | 36.65 |
| KAUH patient flat4 UAR | 62.33 | 38.39 |

例如，HF-on 的 KAUH Macro-F1 36.65 不能与当前三 seed benchmark 的 33.83 拼成“提升”：该实验的实际配对是 51.47 → 36.65。保持原参照后，这组数字可以讨论额外 HF 监督的收益与代价，无须因采用 test-based selection 将整组结果弃用。它仍是单 seed 证据，也不能声称等计算预算的纯标注信息效应。

## 3. 现行协议和解释边界

| 项目 | 当前事实 | 写作处理 |
| --- | --- | --- |
| Table 1 | 保留三 seed、ICBHI-test-selected benchmark | 均值和 sample SD 不变。 |
| Table 2 reference | 复用正式 fresh42 Full，selected epoch17 | 不重跑 Full，不用三 seed 均值填单 seed 行。 |
| 数据划分 | ICBHI subtrain/validation=3,174/968 cycles、63/16 patients；SPR=5,219/1,437 events、194/49 patients | 使用原 seed42 两分；不使用后来准备的 calibration/selection 三分。 |
| 选模 | Full、ICBHI-only、Coarse、Native 采用 ICBHI test Score；SPR-only 采用 SPR inter Score | 表格 dagger 与 §4.1 均注明该差异。另一数据集不参与单源模型的训练、校准或选模。 |
| 优化与暴露 | 原 JH2 epoch cosine，epoch 内 LR 固定；最多50轮，每轮326 updates。Joint每源163，single-source本源326 | 最大总updates相同，但每个source在单源模型中每轮获得两倍updates。不能称每源暴露匹配。 |
| Loss | 管理核对 fresh42 的27轮实际 batch metadata：每个 batch 都有 A/C/W eligible rows，实际节点系数均为1/3 | 当前 controls 显式固定1/3，并在 Coarse 空节点时不重分权重。它保持了 reused Full 的实际有效权重，不改写 Table 1 通用公式。 |
| HF与固定读出 | 当前四项不新增Full+HF或E1训练行；已有独立HF pair和三seed固定读出已在§4.2 | Table 2 为五行。相关已完成分析没有删除，也不伪装成新训练结果。 |
| 旧候选协议 | 新validation三分、active-source validation composite、14/18-run预算均已停用 | 从当前正文移除；不把它们作为继续重跑的条件。 |

这批结果是 seed42 benchmark controls。SPR-only 的 selection objective 不同，source-only 的每源暴露也不同；因此可报告实际 native-task retention/transfer 和取舍，但不能将整个差值解释为来源共享的纯效应。

## 4. 真正的剩余工作与结论影响

### A. 已完成的统计、推理和数字落文

- Table 1、固定读出、三 seed SPR属性与HF/KAUH外评均已落文。
- 独立历史 HF-off/on 的原生、SPR C/W、HF与KAUH配对结果已齐，未与当前Full42或三seed均值混配。
- 旧JH2实际 subtrain/validation eligibility 已完整接收，Data 已按实际 benchmark 分母写明。
- 四个控制条件的双数据集 native 指标与 SPR C/W AUROC 已全部写入 Table 2；相关训练结果的xxx已关闭。
- 本批已要求的既有模型推理缺项已关闭。新四条件的HF/KAUH扩展并非当前执行范围；只有决定比较这些外部端点时，才另行安排相应selected checkpoint的评测，不预设为必做。

### B. 最后两个训练条件已完成

| 条件 | 接收结果 | 本次支持的解释 |
| --- | --- | --- |
| Coarse SPR labels | attempt2完整30轮，采用epoch20的terminal结果 | 隐藏SPR细属性监督后，C/W AUROC和ICBHI Score明显更低，SPR Score变化较小。 |
| Native + attributes | 完整27轮，采用epoch17的terminal结果 | 在匹配C/W辅助监督的条件下，native接口三个报告指标均略高于Full；目标结构和head容量也同时改变。 |

本批四条件训练与已约定的terminal评测缺项已关闭。没有使用旧Coarse partial的best-so-far值，没有合并不同attempt，不补seed0/1，也不恢复旧clean队列。单seed、选择目标与每源暴露的解释边界继续保留。

### C. 文字与图表

Figure 1/§2.2 仍需最终图和对应解释。SPR feature-table的2,683行是recordings；模型events为另一单位。若要cycle/event级、class-matched声学结论，需要对应单位的分析，不能只改图注。

需要与用户复盘主线和最终contribution：当前数据支持fine attribute supervision的作用，但不支持hierarchical readout优于Native+attributes。是否调整主方法或叙事，由用户讨论后决定，本轮只按实际结果落文。

JH3.3 独立变体只保留在资产说明中，没有扩展当前论文主线。Abstract 中旧的validation-selected措辞按用户要求留到最后与Conclusion一起统一，不作为本轮正文改稿的阻塞。

## 5. 当前写作处理

- Section/4evaluation.tex：Table 2五行均已填齐；§4.3写明Coarse相对Full的三个下降量，以及Native相对Full的三个上升量。所有差值使用未舍入原值计算，保留不同selection目标、每源暴露和单seed范围。
- Table 2移除旧的Full+HF和重复Direct C/W行；相关实测结果继续保留在§4.2。PC-MCL训练标签/推理转换的说明移至固定读出分析处，引用保留。
- Section/2data.tex只同步一条过期注释；其已核实的实际benchmark监督覆盖正文未改。
- Table 1和§4.2已有结果保持各自原cohort；Full42三项指标明确区别于三seed均值。
- Intro、Abstract、Conclusion未改。写作任务没有编译、渲染、训练、推理、Git提交/推送或Notion写入。

## 6. 核对来源

- [当前 benchmark 规格](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/EXPERIMENT_SPEC_2026-09-09_zh.md)：seed42四条件、原两分、selection目标、epoch cosine与实际Full节点权重说明。
- [Full42 原生指标](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_42/terminal/selected_native_metrics.json)：Table 2的单seed参照；其SPR属性取下述交付JSON的seed42条目。
- [ICBHI-only 最终指标](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/icbhi_only/seed_42/terminal/native_metrics.json)：另核对同run的run_summary.json与selection.json。
- [SPRSound-only 最终指标](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/sprsound_only/seed_42/terminal/native_metrics.json)：另核对同run的run_summary.json与selection.json。
- [Coarse attempt2 最终结果](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/coarse_spr/seed_42_attempt2/run_summary.json)：selected20；与selection.json、terminal/native_metrics.json核对一致，仅使用此完整attempt。
- [Native 最终结果](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/native_attributes/seed_42/run_summary.json)：selected17；与selection.json、terminal/native_metrics.json核对一致。
- [本地写作交付](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/LOCAL_ASSETS_NO_TRAIN_2026-09-09/WRITING_HANDOFF_zh.md) 与同目录的 LOCAL_ASSETS_NO_TRAIN_SUMMARY.json：主指标、读出 pilot、属性指标、HF 和历史 HF pair。
- [三 seed 外部结果原始聚合](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.json)：补齐 patient-level KAUH Macro-F1/UAR，并核对 eligible HF pool。
- [历史 HF-off/on 全四数据集比较](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH4_JH2_HFaux_seed42_attempt2/JH2_vs_JH4_KAUH_external_comparison.md)：核对同一 pair 的原生与 KAUH 端点。
- [JH3.3 独立外部推理汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_HF_KAUH_external_seed42/run_summary.json)：epoch 21 的 HF/KAUH 结果，未重选 checkpoint 或外部阈值。
- [JH3.3 原始 selection artifact](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_soft_bridge_mvn_seed42/validation_selection.json)：核对其 composite checkpoint-selection 定义。

只读取了交付和上述直接相关汇总，没有重复扫描执行端的全部原始数据。
