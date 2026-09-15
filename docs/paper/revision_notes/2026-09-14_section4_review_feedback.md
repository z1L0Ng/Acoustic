# Section 4 反馈收集

用户要求：先记录独立审阅、Claude、Gemini 三份反馈，全部收到后再思考如何修改。当前不改正文、表格或排版，不提前形成修改方案。

## 独立审阅反馈（用户转交，已收到）

以下记录审阅意见，尚未作为新一轮事实核验或修改决定。

### 总体判断

Section 4 仍偏向“设置 → 指标差值 → 收益、代价或任务差异”的结果播报。需要补充：每项比较让我们对“共享细监督、保留不同原生任务”获得了什么认识。数字应支撑认识，而非替代解释。

### 1. 联合训练

- 将单源跨任务应用较差、联合模型同时承担两个任务、相对专用模型的性能取舍连接起来。
- 在当前共享架构下，具备两个输出接口并不意味着单源训练可以自然覆盖另一原生任务；联合使用两种监督的意义需要明确。
- 不只依次报告 ICBHI 上升、SPRSound 下降和跨源应用较差。

### 2. 外部评估

- 当前正文突出相对 frozen AST 的差距，但更直接检验联合训练的对象是单源控制。
- HF CAS：LSAA 86.44，ICBHI-only 87.05；LSAA 没有在该指标上超过 ICBHI-only。
- KAUH：LSAA 的均值高于两种单源模型。
- 应解释迁移收益与参照来源、评估目标有关；相对冻结参考的表现只提供背景，不能单独证明联合训练改善迁移。

### 3. 细粒度监督

- 原生输出为二分类，并不要求训练监督也只能是二分类粒度。
- 在当前配方下，细声学标注对本数据集的二分类及另一数据集的细分类仍有用，回应 Introduction 的 Wheeze 例子。
- Coarse 的平均差距包含一次 all-Normal 运行；另外两个 seed 的 SPR 增益为 4.57/0.15 pp，不应将平均数表述成均匀、普遍的提升。
- Coarse 同时撤去 C/W 监督和 calibration；需保留配方层面的限定，并讨论运行差异与敏感性。

### 4. Readout

- 共享 C/W 监督与如何实现原生任务预测头，是需要分别评价的设计选择。
- 匹配细监督的替代方案获得更高 ICBHI Score，同时 SPRSound C/W 判别表现相近。
- 细监督有价值，不自动证明当前显式 readout 是最佳原生任务实现。
- 对照重新训练 encoder 和 heads，应解释替代训练与 readout 方案，不能将收益全部归因于单独替换 readout。

### 5. HF 辅助监督

- SPRSound 在两个 seed 中提高；HF CAS AUROC 在三个 seed 中均下降；KAUH 方向不一致。
- 关键现象：增加 HF 来源的监督，也没有改善这里使用的 HF 评估指标。
- 新增监督是 C/W，而 CAS 评估用 Wheeze-head 分数对 Wheeze/Rhonchi/Stridor 联合标签排序；同一数据来源不代表同一监督与评估目标。
- 应分别判断增加了什么监督信息，以及它是否服务于评估目标。
- 目标差异是否造成指标下降尚未得到证明；机制解释只能作为待核实假设。

### 审阅建议的论证链

1. 联合训练解决什么实际问题：一个模型兼顾两项原生任务，并呈现相对单源模型的权衡。
2. 为什么保留细标注：粗粒度输出仍可受益于更细的监督。
3. 共享属性监督是否决定最佳输出方式：readout 与训练方案仍影响原生任务表现。
4. 更多监督是否自然改善迁移：应具体考察监督目标、评估目标和数据条件。

## Claude 反馈

已收到，用户转交。以下为 Claude 的意见与推断，包括尚待核实的数字、机制解释和因果主张；记录不表示已认可或完成核验。暂不执行其建议的新统计。

### 总体框架

Claude 认为各段仍是“变量改变 → 数字”，应明确说明结果对应的 Introduction 主张，以及 Figure 1 的哪项差异与之相关。建议围绕以下四项承诺展开：

1. 一个模型同时保留两项原生任务。
2. SPRSound 的细 C/W 标注可用于 ICBHI，压缩到 binary 会丢失相关信息。
3. 属性层与任务层需要分别评价。
4. 兼容标签下仍存在声学分布差异，共享监督需要容纳这种差异。

### P3：联合训练与单源对照

- 建议先讨论直接跨源应用较差，再解释联合训练的收益和代价。
- 将 ICBHI-only 在 SPRSound 的约 49.5、SPRSound-only 在 ICBHI 的约 47.8 与 Figure 1(b,c) 联系；Claude 将其描述为分布差异的“实验版本”，建议作为 Section 2 takeaway 的验证。
- 用 SPRSound 提供约 1,800 个含 C/W 标注事件、ICBHI Wheeze/Both 为 886/506 个周期联系 Introduction 的 Wheeze 例子。这些数量及所涉划分待核实。
- 恢复或在正文使用 Sp/Se：LSAA Se 47.35 对 ICBHI-only 42.79，Sp 74.98 对 72.77。Claude 将异常子类召回的较大增益解释为细标注帮助子类判别。
- 将 SPRSound-only 92.8 与 frozen BEATs 92.4 接近解释为二分类任务接近饱和、提升空间有限；将 ICBHI frozen 约 51–52 对微调单源 57.8 解释为提升空间较大。
- 建议据此给 frozen references 明确角色：解释两项原生任务的不同提升空间，以及联合收益/代价的不对称性。

### P4：HF/KAUH 外部评估

- 应优先讨论源控制：SPRSound-only 的 HF 79.3、KAUH 67.3 最低；ICBHI-only 的 HF 87.1 最高；LSAA 的 HF 接近 ICBHI-only（差约 −0.6），KAUH 高约 1.7。
- Claude 引用 Figure 1(c) 的声学质心：KAUH 四组约 163/146/161/155，ICBHI 约 148/144/153/147，SPRSound 约 177–215，HF 约 258–286。数值、描述量与图件版本待核实。
- 建议将 KAUH/ICBHI 的接近和 SPRSound 的偏离联系到外部迁移方向，表述为联合训练保留了 ICBHI 的迁移能力并有所补充。
- 建议引入人口学解释：SPRSound 为儿科数据，ICBHI/KAUH/HF 以成人为主；需核实数据集论文与人群范围。

### P5：Coarse SPR

- 指出 Coarse 的 ICBHI 约为 61.17−6.51=54.7，低于 ICBHI-only 的 57.8。
- Claude 据此提出较强结论：“只给 SPR 二分类标签时，联合训练对 ICBHI 有害”“ICBHI 收益完全来自细标签，而非更多音频”，并建议据此回应数据量质疑。这些是待审的因果主张，不是本任务已确认结论。
- 用 SPR C/W AUROC 差 −8.48 补充说明细标签有助于本源属性判别。
- 建议将 all-Normal seed 0 与 L_A/3 的限定保留在结论之后。

### P6：Native + C/W

- Claude 将 SPR C/W AUROC 差约 −0.05 解释为“共享表示没变”，将 ICBHI +2.50 解释为“全部来自决策层”。这些强机制主张须与重新训练 encoder/heads 的事实一起审查，当前只记录。
- 建议明确回答为何仍使用两级 readout：核心贡献是共享属性监督，readout 提供显式、可解释映射，并以约 2.5 pp 的代价体现属性质量与任务精度的区分。

### P7：HF-on

- 强调 CAS −7.41 且三个 seed 均下降，应分析而非仅称为 task-/seed-dependent。
- 提出目标不一致的解释：HF 中没有 Wheeze 的 Rhonchi/Stridor 资格窗口作为 W 负例参与训练，而 CAS 评估将这两类纳入正例；SPRSound 的 Rhonchi/Stridor 又仅监督 A，对 W 属未知。
- Claude 将该差异作为 CAS 下降的原因，并回扣“标签映射影响共享监督”。该解释的代码事实和因果力度均待核实。
- 对 ICBHI −1.90，建议谨慎联系 HF 较远的声学质心及 D 标签的较粗粒度；这些解释尚未核实。
- 建议新增统计：661 个 CAS 阳性录音中有多少没有 Wheeze 区间。用户尚未授权执行该建议，本轮不运行。

### 结构建议

- 合并压缩 setup 两段。
- 每个结果段以问题引导，以 takeaway 回扣 Introduction 或 Section 2。
- 将 frozen references 用于解释任务提升空间，放在联合结果之前。
- 建议至少三次引用 Figure 1：跨源表现、外部迁移方向、HF-on 代价。

### Claude 提出的待决事项

1. 是否将 Sp/Se、AS/HS 放回 Table 1，或在正文保留必要拆分指标。
2. 是否写入 HF Rhonchi/Stridor 的 W 负例与 CAS 正例之间的目标差异。
3. 是否引入儿科/成人的人口学解释，并核实相应数据集来源。

## Gemini 反馈

已收到，用户转交。以下为 Gemini 的意见，不代表已确认的事实或机制。

- 联合训练：建议将跨源低表现与 Figure 1 的谱质心/PCA 差异形成“因果闭环”，并将 LSAA 的表现归因于属性瓶颈提取跨域不变声学基元、避免设备/人群噪声。其引用的 SPRSound-only KAUH 37.25% 与当前表格不符。
- Coarse：建议用 Figure 1(a) Normal 6,199 与异常长尾解释 all-Normal 运行，声称细 C/W 监督通过声学正则化稳定表征。相关训练分布与机制待核实。
- Native + C/W：建议将近似不变的 AUROC 与较高 Score 解释为表征可分性与决策面在机制上解耦；将显式读出称为临床可解释的无参数硬规则，native head 则吸收阈值偏差。
- HF-on：建议将较高谱质心与 derived negatives 联系为冲突梯度、假阴性噪声，并由此证明简单联合不可行、选择性 mask 必要。
- 结构：压缩 setup，把空间留给机制解释与 Figure 1/2 回扣，减少正文逐项复述数字。

## 三份反馈的合并判断（讨论准备，未改稿）

共同有效的写作诊断：正文应说明比较回答什么问题，而不只报告涨跌。可以采用“单源监督覆盖不足 → 联合兼顾原生任务 → 保留细监督 → 区分监督与输出实现 → 按目标解释外部评估”的论证线。

本轮仅阅读现有代码、稿件与已保存汇总，并对已有数值作简单算术；没有启动训练、推理、原始标签重评分或新实验。

### 已核对的事实与解释边界

1. SPRSound-only 的 KAUH BA 为 **67.25%**，不是 Gemini 所写的 37.25%。
2. Coarse 对 ICBHI-only 的平均差距约 −3.12 pp，但逐 seed 为 **−8.1495、−1.7455、+0.5404 pp**。因此不能概括为每个 seed 都有害。监督撤除伴随 calibration 与相对 loss 强度变化，不能说全部收益由细标签造成。
3. Figure 1(c) HF 三组中位数为约 **258.46、285.84、249.43 Hz**。它们是 source-test 的观测 D/W 组合，非 CAS 评价分组，也非 HF 辅助训练分布本身。KAUH Both 只有两名患者。单个描述量接近不能证明迁移机制或人口学因果关系。
4. 相近的 SPRSound C/W AUROC 不能证明共享表示不变，尤其 native-head 对照重新训练了 encoder 和 heads；该指标也不能替代 ICBHI 属性质量证据。
5. HF 代码对含 D/Wheeze/Rhonchi/Stridor 的窗口启用 C/W loss，W target 取决于 Wheeze 标注是否存在。因此 R/S 且无 W 标注的窗口会得到 W=0；SPRSound R/S 则映射为 A=1、C/W 未知。
6. 同一批 957 条 HF 录音的 CAS 阳性为 661、Wheeze 阳性为 405；按已确认 CAS 并集定义，**256 条 CAS 阳性没有 Wheeze 标注**。这是对现有汇总的差值计算，不是新模型评估，且不证明未标注 Wheeze 在声学上不存在。
7. CAS/W 的目标不等价是已确认事实；它是否造成全部 CAS 下降尚未被隔离验证。不能直接称 W 负例为 CAS 训练标注错误，也不能声称实证证明了冲突梯度或假阴性。
8. HF-on 已使用当前 mask 机制；其下降不能用来证明 mask 已解决问题或必不可少。
9. 当前显式 readout 包含验证集选择的阈值与回退规则，没有临床可解释性验证；不能升级为临床可靠性或完全无参数机制的证明。

### 拟讨论的取舍

- 保留现有两 subsection 和 Table 1/2(A/B/C)，不为解释恢复整套 Sp/Se/AS/HS 列。若使用召回拆分，仅在正文保留最必要的观察。
- 外部结果优先比较 LSAA 与单源控制，再把 frozen references 作为背景。
- Figure 1 用于说明输入差异确实存在、提供一致性背景，不用作域偏移、年龄、疾病或设备机制的因果证明，不预设必须引用三次。
- 保留“粗输出仍可利用细监督”与“属性检测和原生任务表现需要分别评估”的实证认识，删除表示不变、完全由决策层造成等无对照支持的推断。
- 将 HF-on 段提升为“同一来源不等于同一监督/评估目标”，并清楚区分观测、定义差异和机制假设。

### 本轮核对来源

- `docs/paper/Overleaf_Sync_final/Section/2data.tex`、`Section/3method.tex`、`Section/4evaluation.tex`、`Section/4tables.tex`。
- `baseline/pafa/joint_hierarchy_hf_auxiliary.py`：`_make_hf_windows`。
- `baseline/pafa/table2_clean_controls.py`：SPRSound 属性映射、Coarse eligibility；`table2_benchmark_controls.py` 复用这些函数。
- `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/multiseed_summary.json`。
- `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/seed_0/hf_metrics.json`。
- `result/reproduce/pafa_joint_hierarchy/PAFA_TABLE2_POSTHOC_HF_CAS_KAUH_20260914/results/hf_cas_kauh_table2_supplement.json`。
- `/Users/zilongzeng/.codex/worktrees/3110/Acoustic/figure1_panel_c_four_sources/README.md` 及 `source_data/four_class_four_source_medians.csv`。

## 后续讨论

三份反馈已齐全，当前向用户反馈综合判断。等待讨论确认后再写回；本轮未修改论文正文、表格、图件或排版。
