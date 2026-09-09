# Story / Contribution / Skeleton 审阅稿

日期：2026-09-07。状态：**工作稿，待用户集中确认；不是最终创新声明或实验验收。**

依据主目录[09-07 Working Plan](/Users/zilongzeng/Research/Acoustic/docs/work_plans/2026-09-07_work_plan_zh.md)。当前执行为并行准备、集中确认，替代09-06讨论稿中的逐项等待流程。

## 建议的论证

我们希望共同利用不同来源的呼吸音标注，同时保留这些标注原本要回答的问题。ICBHI用cycle四分类，SPRSound的主要评价用event二分类；其中部分事件标签能支持Crackle/Wheeze属性，其他标签只支持Abnormal判断。因此，共享学习需要明确监督可以共享到哪一层，以及最终如何回到各自的任务。

现有固定专家checkpoint的SPRSound表现提供实际复用动机，但不能把其表现直接归因于标签不兼容，也不能据此断言learned features不可迁移。方法的回应是共享BEATs表示、按标注支持训练三个节点、保留两种native readout。现有三种子结果展示了同时完成两任务的能力与不均衡的类别表现；是否比matched independent heads或single-dataset training更有价值仍未得到回答。

读者可见主线为：**原生任务不同、监督部分共享 → 具体联合学习问题 → 标注可用性与层级读出的设计 → 当前native benchmark与类别行为 → 尚未被对照回答的方法作用。** 不把已经存在的masking或label harmonization重新命名为全新原理。

工作title保持：*Eligibility-Aware Hierarchical Task Alignment for Heterogeneous Respiratory Sound Learning*。

当前可支持的研究问题：在共同学习ICBHI cycle四分类与SPRSound event二分类时，模型能保留哪些原生预测能力，表现在哪些类别上受到限制？原先“是否超越matched independent-head和single-dataset training”的问题保留为必须由对照回答的部分。

## 三条候选贡献及其强度

| 候选 | 供集中讨论的paper-level表述 | 当前证据与创新边界 | 建议 |
|---|---|---|---|
| C1：具体问题表述 | *We formulate joint learning of respiratory cycles and sound events through annotation-aware sharing of respiratory attributes while retaining their distinct native prediction tasks.* | 当前已实现cycle-flat4/event-binary的任务处理。但DCASE2024已涉及异质音频监督、缺失标签masking和分别评价；Bevandic/Schutera也已有通用方法。具体问题表述不自动成为独立创新。 | 保留方向；若与C2的增量无法区分，合并成一条方法贡献。 |
| C2：方法使什么成为可能 | *We introduce an eligibility-aware hierarchy that jointly learns shared abnormality and supported sound attributes, allowing one respiratory-sound model to retain cycle-level and event-level readouts.* | 已有实现与两任务结果。层级属性、未标注不当阴性、BEATs分别已有先例；尚无matched heads证据证明这套组合的额外价值。 | 作为方法候选；正文只陈述实际设计与能力，不称first或优越。 |
| C3：研究获得什么认识 | *We characterize the native-task behavior of a shared respiratory-sound model, showing that close aggregate scores can coexist with uneven class-specific recognition.* | 三种子汇总与per-class recall支持描述性观察。原先“disentangle joint supervision与hierarchy”需要未完成的matched controls。 | 当前保留为经验分析，是否独立贡献待集中判断；不预设必须凑足三条。 |

最接近先例的已知重叠包括：[DCASE2024 Task4](https://arxiv.org/abs/2406.08056) 的跨数据集标注与masking、[Bevandic WACV2022](https://arxiv.org/abs/2108.11224) 的重叠标签学习、[Schutera 2022](https://doi.org/10.1371/journal.pone.0263656) 的异质标签损失，以及[LungMix](https://arxiv.org/abs/2501.00064) 的呼吸音标签统一与单源泛化。已整合[文献定位稿](/Users/zilongzeng/.codex/worktrees/4e78/Acoustic/docs/literature/PAPER_NOVELTY_POSITIONING_2026-09-07_zh.md)与[缺失证据稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)的必要结论；正式创新强度仍待集中确认。

## 已实际推进的稿件结构

- **Introduction**：原生单位/标注差异；固定head直接复用的问题；承认DCASE、partial-label、LungMix、OPERA、PC-MCL先例；明确本研究的两任务范围。未确认的创新措辞不写成已成立的贡献。
- **Data**：ICBHI official recording split、2,756 test cycles与四类support；SPRSound official-inter 1,429 events；内部group-disjoint validation；节点映射与5s输入。Figure1现在只保留可核实的任务/标签可用性矩阵，未填group-aware separability或coverage数字。
- **Method**：BEATs token mean pooling → 256维共享线性投影 → Level1/Crackle/Wheeze；按eligible样本求节点loss；继承PAFA目标；验证阈值与明确native decoder。Figure2提供可编辑SVG和矢量PDF，HF/KAUH不在主训练路径中。
- **Evaluation**：完整已知训练设置；逐epoch ICBHI-test selection披露；TableI分published/local/transfer/joint三个块；给出Sp/Se和per-class结果；TableII如实标四项validation-selected结果缺失；HF/KAUH仅作有界诊断。
- **Conclusion**：总结已实现的两任务预测及类别差异；明确当前证据不能证明hierarchy superiority或joint learning的因果净收益。

## 数值、实现与来源说明

- JH2仅正式0/1/fresh42：ICBHI Score61.17±0.31%，SPR official Score90.70±0.34%；本审阅布局另列既有AS90.95±0.31%。不是clean estimate。
- TableI published PAFA/BEATs+CE为五run论文值，来自[已有primary-source audit](/Users/zilongzeng/.codex/worktrees/4e78/Acoustic/docs/literature/beats_followup_primary_source_audit_2026-08-29.md)；local checkpoint行及其transfer单列，不把paper row当作本地transfer的同一checkpoint。
- Local ICBHI来源：[已核实checkpoint汇总](/Users/zilongzeng/Research/Acoustic/result/icbhi_strong_method_reproduction/metrics.json)。Transfer AS来源分别为[PAFA](/Users/zilongzeng/Research/Acoustic/result/pafa_sprsound_transfer_20260722_235659/metrics.json)、[SG-SCL](/Users/zilongzeng/Research/Acoustic/result/sg_scl_sprsound_transfer_20260722_235659/metrics.json)、[Patch-Mix](/Users/zilongzeng/Research/Acoustic/result/sprsound_patchmix_frozen_transfer/metrics.json)。
- 正式统计与类别support：[JH2汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md)及[对应JSON](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.json)；外部数字来自[HF/KAUH汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.md)。均直接转写既有值，没有新统计分析或模型重评。
- ICBHI Se是异常细类正确率；Crackle误判Wheeze仍计错。正文未沿用“只损失specificity”或“异常检出完全保留”。
- 层级的三个输出共享输入表示；Crackle/Wheeze是marginal sigmoid，不是假设已经实现的conditional probability factorization。分类头的256维投影与PAFA训练辅助投影分开说明；后者不进入native分类读出。
- SPRSound的native binary endpoint不等于binary-only training：支持的原始event类别还监督Crackle/Wheeze；Rhonchi/Stridor只支持Level1并mask属性。因此，hierarchy与只有SPR binary head的independent设计同时改变了监督信息与分类结构。当前正文明确该混杂；即使四项结果完成，也不能自动称作“纯hierarchy结构”的因果分解。
- 已完成JH2的阈值来自同epoch内部validation，按两dataset的eligible-node F1等权平均选择，随后才访问该epoch ICBHI test。这个历史规则不自动成为新clean suite合同。

## 缺失证据与集中确认内容

1. **方法优势**：缺少共同validation-selected合同下的joint hierarchy、joint independent heads、ICBHI-only和SPRSound-only四项结果；当前不能声称已经分离两种因素的作用。
2. **Figure1最终数值**：缺group-aware/class-matched separability与5s coverage；当前矩阵仅说明事实，不用旧0.9353或PCA充当新定量证据。
3. **学生背景结果**：Hanlin现有材料是未独立核实的seed42背景报告，新增seed0/1尚无完成证据；未纳入TableI正式数字。
4. **创新强度**：需集中决定C1与C2是否合并，以及是否让C3仅承担结果解释。保留具体设计与能力，并不等于已确认方法新颖性。
5. **版面**：通过管理提供的便携Tectonic 0.17.0完成了实际论文编译，修正Times字体替代、公式超宽与TableI浮动位置后，最终[审阅PDF](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-07.pdf)为**4页（含参考文献）**。已逐页渲染检查，无裁切、重叠、未解析引用或超宽盒；Figure1在第2页，Figure2与TableI在第3页，TableII在第4页。日志仍有两条非阻断underfull提示与图PDF版本提示，实际图像显示正常。作者信息尚未填写，当前4页不等于最终投稿版面已冻结。

## 可审阅文件

- [整篇PDF](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-07.pdf)
- [LaTeX入口](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/main.tex)，按顺序载入Section1–5。
- [Figure2可编辑SVG](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/Figure/figure2_method.svg)与[矢量PDF](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/Figure/figure2_method.pdf)；caption在Method中。
- [引用库](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/citation.bib)：新增DCASE引用；排版控制只省略打印的长URL、收起长作者列表，源元数据保持完整。

摘要与引言已按管理最终行文意见，以具体问题、设计、结果为主；摘要保留一次test-selected限定，Protocol与TableI保留完整selection说明，Conclusion只用一句保留matched对照限制。

本轮只修改本worktree的论文文件；主目录稿件、Notion和管理计划保持不变。未训练、validation/test、特征提取/cache、服务器操作、模型重评、Git提交推送或外发。工作稿完成即回报管理供集中审阅，不再设置逐项确认门。
