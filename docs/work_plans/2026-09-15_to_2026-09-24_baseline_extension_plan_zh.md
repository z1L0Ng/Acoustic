# 2026-09-15–09-24 Working Plan：PC-MCL与DCASE补证，Hanlin负责PAFA与既有对照

Notion：[当前工作计划](https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8)。

状态（9/17更新）：DCASE三个seed及四列终点评测完整，成功资产已拉回本地；只将DCASE轻量指标/预测/日志与汇总加入Git，完整checkpoint留本地和服务器。按用户修正，不保留本次PC-MCL失败文件的本地导入副本。当前先提交/推送这些结果与既有代码文档快照，再由“模型设计”接管PC-MCL代码整改。全部训练和heartbeat继续暂停，没有有效PC-MCL三seed基线，不能据此断言作者方法不可复现。

## 1. 本周期目标与责任

当前优先级：保留DCASE完整结果，核查PC-MCL作者配方与5-s适配差异并确认后续方案。PC-MCL队列和监控均已停止，不自动重启。核查与修复见`docs/baseline_design/2026-09-17_pcmcl_numerical_failure_and_source_audit_zh.md`。历史PAFA/BEATs队列不恢复。

- 项目侧：按已批准方案完成三seed，核验四列结果，再与用户讨论论文整合。
- 模型设计：完成当前方案、数据与标签接口、训练/选模/恢复、固定评测、三seed队列、汇总和运行说明；在main提交相关代码及设计文档，不推送，不混入并行稿件或管理文件。
- Hanlin：PAFA三seed，以及已有AST、BEATs、PANNs、OPERA-CT、HeAR五组frozen参考的复跑。PAFA具体recipe和三seed含义仍需明确；材料准备不等于已通知本人或已启动。
- 写作：既定双任务分工保持。当前只形成论文整合方案，写作不等待全部baseline返回；独立reviewer仅由用户明确启动。
- 不设Hanlin个人硬截止。下一工作周期未获反馈的缺项由项目按资源接手。

## 2. 两组实验分别回答什么问题

| 方法 | 训练来源及角色 | 主要问题 | 主比较 |
| --- | --- | --- | --- |
| PC-MCL（5 s适配） | 仅ICBHI源训练；SPR/HF/KAUH固定模型直接迁移 | 呼吸音领域内的N/C/W多标签、多周期输入和患者匹配，在单源训练后能否支持另外三个数据集的任务？ | LSAA主方法 |
| DCASE-style native union + mask | ICBHI+SPR联合；两源原生任务评测，HF/KAUH外部评测 | 已有原生类别并集、语义映射与缺失标签mask的联合学习，与LSAA共享声学属性及任务读出相比表现如何？ | LSAA主方法 |

PC-MCL的单源与LSAA多源差异是研究问题的一部分，不追加LSAA ICBHI-only作为必要新控制。DCASE使比较进一步涉及已有多源异构标签学习。两者都是系统级参照；不预设失败，不把差值单独归因于loss、head或mask。已有Table2控制继续承担LSAA内部机制分析。

## 3. 已批准的PC-MCL方案

- 源训练只用ICBHI，seeds 0/1/42。保留N/C/W additive supervision与patient-matching，源训练更新BEATs及预测模块；随后固定整个模型。
- 两段真实cycles分别repeat-pad/center-crop到2.5 s，在波形域拼成5 s；单unit验证/测试为5 s。这是5-s适配，不使用论文原10-s结果冒充本次数字。
- loss为BCE(N/C/W)+0.1 CE(patient)。每seed完整400轮，关闭patience早停，LR下降点120/160；strict improvement与原Se资格保持。修正runner忽略early_stopping/run_full_epochs配置的问题。
- 保留已核查ICBHI benchmark选模/eligibility，并如实标明test-selected；SPR/HF/KAUH不参与训练、选模、阈值或readout选择。
- ICBHI：固定C/W规则还原四类；SPR：四类读出归并Normal/Adventitious。
- HF：三个5-s窗口maximum p_W，作为CAS ranking proxy。
- KAUH：每view max(p_C,p_W)，同患者B/D/E均值后0.5一次分类；沿用86位兼容患者。
- 三个seed在`result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_400epoch`从预训练初始化；旧`PC_MCL_ICBHI5s`的10轮记录及last checkpoint保留，不复用旧早停状态。

依据：[PC-MCL论文](https://arxiv.org/abs/2601.17080)、[官方代码](https://github.com/wa976/PC-MCL)。当前实现规格：`docs/baseline_design/2026-09-16_pcmcl_icbhi5s_source_transfer_spec_zh.md`。

## 4. 已批准的DCASE呼吸音适配

### 4.1 原生类别并集及mask

九个独立sigmoid输出：

`N_ICBHI, N_SPR, Crackle, Wheeze, Both, FineCrackle, CoarseCrackle, Rhonchi, Stridor`

| 类别关系 | 已批准处理 |
| --- | --- |
| 两源Wheeze | 共享原生Wheeze类别输出 |
| ICBHI Both与SPR Wheeze+Crackle | 共享Both输出 |
| SPR Fine/Coarse Crackle与ICBHI Crackle | 保留三个类别；SPR细类单向对应ICBHI Crackle类别，不能反推细类 |
| 两源Normal | 分别保留N_ICBHI与N_SPR |
| Rhonchi、Stridor | 分别保留；ICBHI未标注这两类的样本不能自动作为阴性 |

Crackle是ICBHI的“有C、无W”原生类别，不是所有含C声音的属性；Both不拆成C/W双阳性。实现必须逐项列明positive/negative/unknown和alias负监督依据，不把未标注填0。

保留frozen BEATs temporal frames、可训练log-Mel CNN、temporal fusion、BiGRU、sigmoid与attention pooling，使用masked BCE和训练时的class mask。推理不能读取目标真值生成mask。保留/省略的Mean Teacher、mixup、SED时间监督与后处理需在实现说明中准确列出；方法称DCASE-inspired respiratory adaptation。

### 4.2 多源验证选模

遵循DCASE结合各源验证表现选择同一checkpoint的原则，呼吸音适配的已批准目标为：

`M = 0.5 × ICBHI native4 validation macro-multilabel-F1@0.5 + 0.5 × SPR native7 validation macro-multilabel-F1@0.5`

- 每源分别计算macro-F1，再等权组合；SPR评测仅原生七类，不重复计alias输出。
- 固定0.5阈值，每epoch验证；max50、patience10、min_delta0，strict improvement，tie保留较早checkpoint。
- 复用适当的既有source-train内部subtrain/validation划分；静态核对每seed样本、患者、类别support和official train/test边界，记录zero_division约定。
- ICBHI/SPR official test与HF/KAUH不用于DCASE选模、早停或阈值拟合。
- 用户当前批准覆盖此前DCASE仅ICBHI选模的限制；PC-MCL原选模不变。
- 原DCASE默认checkpoint monitor是多个验证F1项之和；论文PSDS1+segMPAUC为调参/挑战主指标，不将我们的两源分类F1写成原版SED公式。

依据：[官方训练配置](https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/confs/pretrained.yaml)、[选模实现](https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/local/sed_trainer_pretrained.py)。

### 4.3 四列固定评测

| 数据集 | 读出与指标 |
| --- | --- |
| ICBHI | 在N_ICBHI/C/W/Both原生分数上argmax，报告ICBHI Score |
| SPR | 在N_SPR/FC/CC/W/Both/R/S七类分数上argmax，再归并Normal/Adventitious，报告official Score |
| HF | 每窗max(W,Both,R,S)，再跨三个5-s窗口取max；在既有CAS eligible pool计算AUROC |
| KAUH | 同患者B/D/E的ICBHI四类native scores先平均，再argmax并归正常/异常；86位兼容患者计算BA |

分数为sigmoid及既定attention pooling输出，不额外subset softmax。HF包含Both，Crackle-only不进入CAS；max是排序分数，不称独立sigmoid之和为并集概率。SPR已是训练来源。HF/KAUH均不新建目标监督头。

HF的DCASE分数显式覆盖R/S，PC-MCL和LSAA现有分数是Wheeze proxy；该差异必须披露，不能从AUROC差距单独推出表示学习更强。

## 5. 代码、Git与预算交付

当前状态：DCASE COMPLETE 3/3 / IMPORTED；PC-MCL NUMERICAL FAILURE / STOPPED。DCASE本地完整副本在`result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s/`，Git结果包在`docs/result_exports/2026-09-17_dcase_joint_native_union/`。此前NaN处理与源码核查将随本次资产快照保存；后续代码由“模型设计”在main完成，不能将公开默认参数直接当成产生论文结果的完整命令。

原17–19小时完成估计已失效：三个seed均出现NaN。seed0/1虽执行到400轮并生成summary，不能据此认定有效训练；seed42在40轮中断。新训练时间须在配方确认后重估，不能沿用旧队列完成承诺。

文档收尾已完成：`docs/baseline_design/2026-09-16_source_baseline_implementation_brief_zh.md`的标题及顶部说明已由普通追加commit `3ece467`保存。管理核对该提交仅修改此一个文件，暂存区为空。此前的文档提交审批阻塞已解除，无需再确认；没有amend或改写已有历史。

检查已完成：静态编译/import/config解析；九类positive/negative/unknown映射；DCASE官方类别轴attention与帧概率加权公式；eligible元素归一化BCE；多源F1计算；三seed源内患者划分与cache角色数量；早停/恢复控制逻辑。管理核对了相关代码与已提交本地依赖；没有模型forward或音频解码验证。

元数据核对：ICBHI seed0/1/42的subtrain/validation分别3636/506、2880/1262、3174/968 cycles，各为63/16位患者；SPR为5219/1437 events、194/49位患者。SPR validation的Stridor阳性支持量为0；保留原生七类macro-F1及zero_division=0，记录该类无法由此验证集直接评估的限制。

- 完整接通provider、标签/alias/mask、模型和loss、source validation、checkpoint/恢复/早停、四列评测、逐样本NPZ、三seed队列与mean/sample SD汇总。
- 采用明确配置记录来源采样、每epoch曝光、选模、读出和seed；未完成seed不汇总为n=3。
- 仅作静态语法/import与直接相关纯函数/控制逻辑/metadata检查；不做模型forward、smoke、profile或任何新数据/模型运行。
- 在main提交本次预算、执行控制和相关说明改动，推送后由服务器fast-forward同步。保留其他任务改动及既有结果，不改写历史。
- 历史本地50轮预算为35–87小时，仅保留为原实施阶段估计；不用于当前400轮CUDA排程。
- 旧DCASE I-only flat4、C/W/R/S属性头和strict11草案均不再是当前执行方案；旧30–62小时总预算不用于新排程。

实现入口：`baseline/frozen_method_baselines/`。DCASE当前设计入口：`docs/baseline_design/2026-09-16_dcase_joint_multilabel_candidate_zh.md`（内容由实现任务更新至已批准方案）。预算入口：`docs/baseline_design/2026-09-16_source_baseline_local_runtime_estimate_zh.md`。

## 6. 论文放在哪里、怎么解释

- **Section 1 Introduction**：将PC-MCL作为呼吸音单源多标签学习参照，将DCASE作为异构多源/缺失标签学习参照。准确承认已有多标签与masked joint learning；论文重点是呼吸音异构原生任务中的具体设计与经验发现。引用`jeong2026pcmcl`和`cornell2024dcase`。
- **Section 2 Datasets**：交代训练来源、原生任务和HF/KAUH外部角色；不把SPR在两方法中的角色混写。
- **Section 3 Method**：解释LSAA为何通过A/C/W共享属性和任务读出连接任务。DCASE详细实现放Evaluation baseline说明，不扩写成另一篇方法介绍。
- **Section 4.1 Joint Learning and Transfer / Table 1**：加入“PC-MCL (5 s)”和“DCASE-style union + mask”两行，四列沿用ICBHI Score、SPR official Score、HF CAS AUROC、KAUH patient BA。两行都是三seed实际重跑结果；不能用发表的ICBHI数字拼接本机迁移数字。
- **Table 1 caption及协议说明**：当前“LSAA and frozen-encoder references”需改成能覆盖学习方法的标题；明确PC-MCL源训练更新encoder、DCASE冻结BEATs、训练来源和选模差异。已有五组参考、方法baseline和LSAA可分块，不将全部称为frozen encoder。
- **Table 1后的讨论**：先解释PC-MCL本域与迁移表现，再解释DCASE两源原生任务与外部表现。结果允许优势、持平、取舍或劣势；不预先填写“不能迁移”或“LSAA普遍更好”。
- **Section 4.2 / Table 2**：保持LSAA来源、监督、readout与HF-on消融，承担内部机制证据；不把PC-MCL/DCASE伪装成只改一个组件的消融。
- **Abstract / Conclusion**：待完整结果返回后再据证据调整claim，不预写胜负。

解释边界：PC-MCL与LSAA训练来源不同；DCASE与LSAA还涉及编码器训练方式、架构、输出与选模差异；HF声类覆盖也不同。因此Table 1回答系统表现和适用边界，不能单独证明某个loss、head、mask或hierarchy的因果优势。DCASE可以服务不同原生任务，不能把“已有方法无法联合异构数据”作为结论。

写作责任保持：论文写作负责Sections 1/4/5、Abstract、Table1/2和共享文件；论文写作二负责Sections 2/3与对应图表。本轮管理不直接改稿或触发独立review。

## 7. 推进顺序与完成清单

沿用本周期9/15–9/24计划，内部提交缓冲仍按纽约9/24 07:00安排；Hanlin不设个人硬截止。

1. 9/16–17：完成代码、metadata口径、直接检查、Git提交和更新预算。
2. 9/17：已获两卡启动和PC-MCL 400轮授权；服务器完成PC-MCL三seed，DCASE三seed结果已返回。保留逐样本预测和完整四列结果。
3. 9/21–22：核验本地结果及Hanlin回报，缺项按资源接手。
4. 9/23：结果解释、Table1/caption、正文一致性与四页稿整理；独立review仍单独由用户启动。
5. 9/24：预留最终检查和提交缓冲。

- [x] 确定PC-MCL单源迁移与DCASE异构多源联合学习的不同实验问题。
- [x] 批准DCASE九输出、共享/单向映射/独立类别及四列读出。
- [x] 批准DCASE多源validation macro-F1选模；PC-MCL既有benchmark选模保留。
- [x] 向模型设计下发完整实现、必要检查、预算及相关代码Git提交要求。
- [x] 明确两组实验在Introduction、Evaluation/Table1和Table2机制分析中的分工。
- [x] 完成新DCASE实现，核对PC-MCL与共享执行入口。
- [x] 核对source内validation分区、类别mask、支持量与读出。
- [x] 核验直接检查、更新预算与代码Git提交（53f54eb、8575da1）。
- [x] 完成implementation brief单文件的普通追加提交（3ece467）；已核对范围，审批阻塞解除。
- [x] 保存并推送本地仓库、同步imec main；收到两卡正式执行授权。
- [x] 修复DCASE label-free终点评测错误，服务器确认三个seed四列全部完成。
- [x] 明确PC-MCL每seed400轮、关闭patience早停、恢复120/160节点并保留旧失败记录。
- [x] 用户批准中断PC-MCL，核验进程/GPU释放并暂停原heartbeat；全部产物保留。
- [x] 完成PC-MCL源代码差异核查、NaN终止/预测/旧结果排除修复及7项直接检查。
- [x] 仅导入成功DCASE的47个原始文件及成功运行日志，核对路径/大小与三seed有限训练记录；清理本次PC-MCL导入副本，服务器原件保留。
- [ ] 完成DCASE结果与当前工作区Git快照推送后，将后续代码整改交给“模型设计”。
- [ ] 明确下一轮PC-MCL配方、是否先完成一个正式seed及启动授权；当前不重跑、不改科学超参数。
- [ ] 明确Hanlin PAFA具体recipe/seed含义并核验PAFA及已有五组复跑产物。
- [ ] 根据实际结果更新Table1、讨论、Abstract/Conclusion和最终稿。

已完成的LSAA主模型、核心控制、HF-on和CAS/KAUH结果继续复用，不重复训练。9/14夜间计划保持Done：[9/14夜间计划](https://app.notion.com/p/3db309efda29817f856ff55a4c82e8e3)。学生交接入口：`docs/student_tasks/2026-09-16_hanlin_pafa_handoff_zh.md`、`docs/student_tasks/2026-09-16_hanlin_existing_baseline_rerun_zh.md`。
