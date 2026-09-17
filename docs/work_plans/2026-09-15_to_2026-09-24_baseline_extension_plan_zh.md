# 2026-09-15–09-24 Working Plan：PC-MCL与DCASE补证，Hanlin负责PAFA与既有对照

Notion：[当前工作计划](https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8)。

状态：PC-MCL与DCASE已完成实现、静态/纯函数检查和main上的相关提交。主实现为`53f54eb`，预算与metadata补充为`8575da1`，实施说明收尾为`3ece467`；未push，未启动实验。管理已核对提交范围、关键公式、三seed划分与读出，并更新本计划和Notion；当前不直接改论文。

## 1. 本周期目标与责任

当前优先级已按用户最新指示调整：先完成两组run，再讨论新增结果如何写入paper。现在先保存并推送本地全部可版本管理改动，再由Acoustic服务器任务在imec canonical main上同步代码、检查GPU/磁盘/环境/四数据集与权重路径，形成实际开跑准备。历史PAFA/BEATs队列不恢复；CUDA耗时不能沿用本地MPS估计，后续以正式运行日志更新。当前仅进入服务器准备阶段，尚无新实验执行结果。

- 项目侧：完成PC-MCL与DCASE三seed所需代码、协议和预算；收到启动指令后执行，核验四列结果并整合论文。
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
- loss为BCE(N/C/W)+0.1 CE(patient)。max50、patience10、min_delta0，LR下降点15/20。
- 保留已核查ICBHI benchmark选模/eligibility，并如实标明test-selected；SPR/HF/KAUH不参与训练、选模、阈值或readout选择。
- ICBHI：固定C/W规则还原四类；SPR：四类读出归并Normal/Adventitious。
- HF：三个5-s窗口maximum p_W，作为CAS ranking proxy。
- KAUH：每view max(p_C,p_W)，同患者B/D/E均值后0.5一次分类；沿用86位兼容患者。
- 当前无本次三个已训练source checkpoints，需要正式运行才能产生。

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

当前状态：CODE READY / NOT RUN。主实现`53f54eb`包含18个相关文件；追加`8575da1`统一预算并清理metadata；追加`3ece467`对齐实施说明标题与当前方案。提交未混入论文、学生或管理文件，未push。正式实验仍等待明确启动指令。

文档收尾已完成：`docs/baseline_design/2026-09-16_source_baseline_implementation_brief_zh.md`的标题及顶部说明已由普通追加commit `3ece467`保存。管理核对该提交仅修改此一个文件，暂存区为空。此前的文档提交审批阻塞已解除，无需再确认；没有amend或改写已有历史。

检查已完成：静态编译/import/config解析；九类positive/negative/unknown映射；DCASE官方类别轴attention与帧概率加权公式；eligible元素归一化BCE；多源F1计算；三seed源内患者划分与cache角色数量；早停/恢复控制逻辑。管理核对了相关代码与已提交本地依赖；没有模型forward或音频解码验证。

元数据核对：ICBHI seed0/1/42的subtrain/validation分别3636/506、2880/1262、3174/968 cycles，各为63/16位患者；SPR为5219/1437 events、194/49位患者。SPR validation的Stridor阳性支持量为0；保留原生七类macro-F1及zero_division=0，记录该类无法由此验证集直接评估的限制。

- 完整接通provider、标签/alias/mask、模型和loss、source validation、checkpoint/恢复/早停、四列评测、逐样本NPZ、三seed队列与mean/sample SD汇总。
- 采用明确配置记录来源采样、每epoch曝光、选模、读出和seed；未完成seed不汇总为n=3。
- 仅作静态语法/import与直接相关纯函数/控制逻辑/metadata检查；不做模型forward、smoke、profile或任何新数据/模型运行。
- 在main只提交模型设计负责的相关代码、依赖、配置和设计文档，保留其他任务未提交修改。当前不push。
- 更新预算已交付：PC-MCL三seed训练20–30小时、外评0.75–1.5小时；DCASE一次frame cache约0.6–1.5小时、三seed训练10–43小时、terminal评测0.5–2小时，合计约11–47小时。两方法本地串行约32–79小时，计划中值约52小时；含10%余量按35–87小时（约1.5–3.6天）理解。以上来自历史日志和静态规模外推，本次CRNN/MPS路径未实测，不能承诺早停时间。
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
2. 用户明确启动后：按本地队列执行PC-MCL与DCASE各三个seed；保留原始预测和完整四列结果。
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
- [ ] 获得正式启动指令并完成两方法各三seed及四列评测。
- [ ] 明确Hanlin PAFA具体recipe/seed含义并核验PAFA及已有五组复跑产物。
- [ ] 根据实际结果更新Table1、讨论、Abstract/Conclusion和最终稿。

已完成的LSAA主模型、核心控制、HF-on和CAS/KAUH结果继续复用，不重复训练。9/14夜间计划保持Done：[9/14夜间计划](https://app.notion.com/p/3db309efda29817f856ff55a4c82e8e3)。学生交接入口：`docs/student_tasks/2026-09-16_hanlin_pafa_handoff_zh.md`、`docs/student_tasks/2026-09-16_hanlin_existing_baseline_rerun_zh.md`。
