# Arian反馈后的Work Plan调整建议

初稿：2026-09-17；完成情况更新：2026-09-18。状态：**讨论稿，实验完成事实已核实；新的写作/分析任务尚待讨论，未自动下发。**

反馈原件与完整中文译文已保存。本稿中的分析和建议属于项目管理判断，不属于Arian原文。反馈翻译阶段未排查连接；随后按用户的完成情况查询，通过服务器最终回报及只读JSON核对确认下述结果，没有新增训练、推理或进程操作。

## 9月18日已核实的实验完成情况

PC-MCL新lr1e-4配方的seeds0/1/42均完成400轮及四列固定终点评测，日志、metrics、run_summary数值检查通过，最终汇总n=3且没有排除seed。三seed分别选中epoch3、3、23。最后一个seed于9/17 23:58 CDT完成，最终汇总于9/18 00:12 CDT生成；训练进程已退出，原heartbeat已暂停。DCASE三seed仍保持完成及本地归档。

下表为三seed均值±样本SD，单位为百分比。

| 方法 | ICBHI Score | SPR official Score | HF CAS AUROC | KAUH BA |
|---|---:|---:|---:|---:|
| LSAA主方法 | 61.17±0.31 | 90.70±0.34 | 86.44±2.13 | 72.17±1.64 |
| PC-MCL 5-s适配 | 59.81±0.80 | 48.91±2.59 | 78.64±7.00 | 75.39±4.55 |
| DCASE-style适配 | 54.42±1.25 | 91.42±0.39 | 80.97±6.42 | 77.40±4.38 |

LSAA减PC-MCL的均值差依次为+1.36、+41.79、+7.80、−3.22 pp，按未取整数值计算。PC-MCL仅用ICBHI训练；LSAA与DCASE都使用ICBHI+SPR，所以SPR差距包含训练来源差异，不能单独归因于A/C/W或层级读出。PC-MCL和LSAA都微调BEATs，增加了一个可训练骨干的领域内多标签方法参照，但仍没有单独分离PAFA、数据来源与读出效果。HF/KAUH存在明确任务取舍，不能宣称所有外部数据集均占优。

这说明本次5-s适配已产生数值有效的完整实验结果，不等于精确复现作者原论文配方或发表数字。新代码与学习率都发生过调整，不能把成功归因于学习率这一个因素。

PC-MCL原始成功结果仍在服务器`/files1/Zilong/Acoustic/result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_lr1e4_20260917/`，本次仅只读核对JSON，没有下载checkpoint或失败文件。DCASE继续使用已归档成功结果。

## 建议将当前计划转为写作与结果分析

1. **关闭项目侧这轮补跑任务。** 将PC-MCL和DCASE标为完成，停止等待这两组新数字，不自动扩大seed、配方或基线清单。Hanlin负责的PAFA及已有参照回报另行核验，不能随本轮一起自动勾选。
2. **先讨论C2/C3的overall论点。** 明确Table 1是系统比较、Table 2各控制具体回答什么；认可Native + C/W是有竞争力的替代读出，保留KAUH上的不占优结果。
3. **并行准备C1/C4的协议说明与Table 1更新稿。** 来源、梯度更新、阈值拟合、checkpoint/早停、终点评测各自用途写清；分别披露LSAA、PC-MCL、DCASE的训练来源、编码器和选模差异。
4. **使用现有结果回答统计问题。** 先固定比较与配对单位，再计算逐seed差值、区间及适当的配对检验；不将四舍五入后的表格作为原始数据，不为了显著性自动追加seed。
5. **最后决定是否需要新的匹配消融。** 仅当仍要保留关于PAFA以外独立因果贡献的claim、且现有控制不足时，再讨论无PAFA等匹配实验的价值与预算。完成主体后统一Abstract、Conclusion、图注与四页篇幅。

以上是建议顺序，不是新的实验、写作或独立审阅启动指令。

## 建议优先解决的核心问题

本轮优先级建议调整为：先把数据划分、训练配方和实验可支持的论点讲清，再决定是否需要补新的机制对照。继续增加Table 1 baseline不能自动解决C2的归因问题；统计检验也不能替代C3对读出必要性的概念判断。

当前论文可以讨论共享细粒度监督如何连接不同原生任务，以及不同读出的性能取舍。现有Native + C/W结果不足以支持“LSAA层级读出是必要或最佳选择”。是否继续把现有LSAA读出作为主方法，或重新组织主方法与变体，应先与用户讨论，不自动替换主方法。

## C1与C4先完成协议说明

**已核对事实：**LSAA主方法的seed0配置和训练入口区分了subtrain与core validation；梯度更新使用subtrain，C/W阈值来自当轮core validation预测，阈值冻结后才访问ICBHI official test。最终checkpoint和patience-10早停则依据ICBHI official-test Score。当前稿件只写“Thresholds use core validation”，缺少这个分工的清晰定义。

修订应解释真实流程，不把test-selected结果改写成validation-only结果。Arian的C1是在询问定义和区别，并没有要求我们立即更换选模协议。

建议准备一份简明的协议说明供写作二使用：训练数据来自哪里，内部验证如何划分，哪些样本参与梯度更新，C/W阈值怎样确定，哪个集合用于checkpoint/早停，SPR/HF/KAUH何时评估。需保留官方recording split与严格patient-disjoint评估的区别。

C4要求的实现信息大多已有记录，可以先从现有资产整理，不需要重训。初步核对的LSAA主方法字段包括：

- 数据源同质batch，即一个batch只来自一个数据源；每epoch两源batch数相同。
- batch size 32；按当前主方法数据规模每源163个batch、共326次更新。较小来源使用随机排列后循环补齐曝光；这不是按类别频率做class-balanced sampling。
- 全量微调BEATs；Adam，初始学习率5e-5，weight decay 1e-6，cosine调度，EMA系数0.5。
- 最多50轮，patience 10；输入为单声道16 kHz、5 s重复补齐／前端截断；没有SpecAugment。
- PCSL系数50、GPAL系数5e-4；HF-on是独立辅助条件。

以上来自主方法配置与脚本，正式写入前仍应逐项确认其他两个seed及各对照。不要把新PC-MCL的1e-4、400轮或增强配方混进LSAA段落。C1/C4可集中写成一段实现说明，必要时辅以精简表注，兼顾四页篇幅。

## C2与C3先明确论点和比较角色

**C2：**Table 1中的冻结编码器参照、DCASE-style以及PC-MCL，属于系统层面的参照。它们不能单独证明共享属性、编码器微调或PAFA正则各自贡献了多少。已有Table 2控制应承担其能支持的具体问题，例如在既定配方下保留细粒度监督的影响、以及不同读出的表现。权重λp=50本身也不能量化PAFA对最终性能的贡献。

**C3：**现有三seed汇总中，Native + C/W相对Full为ICBHI +2.50 pp、SPR +0.16 pp、C/W AUROC −0.05 pp。当前Evaluation已经承认native heads是一种替代方案，但Introduction、Method、Abstract和Conclusion仍需共同解释：共享属性监督的价值和具体层级读出的必要性是两个不同问题。

建议保持有边界的论点：细粒度监督可用于不同原生任务；明确的属性到任务映射是LSAA采用的一种实现；原生分类头也是可行读出，并在当前ICBHI比较中有更高均值。DCASE在SPR/KAUH均值更高这一事实也应如实保留，不能预设“二分类简单、复杂任务必需LSAA”。

若用户希望继续主张PAFA以外的独立属性机制优势，先盘点是否已有严格匹配的无PAFA对照。只有在现有证据不足、该因果claim仍需保留时，再讨论一组匹配的正则消融；本反馈不直接授权新增训练。

## 统计问题单独形成分析方案

Arian把paired t-test作为例子提出。建议先确认需要回答的比较，优先Native + C/W对Full，并列清同seed原始数值、对应split、选模和参照路径；不要从论文中四舍五入后的均值和SD反推结论。

如果以三个训练seed构成三对观测，配对t检验的自由度只有2。可同时报告逐seed差值、效应量、区间和p值，但需明确它描述的训练随机性及很小的重复次数，不能凭一个显著性阈值决定方法价值。未拒绝零差异也不等于证明等效。参考：[SciPy配对t检验定义及区间](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)、[ASA统计显著性与p值声明](https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf)。

如需评估测试样本不确定性，再讨论按患者／录音聚类的配对重采样；不能把同一患者的多个cycle、窗口或KAUH的B/D/E滤波版本当成独立患者。训练seed变化和测试集采样变化应分开解释。多个比较应预先列清，避免只挑显著项；统计程序也不能消除official-test选模带来的解释限制。

本轮没有执行t检验或bootstrap，也没有新增seed。统计分析方案确认后可以使用已有预测和指标完成，不需要训练GPU。

## 建议的任务顺序和责任

| 优先级 | 对应反馈 | 交付物 | 建议负责方 | 是否需要新训练 |
|---|---|---|---|---|
| P0 | C1 C4 | 数据划分与用途说明、完整实现参数段 | 管理核对；写作二负责Sections 2/3；写作一统一Abstract/Evaluation口径 | 否 |
| P0 | C2 C3 | overall claim与证据对应、Table 1/2角色、readout取舍 | 先与用户讨论；写作一牵头，写作二同步方法表述 | 否 |
| P0 | C3附注 | 同seed比较清单及统计分析方案，确认后计算 | 现有本地训练任务做结果分析；写作一解释 | 否，使用已有结果 |
| P1条件项 | C2 | 已有匹配控制盘点；若必要再提出无PAFA消融 | 管理和模型设计 | 仅经另行批准才可能需要 |
| P0 | 新基线整合 | 已完成DCASE与PC-MCL三seed结果、训练/选模差异及任务取舍 | 写作一，管理提供已验证结果 | 否 |
| 收尾 | 全文一致性 | 统一术语、贡献、表注、Abstract/Conclusion及篇幅 | 两写作任务按既定分工 | 否 |

建议先并行准备C1/C4材料与C2/C3论点方案，在用户确认overall思路后再由对应写作任务逐章修改。图件只在文字不足以解释流程时由章节负责人协调绘图任务；不因本次反馈自动重画或编译论文。独立审阅任务仍保持等待，不能由计划时段自动唤醒。

## 对现有Work Plan的具体调整建议

- 保留DCASE三seed完成及本地归档状态。
- PC-MCL新配方三seed已完成，SSH读取恢复，监控已暂停；不新增seed、变体或重试。
- 把上述C1/C4协议说明、C2/C3论点讨论、统计分析方案及两组新基线整合列为当前主线，取消等待PC-MCL的依赖。
- 仅把“是否需要匹配的无PAFA对照”作为待决定项，不直接加入GPU队列。
- 保留原提交时间边界，不给本科生新增个人硬截止。本讨论稿不自行改写Notion正式计划、不发邮件、不向写作/审阅任务下发新指令。

## 本次核对依据

- 反馈原件：`docs/source_materials/collaborator_feedback/2026-09-17_arian_azarang/ICASSP-Comments.docx`。
- 当前稿件：`docs/paper/Overleaf_Sync_final/Section/1introduction.tex`、`3method.tex`、`4evaluation.tex`、`4tables.tex`、`5conclusion.tex`。
- 主方法记录：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_0/config.json`及`selection_split_summary.json`。
- 训练/阈值流程：`baseline/pafa/joint_hierarchy_main_multiseed.py`与`baseline/pafa/joint_hierarchy.py`中的`_balanced_epoch_batches`。
- 既有配对结果：`docs/review_packages/2026-09-15_0201_EDT_independent_review/optional_evidence/core_benchmark_summary.md`，其原始JSON路径已列在文件中。
- DCASE成功结果：`docs/result_exports/2026-09-17_dcase_joint_native_union/`。
