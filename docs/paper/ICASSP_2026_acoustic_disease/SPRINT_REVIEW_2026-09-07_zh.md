# 09-07 论文冲刺｜集中审阅包

状态：三条工作线的可审阅产物已齐，等待用户集中确认。论文已从placeholder skeleton推进为连续正文，四页审阅PDF已实际编译并逐页检查。所有产物仍是工作稿，不是最终创新声明或新实验验收。

## 1. 产物入口

- [Story / contribution / skeleton 工作稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/STORY_CONTRIBUTIONS_DRAFT_2026-09-07_zh.md)
- [论文源码入口](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/main.tex)
- [最接近文献定位](/Users/zilongzeng/.codex/worktrees/4e78/Acoustic/docs/literature/PAPER_NOVELTY_POSITIONING_2026-09-07_zh.md)
- [15项缺失数据与证据清单](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)
- [Figure 2 可编辑矢量源图](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/Figure/figure2_method.svg)
- [四页审阅 PDF](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-07.pdf)

编译采用临时目录中的便携Tectonic，未安装完整系统TeX环境。已核对4页输出、Figure2及页面显示，未见文字/表格裁切、重叠或未解析引用。作者信息尚未设置；TableII明确保留未完成对照状态，因此这是review draft而不是submission-ready paper。

## 2. 今天已经明确的研究判断

### 创新定位

C1/C2不能作为新的通用masking、hierarchy或native evaluation原则。最接近先例包括DCASE2024、Bevandic、Schutera、LungMix、SPRSound data fusion、PC-MCL、OPERA和BTS-CARD。

当前可讨论的增量是：在ICBHI cycle-flat4和SPRSound event-binary两种非等价呼吸音任务中，使用source-grounded eligibility与共享属性读出共同学习，并观察同一模型的native-task行为。仅换一个应用领域，或报告多个native metrics，本身不自动成立为方法创新。方法的相对收益仍需受控证据。

### 最重要的对照限制

当前hierarchy使用SPRSound原始event标签中有支持的Crackle/Wheeze监督，Rhonchi/Stridor的属性才被mask；最终binary readout不等于binary-only training。

因此，joint hierarchy与只有SPR-binary head的independent模型同时改变了监督信息和classification interface。保持当前比较时，应称为整个eligibility-aware supervision/classification interface的对照；若要主张纯hierarchy结构效应，需要先另行定义matched-supervision设计。今天没有擅自增加实验轴或冻结这个选择。

### 已有结果可以支持什么

- JH2正式0/1/fresh42：ICBHI Score61.17±0.31%，SPR official Score90.70±0.34%；这是ICBHI-test-selected benchmark，可展示两项native readout及类别行为。
- 三份固定ICBHI checkpoint的SPR transfer AS为55.82–59.98%，all-Normal AS为50.00；只能作为这些checkpoint在该规则下的直接复用动机。
- JH2 SPR AS另为90.95±0.31%，与official Score分开。不能将异指标或异训练条件的差直接解释为joint收益。
- ICBHI Se是异常细类正确率，不能表述成binary异常检出率；与local PAFA比较时JH2的Sp、Se均较低，不能无条件写“只损失specificity”。
- HF/KAUH保持post-hoc supporting diagnostics；不同任务分数不直接相减为transfer成败。

## 3. Paper还缺哪些数据

| 优先级 | 具体缺口 | 为什么需要 | 当前怎么写 |
|---|---|---|---|
| 方法收益必须补 | R1：prospective clean joint hierarchy | 新validation-selected主参考 | 旧JH2仅放test-selected benchmark，不填到clean行 |
| 方法收益必须补 | R2：matched joint independent heads，并明确supervision matching范围 | 检验整个对齐interface的额外作用 | 不写hierarchy已优越或纯结构因果 |
| Joint收益必须补 | R3：matched ICBHI-only hierarchy；R4：matched SPR-only hierarchy | 判断加入另一数据集的作用与native retention | 专家checkpoint的zero-target transfer不能替代这两项 |
| 对应claim保留时必须 | Figure1 group-aware/class-matched quantitative separability | 支撑超出描述性图的domain-separation论断 | 先用native-task/availability矩阵，不填未有数字 |
| 有相应论断时再补 | Dataset-wide 5s coverage、group uncertainty | 支撑coverage充分、事件未丢失或统计推断 | 可直接写5s preprocessing事实，去掉上述未证实解释 |
| 学生新数据待回 | Hanlin seed0/1及与seed42相同协议的完整产物 | 形成可核实的multi-seed背景 | 不把seed42报告或HF folds当作multi-seed，不合并不同selection |
| 外部扩展依赖 | HF/KAUH可评价label/task/mapping与同条件参考 | 若要比较三目标数据集transfer或四数据集能力 | 无法原生评价不等于性能失败；仅保留合法diagnostic范围 |

R1–R4仍需独立canonical train split、共同selection、预算与loss/interface定义；旧LocalCleanQueue不作为新suite参考。本文只是缺口清单，未启动实现或实验。

## 4. 稿件应当达到的当前范围

- Data/Method说明实际数据、标注支持、BEATs/PAFA、层级预测与native decoder。
- Figure1先说明native-task/label availability；Figure2展示方法流程。
- TableI将published、local checkpoint、zero-target transfer和JH2三seed分开，并明确AS/official Score与test selection。
- TableII如实列出四项尚无结果的对照，不填历史结果或预期值。
- 中文story稿保留三条候选与强度判断；reader-facing正文以已实现设计和已有证据为主，必要限制集中说明。

## 5. 集中确认时最需要决定的三件事

1. C1/C2是否分别作为问题表述与方法能力，或合并以避免重叠；C3当前仅保留描述性行为，不能写成已完成的因果分解。
2. 是否将R1/R2按当前native independent定义限定为整体interface对照；若要纯结构效果，需要另行决定匹配监督设计。
3. 是否采用当前两core数据集、Figure1最小事实版本和TableI分块作为本轮审阅稿范围，再据缺口清单单独安排必须实验。

这些决定尚未自动冻结。集中确认后由管理更新计划验收状态。

## 6. 沟通与学生状态

用户已要求Hanlin补seed0和seed1；只记录请求，不宣称已开跑或完成。两条学生范围保留，且不设具体期限。

合作者进度邮件与Jingping汇报仍按用户要求后置：今天研究、稿件和集中讨论完成后再进入该环节。当前未准备发送、未联系任何人；收件人、渠道和具体正文待该阶段处理。

本轮没有项目侧新训练、validation/test、模型重评、feature/cache、服务器或GPU操作，也没有Git提交/push。文献和方法输出为工作稿；稿件修改限于写作worktree。
