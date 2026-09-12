# Acoustic｜2–3天补齐数字的复盘草案

日期：2026-09-09  
状态：**用户已明确批准四条件benchmark串行训练；启动已下发，运行状态见当前Work Plan。旧clean队列及下文其他历史备选仍暂停。**

最新顺序：用户要求先由本地训练任务完成已有本地资产的无训练后处理，直接交付论文写作任务，再按实际剩余缺项讨论补跑。新的validation协议本身也在复盘；下文六条件、A/B和对应时间只是保留原设计时的备选估算，不能当作当前已确定的必跑清单。当前执行与交接状态见[Work Plan](../WORK_PLAN.md)。

## 最新最小方案：四项实际启动已获批准

本轮已明确的仅推理缺项及数字落文已完成。用户已认可的训练优先顺序为：

| 顺序 | 条件 | 本轮重复数 | 要回答的问题 |
|---|---|---:|---|
| 1 | ICBHI-only | seed42一次 | 联合模型与ICBHI单源参照的原生表现 |
| 1 | SPRSound-only | seed42一次 | 联合模型与SPR单源参照的原生表现 |
| 2 | Coarse SPR | seed42一次 | 保留SPR音频与A标签时，额外C/W标签的作用 |
| 3 | Native+attributes | seed42一次 | 匹配属性监督后，分类接口的作用 |

采用推荐的延续benchmark方向，优先核对并复用现有fresh42 Full作为单seed对照，不能拿三seed均值代替。不默认增加Full或HF训练。四个常规run按历史MPS每run4.4–5.2小时估算约18–21小时，另计实现和写作；2–3天是正常早停下的项目预算目标。若准备中出现无法复用Full的实质差异，先讨论，不自动扩大预算。

实施规格按本源原生benchmark选模准备：ICBHI-only、Coarse SPR、Native+attributes使用ICBHI official-test native Score；SPRSound-only使用其本源SPR official inter Score，不拿ICBHI目标test选模。单源C/W阈值仅来自本源validation，另一目标在选模和阈值冻结后评估。保留的Full仍是ICBHI-selected，因此SPR-only与Full的selection目标不同，必须披露，不能把差值全部归因于来源共享。四项benchmark训练已获启动授权；统一源域validation路线退为历史备选。

模型设计任务负责更新 EXPERIMENT_SPEC_2026-09-09_zh.md 和最小benchmark入口，复用原fresh42 subtrain/validation，不启用新三分manifest；本地任务只读核对执行路径。实现和只读核对已完成；用户已明确批准本地按四项顺序执行。服务器任务与旧clean队列不在授权内。

仅metadata核对发现：现有fresh42 Full的27个epochs中，每源4401个batch均有A/C/W eligible样本，因此其available-node平均损失实际等于每节点1/3。公式文字差异本身不足以排除Full复用；仍需核对原划分、选模、预算和实现。Coarse隐藏C/W后，A权重应按最终规格明确，避免自动重归一造成额外变化。

Hanlin当前资源与接收情况未确认，不将其返回计入必达路径，不设学生个人固定截止时间；需要额外seed或独立环境复核时，再给出含相应Full参照的完整工作包。下文旧6条件和13/14-run方案保留作历史备选。

## 1. 暂停与保留内容

Full42的最后console记录为update448；完整持久保存至validation point1/update326，official test未访问。Python59399已因Ctrl-C退出，旧queue controller59395随后结束，两个PID均无存活条目；ICBHI-only42与SPRSound-only42均未启动。

`result/reproduce/pafa_joint_hierarchy/Table2_clean_controls/full_hf_off/seed_42/`中的best/last checkpoint、config、split snapshot、calibration/selection NPZ及日志全部保留。该run是partial；现有checkpoint不含完整optimizer/RNG状态，不称为可无损续跑，也不作为完成的seed。后续若需正式重启，应保留此目录并按确认的初始化/输出策略执行。

此前14/18-run预算和seed42自动串行授权暂停。现有split、loss、selection定义暂不改变，先重新决定重复次数和分工。

## 2. 真正需要哪些数字

在保留当前论文全部比较的前提下，需要六种训练条件；不等于必须本地顺序跑十四次。

| 条件 | 必须回答的问题 | 最小新增训练 |
|---|---|---:|
| Full HF-off | 新validation协议下的共同参照；ICBHI/SPR原生表现和SPR C/W AUROC | 1 |
| ICBHI-only | 与Full在同协议下比较ICBHI表现，并报告固定模型到SPR的结果 | 1 |
| SPRSound-only | 与Full在同协议下比较SPR表现，并报告固定模型到ICBHI的结果 | 1 |
| Coarse SPR | 撤掉SPR细属性监督后，属性复用与原生表现如何变化 | 1 |
| Native+attributes | 同等属性信息下，换用native分类接口的作用 | 1 |
| Full+HF | 相对同环境Full，加入HF辅助监督后的变化 | 1 |

以下不增加训练次数：

- Direct C/W：复用新Full的同一概率与阈值；旧E1已有，但不能代替新Table2行。
- SPR C/W AUROC、per-class recall/confusion、paired差值：使用每个run已保存的prediction artifacts。
- HF/KAUH外评估：对新Full和Full+HF的selected checkpoints做外部推理，不重新训练；旧模型的外评不能填入新模型的before/after。
- mask数量、数据集/分组支持、窗口覆盖与Figure1基础统计：metadata/声学统计工作，可与训练并行。Figure1具体内容仍需按正文需求收敛，不在本轮默认扩展新的foundation-model特征实验。

Table1现有文献与frozen背景数据不重跑。Hanlin旧2s frozen实验的额外seed不替代本轮full-fine-tuning对照。

## 3. 方案A：本地六条件seed42，优先保证全表有数

六个条件均在当前MPS/Beats环境运行seed42。Full作为其余五项的同环境参照，保持相同split、初始化、physical batch32、loss/selection与最大更新预算。

- 训练量：5个常规run + 1个HF run，历史速度参照约29–36小时。
- 再计新HF/KAUH外评、汇总和核对；Figure1基础统计并行。正常早停情况下，2–3天是可行目标。
- 这不是硬时限承诺：多数run若达到50-point上限或需要修复重跑，可能超出3天。
- 科学代价：新Table2所有训练行都是single-seed点估计，必须修改表头/标注；不能写mean±SD、稳定提升或显著性。

优点是协调依赖最少。Hanlin资源不明或下一工作周期仍无可用反馈时，不继续将其产出计入必达路径，按本方案重新安排。

## 4. 方案B：Hanlin承担完整CUDA核心组，本地承担MPS机制组

**前提：Hanlin确认有可用CUDA资源，能保持physical batch32/FP32，并能接收和执行完整代码/数据合同。当前GPU型号、显存、环境和可用性均未知，不能据此承诺速度。**

| 执行方 | 完整工作包 | 物理训练run数 |
|---|---|---:|
| Hanlin | Full、ICBHI-only、SPRSound-only，各seed0/1/42；同一CUDA环境 | 9 |
| 本地 | 本地Full42参照、Coarse42、Native42、Full+HF42 | 4 |

总计13个物理run，但两边并行。本地约20–26小时训练；项目能否在2–3天内收齐，取决于Hanlin的实际资源、吞吐与反馈，不能沿用未经核实的75min/run估计。

报告必须分块：

- CUDA核心块：三条件各三seed，在同环境内给mean±SD和逐seed配对差值。
- MPS机制块：本地Full42分别与Coarse42、Native42、HF42配对，只报单seed方向。
- 两个Full42是各自环境的参照，不合并成同一行或“额外一个seed”。不把MPS seed42与CUDA seed0/1直接凑成三个seed。
- HF/KAUH机制before/after使用本地Full42与HF42；readout分析也复用对应块自己的Full概率。

本方案保留核心问题的多seed证据，但表格需要明确区分两块。若要求全表同一环境，应由一个执行环境承担整套条件，不按模型名称把对照拆到两台机器。

## 5. Hanlin任务准备边界

- 他没有repo访问权限，需要我们提供完整代码/依赖、BEATs初始化、ICBHI/SPR数据路径约定、固定split manifest、准确命令和输出说明。
- 当前runner限定MPS，尚不能直接发一条CUDA命令让他跑。CUDA适配需在分工确认后完成。
- CUDA保持FP32、关闭AMP/TF32，使用同一Adam/EMA/cosine；physical batch必须32。PAFA的batch内患者关系使microbatch梯度累积不能直接替代physical batch32。
- Hanlin承担完整Full+两个single-source组，不单独交给他两个single-source却用本地Full作参照。
- 每run交付config、split snapshot、日志、checkpoint、calibration/selection/terminal predictions、native/attribute metrics。只给表格数字不够。
- 2–3天是项目整体目标，不自动转成学生硬期限；不反复催问。资源/接收意愿未知时不把他视为已承诺。

可转发的规划说明见[Hanlin核心对照任务草案](../student_tasks/hanlin_table2_core_cuda_draft_2026-09-09.md)。草案尚未发送，不是立即开跑指令。

## 6. 本次建议与待决定点

优先考虑**有明确资源前提的方案B**，以保留核心三条件的多seed证据；若Hanlin尚不能确认资源或反馈，采用方案A作为不依赖学生的完整数字版本，同时接受single-seed证据范围。

目前只需先确认Hanlin的GPU/显存与可接任务情况，再决定是否接受“CUDA核心块＋MPS机制块”的呈现方式。既有训练保持暂停；不提前适配、转发任务或开跑，直到用户确认新分工。

Abstract/Conclusion继续留到最后；不因本次复盘自动编译论文、修改已确认文案、提交Git、写Notion或联系外部人员。
