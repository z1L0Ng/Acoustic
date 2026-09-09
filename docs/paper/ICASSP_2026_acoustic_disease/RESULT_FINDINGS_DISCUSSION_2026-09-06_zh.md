# 结果 → 候选发现｜Todo 1 讨论稿

日期：2026-09-06。状态：**等待用户讨论/确认，尚未验收**。

本稿依据主目录[最新 Work Plan 第 0 节](/Users/zilongzeng/Research/Acoustic/docs/work_plans/2026-09-05_to_2026-09-06_work_plan_zh.md)与[09-04 短会记录](/Users/zilongzeng/Research/Acoustic/docs/meeting_records/2026-09-04_paper_story_short_meeting_record_zh.md)，只从已有结果提炼发现和可能的研究价值。下面四项是供取舍的讨论对象，不是四条 contribution，也不预设必须保留三条。

建议优先讨论 A 的问题发现与 B 的已实现能力；C 合并到 B，说明能力的具体边界；D 保留为 supporting diagnostics。是否具有新颖性，留给用户确认后的 Todo 2 核对。

## 共同证据口径

- **Verified Result** 指本轮从既有汇总与 JSON 核对的结果；没有重新评分、重算统计或验证模型。均值与标准差直接采用现有产物；标准差不是置信区间。
- JH2 正式统计仅含 **0、1、fresh42**，不含历史 seed42 或未完成 seed2。各 seed 用 **ICBHI official-test Score 选择 epoch**，随后对该 checkpoint 进行一次 SPRSound official-inter 评估。这是 PAFA-protocol-matched、ICBHI-test-selected benchmark，不能表述成 clean generalization estimate。
- ICBHI 四分类 Score = (Sp + Se) / 2。Sp 是 Normal recall；Se 是异常样本中**异常细类也被正确判定**的比例。例如 Crackle 判成 Wheeze 仍计错，不能把该 Se 简称为二分类“异常检出率”。
- SPRSound 二分类 AS = (Se + Sp) / 2；HS 为二者的调和平均；official Score = (AS + HS) / 2。Transfer 的 55.82/59.98/59.38 与 all-Normal 50.00 是 **AS**。JH2 的 **90.70±0.34% 是 official Score**，既有 **AS 另为 90.95±0.31%**。本稿不混列、不计算 joint 净收益，也不决定正式表格是否附列 AS。

## A｜源任务表现不能代替固定模型的跨 benchmark 验证

**1. 候选发现。** 在已检查的三个 ICBHI 专门化 checkpoint 上，保留源模型的固定输出规则、直接用于 SPRSound 时，二分类 AS 为 55.82–59.98%，高于 all-Normal 的 50.00%，但仍显示出跨 benchmark 直接复用的局限。

**2. Verified Result 与背景。** 以下是每种方法一个既有 checkpoint 的结果，两个任务分别列示，不能相减为跨任务退化：

| 固定 source checkpoint | ICBHI 原生四分类 Score（%） | SPRSound transfer AS（%） |
|---|---:|---:|
| PAFA | 64.14 | 55.82 |
| SG-SCL | 60.98 | 59.98 |
| Patch-Mix | 62.17 | 59.38 |
| all-Normal 参考 | 不适用 | 50.00 |

ICBHI 来源为 local author-checkpoint evaluation，覆盖 official test 的 2,756 cycles；这些 checkpoint 为 ICBHI-test-selected。SPRSound 为 official-inter 的 1,429 events（Normal 1,040、Adventitious 389）。目标端不训练、不校准、不选阈值；决策规则为源四分类 argmax 为 Normal 才输出 Normal，其余映射为 Abnormal。SG-SCL 源训练还包含 device/metadata 信息，三种方法不是只改变一个因素的 matched controls。

来源：[ICBHI 既有结果](/Users/zilongzeng/Research/Acoustic/result/icbhi_strong_method_reproduction/metrics.json)、[PAFA transfer](/Users/zilongzeng/Research/Acoustic/result/pafa_sprsound_transfer_20260722_235659/metrics.json)、[SG-SCL transfer](/Users/zilongzeng/Research/Acoustic/result/sg_scl_sprsound_transfer_20260722_235659/metrics.json)、[Patch-Mix transfer](/Users/zilongzeng/Research/Acoustic/result/sprsound_patchmix_frozen_transfer/metrics.json)。

**3. 可能的 paper merit。** 把“数据集存在差异”具体化为一个实际复用问题：在没有目标标注参与适配时，源任务上已有表现的模型，能否直接回答另一个 benchmark 的问题。价值可能在于明确这种使用条件下的问题与证据，而非列出更多 baseline。

**4. 当前可声称的范围。** 限于上述三份 checkpoint、ICBHI → SPRSound 方向及固定 head/固定映射。可以报告受限的目标任务表现；不能概括为所有专家模型失败，也不能用与 joint 结果的距离证明联合训练净收益。

**5. Interpretation / Proposed Claim 与缺口。** 局限可能涉及源决策边界、预处理、预测单元或采集条件；当前结果不能分离原因。**固定 source head 受限不等于 learned features 不可迁移**；冻结 encoder 后训练 target-native head 属于另一类、使用目标监督的适配证据。若要把本项升级为 gap-identification contribution，需要 Todo 2 核对文献是否已在可比条件下揭示同类问题，以及本研究具体增加了什么证据。本轮不开展检索或设计新实验。

**6. 取舍建议。** **保留为问题发现候选**；如果文献已明确揭示同类问题，降级为本方法的 motivation。删除“首次发现”“表征不可迁移”及“专家普遍失败”这类超出当前证据的表述。

## B｜同一个联合模型已能保留 cycle 四分类与 event 二分类两种任务

**1. 候选能力。** 当前系统已实现：共同学习两种来源的呼吸音标注，同时保留 ICBHI 的 cycle 四分类和 SPRSound 的 event 二分类输出；共享学习没有要求把 ICBHI 的四类区分压缩为统一的二分类。

**2. Verified Result 与背景。** 每个 seed 使用同一 selected checkpoint 分别完成两个原生任务，正式三种子结果为：

| 原生任务 | 既有结果（mean ± sample std，%） |
|---|---:|
| ICBHI cycle Normal / Crackle / Wheeze / Both | Score **61.17±0.31** |
| SPRSound event Task1-1 Normal / Adventitious | official Score **90.70±0.34** |

ICBHI 为 official recording split、2,756 test cycles，不称作严格 patient-held-out test；SPRSound 为 official-inter、1,429 events。训练是 ICBHI+SPRSound、BEATs full fine-tuning、5 s 输入；selection 边界见上文。

来源：[JH2 汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md)及[对应 JSON](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.json)。只读实现同时确认：节点分类损失只使用被标为 eligible 的目标；ICBHI 从层级输出解码四类，SPRSound 使用 Level-1 二分类。参见[目标与损失处理](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:246)、[ICBHI 解码](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:577)及[SPRSound 读出](/Users/zilongzeng/Research/Acoustic/baseline/pafa/joint_hierarchy_main_multiseed.py:268)。

**3. 可能的 paper merit。** 方法提供的是一种共同利用标注、仍保留各数据集所问问题的能力：跨来源共享学习时，ICBHI 仍可回答异常属于哪一类，SPRSound 仍按自己的事件任务评价。未提供的节点目标不进入该节点分类损失。这比“使用了三个 classifier”更接近方法价值，但代码能执行这种处理本身还不能证明它优于其他设计。

**4. 当前可声称的范围。** 可以说“在两项原生任务上展示了同一联合模型的兼容性与上述 benchmark 表现”。不能说“性能已与各自单数据集模型相当”“hierarchy 优于独立 heads”或“四数据集能力已实现”。

**5. Interpretation / Proposed Claim 与缺口。** “任务对齐带来额外学习收益”仍是待检验解释；“这种能力离不开本方法”尚无直接证据。需要 matched joint independent heads 判断 hierarchy 的额外作用，需要 matched single-dataset 对照判断联合训练的作用；两者不能由 A 的固定专家 checkpoint 替代。方法相对文献的独特价值也待 Todo 2 核对。本轮只指出缺少哪类证据，不制定这些对照的实现合同。

**6. 取舍建议。** **保留为主要能力候选，并收窄措辞**：先陈述已实现的两任务兼容性，把相对优势留作待证实的 claim。C 并入本项，避免把“同时有输出”误写成“各类表现均被保持”。

## C｜接近的 ICBHI 总分伴随不同的组分表现，细类识别仍不均衡

**1. 候选发现。** 在当前联合训练 benchmark 中，ICBHI 三种子总分接近，并不意味着 Normal 与异常细类的识别表现同样稳定；Wheeze 与 Both 的 recall 仍低于 Normal 和 Crackle。

**2. Verified Result 与背景。** 沿用 B 的正式三种子及 test-selected 设置。既有 ICBHI 汇总为 Score 61.17±0.31%、Sp 74.98±5.69%、Se 47.35±5.35%。分类别结果为：

| ICBHI 类别 | Test cycles | Recall（mean ± sample std，%） |
|---|---:|---:|
| Normal | 1,579 | 74.98±5.69 |
| Crackle | 649 | 60.20±8.48 |
| Wheeze | 385 | 29.61±2.08 |
| Both | 143 | 36.83±11.37 |

来源：[JH2 现有汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md)与[既有 per-seed support/recall](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.json)。数字仅转写为百分比，没有新增统计分析。

**3. 可能的 paper merit。** 对已实现能力给出具体回答：当前模型在哪些判断上仍受限，以及为什么一个接近的平均分不足以描述全部表现。这里有价值的对象是已观察到的类别差异，而不是“我们报告了 per-class recall”这一操作。

**4. 当前可声称的范围。** 仅描述本设置下的组分与类别差异，不宣称显著性、普遍稳定性或这些差异由 joint/hierarchy 导致。ICBHI Se 的定义见前文与[指标实现](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/posthoc_native_readout.py:23)。

**5. Interpretation / Proposed Claim 与缺口。** “共享模型的能力在异常细类上受到限制”是可讨论的概括；原因尚不能归到类别支持、阈值或跨数据集监督。需要 matched controls 的相同指标才能判断是否为联合学习带来的收益或代价。特别是既有 local PAFA 的 Sp/Se 为 76.88%/51.40%，JH2 两个分量均较低，因此不能沿用无条件的“只损失 specificity”或“异常检出完全保留”。Published PAFA rows 与 local checkpoint 的比较对象必须分开。

**6. 取舍建议。** **合并到 B 的能力边界与经验分析**，暂不强行独立列为 contribution。若用户认为“平均分掩盖具体能力差异”值得独立讨论，再由后续文献与 matched 证据判断其 merit。

## D｜外部诊断呈现属性与读出差异，尚不能归纳为整体迁移成功或失败

**1. 候选发现。** 固定的 JH2 checkpoints 在外部数据上的表现随属性与读出任务变化，现有证据只支持局部诊断，尚未建立四数据集原生任务的整体性能保持。

**2. Verified Result 与背景。** 以下直接来自[HF/KAUH 既有三种子外部汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.md)，保留原始 0–1 尺度，所有数值为 mean ± sample std：

- HF：源标注 `D`（此诊断用 Crackle 输出评分，汇总字段为 `d`）的 positive-interval recall 为 0.582781±0.126307，Wheeze 为 0.854312±0.010682；对应 recording AUROC 为 0.678225±0.044249 和 0.863603±0.013765。这里的 interval recall 是已标注正区间中至少一个重叠窗口超过固定阈值的比例，不等同于精确事件定位；recording AUROC 则衡量有标注支持的 recording 层面区分。参见[外部诊断读出](/Users/zilongzeng/Research/Acoustic/baseline/pafa/jh2_hf_kauh_external.py:360)。
- KAUH：patient Level-1 binary average score 为 0.721662±0.016411，patient flat4 average score 为 0.470028±0.082964。这是两个不同读出任务，不能相减为同任务收益，也不能仅凭两值判定“粗粒度可迁移、细粒度不可迁移”。

均使用 B 的 selected checkpoints 和既有验证阈值，没有 HF/KAUH 调参。HF 按每条 15 s recording 的三个连续 5 s windows 评价，gap/unknown、unsupported labels 与 Level-1 Normal 不评分。KAUH 为 336 recordings / 112 patients，B/D/E 概率取患者均值；compatible overlay 排除未闭合的 Crep、Bronchial 与 I C B，不是 raw9 native reproduction。

**3. 可能的 paper merit。** 说明同一模型扩展到外部数据时，需要按真正有标注支持的属性和任务判断能力；当前数值可帮助界定后续适配问题，不能替代主任务对照。

**4. 当前可声称的范围。** 限于既有 fixed-checkpoint post-hoc diagnostics；HF gaps/empty 不当负例，KAUH B/D/E 不当独立患者。老师提出的“四 dataset 与各自模型可比”仍是有条件的价值示例，尚非已实现结论。

**5. Interpretation / Proposed Claim 与缺口。** 外部结果为何随属性、读出而不同尚未确定；若要讨论性能保持，需要相同 native task、split、grouping 和 adaptation 条件下的单数据集参考。目前没有这种 matched 证据，不能据此扩大核心 claim。

**6. 取舍建议。** **降级为 supporting diagnostics，排除出主要贡献候选**；篇幅或论证不需要时可从主文删除，保留原始产物即可。

## 本轮待用户决定

当前建议为：**保留 A；保留并收窄 B；将 C 合并到 B；D 仅作 supporting，必要时移出主文。** 这只是 Todo 1 的取舍建议，尚未形成新的 story、section skeleton 或图表/对照方案。

请先讨论这些陈述是否准确捕捉了你认为最有价值的发现，以及哪些应保留、合并、降级或删除。用户确认后才回报管理；Todo 2–4 保持等待。

本轮仅新增此讨论稿；未修改 main.tex/Section、图表、主目录计划或 Notion，未检索文献、开展新统计分析/模型重评/实验、编译、操作服务器、提交推送 Git 或外发，也未创建其他审查材料。
