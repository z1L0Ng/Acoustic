# Feedback 回应：JH2 主表、选模依据与结果分支

本轮按用户意见将 JH2 三次完成运行放回 Table 1，并统一摘要与实验协议。Claude 的意见作为审稿建议评估，不自动视为实验启动或删除 Figure 1 的指令。

## JH2 可以用于哪一种比较

Table 1 作为 benchmark 汇报：JH2 用 ICBHI official-test Score 选 checkpoint 和 early stopping，C/W 阈值来自内部验证数据，SPRSound inter 在每个选定 checkpoint 上评测一次。统计只含 seed 0、1、fresh42，分别选 epoch 15、11、17。其 ICBHI Score 为 61.17±0.31%，SPRSound official Score 为 90.70±0.34%。来源为当前主目录的 result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md。

这能解决摘要与 Table 1 的协议不一致，但不能把这组三 seed 自动当作 validation-selected 的 matched reference。按后续篇幅调整，HF on/off 已并入消融表，合并后的表编号为 Table 2。该表保留验证集选模的独立对照，以 Full model (HF-off) 为共同参考，增加 Full model + HF 一行；它们不引用 Table 1 的 JH2 作为同一结果。HF/KAUH 的关键比较转入正文。

## 对“文献普遍用 test Score 选权重”的核对

2026-09-09 读取三份官方实现的 main.py 和 util/icbhi_dataset.py，均找到完整调用关系：official split 下 train_flag=False 读取 test 记录；这个数据集被命名为 val_dataset/val_loader；每个 epoch 的评测 Score 用于更新 best checkpoint。

- [PAFA main.py](https://github.com/wa976/PAFA/blob/main/main.py)：206 行创建 val_dataset，522 行比较 Score，660 行逐 epoch 评测；[dataset loader](https://github.com/wa976/PAFA/blob/main/util/icbhi_dataset.py) 的 108–120 行区分 official train/test。
- [SG-SCL main.py](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/main/main.py)：213、517、581 行对应上述逻辑；[dataset loader](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/main/util/icbhi_dataset.py) 的 97–101 行选出 test。
- [Patch-Mix main.py](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/main/main.py)：215、460、521 行对应上述逻辑；[dataset loader](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/main/util/icbhi_dataset.py) 的 105–117 行选出 official test。

正文因此使用“following the test-based selection approach in the released PAFA, SG-SCL, and Patch-Mix implementations”。不扩大为整个领域的普遍做法。三份实现还有最低 Se 条件，阈值并不完全相同，因此也不声称我们的全部选模代码与作者完全一致。这里的依据主要来自公开代码；仅有“official split”这几个字不能证明论文使用 test-based selection。

## Table 1 的强参考与 context

主表恢复 published ICBHI references，并保留用户要求的 Hanlin 五种基础 baseline，后者放在独立 frozen-context 分区，不作为监督/分类设计的 matched ablation。

| 方法 | 论文四分类 Score（%） | 原始来源 |
| --- | --- | --- |
| Patch-Mix CL | 62.37±0.61 | [原文 Table 2/3](https://www.isca-archive.org/interspeech_2023/bae23b_interspeech.pdf) |
| SG-SCL | 61.71±1.61 | [原文 Table 2/3](https://arxiv.org/html/2312.09603) |
| PAFA | 64.84±0.60 | [原文 Table 1/2](https://www.isca-archive.org/interspeech_2025/jeong25_interspeech.pdf) |
| PC-MCL, BEATs | 65.37±0.73 | [原文 Table 1](https://arxiv.org/html/2601.17080) |

64.14、60.98、62.17 是本地强 checkpoint 的评测值，不是这三篇论文的多次运行均值。PAFA 的 Table 3 还有 Se/Sp 表头对调问题，本稿采用其 Table 1/2。PC-MCL 使用 10 s concatenation 设置；本轮核对了其任务、official split、数值和五次运行说明，没有据此声称其 checkpoint selection 与 JH2 相同。Patch-Mix 的 68.71 是二分类评测值，不能放入这里的 ICBHI 四分类列。

Hanlin 的 frozen BEATs 92.00% SPRSound Score 保留在表中，正文也明确提到；没有因其高于 JH2 而删去。

## P2、contribution 与 Figure 1

- 按用户后续决定，Intro 采用“领域问题—已有工作—标注与任务之间的问题—相近方法—具体缺口—问题定义与研究内容”的论证顺序。Intro 不再包含实验数值或 xxx 结果槽位；matched 比较的数字与发现放在 Evaluation 中，不再要求 P2 提前展示结果。
- 原三条贡献对应的研究内容融入 Intro 末段：native-task retention、属性信息复用，以及监督信息和分类设计的作用。末段概括本文做了什么，不预先宣称 two-level readout 优于其他读出。
- Figure 1 按用户此前要求保留在第 2 页，其待补内容改为同类 C/W 声学线索与标注粒度的关系，连接 coarse-label 实验。未提供的图和分析没有被补造；仅靠按数据集着色的 PCA 不能说明共享属性，亦不能解释迁移结果的因果来源。
- Table 2 的 Direct C/W thresholding 行只保留 ICBHI 结果槽位。SPR Score 和 C/W AUROC 均为破折号，表示未改变的输出不重复报告。

## PC-MCL 与 frozen-encoder 引用的进一步核对

- 将行名改为 Direct C/W thresholding，以操作命名，保留 PC-MCL 引用及其与离散转换规则的关系。[PC-MCL §2.1](https://arxiv.org/html/2601.17080) 的三标签设计用于训练中保留 Normal 成分；§2.3 在推理中仍按 C/W 是否出现返回 Both、Crackle、Wheeze 或 Normal。因此不采用 Gemini 提议的“unlike PC-MCL”解码对比句。当前消融固定我们自己的模型、监督和验证阈值，不是 PC-MCL 三标签训练的复现或消融。
- §4.2 直接引用 [Niizumi 等的 frozen-encoder 评测](https://arxiv.org/html/2504.18004)：其冻结 backbone、训练任务网络，并比较 MLP 与 Transformer 任务头。该引用用于任务/任务头相关的评测背景，Table 1 的 Hanlin 数字仍有独立来源。噪声解释在原文中属于对跨任务差异的讨论，不能据此将我们的分差归因于噪声。

## 三种结果分支

| 结果 | 可以形成的叙事 | 必须同时满足的证据 |
| --- | --- | --- |
| two-level 对固定 C/W 读出及同等监督的 native heads 有稳定收益 | 分类设计在这些条件下有额外价值 | 监督、训练配方和选模匹配；同时报告各类代价，不能只看总体 Score |
| readout 没有明确收益 | 若细标注对照显示收益，可将发现落在共享细粒度监督上，降低 readout 的贡献地位 | “均值接近”或“不显著”不等于等效；不能仅凭 readout 无收益就断言监督有收益 |
| readout 更差 | 报告具体代价，并用 source-only 对照判断是否仍有 retention 或属性学习价值 | 不隐藏负结果，也不自动假定 retention 成立；标题保持当前不绑定 readout 优势的版本 |

固定分数对照可以回答当前模型的决策规则作用，但单独这一项不能证明相对 PC-MCL 的全部新颖性。PC-MCL 还涉及多周期训练、标签构造和 patient-matching regularization。

## 实验数量与执行范围

合并后的 Table 2 有六个需训练的条件（含 HF-on）、一个固定分数读出条件。若全做三 seed，为 18 次训练；JH2 不能替代 validation-selected full joint reference。按反馈假设的 75 min/run 是 22.5 GPU 小时，仅是算术估计。未核验现时 GPU、运行速度或可用窗口，也没有启动训练、重评分或服务器工作。

在 JH2 现有分数上做 fixed-readout 分析本身可以零训练，但属于 test-selected checkpoint 上的条件性诊断，不能填进 validation-selected Table 2。若最后改用不同 seed 数量或不同选模协议，表注和比较范围也要对应更新。

## 引用的补充原则

实际被引用的不同论文由 19 增至 26。新增使用：AST、PANNs、HeAR（baseline 来源）；audio-foundation-model evaluation（适配条件）；AudioSet（预训练数据）；marginal/exclusion loss 与 single-positive partial-label learning（部分标注语义）。HeAR 的 BibTeX 按 arXiv 原始作者元数据新增，其余来自现有文献池。未使用 nocite 强行填充，也未为达到数量引入无关实验模块。
