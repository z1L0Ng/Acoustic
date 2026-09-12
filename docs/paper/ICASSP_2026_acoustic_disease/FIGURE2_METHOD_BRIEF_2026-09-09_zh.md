# Figure 2：共享属性学习与原生任务读出

## 目标与协作方式

为 Section 3 绘制可以放入 ICASSP 双栏稿件的方法图，解释不同来源的标注如何支持共享声学属性，以及同一组预测如何还原原生任务。Figure 1 留给四个数据集的样本声学特征；Figure 2 不重复数据集介绍表。

用户已授权管理任务以 GPT-6 Astra、max 创建专门绘图任务。绘图任务直接与写作任务 01a08442-92e6-7110-8399-e42eca520ea8 沟通，以用户意见为准，无需管理逐轮转发。本任务本身已经是专门绘图任务，不再创建任务或子任务。

已创建的绘图任务为「论文绘图」，threadId 为 01a084ea-4764-79a1-815b-e3b79948caca，hostId 为 local，工作目录为 /Users/zilongzeng/.codex/worktrees/3110/Acoustic；模型设置为 gpt-6-astra / max。讨论用首稿已集成至主目录稿件的 Figure/figure2_method.svg、.pdf、.png，后续根据用户意见直接迭代。

允许使用 Inkscape、原生 SVG 等矢量工具。交付独立的 figure2_method_draft.svg、figure2_method_draft.pdf 和 figure2_method_draft.png，并提供绝对路径。保留可编辑文字和矢量对象，不打包。用户已授权单独图件导出；不编译整篇论文，不生成 review PDF，不执行模型、实验、特征提取、缓存、smoke 或哈希，不提交推送、不写 Notion。若运行在独立 worktree，由写作任务将确认后的图稿集成到主目录，不切换主目录分支。

## 当前来源与视觉参考

当前稿件与代码以 /Users/zilongzeng/Research/Acoustic 为准。

- 方法：docs/paper/ICASSP_2026_acoustic_disease/Section/3method.tex
- 现有方法草图：docs/paper/ICASSP_2026_acoustic_disease/Figure/figure2_method.svg
- 旧视觉参考：figures/pipeline/acoustic_end_to_end_pipeline_v2.8.svg 和同名 PNG
- 旧参考说明：docs/pipeline/end_to_end_pipeline_v2.8_supplement.md

旧图的分区、柔和配色和流向可以参考。不能继承旧图中的多种候选编码器、Other 预测头、四套独立任务头、HF 完整时序预测头、KAUH raw9 分类头、旧的 positive-only 规则或工程状态标签。

## 图的核心逻辑

建议以共享学习和任务读出为主，HF/KAUH 作为紧凑的辅助条件及评测支路。可以用两个主要面板，但不强制面板数量。

1. ICBHI cycle 与 SPRSound event 的音频进入共享表示模块。
2. 与音频路径分开的标注路径产生 A/C/W 目标及 availability mask，进入对应损失；标签不输入音频编码器。
3. 共享 BEATs、token mean pooling、线性投影之后，三个并行预测头输出 A、C、W。
4. 原生任务读出：SPRSound 用 A 输出 Normal/Adventitious；ICBHI 用 A 的 Normal/Abnormal 门控及 C/W 组合输出四类。
5. HF source-train 只出现在独立的辅助监督条件；HF source-test 用于评测；KAUH 仅外部评测。四个数据集不是四源共同主训练。

HF/KAUH 的评测音频同样经过共享模型；通过实线/虚线、分区或文字说明其角色，不能画成外部样本绕过编码器直接进入预测头。避免交叉箭头和把全部实验对照都塞进方法图。

## 必须遵守的科学细节

### 共享模型

- 主训练源是 ICBHI 和 SPRSound；音频为 mono、16 kHz、5 s。输入处理可以简写，不必画完整预处理流水线。
- BEATs iter3+ AS2M 全量微调。768 维 token mean pooling 后，经一个 768→256 线性投影得到共享表示。
- 三个并行线性头：A 是两个 logits 加 softmax；C/W 各一个 logit 加 sigmoid。
- 两层结构是语义和决策层级。不能画成 A 的预测作为 C/W 的特征输入，也不能标成条件概率 P(C|A)、P(W|A)。
- 无额外 Other 头，无独立 flat4 概率头，无显式父子概率一致性模块。
- 如画 PAFA，用从 encoder token sequence 引出的独立 auxiliary projection 支路表示训练时的 patient regularization。不能把分类用的 256 维投影误当成 PAFA 投影，也不能让正则分支进入任务读出。权重留在正文即可。

### 可用标注与损失

- ICBHI 提供完整 A/C/W 目标。可兼容的 SPRSound 标签监督三节点；Rhonchi/Stridor 只监督 A，C/W 被 mask。
- 对每个节点，在该 batch 内仅对可用目标求平均损失，再对有监督的节点取平均。A 用 CE；C/W 用 BCE。
- PAFA 的 patient cohesion-separation 和 global patient alignment 是保留的辅助正则，不是本论文新发明的算法。

### 两层原生任务读出

- A 的 argmax 给出 Normal/Abnormal；SPRSound 对应 Normal/Adventitious。
- ICBHI 中 A=Normal 则输出 Normal；否则由 C/W 阈值结果产生 Crackle、Wheeze 或 Both。
- 若异常但 C/W 都未过阈值，由较大的 probability-minus-threshold margin 选择 Crackle 或 Wheeze。这一边界情况可留正文，不必占满图。
- 阈值由 core validation 固定；HF/KAUH 不参与选阈值、模型选择或校准。

### HF

- 单独的 HF auxiliary-supervision condition 才使用 HF source-train，不把它描述为 core 训练的第三主源。
- 每个 15 s 录音分为三个 5 s 窗口；与 D/Wheeze/Rhonchi/Stridor 任一标注重叠的窗口可监督 C/W。
- 目标是是否与 D、Wheeze 重叠，因此该条件包含 annotation-derived negative targets；不是纯 positive-only。纯 gap、empty、phase-only 窗口不监督，也没有 HF 的 A loss。
- 图中可简写 HF auxiliary condition: C/W only；具体负例假设放正文。
- HF source-test 采用固定模型评测 annotated-interval coverage 和 recording-level presence，不声称精确 onset/offset localization 或完整 I/E/CAS/DAS 原生任务恢复。

### KAUH

- KAUH 仅外部评测。相同患者的 B/D/E 文件概率先求均值，再用相同规则输出 binary / compatible-four-class 结果。
- 不能画 KAUH 训练梯度、独立分类头、参数校准或 raw9 输出。

## 出版与交付

以双栏通栏约 178 mm 宽设计，目标高度约 55–70 mm；若可读性不够，优先简化信息，再提出高度取舍。按最终放置尺寸检查可读性，文字尽量不小于 9 pt；不要用巨大的 SVG 画布掩盖缩放后字体过小的问题。

白底，少量稳定颜色；灰度下也能凭文字、边界和线型区分。图内仅保留科学概念和必要缩写，不放管理状态、任务 ID、文件路径、运行日志或未经确认的结果。

交付时简要说明信息组织、最终尺寸和仍需作者确认的取舍。先给可以直接查看的第一版图稿，再与写作任务直接迭代。

## 代码核对入口

- baseline/pafa/joint_hierarchy.py：PAFAJointHierarchyModel.forward
- baseline/multidataset_pipeline/core2_hf_positive_kauh_external.py：Core2Head
- baseline/multidataset_pipeline/beats_nal_protocol.py：hierarchical_loss、decode_icbhi_hierarchical_flat4
- baseline/pafa/joint_hierarchy_hf_auxiliary.py：_make_hf_windows、_hf_loss
