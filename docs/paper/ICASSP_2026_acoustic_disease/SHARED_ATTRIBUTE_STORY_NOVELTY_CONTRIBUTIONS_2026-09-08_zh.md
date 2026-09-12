# 共享呼吸音属性监督与原生任务保持

本轮补充后的论文范围为完整四数据集研究：ICBHI 与 SPRSound 负责主模型训练和原生任务评测；HF 负责单独的属性辅助监督测试及 source-test 评测；KAUH 负责外部评测。两源主模型与 HF 辅助条件分别呈现，HF test 和 KAUH 不用于共享模型训练、选优或阈值拟合。下文关于“两项原生任务”的范围仅指主训练端点，不应据此省略 HF/KAUH 的数据、方法与结果章节。

HF 历史监督规则已按保存配置与代码核对：在含 D/Wheeze/Rhonchi/Stridor 标注的窗口内，对 C/W 同时计算 BCE，对应标签未出现时使用派生负目标；只有 gap/empty/phase-only 窗口整体屏蔽。因此现有 JH4 不能称为严格 positive-only。论文分别报告历史单 seed HF-on/off 对照和正式主模型三 seed 的 HF/KAUH 外部结果，不把两组基线混合。


这条故事可以作为论文主线推进：不同呼吸音数据集的标注虽然不完全相同，仍可能共同监督一组声音属性，并通过这些属性完成预先指定的原生分类任务。现有模型已经提供了这一能力的实例。论文还需要用受控比较说明，共享监督学到了什么有用信息，以及两层分类设计对任务表现有什么作用。

新颖性的判断需要分层。异质标注联合学习、共享属性以及从共同预测空间返回各数据集标签，都有明确先例；当前不能认领这些一般原则的首创。可争取的贡献是一个具体的呼吸音学习方案，以及它在标注粒度、属性标注覆盖和原生任务读出之间揭示的实证关系。消融能证明方案的价值和作用范围，不能使已有的方法原则变成新原则。[^1][^2][^5][^6]

## 主线与研究对象

建议使用下面的研究问题：

> 面对标注单位、类别粒度和标注覆盖不同的呼吸音数据集，模型能否利用兼容的共享属性监督，学习可供多个任务使用的声学信息，并通过两层决策完成各数据集指定的原生分类目标？

其中，“声学信息”首先指模型对异常性、Crackle 和 Wheeze 的识别信息。它不是已经证明与设备、患者或来源无关的表示，也不等于已经学习到可分离的物理声源。当前 Both 标签表示一个监督单元中 C/W 的共同存在；没有时间定位或源分离证据时，不将其解释为已验证的同步声学叠加。

“还原原生任务”应指返回任务定义的预测单位和类别，而不是恢复全部原始标注。当前模型的明确范围是 ICBHI cycle 四分类与 SPRSound event Task 1-1 二分类。SPRSound 的七类任务、HF 的阶段与事件时间检测、KAUH 的原始九分类，不能由现有三个输出全部还原。[^10][^11]

| 条件 | 当前方法实际处理的内容 | 尚需另行证明的强表述 |
|---|---|---|
| 标注单位不同 | 保留 cycle/event 的输入与标签对应，再转换为统一长度输入 | 已解决跨时间粒度的联合定位或时序建模 |
| 类别粒度不同 | ICBHI 四类与 SPR 二类作为指定端点；SPR 支持的原始细标签仍用于训练 | 任意粗细标签任务都可无损恢复 |
| 属性标注覆盖不同 | 仅计算有依据的属性目标，屏蔽不可用目标 | 面对普遍或严重缺标仍能稳定识别属性 |
| 人群、设备和录音条件不同 | 两来源数据共同进入模型 | 学到了来源不变、设备不变或临床可泛化的表示 |

这个范围使论文可以从“模型有两个分数”进一步讨论“属性监督是否足以支撑不同的目标”，同时避免把未研究的困难都写成已经解决的问题。

## 属性与任务之间的具体关系

用 A、C、W 分别表示异常性、Crackle 与 Wheeze。当前监督关系如下；“不可用”表示该目标不进入对应损失，并不表示其真实值必然为零。[^10]

| 来源标签 | A | C | W | 当前指定任务所需输出 |
|---|---:|---:|---:|---|
| ICBHI Normal | 0 | 0 | 0 | Normal |
| ICBHI Crackle | 1 | 1 | 0 | Crackle |
| ICBHI Wheeze | 1 | 0 | 1 | Wheeze |
| ICBHI Both | 1 | 1 | 1 | Both |
| SPR Normal | 0 | 0 | 0 | Normal |
| SPR Coarse/Fine Crackle | 1 | 1 | 0 | Adventitious |
| SPR Wheeze | 1 | 0 | 1 | Adventitious |
| SPR Wheeze+Crackle | 1 | 1 | 1 | Adventitious |
| SPR Rhonchi/Stridor | 1 | 不可用 | 不可用 | Adventitious |

两层决策的用途很具体：第一层判断 Normal/Abnormal；第二层提供 C/W 判断。SPRSound 二分类读取第一层，ICBHI 四分类结合第一层与属性判断。对 ICBHI，异常但两个属性均未过阈值时，现有规则按相对阈值的分数选择 Crackle 或 Wheeze。该规则是设计的一部分，其影响需要单独评价。

当前三个预测节点共享表示并行计算。C/W 是边缘 sigmoid 输出，没有由异常概率乘出的条件概率分解，也没有显式的父子概率一致性损失。因此，“两层分类”适合描述语义组织和决策流程；它不能替代新的层级概率学习机制的证据。C-HMCNN 已提供利用层级约束生成一致预测的先例。[^9][^10]

可以用一个简单标准界定适用任务：原生目标必须能够由保留的属性表达。ICBHI 四类和 SPR 二类满足当前设计要求；粗/细 Crackle 经过合并后已经失去区别，因而不能仅靠这三个输出重建 SPR 七类。这是对当前标签映射的分析，不是一项新的理论贡献。

## 最接近文献与可保留的区别

以下比较以原始论文中的学习设置、监督和读出为依据，不用跨协议的分数高低判断方法优劣。

| 工作 | 已经覆盖的内容 | 与当前故事的关系 |
|---|---|---|
| DCASE 2024 Task 4 | 联合不同域与不同粒度音频标注；跨类别映射、缺标 masking；分别评价；移除数据来源与 CrossMap 的消融 | 已覆盖一般研究问题及核心组织原则。呼吸音任务的特定价值必须由相应证据说明。[^1] |
| Bevandić 等，WACV 2022 | 建立共同 taxonomy，在异质标签上训练，再映回各数据集目标 | “共享内部预测空间、保留原生标签输出”本身不是新贡献；其互斥类别求和与我们的可共存属性损失不同。[^2] |
| SPRSound 数据库论文，2022 | 将 ICBHI 与 SPR 的训练集、测试集分别合并，并调整融合类别进行比较 | 排除首次融合两数据集。融合任务与当前两个指定原生端点不同，但差异本身还不等于方法优势。[^3] |
| LungMix，ICASSP 2025 | 呼吸音单源泛化、语义混合，并将 ICBHI/SPR/HF 映射为共同四类 | 我们的区别是联合监督并保留不同端点；不能声称此前没有跨来源属性或标签语义研究。[^4] |
| PC-MCL，2026 | 显式 Normal/C/W 多标签监督与确定性四类读出，解决多 cycle 拼接中的正常成分丢失 | 排除“属性分解再回到四类”这一宽泛创新。其 Normal 表示正常成分存在，与当前互斥 Normal/Abnormal 决策不同。[^5] |
| OCAD，2026 | C/W 因子、显式组合模块、结构化类别解码和层级推断；SPR 用于映射后的外部评价 | 当前 JH2 没有该组合表示机制；区别在跨来源联合监督与指定原生端点。[^6] |
| OPERA，NeurIPS 2024 | 多来源呼吸音表示预训练、多个下游任务及线性评价 | 排除“共享呼吸音表示可以服务多个任务”的一般首创；当前关注不同来源的标注如何共同监督同一个任务模型。[^7] |
| REACH，2026 年 8 月预印本 | 用元数据合成文本进行音频语义对齐，在六个数据集的九项任务上进行零样本及表征评价 | 进一步说明共享语义空间与多任务适用性已有近邻；其音频文本对齐及疾病等任务，与这里的属性监督和两层读出不同。[^8] |

上述对照支持的判断是：**研究对象和具体任务方案存在区别，但尚未确立一个独特的通用算法原则。** 尤其 DCASE 与 universal-taxonomy 方法的重叠不能仅以“不是呼吸音”排除。当前有价值的研究空间，是在呼吸音标签的具体关系和录音差异下，哪些监督可以有效共享、共享信息是否可复用、任务读出又增加了什么。

本轮近邻核对没有给出某篇工作已经完整回答这一具体呼吸音实证问题的证据；这不是“全球没有类似研究”的证明。额外发现的 SOHFL 2026 文献涉及多数据集、标签稀缺和开放集联邦学习；出版商全文未能直接读取，本报告不据此作精细机制或协议排除。它也提示不宜将“多种异质条件同时存在”当作充分的新颖性依据。[^15]

## 三条 contribution 候选

建议把共享监督与两层任务读出作为一条完整的方法贡献，另外两条分别对应属性学习的发现和分类设计的作用。这样不会将同一方法拆成两条近义的创新。这三条是研究贡献候选：C1 有实现与初步结果，C2/C3 的结论需要新对照，不能作为已完成发现写入摘要。

### C1：通过共享属性连接不同原生任务

> We develop an annotation-aware framework that jointly learns shared respiratory-sound attributes from cycle- and event-level annotations and maps them to dataset-native classification targets through a two-level decision scheme.

中文：提出一个依据标注支持共享呼吸音属性监督的联合学习框架，并通过两层决策连接 cycle 与 event 的原生分类目标。

这条贡献描述方法带来的完整能力。它的价值在于具体监督与任务的组织，不以 BEATs、masking 或“两层”本身作为新原理。相对于联合编码器加原生分类头的合理替代方案，是否获得有价值的表现，仍需要证据。

### C2：揭示标注变化下属性信息的跨来源复用

> We characterize cross-dataset respiratory attribute learning under coarse and partially available annotations, identifying the extent to which supervision from another source supports fine-grained recognition.

中文：揭示在粗粒度或部分属性标注条件下，另一来源的监督能够支持哪些细粒度呼吸音判断，以及这种复用的边界。

这条最直接对应故事中的“学到 acoustic feature 的信息”。例如，训练时不使用 SPR 的 C/W 细标注、仍保留其异常二分类监督，考察从 ICBHI 获得的属性监督能否支持 SPR 的兼容 C/W 判断，并与 source-only 模型和完整标注条件比较。这里目标数据集参与了粗任务训练，因此不能称为完全零样本或未见域泛化。

这是一项待回答的问题，不预设复用一定成功。若只得到两个 native Score，没有属性层面且标注使用明确的对照，这条还不足以独立成立。仅证明共享头能输出 C/W，也不足以形成有意义的发现。

### C3：区分监督信息与两层分类设计的作用

> We separately assess the effects of attribute supervision and two-level task decoding through supervision-matched comparisons, revealing their contributions and trade-offs on native respiratory-sound tasks.

中文：通过标注信息匹配的对照，区分额外属性监督与两层任务读出对原生任务表现的作用及代价。

这里的知识增量应是“分类设计在哪些判断上起作用”，例如异常检出、单属性识别和 Both 决策各自如何变化。C2 关注标注变化时可复用的属性信息，C3 关注相同信息如何被任务设计转化为最终预测。单纯列出一个 ablation table 不构成这条贡献；最终表述应由实际发现决定。

如果相同信息下原生分类头与两层读出表现相近，结果仍可能说明主要价值来自监督组织。此时应降低两层设计的优势主张，而不是回避对照。若 C2 与 C3 没有得到可区分的发现，最终贡献可以合并，不必坚持三条。

## 现有结果在故事中的位置

| 证据 | 已知内容 | 在这条主线中的用途 |
|---|---|---|
| JH2 正式三种子 | 同一联合模型：ICBHI Score 61.17±0.31%；SPR official Score 90.70±0.34% | 支持指定双任务能力；尚不证明共享、属性信息或两层设计的相对价值。[^11] |
| 本科生 BEATs 报告 | 分别训练、冻结编码器、seed 42：ICBHI 51.37%；SPR inter Score 92.00% | 提供单数据集背景；冻结范围、输入与选优方式不同，不能将分数差归因于共享。[^12] |
| 已有单源 PAFA | ICBHI Score 64.14%，2,756 cycles | 说明收益判断不能只依赖较弱的冻结基线；同样不能反过来用未匹配差值证明共享有害。[^13] |
| 本科生 event7、HF、KAUH | 分别是七类事件、时间多标签、九类 recording 任务 | 展示更广的任务背景，不能作为现有三属性模型完成这些任务的证据。[^12] |

JH2 使用 ICBHI official-test Score 选 checkpoint。学生报告中 ICBHI 同样按 official-test Score 选优，SPRSound 的 intra/inter 则分别选择。这些历史结果不替代使用验证集选优的前瞻对照。学生材料是报告中的结果与设置，本报告没有重新运行模型或独立复算其预测。[^11][^12]

原生任务保持和性能保持也应分开。前者要求输出符合指定任务定义，当前已实现；后者要求相对合适单任务模型保留多少性能，需要直接对照。三种子的均值和标准差不会自动解决模型选择或比较条件不同的问题。

## 与主张对应的消融

以下列出需要回答的比较问题，尚不是已批准执行的实验矩阵。

| 要回答的问题 | 对照应固定什么、改变什么 | 可以得到的结论 |
|---|---|---|
| 加入另一来源是否有价值 | 相同方法、输入与优化设置下，比较联合训练和两个单来源条件；明确训练预算的比较口径 | 新来源加入后的整体收益与代价 |
| 更多细属性标注提供了什么 | 固定联合音频、结构与训练安排，只改变指定来源中 C/W 标注是否用于训练 | 该设置中细属性监督的增量 |
| 属性信息是否能跨来源复用 | 在目标来源细属性标签不用于训练的条件下，评价兼容属性；比较 source-only、联合粗标注与完整标注参考 | 粗任务训练和外部细属性监督能支持多少目标属性识别 |
| 整体分类设计是否有价值 | 比较共享属性方案与原生分类头方案；两边都允许使用相同的原始属性标注，例如为原生头对照保留同样的属性辅助监督 | 信息匹配条件下分类接口的效果；并非只改变网络拓扑 |
| 两层 gate 本身改变了什么 | 固定同一个模型的预测分数，比较预先定义的有 gate 和 C/W-only 四类读出；阈值依据验证集确定 | 最终决策规则的作用，不能解释为编码器学到了不同信息 |

当前 R1/R2 不能直接完成第四行：R1 使用 SPR 支持的 C/W 细标注，原生二分类 R2 则没有同等信息。即使得到差距，也会同时包含监督和分类设计的变化。[^14]

细属性删除实验还需要注意一个直接的实现问题：当前损失按可用节点平均。删除某来源的 C/W 后，如果将其 A 损失从三节点平均改为单节点损失，剩余 A 项的相对权重也会变化。若要解释为属性信息的作用，应预先固定保留项的权重与归一化，而不能只保持学习率和 epoch 数相同。[^10]

不需要一开始就展开全部缺标比例、全部数据集和全部 backbone。先围绕一个明确的标注差异与两个指定原生任务建立有解释力的对照，再依据结果决定是否扩展。若进一步主张编码器的表示质量改善，可以在相同读出训练条件下比较冻结表示；仅更换解码后的 Score 提升不足以支持这一更强主张。

新比较应事先确定训练/验证划分、模型选择、阈值与数据使用范围，官方测试集用于最终评价。补学生 seed 0/1 能增加重复性，但不能替代监督和训练设置的匹配。

## 结果如何决定最终叙事

若共享方案在两个任务上都改善，可以报告该具体设置中的共同收益；若一项改善、一项下降，应解释取舍集中在哪些判断上。两种结果都不自动支持更广泛的域泛化结论。

若细属性监督带来收益，而信息匹配后分类设计差异很小，论文的重点应落在监督信息的利用。若两层规则只改变 Normal/Abnormal 与 Both 的工作点，应把它写成决策层的效果。若属性缺标后能够维持有用的跨来源识别，则能更直接支撑声学属性复用这条研究发现；若不能，需要说明适用条件。

基于当前文献与实现，最有希望的文章定位是：**利用呼吸音属性监督连接不同原生任务，并通过受控实验说明共享信息的可复用性与分类设计的作用。** 这可以形成方法与领域实证相结合的贡献。目前还没有足够依据将其定为一种新的通用层级学习算法，也没有必要为此临时增加新架构。

## Sources

文献范围为截至 2026 年 9 月 8 日可获得的原始论文与作者公开稿。表中的已有工作内容是来源事实，对相似性和研究空间的判断是分析。OCAD 使用出版商正文索引中可读取的方法与外部评价说明；不将其异协议分数用于排名。

[^1]: Cornell, S., et al. **DCASE 2024 Task 4: Sound Event Detection with Heterogeneous Data and Missing Labels.** 2024. §2.1、§6.1、§7.2、Table 2。[原文](https://arxiv.org/html/2406.08056v1)。
[^2]: Bevandić, P., et al. **Multi-Domain Semantic Segmentation with Overlapping Labels.** WACV, 2022. §3.4–3.6，共同 taxonomy、数据集目标映射与 NLL+。[原文](https://arxiv.org/html/2108.11224v2)。
[^3]: Zhang, Q., et al. **SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database.** IEEE TBioCAS, 2022. 作者公开稿 §V-C、Table V，PDF 第10–11页。[作者公开稿](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound/blob/main/TBioCAS_SPRSound_Paper.pdf)。
[^4]: Ge, S., et al. **LungMix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification.** ICASSP, 2025. §II-A、§III-A–C。[原文](https://arxiv.org/html/2501.00064v1)。
[^5]: Jeong, S. G., and Kim, S.-E. **PC-MCL: Patient-Consistent Multi-Cycle Learning with Multi-Label Bias Correction for Respiratory Sound Classification.** 2026. §2.1、§2.3、§3.3。[原文](https://arxiv.org/html/2601.17080v1)。
[^6]: Zhang, X., Zhao, W., and Liang, H. **OCAD: Overlap Composition and Class-Atom Decoding for Respiratory-Sound Classification.** Applied Sciences 16(13), 6832, 2026. §3、§4.5。[期刊正文](https://www.mdpi.com/2076-3417/16/13/6832)。
[^7]: Zhang, Y., et al. **Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking.** NeurIPS Datasets and Benchmarks, 2024. 下游 evaluation protocol、Appendix A.4。[原文](https://arxiv.org/html/2406.16148v2)。
[^8]: İlerisoy, M. T., Pham, H. M., Funk, M., Pechenizkiy, M., and Saeed, A. **Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment.** arXiv:2609.00055v1，2026 年 8 月 30 日，预印本。§2、§3、Table 1。[原文](https://arxiv.org/html/2609.00055v1)。
[^9]: Giunchiglia, E., and Lukasiewicz, T. **Coherent Hierarchical Multi-Label Classification Networks.** NeurIPS, 2020。[官方论文页](https://proceedings.neurips.cc/paper/2020/hash/6dd4e10e3296fa63738371ec0d5df818-Abstract.html)。
[^10]: **JH2 当前实现。** [共享预测节点](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/core2_hf_positive_kauh_external.py:54)、[标签映射](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/m_unified.py:114)、[损失](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:279)、[四类读出](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:577)、[模型 forward](/Users/zilongzeng/Research/Acoustic/baseline/pafa/joint_hierarchy.py:151)。
[^11]: **JH2 正式三种子汇总。** [原始汇总](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md)。数值为 mean ± sample standard deviation；保留原有 test-selection 含义。
[^12]: **本科生 forward_material 报告。** [设置说明](/private/tmp/acoustic-hanlin-report-20260908-3nckogj7/forward_material/comparison_notes.md)、[原始 LaTeX 结果表](/private/tmp/acoustic-hanlin-report-20260908-3nckogj7/forward_material/latex/foundation_tables_content.tex:75)。文件是收到的报告解压件；上游训练与预测未在本报告中独立重评。
[^13]: **本地单源 checkpoint 汇总。** [metrics.json](/Users/zilongzeng/Research/Acoustic/result/icbhi_strong_method_reproduction/metrics.json)。本文只引用 PAFA 的 2,756-cycle 行。
[^14]: **此前的问题与实现定位。** [已有深度定位稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/DEEP_NOVELTY_GAP_REVIEW_2026-09-08_zh.md)，§4.2、§6；监督差异同时由当前映射与方法定义核对。
[^15]: Cho, W.-Y., Chang, H., and Lee, S. **Semi-Supervised Federated Learning for Open-Set Respiratory Sound Classification.** IEEE Access 14, 89898–89913, 2026，DOI:10.1109/ACCESS.2026.3703086。[作者稿公开索引](https://www.researchgate.net/publication/407017129_Semi-Supervised_Federated_Learning_for_Open-Set_Respiratory_Sound_Classification)。仅作新增文献线索与宽泛问题范围参考，不承担本报告的精细机制排除结论。
