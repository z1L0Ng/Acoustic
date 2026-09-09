# 呼吸音联合学习的研究定位与可成立贡献

## 核心判断

当前 JH2 可以被准确地描述为：**针对 ICBHI cycle 四分类与 SPRSound event 二分类，使用共享呼吸音属性、标注可用性掩码和原生读出的联合学习实例。** 它已经展示同一 checkpoint 同时完成两项任务的能力，但尚未建立独立于既有方法的通用算法创新，也尚未证明联合学习或整个分类设计的相对收益。这个判断针对当前实现与证据，不等于断言该研究没有论文价值。[^1][^2][^17][^18]

现有文献的重叠比“也使用了 masking”更深。DCASE 已联合处理不同音频标注、映射部分标签、屏蔽不可用监督、分别评价数据集，并比较联合与单数据集训练；呼吸音文献已有多标签属性表达、flat4 重建、跨数据集映射，OCAD 进一步提出显式组合表示和结构化解码。将这些已有原则换成“supervision compatibility”一词，不会自动产生新的学习原则。[^1][^7][^8]

**最值得争取的贡献目前是新的、受控的呼吸音实证认识**：在两项原生任务上，共享监督究竟带来什么收益与代价；观察到的变化来自额外属性标注、参数共享、分类目标还是解码选择；哪些类别受益，哪些没有。这类研究不要求发明一个全新神经模块，但需要合理替代方案下的证据，不能以“做了四个实验”本身代替研究发现。

因此，建议把 C1/C2 作为一条完整的、范围具体的方法设计候选，暂缓将其拆成两条独立 innovation；C3 保留为需要对照回答的研究问题，待结果出现后提炼真正发现。论文贡献不必证明“只有本方法能够做到”，但必须说明相较合理已有方案，它增加了什么实质价值。

## 1. 先确定被比较的实际方法

### 1.1 当前实现提供了什么

JH2 对两个数据集都先截取原生单元，再做 5 秒 repeat-padding/front-truncation；输入共享 BEATs。token 特征取均值，经共享线性投影后得到 Normal/Abnormal、Crackle、Wheeze 三个输出。分类损失按可用节点求平均，再叠加继承的 PAFA 患者相关目标。[^17]

| 层面 | 当前实现事实 | 可以说明的能力 |
|---|---|---|
| 输入 | cycle 或 event 均转换为一个 5 秒输入 | 保留原始监督单位及一单元一预测的对应 |
| 表示 | 共享 BEATs 与分类投影 | 两来源共同更新表示 |
| 监督 | Level-1 CE；Crackle/Wheeze BCE；不可用目标不进入对应 loss | 使用有标注支持的属性，避免为这些被屏蔽目标制造负标签 |
| 输出 | 三个节点分别产生分数 | 显式表达异常性及两种声音属性 |
| 读出 | ICBHI 用 Level-1 gate、属性阈值及确定性规则得到四类；SPR 用 Level-1 二分类 | 同一 checkpoint 返回两种原生任务输出 |
| 训练辅助 | PAFA 患者目标与其投影分支 | 继承的表示正则化；不是新层级关系约束 |

这里没有新的时间对齐网络、跨分辨率特征交互或事件定位机制。**保留 cycle/event 的标注单位有研究意义，但不等于模型已经解决了不同时间粒度的联合时序建模。** DCASE 本身还要处理时间强/弱、软/硬标注和长录音重建，不能用“我们的预测单元不同”就排除它的相关性。[^1][^2]

### 1.2 “层级”在哪一层起作用

设共享表示为 $h$，Level-1 两个 logits 为 $z_0,z_1$，则

\[
p_A=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}=\sigma(z_1-z_0).
\]

因此，Level-1 的二类交叉熵可以等价写为该 logit 差上的二元交叉熵。加上 $p_C=\sigma(z_C)$、$p_W=\sigma(z_W)$，分类部分属于带掩码与节点权重的多标签目标。这里 $B_k$ 是batch $B$ 中节点 $k$ 有可用target的样本子集，$K_B$ 是这些子集非空的节点集合：

\[
\mathcal L_{cls}=\frac{1}{|K_B|}\sum_{k\in K_B}
\frac{1}{|B_k|}\sum_{i\in B_k}\operatorname{BCE}(p_{ik},y_{ik}).
\]

这是依据当前代码的**代数分析**，不是一次模型实验。它说明预测函数与分类目标的类型；不声称更换参数化后优化轨迹、正则化或最终性能完全相同。三个节点仍通过共享表示产生梯度耦合，不能据此说它们的表征互不相关或属性在统计上独立。[^17]

对给定的BEATs pooled features而言，256维线性投影后仍接线性heads，中间没有非线性激活；因此head部分仍可写成这些features的仿射映射。参数化、优化和共享正则化可能影响学习，但该投影本身没有增加新的非线性层级表达；这不意味着整个BEATs模型是线性的。[^17]

当前分类目标没有显式要求 $p_C\leq p_A$ 或 $p_W\leq p_A$，也没有单独的 joint flat4 概率头、属性—joint 边缘一致性损失，或学习得到的 overlap composition 模块。ICBHI 最终输出的合法性主要由确定性解码保证。C-HMCNN 明确区分一般多标签预测加后处理，与在模型和训练中施加层级一致性；这一先例说明“用了 hierarchy”需要具体到约束如何进入学习。[^12]

**定位含义：**JH2 的贡献候选应落在任务特定的监督组织与读出设计，以及其经验价值上。现有实现不足以支持“新的层级概率模型”“新的组合表示学习”或“新的层级一致性机制”。

### 1.3 不能继承早期提案的区别点

早期属性方案曾讨论 attribute head 与 joint flat4 head 并存、marginal consistency，以及更多数据集的监督。当前 JH2 主方法没有这些组件，也没有 HF 辅助训练。以早期完整组合为对象得出的“未见完全重叠”，不能作为当前 JH2 新颖性的依据；每个区别必须在实际模型中存在。[^21]

## 2. 最接近的工作：重叠到什么程度

### 2.1 DCASE 是方法级近邻

Cornell 等人的 DCASE 2024 Task 4 将 DESED 与 MAESTRO 放入一个训练系统，处理标注粒度、类别范围与置信度差异；部分 MAESTRO 类向 DESED 父类映射，不可用类别的 loss 与 attention 被屏蔽，同一系统按两套标签与指标评价。论文还有移除数据集和 CrossMap 的对照。[^1][^2]

这已覆盖“异质监督共享、缺失标签处理、一个模型、分别评价、以单数据集系统判断联合收益”的大框架。其 baseline 使用冻结 BEATs；同届 Schmid 等人的系统进一步对 joint DESED/MAESTRO 上的 BEATs 等 Transformer 进行微调，所以 full fine-tuning 本身也不是可独立主张的增量。[^13]

两者仍有实际区别：DCASE 的任务是时间声音事件检测，JH2 是预分割呼吸单元分类，并用 Abnormal/Crackle/Wheeze 表达标签关系。**这些区别说明研究对象不同，却尚不能证明已有方法无法覆盖这个实例。** 当前需要回答的是这种具体监督组织是否有额外价值，而不是反复强调领域名称不同。

### 2.2 呼吸音近邻

| 工作 | 已覆盖的核心内容 | 与当前 JH2 的具体差别 | 定位作用 |
|---|---|---|---|
| SPRSound data fusion | 合并 ICBHI/SPRSound 训练与测试集合，合并 coarse/fine crackle，并在融合任务上评价 | 未展示当前同一 jointly trained checkpoint 的 ICBHI flat4 与 SPR Task1-1 双端点设计 | 排除“首次融合两数据集”；保留具体任务方案的比较空间 [^3] |
| LungMix | 三个呼吸音域的标签统一、语义 OR 混合与单源泛化 | 将任务映射到共同 flat4；JH2 使用联合监督及不同 native readout | 其跨域问题已有；区别在学习设置与标注使用，不是发现“存在域差异” [^4] |
| BTS-CARD | ICBHI→SPRSound OOD 与 metadata debiasing | 使用映射后的 flat4，目标不是当前联合训练的 event-binary endpoint | 固定专家迁移可以补充背景，不能单独成为新 gap [^5] |
| Chua & Cheng | Crackle/Wheeze 二属性、多标签训练、阈值重建四类 | ICBHI-only，Normal 隐式为两属性皆无 | 排除二属性分解与 flat4 重建本身的新颖性 [^6] |
| PC-MCL | 显式 Normal/Crackle/Wheeze 标签、multi-cycle OR、确定性读出和组件对照 | Normal 表示混合样本中正常成分的存在；JH2 的 Normal/Abnormal 决策互斥，且不做该多cycle拼接 | 不完全相同，但“显式 Normal＋属性标签”已有；差别需说明语义与用途 [^7] |
| OCAD | 独立事件因子、显式 overlap composition、class-atom decoder 与层级推断 | JH2 没有这些组合表示组件；其主要区别是两源联合监督与 native readout | 进一步排除“显式组合 Crackle/Wheeze 并解码 Both”这一宽泛新颖性 [^8] |
| OPERA | 多来源自监督表示与任务特定的下游评价 | 下游任务分别训练/评价，不是当前两源标注共同训练的三节点模型 | “共享呼吸音表示服务不同任务”已有，联合监督是更窄的区别 [^9] |
| Tran-Anh 等 | 呼吸音检测/分类多任务网络，在 BreathSet 与 ICBHI 上评价 | 原文明确两数据集分别训练其他模型 | “一个网络做呼吸音多任务”已有；该工作不是跨两数据集共同训练 [^10] |

**PC-MCL 的可借鉴点在于问题与机制对应。** 它先指出 multi-cycle 拼接下二属性标签会丢失正常成分，再用独立 Normal 标签处理该问题，并给出对应比较。JH2 没有这一拼接问题，因此不能沿用它的 Normal-preservation 解释；可以借鉴的是提出明确问题、选择针对性设计、用相应证据回答的论证方式。[^7]

**OCAD 是重要的新增近邻。** 它的结构化组合表示与当前JH2不同，SPRSound也仅用于映射后的外部评价；这些差别不支持跨协议分数排名，更不能把它已实现的组合机制当作JH2的能力。[^8]

### 2.3 通用方法先例

| 方法家族 / 代表 | 已有原则 | 对当前定位的约束 |
|---|---|---|
| Overlapping taxonomy：Bevandić 等 | 将不同标签表达为互斥 universal classes 的集合，通过概率求和学习，再映回原生 taxonomy | “统一内部表示并保留外部任务标签”已有系统方法 [^11] |
| Heterogeneous labels：Schutera 等 | 根据某类标注是否存在屏蔽并归一化损失；指出 missing 不等于 background | 样本—类别可用性 mask 是既有方法原则 [^14] |
| Hierarchical multi-label：C-HMCNN | 用 constraint module 和相应 loss 使预测满足父子层级关系 | 必须区分训练期层级约束与最终规则解码；当前 JH2 主要属于后者 [^12] |
| Heterogeneous supervision：YOLO9000 | 使用 WordTree 结合分类和检测标注，在有监督的层级上反传 | “不同预测任务＋部分层级监督＋共同模型”作为一般思想已有；它不等同于我们的 native 双端点评价 [^15] |
| Medical partial labels：LENS | 针对不同医学数据集的标注范围与漏标问题，采用多任务检测和缺失标注挖掘 | 医学数据中的错误负监督与跨数据集互补不是新问题；LENS 还包含 JH2 未做的标注挖掘 [^16] |

Bevandić 的 partial label 指一个粗标签对应多个可能的细类；JH2 的 mask 指某个二元属性没有可用目标。二者不是同一个损失，却都说明标签差异可以通过明确的标签语义与监督关系处理。Schutera 的额外 class-asymmetric loss依赖像素类别互斥，不能直接套到可共存的 Crackle/Wheeze 属性；应比较原则与适用条件，而不是仅凭关键词宣布“完全一样”。[^11][^14]

## 3. 当前区别：哪些有意义，哪些还不足以成为贡献

### 3.1 区别与研究价值不是同一件事

| 当前区别 | 是否真实存在 | 为什么还不能直接等同于 innovation | 能使其形成贡献的证据 |
|---|---|---|---|
| 两个具体 respiratory benchmarks 共同训练 | 是 | 特定数据组合本身不是新学习原理 | 原生任务上的可信收益、代价或清晰的新经验发现 |
| cycle-flat4 与 event-binary 两种端点 | 是 | 两种输入都经过同一5秒分类流程；在共同标签子集上，binary是四类的粗化 | 保留原生端点相对合理共同任务/独立任务方案增加的价值 |
| SPR 原始事件标签监督属性 | 是 | 利用已提供的细粒度标注是合理选择；需要分清信息量与结构作用 | 固定标注使用范围后的设计比较，或明确把收益归于整个监督设计 |
| 对 Rhonchi/Stridor 的 C/W target 设为不可用 | 是 | sample-class masking已有；该规则的性能价值尚未验证 | 与明确、合理替代规则的受控比较 |
| Normal/Abnormal gate 加 C/W 解码 | 是 | 类属性与层级后处理已有；当前未加入新的概率一致性约束 | 相同监督、相近容量与选择规则下，读出/目标设计的作用 |
| 全量微调 BEATs并继承PAFA | 是 | 都是已有训练组件 | 作为控制条件固定，不单独包装为贡献 |
| 分别报告 native metrics | 是 | 这是评价定义的保留，DCASE等已有 | 最终由不同任务的实际行为形成认识，不把“不求pooled score”当算法创新 |

“保留任务”有两层含义，必须分开。**Task preservation** 是保留预测单元、标签和评价定义，当前已实现；**performance retention** 是相对单任务训练保留多少性能，需要合适的单任务参考。两个词不能互换，现有同一 checkpoint 的两个分数也不构成性能等价证明。

### 3.2 标注可用性的实际范围较窄

当前映射把 ICBHI 的四类全部转换为完整的 $A,C,W$ 监督；SPR 的 Normal、Coarse/Fine Crackle、Wheeze、Wheeze+Crackle也提供完整三节点目标，Rhonchi/Stridor只提供 $A=1$，两个属性被mask。SPR binary endpoint并不意味着它在训练中只提供二分类标签。[^17]

现有协议的官方training支持表列出 Rhonchi 39、Stridor 15；这些样本随后还要分入 fitting/validation，不能把两数直接当作实际训练batch里的数量。主 inter 支持表未含这两类。这里引用的是已有协议常量，没有重新统计原始数据。[^19]

这使“解决广泛缺失标注学习”成为过强叙事：当前核心数据主要拥有可映射的属性标注，也没有训练使用HF正向区间。mask是必要的语义处理，但其影响大小、面对更严重缺标时是否稳定，均未由现有主结果回答。不能为了强化该叙事，自动把HF加入主训练或改用另一测试划分。

### 3.3 三个不应凭直觉创造的 gap

**Normal 定义不兼容不能被直接宣布为已发现问题。** ICBHI 的文件确实只编码C/W两列，但官方说明将正常cycle描述为没有adventitious sounds。因此不能仅凭文件字段数量，断言其Normal包含其他异常或构成错误负标签。跨来源Normal等价性可以作为语义核查项，却尚不是已观察到的冲突。[^20]

**Both 不自动等于时间上的声音叠加。** 当前cycle标签表示同一监督单元内存在C/W，未提供两种事件精确同步叠加的证据。用“co-presence”表达标签事实通常比宣称已经学习声学overlap composition更准确；当前也没有定位或源分离结果。

**native任务保留不意味着融合必须失败。** 在共同四类子集上，可将四类确定性地粗化为Normal/Abnormal；SPRSound自身已有融合实验。应实证比较不同监督组织，而不预设统一标签方案不可执行，或必然牺牲原生任务。[^3]

## 4. 真正值得回答的研究问题

### 4.1 首选：共享监督在两项原生任务上何时有价值

已有文献说明跨数据集误差、标注异质性和联合数据的潜在收益。更具体、仍值得回答的问题是：**在同一输入、backbone、预算和模型选择条件下，共享ICBHI cycle与SPR event监督，是否能在两项原生任务上获得有价值的性能组合，代价集中在哪些判断上？**

这是一项呼吸音场景中的经验问题，而不是一个尚无人提出的普遍问题。它可能形成论文贡献，是因为受控结果可以告诉读者什么时候值得使用统一模型，以及需要接受什么代价。价值由实际结果产生，不能由“同时用了两个数据集”推导。

若两个任务均改善，可以讨论该设定中的协同收益；若一项改善、另一项下降，应报告具体trade-off；若没有实质增量，就应收窄或删除优势主张。单个seed的控制结果不能自动支持稳定性、显著性或临床等价性。

### 4.2 关键机制问题：额外标注信息，还是分类设计

当前 R1 使用 SPR 原始event类别提供的 C/W监督，R2 的SPR independent binary head会丢弃这部分信息。因此，R1/R2同时改变了监督信息和分类结构。即使两行最终有差距，也只能先解释为**整体监督/分类设计**的差别，不能直接归为纯hierarchy结构。[^22]

更有研究价值的认识可能是：细粒度属性监督究竟如何帮助原生二分类及另一个数据集的四分类；共享属性与仅保留dataset-native heads各自在哪些情况下有利。若需要结构级结论，必须先固定两边可以利用的标签信息，再定义要改变的单一设计因素。现有四行仍有价值，但不会天然产生纯因素分解。

这一问题提供了一个明确、可被结果支持或否定的贡献方向。它也解释了为什么仅把C1与C2换一个名字不够：需要找到设计、信息和结果之间可以区分的关系。

### 4.3 次级：总体分数与具体识别能力如何分离

既有JH2三种子ICBHI Score为61.17±0.31%，但Sp/Se标准差分别为5.69/5.35个百分点；Wheeze/Both recall低于Normal/Crackle。它支持当前设置中的描述性观察，尚不能解释差异来自共享监督、阈值、类别构成还是任务本身。[^18]

如果匹配对照显示某种监督设计持续改变特定类别或Normal/Abnormal判别，这可以成为值得报告的经验发现；如果只是当前模型上的一次误差描述，则适合作为结果分析。不能把“报告per-class recall”或“做error analysis”本身当贡献。

### 4.4 当前不宜作为主 gap 的方向

- “专家模型不能跨dataset直接使用”：已有LungMix与BTS-CARD先例，现有三份checkpoint提供的是特定条件下的补充动机。[^4][^5]
- “未标注不应当作阴性”：已有明确方法原则，当前mask实现不构成一般性首创。[^2][^14]
- “单一模型保留不同数据集评价”：已有DCASE、universal taxonomy和task-specific probing先例。[^1][^9][^11]
- “Crackle/Wheeze可组合为Both”：已有Chua、PC-MCL与OCAD，且我们的训练中没有新增组合表示机制。[^6][^7][^8]
- “四数据集性能可比”：HF/KAUH目前只是固定checkpoint的有界诊断，无法提供这一结论。[^18][^23]

## 5. 已有结果究竟支撑到哪里

| 证据 | 已有内容 | 可以支撑 | 尚不能支撑 |
|---|---|---|---|
| JH2正式0/1/fresh42 | ICBHI Score61.17±0.31%；SPR official Score90.70±0.34%，AS另为90.95±0.31% | 指定benchmark协议下同一模型的双任务表现 | clean generalization、优于分别训练、hierarchy因果收益 |
| 三个固定专家checkpoint | SPR AS55.82/59.98/59.38%，all-Normal AS50.00% | 无目标适配、固定head/规则的直接复用动机 | learned features不可迁移，或joint训练净收益 |
| Native per-class结果 | 四类ICBHI与SPR二类recall | 当前模型的类别行为描述 | 性能变化的原因，或普遍的dataset规律 |
| HF/KAUH | Fixed-checkpoint post-hoc diagnostics | 对可评价标注与读出范围的支持性观察 | native reproduction、四域稳定性能保留 |

正式三种子checkpoints由ICBHI official-test Score选择，均值±sample standard deviation没有消除该选择机制的影响。SPRSound在每个selected checkpoint后评价一次，也不能反向把整套研究称为clean validation-selected。ICBHI Se计算异常细类的正确分类比例，不能与SPR二分类异常recall直接混同；AS与official Score也必须分列。[^18][^22][^23]

## 6. 最小证据应该回答什么

以下是研究判断条件，不是新增执行授权或已冻结实验合同。

| 比较问题 | 最小方向 | 结论应怎样限定 |
|---|---|---|
| Joint是否优于相同配方的单源学习 | prospective joint与两个source-only条件，固定允许的数据、预算和选择方式 | 在两项native任务上分别报告，不使用异协议paper/transfer差值代替 |
| 当前整体设计是否优于直接native heads | R1/R2在共同协议下比较，明确SPR属性信息差异 | 可以回答整体interface的价值；不能自动称纯结构效果 |
| 收益是否仅来自额外SPR属性监督 | 若机制claim确有必要，先定义一个使用相同原始标注信息的对照 | 不自动增加大矩阵；必须说明剩余变化究竟是什么 |
| Mask是否带来经验收益 | 只有保留该收益claim时，才需要明确标注可用性变化的受控问题 | 数学上不计算未知target的loss，与实测改善是两个层面 |
| 是否真的“保留性能” | 有可解释的单任务参考；若声称等价/非劣，事前定义其标准 | 不能用“差得不多”替代论证，也不能从未显著得出等价 |

验证设置应与要回答的问题对应。相同seed并不足以消除任务监督、update exposure、类别比例、loss归一化、阈值拟合和checkpoint选择的差别。反过来，也没有必要为所有可能因素构建巨大实验矩阵；先决定最需要成立的一项claim，再选能区分它的最小比较。[^22]

R1–R4的新split、criterion、训练预算和监督匹配尚未闭合。本报告不采用旧LocalCleanQueue作为新合同，不启动训练、模型重评或统计分析。现有test-selected结果继续作为背景benchmark，而非填入缺失的clean行。

## 7. 建议的论文定位

### 7.1 当前最合理的路线

建议把论文定位为：**对呼吸音原生任务联合学习进行具体设计和受控实证研究，解释共享监督的收益与边界。** 方法部分用一条完整贡献呈现任务定义、属性监督和读出设计；真正的经验贡献由对照揭示的发现来命名。C1与C2目前不宜被拆成两条近义的方法创新。

这个定位仍有明确价值。临床声学数据的获取成本高，单模型复用是否有实际收益需要可靠答案；而保留原生端点能避免把任务改写后的高分当作原问题已经解决。不过，“重要且值得研究”与“当前已获得新发现”需要分开，后者依赖尚未具备的控制结果。

### 7.2 若希望坚持算法创新

需要指出当前一般masked多标签方案无法妥善处理的具体机制问题，并给出有实质区别的设计与证据。单纯更换术语、增加一个已有形式的loss、或把全部旧模块重新组合，都不能替代这一论证。C-HMCNN、OCAD和universal-taxonomy方法分别展示了如何把约束、组合或标签集合关系落实到学习中；它们提供比较对象，不是自动可移植的新贡献。[^8][^11][^12]

当前不建议为了标题中的“hierarchical”临时增加架构。应先判断现有任务中是否真的出现了需要该约束的问题，再决定是否值得改变模型。这会形成另一项方法研究，而不是当前稿件措辞的微调。

### 7.3 若没有新增结果的时间

现有材料仍能构成一个有边界的联合模型benchmark与方法实例，但不能用更强的文字代替缺失的相对证据。可以呈现所实现的能力、明确既有方法来源，并如实展示类别表现；不宜宣布已经解决共享监督的普遍gap，也不宜预先写成“disentangle”或“outperform”。

## 8. 对“unique contribution”的直接回答

**当前确实新增了一个具体、可运行并已评价的两任务实例。** 它把ICBHI cycle-flat4与SPR event-binary纳入同一训练模型，利用SPR支持的细粒度事件标注，并保留各自读出。这是当前可以核实的设计与能力，但“完整配置此前未见”不是充分的新颖性论证。

**当前尚未确立独特的算法原则。** 核心分类目标、masking、属性表达、确定性读出和共享模型均有明确先例；代码也没有实现早期提案中的joint概率头或marginal consistency。更窄的数据与任务组合不能自动补上这个差距。

**最可能形成实质贡献的，是目前仍待回答的经验问题。** 如果公平比较揭示：某种标注共享方式在原生任务上带来明确、可解释的收益或代价，并能区分额外监督与分类设计的作用，那么“发现了什么”将比“用了哪些组件”更有说服力。当前应优先确定这项科学问题及其最小证据，而不是先冻结三条漂亮的贡献句。

## 9. 证据范围与来源

本报告以2026年9月8日可获得的原始论文、官方任务/代码说明、当前JH2实现与既有结果为范围。外部论文的性能叙述属于作者报告，本报告没有复现实验；模型结构判断来自代码阅读及代数推导。本文没有给出“全球没有完全相同工作”的证明，也不以检索未命中支持first claim。

OCAD的期刊正文索引可读取方法和实验设置，但直接网页下载受访问限制；本报告只使用这些可核对的机制与任务描述，不采用其分数作优势判断。SPRSound采用作者官方仓库公开稿的§V-C与Table V定位，表号可能与排版后的期刊版本不同。LENS只用于已公开摘要能够支持的一般问题范围，不承担精细机制等价的判断。

### Sources

[^1]: Cornell, S., et al. **DCASE 2024 Task 4: Sound Event Detection with Heterogeneous Data and Missing Labels.** DCASE Workshop, 2024. 重点：§2.1、§5、§6.1、§7.2、Table 2。[原文](https://www.merl.com/publications/docs/TR2024-146.pdf)。
[^2]: DCASE. **2024 Task 4 official task and baseline.** 重点：Baseline Novelties Short Description；loss/attention masking、CrossMap与dataset-specific evaluation。[官方任务](https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels)；[官方baseline](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline)。
[^3]: Zhang, Q., et al. **SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database.** IEEE TBioCAS 16(5), 867–881, 2022. DOI:10.1109/TBCAS.2022.3204910。重点：作者稿§V-C、Table V。[作者公开PDF](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound/blob/main/TBioCAS_SPRSound_Paper.pdf)。
[^4]: Ge, S., et al. **LungMix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification.** ICASSP, 2025. 重点：§II-A、§II-C、§III-A及COMB选择说明。[原文](https://arxiv.org/html/2501.00064v1)。
[^5]: Koo, H., et al. **Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial Debiasing for Out-of-Distribution Robustness.** ICASSP, 2026. 重点：ICBHI/SPR设置与§3.2。[作者稿](https://arxiv.org/html/2510.22263v1)。
[^6]: Chua, Y.-W., and Cheng, Y.-C. **Towards Enhanced Classification of Abnormal Lung Sound in Multi-breath: A Light Weight Multi-label and Multi-head Attention Classification Method.** arXiv:2407.10828, 2024. 重点：§3.4.1、Fig.4，PDF第6页。[原文](https://arxiv.org/abs/2407.10828)。
[^7]: Jeong, S. G., and Kim, S.-E. **PC-MCL: Patient-Consistent Multi-Cycle Learning with Multi-Label Bias Correction for Respiratory Sound Classification.** ICASSP, 2026. 重点：§2的标签与读出、§3.3对照。[原文](https://arxiv.org/html/2601.17080v1)。
[^8]: Zhang, X., Zhao, W., and Liang, H. **OCAD: Overlap Composition and Class-Atom Decoding for Respiratory-Sound Classification.** Applied Sciences 16(13), 6832, 2026. DOI:10.3390/app16136832。重点：§3.1、§4.1–4.2与SPRSound映射说明；出版商正文索引核对。[期刊页](https://www.mdpi.com/2076-3417/16/13/6832)。
[^9]: Zhang, Y., et al. **Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking.** NeurIPS Datasets and Benchmarks, 2024. 重点：§3–5的预训练与下游probe区分。[原文](https://arxiv.org/html/2406.16148v2)。
[^10]: Tran-Anh, D., et al. **Multi-task Learning Neural Networks for Breath Sound Detection and Classification in Pervasive Healthcare.** Pervasive and Mobile Computing 86, 101685, 2022. 重点：§4.2、§5.2明确两数据集分别训练。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9419997/)；[出版商](https://doi.org/10.1016/j.pmcj.2022.101685)。
[^11]: Bevandić, P., et al. **Multi-Domain Semantic Segmentation with Overlapping Labels.** WACV, 2615–2624, 2022. 重点：§3.4–3.6、原生taxonomy映射与NLL+。[原文](https://openaccess.thecvf.com/content/WACV2022/papers/Bevandic_Multi-Domain_Semantic_Segmentation_With_Overlapping_Labels_WACV_2022_paper.pdf)。
[^12]: Giunchiglia, E., and Lukasiewicz, T. **Coherent Hierarchical Multi-Label Classification Networks.** NeurIPS, 2020. 重点：§2–4的coherence、后处理与训练约束区别。[原文](https://proceedings.neurips.cc/paper/2020/file/6dd4e10e3296fa63738371ec0d5df818-Paper.pdf)。
[^13]: Schmid, F., et al. **Improving Audio Spectrogram Transformers for Sound Event Detection through Multi-Stage Training.** DCASE Challenge technical report, 2024. 重点：joint DESED/MAESTRO Transformer fine-tuning。[原文](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Schmid_75_t4.pdf)。
[^14]: Schutera, M., et al. **Methods for the Frugal Labeler: Multi-Class Semantic Segmentation on Heterogeneous Labels.** PLOS ONE 17(2), e0263656, 2022. 重点：§2.4的label-mask、normalization与background边界。[原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263656)。
[^15]: Redmon, J., and Farhadi, A. **YOLO9000: Better, Faster, Stronger.** CVPR, 2017. 重点：§4、WordTree及joint classification/detection，PDF第6–8页。[作者原文](https://pjreddie.com/static/papers/YOLO9000.pdf)。
[^16]: Yan, K., et al. **Learning From Multiple Datasets With Heterogeneous and Partial Labels for Universal Lesion Detection in CT.** IEEE TMI 40(10), 2759–2770, 2021. DOI:10.1109/TMI.2020.3047598。本文使用其公开摘要中的问题及方法范围。[作者稿](https://arxiv.org/abs/2009.02577)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/33370236/)。
[^17]: **JH2当前实现。** [forward](/Users/zilongzeng/Research/Acoustic/baseline/pafa/joint_hierarchy.py:151)、[三节点head](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/core2_hf_positive_kauh_external.py:54)、[masked loss](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:279)、[native mapping](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/m_unified.py:114)、[ICBHI decoder](/Users/zilongzeng/Research/Acoustic/baseline/multidataset_pipeline/beats_nal_protocol.py:577)、[训练总目标](/Users/zilongzeng/Research/Acoustic/baseline/pafa/joint_hierarchy_main_multiseed.py:405)。只读核对。
[^18]: **正式JH2三种子既有汇总。** [结果文件](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md)。保留原有selection与mean/sample-SD含义。
[^19]: **SPRSound现有协议支持表。** [protocol.py](/Users/zilongzeng/Research/Acoustic/baseline/shared_encoder_native_heads/protocol.py:68)、[loader与内部split](/Users/zilongzeng/Research/Acoustic/baseline/four_dataset_frozen_encoder/data.py:407)。本报告未重新统计音频或manifest。
[^20]: ICBHI. **Official Respiratory Sound Database description.** 重点：专家标注定义与四列annotation格式。[官方说明](https://bhichallenge.med.auth.gr/)。
[^21]: **早期共享属性提案的范围。** [2026-07-29文献与设计讨论](/Users/zilongzeng/Research/Acoustic/docs/surveys/survey_shared_crackle_wheeze_attribute_prior_art_2026-07-29.md)。仅用于区分提案版本，不能代表当前JH2实现。
[^22]: **当前比较与证据边界。** [缺失证据稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)、[现有结果讨论](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/RESULT_FINDINGS_DISCUSSION_2026-09-06_zh.md)。
[^23]: **JH2 HF/KAUH既有外部汇总。** [结果文件](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.md)。仅固定checkpoint的支持性诊断。
