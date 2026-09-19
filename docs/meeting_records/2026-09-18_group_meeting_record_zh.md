# 2026-09-18 组会论文讨论：完整会议记录与修订要求

Notion：[完整会议记录](https://app.notion.com/p/3df309efda29810c84bdc20f5462b88f)。

## 记录说明

会议日期：2026-09-18（周五）。来源为用户提供的完整中文会议意译稿；并非重新听录音生成的逐字转录。下方完整讨论保留原有顺序与实质信息，仅统一明显术语：Aaron → Arian，PFA → PAFA，DCAS-style → DCASE-style。原始附件已原样存档。

## 老师明确要求

- 从第一页及Figure 1开始区分核心训练、可选辅助训练和外部评测，避免读者误以为四个数据集全部参与联合训练。
- Figure 2必须先由作者修到逻辑正确、符号和维度一致；D与768、T_i、attention输入输出和所有箭头都要能清楚解释。
- 各dataset-specific Score需给出定义或原始协议引用；不同数据集的Score不能因为名称相同就直接横向比较。
- 将现有ablation、新baseline及Arian comments整理成连贯、完整、可直接编辑的论文段落。
- 准确表达Native+C/W的竞争力及HF辅助监督的不稳定收益；不宣称所有任务SOTA或全面胜过其他方法。
- Paper和figures修好后再通知老师，请老师进行最终editing/formatting；不能把未完成图件留给老师重做。

## 完整会议讨论

这次会议主要围绕当前 respiratory sound cross-dataset 论文的整体设计、Figure 1/2、实验结果、ablation、baseline comparison，以及论文提交前还需要完成的修改展开。

首先介绍了论文当前的研究主题。论文暂定标题是 Learning the Shared Acoustic Attributes Across Respiratory Sound Datasets with Different Annotations。核心问题是：不同 respiratory sound datasets 的 annotation schema 并不统一，因此无法简单地把多个数据集直接合并成一个统一分类任务。但这些数据集之间又存在共同的声学现象，例如 crackle、wheeze 和 abnormality。因此论文希望研究，是否能够把这些共同的 acoustic attributes 作为跨数据集共享的 supervision，使模型从不同标注体系的数据中学习共同表示。

目前主要涉及四个数据集：ICBHI、SPRSound、HF_Lung 和 KAUH。Figure 1 的作用就是说明这些数据集之间的 annotation relationship。ICBHI 本身包含四分类；SPRSound 中也包含 normal、不同类型的 crackle、wheeze，以及 crackle 与 wheeze 同时出现的情况，因此其中部分标签可以和 ICBHI 的 crackle、wheeze、both 等概念建立对应。HF_Lung 也包含可以对应到共享 acoustic attribute 的信息，例如 wheeze；KAUH 中的 normal、crackle 等也能够与其他数据集中的部分标签建立对应关系。

论文的关键思想不是强行把四个数据集映射成完全相同的 classification task，而是抽取其中可以共享的属性。例如，SPRSound 的 native evaluation task 可能最终只要求 normal vs. abnormal，但它的数据中实际上仍然包含 crackle samples。因此，这些 crackle samples 仍然可以用来监督一个共享的 crackle attribute，使这部分监督能够和 ICBHI 中的 crackle information 共同参与模型训练。

基于这个思路，目前系统主要构造了 abnormal、crackle 和 wheeze 三类 supervision/loss，再根据这些共享属性重新构建不同数据集自己的 native task。例如，对 ICBHI，可以结合 abnormal gate、crackle 和 wheeze 的输出恢复其四分类结果：crackle、wheeze、both，以及没有 crackle/wheeze 时对应的 normal。对于 SPRSound，则可以根据这些属性恢复其 binary normal/abnormal task。HF_Lung 和 KAUH 也根据各自可用的标签采用相应的映射方式。

老师在这里首先指出了一个论文叙事上的重要问题：必须从论文第一页开始就非常明确地区分哪些数据集用于 training，哪些用于 inference/external evaluation。

目前的 presentation 容易给读者一种“四个数据集从一开始就一起训练”的印象，但实际上它们承担的角色并不完全一样。论文不能等到第二页或者 Method 后面才解释这个问题，因为读者第一次看到 Figure 1 和四个数据集的时候，就会自然地问为什么某些数据集参与训练、另一些没有参与。因此 training/inference 的区别应该在最开始介绍 datasets 和 Figure 1 时就明确表达，并且最好在图中也直接体现出来。

当前主要训练数据是 ICBHI 和 SPRSound。KAUH 因为数据量很小，因此主要作为 external testing dataset，而不是加入训练。HF_Lung 则被进一步尝试作为 optional auxiliary supervision，实验中会验证把它加入 crackle/wheeze 等属性监督后是否真正有帮助。

接下来讨论 Figure 2，也就是整个 model architecture。当前方法的部分 loss design 参考了 PAFA 类工作，包括 patient/group-level representation 等设计，同时使用已有 pretrained/base audio encoders。除此之外，目前系统还为 HF_Lung 设计了 optional auxiliary loss，用于测试 HF_Lung 是否能够进一步提供 crackle 或 wheeze 的监督。

这里老师对 Figure 2 提出了非常明确的修改要求。现在图里的 dimensionality、variable notation 和具体数字之间存在不一致。例如某个地方使用 D，另一个地方又直接写 768；attention module 的输入输出、A_i、T_i 等变量具体代表什么也不够直观。老师认为，读者不应该依赖正文才能猜出 figure 里的 tensor dimension 和 intermediate representation。

尤其是，如果 D 实际上就是 base encoder 输出的 embedding dimension，而且当前 encoder 对应的是 768，那么 figure 必须统一表达。可以全部使用 D，并在图中或 caption 中定义 D=768；也可以在所有对应位置直接写 768。不能一个位置写 D，另一个对应位置突然写 768，造成读者怀疑这两个 representation 是否具有不同 dimensionality。

老师对此比较严厉，因为当前 figure 还明显处于未完成状态。当提到“之后再解释”或者依赖正文解释时，老师明确表示现在已经没有时间把这些基础问题留到以后讨论。Figure 本身必须先由论文作者整理清楚，不能期待老师最后帮忙重新设计或把所有图画完。老师强调这是学生自己的 paper，老师可以帮助做最终 editing 和 formatting，但作者首先必须能够解释清楚图中的每一个 component、dimension 和 arrow，并完成一个基本正确的版本。

随后讨论 baseline comparison。当前 baseline 主要包括若干 open-source/pretrained encoders，并针对各数据集训练对应 classifier。现有结果显示，当前方法在多个 native task 上具有较好的表现，特别是 ICBHI 和 HF_Lung 的结果。和针对单一数据集优化的 SOTA 工作相比，目前部分结果大约低 2–3 个百分点；但和 frozen encoder baselines 相比，当前方法有比较明显的提升。因此目前论文不能简单把贡献写成“所有数据集都达到 SOTA”，更合理的证据是：统一的 shared-attribute architecture 在跨数据集设置下能够保持较强性能，并明显优于直接使用 frozen representations 的 baseline。

接下来重点讨论了 ablation study。目前 ablation 基本已经完成，但部分结果还没有正式整理进论文，因为 Arian 对其中一些内容还有 comments，需要先回应和修改。

Ablation Table A 的主要目的，是比较 joint-dataset training 和 separate/single-dataset training。例如只使用 ICBHI 训练相同 architecture 和相关 supervision/loss 时，结果仍然不如 ICBHI + SPRSound 的 joint training。与此同时，当模型只看到单一数据集时，它在另一个数据集上的表现会明显下降，SPRSound 上甚至可能接近随机水平，KAUH external evaluation 也低于预期。因此这个实验主要用于证明：joint training 本身确实带来了额外信息，而不是 architecture 单独就足以产生当前结果。

老师随后追问了一个非常重要的 evaluation 问题：表格中的这些 “score” 到底是不是同一个东西。因为不同 dataset 的 native task 和官方 evaluation protocol 不完全相同，如果只是把多个数字并排放进同一个 table，读者可能会错误地认为这些 score 可以横向直接比较。

虽然部分 score 都可能采用类似 (SP+SE)/2 的形式，但老师强调，“相似”是不够的，关键在于定义是否完全一致。如果不是完全相同的 metric definition，就必须明确标出来。对于 AUC、MAE、MSE 这类通用指标，一般不需要专门解释；但是 dataset-specific score 如果来自不同 benchmark protocol，就应该在 table footnote、caption 或正文中给出定义或引用原始论文。这样读者才能知道每个数字应该在各自 benchmark 内部理解，而不是直接把 ICBHI 的某个 score 和 SPRSound 的某个 score 当作同一量纲比较。

因此，这里需要给所有非标准 native metrics 加 footnote/reference，并明确哪些指标定义一致、哪些虽然名字类似但 evaluation protocol 不同。

Ablation Table B 则主要验证 shared acoustic supervision 的贡献。其中一个设置是在保持 joint-training architecture 的情况下，去掉 crackle 和 wheeze supervision。这个设置下性能明显下降，因此可以用来支持 crackle/wheeze shared supervision 对模型有效这一论点。

另一个设置是 Native Class。这个版本除了 shared supervision 外，再针对每个数据集使用 dataset-specific native classification head。结果显示，如果针对单个数据集直接优化 native task，某些性能可以比当前统一 shared-attribute reconstruction 稍高。这一点并不一定否定当前设计，反而可以帮助界定论文的目标：当前 architecture 不是为了在某一个特定 dataset 上把 native score 优化到最高，而是为了学习可以跨 annotation schema 使用的 shared attributes。如果实际应用中希望针对某一个 dataset 最大化性能，可以在 shared representation 上简单增加 native classifier head，并恢复甚至进一步提高该数据集的 performance。

Table C 讨论的是 HF_Lung auxiliary supervision 是否值得加入 training。实验尝试把 HF_Lung 中能够映射的 samples 加入 auxiliary loss，希望这些数据进一步监督 crackle/wheeze attribute learning。结果并不一致：两个数据集有所改善，但另外两个数据集反而下降。因此当前证据并不能说明“更多 HF_Lung 数据一定更好”。

更合理的结论是：HF_Lung auxiliary supervision 在当前设置下不是必要组成部分。由于 dataset size、distribution、split 以及其他 dataset-specific differences，它提供的额外 supervision 并没有稳定改善所有目标数据集。因此 HF_Lung 不需要作为核心 training dataset，相关实验可以作为 ablation，说明为什么最终 system 不依赖这部分 supervision。

会议后半部分又讨论了一组当天刚完成、还没有加入 paper 的 method-level baseline comparison。

其中一个比较对象是最近的 multi-label respiratory sound 方法。该工作同样把不同标签转换为某种 multi-label representation，例如 normal、crackle 和 wheeze。它的动机与当前论文并不完全相同：原方法会把多个固定长度 audio samples 拼接/augment 成一个更长 sample。例如一个 5 秒 normal sample 和一个 5 秒 crackle sample 拼成 10 秒后，新 sample 中同时存在这些信息，因此其 label design 会考虑 crackle 对 normal 状态的覆盖或组合关系。

之所以把这个方法加入 baseline，是为了回答一个潜在 reviewer question：如果只是把 respiratory sound problem 转成 multi-label learning，是否已经足够解决当前问题？现有实验显示，直接采用这种 multi-label formulation 的结果并没有超过当前方法，因此它可以作为一个有意义的 baseline，帮助说明当前 shared-attribute formulation 不只是简单的 label remapping。

另一个比较对象是 DCASE-style adaptation。这类方法原本更多用于其他场景，例如 environmental sound classification。基本思想是，当两个 dataset 拥有不同 label spaces 时，把它们组合成一个更大的 label space；对某个 dataset 不具备的 labels，在训练时进行 masking，只优化该 dataset 实际具有的 label。

例如一个数据集有四类、另一个有七类，就保留整体 label space，但针对每个 dataset mask 掉不存在的 categories。把这种方法 adapted 到当前 respiratory sound setting 后，当前方法在 ICBHI 和 HF_Lung 上表现更好，而对方在 SPRSound 上略好。

这里需要谨慎解释。因为 DCASE-style method 原本不是针对当前 respiratory sound problem 设计的，而且 adaptation 后的 task formulation 与其原始论文存在明显差异。因此论文不能简单地根据这个表格声称某个方法全面优于另一个。这个 baseline 更适合说明：另一种处理 heterogeneous label spaces 的通用策略，在当前任务上会得到怎样的结果，以及当前 shared-attribute formulation 与简单 label masking 的区别。

最后，老师总结了提交前的工作要求。

第一，当前很多修改内容还是 fragmented sentences，来自 chat、cloud notes 或零散 comments。老师要求先把这些内容整理成完整、连续、可以直接 copy/paste 和编辑的句子。如果内容一直是碎片化的，老师无法有效进行最终编辑。

第二，ablation 实验本身现在基本已经完成，剩余的主要工作是处理 Arian 的 comments，把对应结果和解释整理进 paper。因此接下来的重点已经从“继续无限增加实验”转向“把现有结果完整整理成论文”。

第三，也是老师强调最多的一点：在老师开始最终 formatting/editing 之前，必须先把所有 figures 自己修好。 特别是 architecture figure，要统一 variable/numerical notation，补清楚 dimensions、attention outputs、tensor shapes 和模块之间的关系。不能把一个明显没有完成的 figure 留给老师最后重画。

你表示这些修改基本可以在当天晚上完成。

老师最后再次明确：他可以帮助修改最终版本，也可以提供 formatting/editing 上的支持，但没有责任替你重新完成所有原始 figures。不要期待老师帮你把每一张图从头画完。你应该先把 figures 修改到内容完整、逻辑正确、notation consistent 的状态，然后再把最终版本交给他。

最后双方确认：等 paper 和 figures 都修好之后，再通知老师；老师届时会查看 final version 并进行后续编辑。

## 执行时需要核对的口述简写

以下是依据当前实现和已核实结果补充的管理核对说明，不是新增的会议发言：

- 当前主方法核心训练为ICBHI＋SPRSound；HF source-train只在独立HF-on条件参与辅助监督，HF source-test和KAUH承担外部评测。应在Intro、Figure 1、Figure 2与Evaluation保持相同角色定义。
- 会中ICBHI读出的口述为简写。当前实现是先用A判Normal/Abnormal；A判异常而C/W均未过阈值时，仍按概率减阈值的较大余量选Crackle或Wheeze。不能在图中简化成“C/W都不阳性就必定Normal”。
- 已完成Coarse SPR只关闭SPRSound训练侧的C/W监督，不是删除所有数据源的A/C/W监督；本次另批的Native-only是保留原生分类与PAFA、关闭辅助C/W的不同对照。
- Native+C/W表示带原生分类损失和辅助C/W监督的联合训练模型；不能把“可增加native head”写成给已训练模型临时接一个未训练头便能保证增益。
- PC-MCL与DCASE均为本项目适配结果。PC-MCL是ICBHI-only，不能把其SPR差距单独归因于readout；本次输入是两段2.5 s拼成5 s。DCASE是九类并集的masked multilabel输出，保留SPR细标签，不能称为binary-only。
- 现有三seed均值中LSAA在ICBHI/HF优于这两组适配；DCASE在SPR/KAUH更高，PC-MCL的KAUH均值也高于LSAA。“更广声学结构”仍是解释假设，不能写成已验证的单一因果。
- Native-only与LSAA without PAFA两组三seed已获用户批准并在GPU0/1启动；最近回执尚未提供完整三seed终点。这些待返回结果不应因会中“ablation基本完成”而被记为已完成。

## 关联材料

本次会后计划：[9/18–9/19工作计划](https://app.notion.com/p/3df309efda2981a68624c85b689f596b)。

上一周期：<mention-page url="https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8"/>
组会中英文讲稿：<mention-page url="https://app.notion.com/p/3df309efda2981f5bf1ec12b9eea20eb"/>
Arian原始反馈：`docs/source_materials/collaborator_feedback/2026-09-17_arian_azarang/ICASSP-Comments.docx`。
Arian中文译文：`docs/paper/revision_notes/2026-09-17_arian_feedback_zh.md`。
本次原始意译稿：`docs/source_materials/meeting_records/2026-09-18_group_meeting_user_record.txt`。
