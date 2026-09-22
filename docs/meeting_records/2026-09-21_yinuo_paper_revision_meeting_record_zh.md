# 2026-09-21 与Yinuo讨论：主方法、统计报告与ICBHI选模

Notion：[完整Meeting Record](https://app.notion.com/p/3e2309efda2981f2b46fc0ea762b320e)。

原始附件：[片段1](source_materials/2026-09-21_yinuo/part_1.txt)、[片段2](source_materials/2026-09-21_yinuo/part_2.txt)、[片段3](source_materials/2026-09-21_yinuo/part_3.txt)。三份均原样保存；下文仅接回两处截断并增加结构标题。

## 记录说明
会议日期：2026-09-21（周一）；参会人：Zilong、Yinuo。来源为用户提供的三段连续中文会议纪要，不是本次重新听录音生成的逐字转录。已接回“相应地，”及“Table 2、Table 3”两处截断，保留全部26节及讨论过程；正文中的第一人称沿用用户原文。
主线是：异构标注与不同native tasks → 保留原生预测结构并共享声学监督 → 完整模型的native/external表现 → 组件作用与跨数据集适用边界。论文价值不以每一列均值第一或给较弱base逐项加模块来表述。
## 会议形成的修改方向
- 将“native task heads＋共享C/W监督”的已有配置作为新主方法方向；原文早段建议与Jingping确认，后续讨论倾向按此推进。方法最终命名/定位与Jingping的确认状态仍需记录，不能把“会后要询问”写成已同意。
- 先统一Figure 2、Method与Tables 1–3的主方法、配置和数字，删除与新main重复的展示行，再组织Results。仅修改名称不能改变某个历史run实际采用的结构。
- 保留HF-on作为向第三个异构数据源扩展的边界分析，不以没有提升为由删除；避免把推测的原因写成已证实因果。
- 统一HF/KAUH subset及其角色rationale在Section 2的出现位置；Figure 1C补纵轴Frequency (Hz)。Yinuo会后继续看Dataset/Method的语言、abnormal语义与optional supervision描述，是否已返回修改需另行确认。
- 核对每组“±”的实际定义，计算SD和SEM并做适合现有原始数据的统计分析，再决定表注/显著性标记和Results用语。Abstract/Conclusion最后对齐，Introduction只调整受最终方法变化影响的内容。
- ICBHI checkpoint-selection尚未定论：先把代表论文与官方代码材料给Yinuo讨论，再决定透明重述或调整协议；会议没有直接批准整批重跑。
## 记录之外的会后技术核对
以下是管理依据当前本地稿件/资产的核对，不是替换或改写上面的会议原话。
1. **SEM/SD存在待澄清差异。** 会议口述为SEM；当前Section 4称sample standard deviations，已有本地汇总使用sample_std/sample_sd（新Native+C/W外评还明确ddof=1）。因此先查逐seed值，不能把现有“±”原样改名SEM；Hanlin每组统计口径也需单独核验。
2. **主方法切换后需重新核对消融比较对象。** 新main候选是Native+C/W；Native-only是无C/W监督的结构匹配对照候选，仍需核对CUDA/MPS、划分及recipe差别。现有single-source、Coarse SPR、without-PAFA、HF-on主要是旧explicit-readout分支，不能直接重命名为新native-head main的单因素消融。应先按实际config归类，并决定展示在同一框架的哪个分块；缺匹配run不自动新增训练。
3. **会议中的baseline简述需按实现复核。** 当前DCASE respiratory adaptation用native-class union＋mask，不是只训练两个数据集的交集；PC-MCL原文拼接与本项目2.5 s＋2.5 s的5 s适配要区分。Table 1 frozen references各列是否使用目标数据监督，仍以Hanlin原始实验材料核验，不能直接把口述external当作已证实zero-target。
4. **prior-work核查已准备好，但不代替Yinuo的协议决定。** 四篇论文未明确写test-score checkpoint selection；官方代码可证明该选模方式。PC-MCL提及validation用于alpha搜索，却未定义集合。Yang 2020明确另拆validation。材料在docs/literature/2026-09-21_icbhi_checkpoint_selection/。
5. **统计语言需以分析结果为准。** 误差条重叠不能代替检验；没有显著证据不等于等效；较小SD/SEM可以描述观察到的变异，不能仅凭三个seed直接证明更稳定或更好泛化。统计比较与单位先约定，再计算和写结论。
本次仅保存会议与制定计划，不直接重写paper或启动新实验。今天的SPR4/HF-DAS支线仍因正式执行审批待用户确认而未启动，不是已有新结果，也不阻塞本轮写作。
## 关联材料
今日工作计划：[2026-09-21 Work Plan](https://app.notion.com/p/3e2309efda2981d4927cf7476ef18691)。
上一周期：[关联页面](https://app.notion.com/p/3df309efda2981a68624c85b689f596b)。
术语索引（9/20快照，主方法确定后需刷新）：[关联页面](https://app.notion.com/p/3e1309efda2981549909fd773f5e8623)。
本次完整附件原文同时保存在本地会议记录目录；不替换用户原始附件。
## 完整会议纪要（26节）

参会人：Zilong、Yinuo
时间：2026 年 9 月 21 日
讨论主题：论文提交前整体检查、Method 与 Evaluation 修改、主方法重新定义、ICBHI checkpoint selection、统计显著性及 Results 论述

### 一、会议整体安排与论文检查方式

会议开始后，我共享了屏幕，原本计划从 Abstract 开始和 Yinuo 一起逐段检查论文。不过在正式进入正文之前，我先提出了目前论文中一个比较重要的问题：在 Section 4 的 Table 3 中，我们目前定义的主方法实际上不如其中一个 variant 表现好。因此我今天上午一直在考虑，是否应该直接把这个表现更好、结构上也更合理的 variant 提升成新的主方法。

这个调整本身基本不需要重新补实验，主要涉及方法名称、Table 中数字以及围绕方法设计的一部分论述发生变化。Introduction 基本不会受到影响，Discussion 的核心结论预计也不会有太大变化。

Yinuo 的个人意见是倾向于修改。既然这个 variant 本身更符合我们真正想表达的方法设计，而且结果也更好，把它定义成最终主方法会更加合理。不过她建议我还是和 Jingping 确认一下，因为这涉及项目内部对于最终方法定义的 preference。

我表示可以之后再问 Jingping，而且即使最终决定修改，也不影响我们先把论文其他部分的逻辑理顺。等讨论到 Table 3 时，我会再具体解释目前主方法和这个 variant 的区别。

原本我准备从 Abstract 开始逐句解释，但 Yinuo 认为没有必要逐句过。她已经大致看过论文，而我作为作者也很清楚自己写了什么，因此当前更高效的方式应该是集中讨论真正存在疑问的地方。她认为最值得重点检查的是 Method 和 Results：Method 需要确保她作为外部读者理解到的内容和我实际实现的方法一致；Results 则需要检查每张表到底支持什么结论，以及目前的论述有没有过度解释或者不够清楚的地方。

我说明 Abstract 和 Introduction 昨天已经基本检查完，而且相关问题也已经和 Jingping 讨论过，目前我对这两部分比较有把握。因此我们决定不再逐句检查 Abstract 和 Introduction。

Yinuo同时提醒我把 Overleaf 中已经确认的 changes 及时 accept，需要删除的删除，需要保留的保留，避免 revision marks 一直堆积。

### 二、Section 2：Datasets 与 Data Structure

对于 Section 2，我解释说这一部分主要介绍不同 respiratory acoustic datasets 的数据结构和 characteristics，本身不是为了提出新的实验结论，而是为后面的 cross-dataset results 提供背景。

这一部分主要讲两个问题。

第一是四个数据集不同的 labels 和 annotation 方式，对应 Figure 1A。这里主要希望让读者看到，不同 respiratory datasets 的原生任务和 annotation granularity 并不一致，因此不能简单把所有数据直接合并成一个传统 classification dataset。

第二是不同 datasets 的 characteristics，以及这些差异为什么构成我们研究 cross-dataset learning 的基础。这些 characteristics 会和 Results 中观察到的 dataset-dependent performance 形成前后呼应。

剩余部分主要是技术性说明，例如 Figure 中的内容如何计算、使用了哪些分析方法等。因此 Section 2 整体并不长，大约只有半栏。

Yinuo表示她之前还没有仔细检查 Section 2，会在会议结束之后快速完整读一遍，重点检查语言和逻辑。

她同时发现 Figure 1C 有一个具体问题：Y 轴没有明确标注含义。我解释纵轴实际上是频率，单位为 Hz。她建议直接在纵轴标注 Frequency (Hz)，不要让读者自己判断。我确认会修改。

### 三、Section 3：Method 的整体结构

之后我们进入 Method。Method 主要跟随 Figure 2 的 architecture 展开，而且昨天已经根据 Jingping 的意见修改过不少内容。

Yinuo希望我不需要一句一句解释，而是重点告诉她：最近改了什么、哪些地方我自己也觉得可能让 reader confused，以及哪些地方需要她从另一个视角重新检查。

#### 3.1 从输入到 PI，以及 abnormal label 的解释

Method 前半部分从 input 到 PI 的流程目前比较清楚，我认为没有明显问题。

真正比较难写的是后面关于 abnormal 类别以及不同 datasets 之间 label relationship 的部分。

这里需要解释为什么我们要设置 abnormal，以及为什么 abnormal 可以作为一个更 broad 的类别，覆盖其他数据集中更具体的异常事件。这个逻辑本身对整个 label harmonization 很重要，但是目前这段文字读起来有些卡。

我认为这里的问题可能不只是单独一句话，而是上下几句话都需要一起调整。因此我在文稿中标记了这一段，希望 Yinuo 会后重点看。

Yinuo表示她会完整 go through 这一部分。如果可以直接改，她会直接修改；如果存在不确定的地方，她会在 Overleaf 留 comment，让我决定 accept、进一步修改还是删除。

#### 3.2 Classification / Linear Heads 与 external testing

接下来的部分主要解释模型中的三个 classification / linear heads，以及为什么这样选择。

之后介绍模型如何应用到另外两个 external testing sites，以及这些 external datasets 上的分数如何得到。

因此 3.1 整体承担的是解释 architecture 从输入、representation、classification heads 到 external evaluation 的完整流程。

#### 3.3 Loss functions 与 optional supervision

#### 3.2 主要介绍训练过程中各个 loss function 的计算，其中重点是我们的 core loss 和额外 supervision。

我特别提出 optional component 的表述需要再检查。因为虽然这个组件是 optional 的，但 Figure 2 中仍然把它画在整体 architecture 里，因此正文必须明确告诉 reader 哪些是核心组件、哪些属于 optional supervision，以及什么时候启用。

另外，其中 patient-related 的部分来自已有工作。相关设计参考了一篇在该方向上表现较强的工作，我们的参数设置也基本 follow 了他们的设定，同时我也针对这个组件进行了 ablation。

因此 implementation 本身应该没有太大问题，当前主要需要确认的是文字是否足够准确。

#### 3.4 Training details

Method 最后一部分主要描述具体 training parameters，包括每个 epoch 中的操作和训练方式。这一部分是根据之前反馈补充的。

我说明昨天已经按照代码重新核对过这些参数，因此参数本身可以确认是正确的。如果还有问题，主要会是文字描述不够清晰，而不是 manuscript 与 implementation 不一致。

### 四、Method 中最大的风险：ICBHI 的 checkpoint selection

接下来我们讨论了整场会议中最重要的问题之一：ICBHI 的 model/checkpoint selection protocol。

我解释，ICBHI 官方 split 只有 train 和 test，大约是 60/40，并没有官方 validation set。

问题在于，一些已有 ICBHI 工作在训练过程中实际上直接使用 official test set 的 score 来进行 checkpoint selection。也就是说，他们不是从 training set 中进一步拆出 validation set，然后根据 validation performance 选择模型，而是直接根据 official test set 上的 score 保存最好的 checkpoint。

我认为这实际上可能造成 test-set leakage，所以一直不知道论文里应该如何描述。

Yinuo先重新确认了整个流程。ICBHI 官方只有 train 和 test，没有 validation，这一点本身没有问题。但 validation 的作用就是进行 model selection：例如 model A 和 model B 都在 training set 上训练，然后应该在独立 validation set 上比较，决定哪个 checkpoint 更好。最后 test set 只用于最终 evaluation。

如果直接根据 test set performance 选择 checkpoint，那么 test set 实际上已经参与了训练过程中的 model selection。

在确认这确实是我们当前 follow 的 protocol 后，Yinuo认为从标准 machine-learning methodology 来说，这种做法非常可疑，甚至可以明确理解为一种 test leakage。

她用考试作类比：正常情况下，training 相当于平时学习，test 相当于最终考试。正常流程是不能提前看到考试卷子的。如果一个人可以连续考很多次，然后从这些考试里挑最高分的一次作为最终成绩，那么 test 已经不再是独立评价。

我进一步解释，这并不是我们自己随意设计的，而是 ICBHI benchmark 中相当常见的做法。

例如我之前提到的一个韩国团队，他们长期在 ICBHI 上优化 performance，从早期 50 多分一直做到现在大约 64、65 左右。他们公开 implementation 中直接把 official test split 放进名为 validation loader 的变量，然后根据这个 loader 的 score 更新 checkpoints。

而且并不是只有他们一组这么做。我看到的很多 ICBHI 工作都有类似 protocol。因此我真正纠结的是：如果我们完全按照更严格的 machine-learning protocol 重新从 training set 中拆 validation，虽然方法学上更规范，但我们的 training data 会进一步减少，而且最终 performance 也很难与 prior work 的 reported results 公平比较。

Yinuo的观点是，不能因为 previous work 这么做，就默认我们也应该这样做。如果我们自己明确认为从 training set 中再拆出例如 70/30 的 train-validation 才是正确的方法，那么完全可以选择正确的方法，而不应该仅仅为了和别人一致就继续一个有问题的 protocol。

她同时指出，很多 benchmark-oriented work 为了不断提高 reported score，evaluation protocol 未必经得住严格检查，有些工作可能默认 reviewer 不会深入查看 implementation 或复现整个过程。

不过她也认可这里存在 fair comparison 的现实问题。因此没有立即要求我全部重跑，而是提出先检查 representative prior work。

我会把一到两篇最有代表性的论文发给她，也可以把 repo 一并发过去。如果有必要，可以提供三四篇类似工作。她会重点检查这些论文到底如何描述 model selection，以及论文描述和代码 implementation 是否一致。

如果 prior work 明确说明就是根据 official test score 选择 checkpoint，而且这个 protocol 已经成为该 benchmark 明确使用的 convention，那么我们可以考虑为了 fair comparison 保持一致，但 manuscript 必须准确、透明地写清楚。

如果这些论文本身也没有解释清楚，或者 implementation 暴露出明显的 evaluation problem，那么我们就需要重新考虑当前 protocol，甚至可能需要安排时间重新运行部分实验。

我也提到，更早期的一些工作其实会从 official training set 中进一步按照例如 70/30 拆分 train 和 validation，这在方法学上明显更加正常。只是后来的很多工作逐渐开始直接使用 test score 进行 checkpoint selection。

因此，这一问题最终暂时没有直接定论。Yinuo 会先检查我提供的 prior work，然后再决定我们是只需要 rephrase，还是需要对实验 protocol 本身进行修改。

考虑到 deadline 已经非常接近，我们判断最现实的情况可能是重新明确措辞，但在检查相关论文之前不能直接假定当前做法没有问题。

这被我们共同认为是当前 Method 中风险最高的部分。

### 五、Evaluation Setup：metrics 和 dataset subsets

随后进入 Evaluation。

Evaluation 开头主要解释每个 dataset 最终 report 的 score 是如何计算的。这一部分是根据 Jingping 前一天的意见补充的。

对于 ICBHI，我们明确写出 score 的计算方式，并解释 SP、SE 等指标分别代表什么。

对于 SPRSound，也说明对应 metric 的计算方式。

对于 HF 和 KAUH，则需要说明我们没有使用完整数据，而是选择了与当前任务兼容的 subset。

我指出，KAUH 使用 subset 的事情已经在 Section 2 的 dataset description 中明确过，所以 Evaluation 中没有再次重复；但是 HF 的 subset information 目前又放在 Evaluation 这里，因此两个 external datasets 的结构并不完全一致。

Yinuo建议把 HF 的相关说明也移动到 Section 2，这样 dataset selection rationale 都统一在 Dataset Section 中说明。

她进一步确认我有没有解释为什么使用 subset。我说明已经解释过：部分 labels 和我们的 target label space 不兼容，因此只选择能够和当前任务对应的部分。

Yinuo认为这个理由本身没有问题，主要需要统一信息出现的位置。

### 六、Table 1：Frozen Encoder Baselines

接下来我们开始逐张表讨论 Results。

Table 1 是比较基础的 baseline comparison。相关实验由本科生根据我提供的实验参数运行，主要比较几个 frozen encoders 和我们的方法。

对于 ICBHI，baseline 只在 ICBHI 上训练；对于 SPRSound，则只在 SPRSound 上训练。这里主要想观察 frozen encoder 在各自 native task 上能够得到什么样的 performance。

HF 和 KAUH 作为 external evaluation datasets，用训练好的模型进行测试。

所有结果都是三次运行的均值，并附带相应的 variation / uncertainty statistic。

这一张表主要支持一个比较直接的结论：相比单纯使用 frozen encoder，我们通过联合学习、fine-tuning encoder 以及加入不同 supervision heads，可以获得更好的整体表现。

因此 Table 1 本质上属于 relatively simple baseline comparison，并不是论文最复杂的实验。

### 七、Table 2：与其他 multi-dataset / label-learning 方法比较

Table 2 是根据另一位老师之前的意见新增的。

对方提出，既然我们的工作本身也是一个 multi-dataset / multi-label learning framework，就应该与已有类似方法比较，而不能只和 frozen encoder baseline 比。

因此我们加入了两个性质不同的 baseline。

第一个是 DCASE / environmental sound detection 方向的方法。

这个工作原本不是针对 respiratory acoustics，而是针对 environmental sound detection。面对多个 datasets 时，它会寻找不同 datasets 之间能够形成共同集合的 classes，只使用这些共有类别训练模型，并 mask 掉不能统一的 labels。

我们把这种方法迁移到当前 respiratory datasets，并在和我们相同的 training setting 下运行。

第二个是 PC-MCL，也就是前面提到的韩国团队相关工作。

他们主要关注 ICBHI benchmark，并通过 sample concatenation 等 augmentation 提高 performance。例如将两个大约 5 秒的 samples 拼接起来训练。

这种做法会带来 label hierarchy 的问题：如果拼接后的 sample 同时包含 normal 和 abnormal，那么显然不能把整个 sample 继续定义为 normal，因此他们进一步设计了 label decomposition / handling strategy。

在我们的 Table 2 中，PC-MCL 只在 ICBHI 上训练，然后测试它在其他三个 datasets 上的迁移表现。

因此 Table 2 实际上在比较两种不同的问题。

对于 DCASE-derived baseline，我们比较的是：在相同 cross-dataset training setting 下，不同 multi-dataset / label harmonization strategy 的表现。

对于 PC-MCL，我们比较的是：已有 ICBHI-focused 方法虽然能够优化 native benchmark performance，但在没有专门进行 multi-dataset training 的情况下，其 cross-dataset generalization 能力如何。

因此这两个 baseline 虽然放在同一张 Table 中，但性质并不完全一样。正文必须明确解释这个区别，不能让 reader 误以为它们是两个完全同类的方法。

### 八、Table 3：Ablation Study 与主方法重新定义

Table 3 是本次会议另一个最重要的讨论点。

首先我们逐个确认不同 variants 的含义。

其中一个 variant 只使用 ICBHI 训练，另一个只使用 SPRSound 训练。它们的方法结构基本与主方法一致，但 training data 被限制在单一 dataset。

结果显示，这些方法在自己的 native task 上表现还可以，但迁移到其他 datasets 后 performance 明显下降。

这支持我们的一个核心观察：只针对单一 respiratory dataset 进行优化，即使 native performance 不错，也无法保证稳定的 cross-dataset generalization。

之后我们讨论了一个使用 native-task heads 的配置。

这个配置和当前主方法的重要区别在于，它保留两个 native tasks 对应的独立 classification heads，然后再通过这些 native outputs 映射到 HF 和 KAUH，而不是完全使用当前主方法的统一 readout design。

在这个配置基础上加入我们提出的额外 supervision 后，它实际上比当前定义的主方法表现更好。

Yinuo问我为什么 native-task-oriented 的配置反而会优于当前 base model。

我解释，尤其对于 ICBHI 来说，这其实是合理的。Native task 本身就更加贴合 ICBHI 的原始 prediction structure。我们的整个方法实际上也是从这些 native task 的结构出发，然后进一步设计不同 datasets 之间如何共享 supervision，以及如何迁移到 external datasets。

因此，保留 native-task readout，同时加入我们提出的 cross-dataset supervision，从逻辑上反而更加自然。

这也是为什么我想把它直接提升为新的主方法：它既能够服务 native task，又使用了我们提出的额外 supervision，因此比目前主方法更符合论文真正想表达的 cross-dataset learning story。

Yinuo认为这样更 make sense。

她指出，我们这篇论文真正的 main claim 并不是：

“我们先设计一个 base model，然后在这个 base model 上加入一些 regularization，发现 performance 又提高了一点。”

真正应该表达的是：

“我们设计了一个完整的 model，这个 model 包含这些 supervision、regularization 和 architectural features；这是完整模型的 performance；然后我们通过 ablation 分析这些 components 分别有什么作用。”

也就是说，论文里的 base/main model 应该直接是包含完整设计、能够达到最佳合理 performance 的模型，而不是人为把一个较弱的中间版本定义成主方法。

因此，如果新的配置已经包含我们真正想贡献的设计，而且结果更好，那么它更适合作为主方法。

同时，这意味着 Table 3 中原来的一些 variants 会变得重复。

如果新的 main method 本身已经包含某些 supervision，那么原来一个和它实际上等价的 variant 就应该直接删除，而不是继续作为一个单独的 ablation row。

我们因此基本达成共识：倾向于把这个表现更好、结构也更完整的配置提升为新的主方法，同时删除和新主方法重复的 variant。

相应地，Figure 2、Method Section、Table 1、Table 2、Table 3，以及 Results 中所有涉及主方法名称、结构和数字的地方都需要同步更新。这个修改本身不会要求我们重新设计一套实验，但必须保证全文的方法定义保持一致，不能出现 Method 中描述的是旧主方法，而 Results 中已经使用新主方法的情况。

### 九、为什么更换主方法不会改变论文的核心贡献

我们随后进一步讨论了：如果把这个 variant 提升为主方法，并删除原来的一部分 readout design，会不会影响论文原本的 contribution。

Yinuo认为不会，而且这样反而能让论文的 story 更清楚。

她强调，我们最终想说的并不是“我们有一个基础模型，然后在基础模型上不断增加 regularization，最后发现性能有所提升”。如果按照这种方式组织，reader 很容易把论文理解成一个 incremental model improvement。

真正应该强调的是，我们设计了一个完整的 model，这个 model 本身就包含解决当前 heterogeneous respiratory datasets 问题所需要的 supervision 和 feature。然后我们展示完整模型在 native datasets 和 external datasets 上的 performance，再通过 ablation study 分析其中不同 components 的作用。

因此，最终定义的 base/main model 就应该是包含完整设计、整体表现最合理的版本。

换句话说，原来那个较弱的 model 没有必要因为历史上最先被我们定义为“base model”，就一直保留为论文主方法。如果后续实验已经证明另一个配置更加符合方法设计，也有更好的 performance，那么直接把后者作为完整模型更合理。

我确认理解这个逻辑，并决定按照这个方向修改：删除与新主方法重复的配置，把新的完整 configuration 提升成主方法，然后同步修改 Method、Figure 和 Results。

### 十、结果应该报告 SEM 还是 Standard Deviation

在讨论 Table 3 时，Yinuo又注意到一个重要问题：每张表里均值后面的 “±” 到底表示 standard deviation 还是 standard error of the mean。

我说明目前报告的是 standard error of the mean，SEM。

Yinuo指出，如果这些数字是 SEM，那么目前很多看起来有数值差异的结果实际上未必具有 statistical significance。

例如，一个结果是 91.55，另一个是 90.72。如果后面的 uncertainty 是 SEM，而且两个结果的 uncertainty intervals 有明显 overlap，那么仅仅因为 91.55 数值更高，并不能直接声称它 statistically significantly better。

这使得当前 Results 中一些“higher performance”之类的描述需要更加谨慎。

我最初提出，那是否应该直接把所有 SEM 换成 standard deviation。

Yinuo认为两种方式其实都可以，但选择取决于论文想面对的 target audience。

如果 target audience 更偏 machine learning community，那么报告 mean ± standard deviation 会更加符合很多 ML papers 的常见写法。

但如果主要 target audience 是 respiratory acoustics / signal processing / biomedical acoustic analysis，那么保留 SEM 也有意义，因为这类读者可能更关心不同模型 performance estimate 的 uncertainty 以及 differences 是否 statistically significant。

如果继续使用 SEM，那么 Results 和 Discussion 中就不能只看 mean 的大小来判断方法优劣，而应该明确区分：

某个方法的 mean 更高；

以及这个 improvement 是否 statistically significant。

如果一个方法 mean 更高，但是 difference not significant，那么正文应该直接说明这一点。

如果某个 improvement 确实达到 significance，则可以考虑在 Table 中通过星号等方式标记。

例如，最高 mean 仍然可以 bold，但是如果和 comparison method 的差异不显著，就不要额外标 significance star；只有统计检验确认 significant 的结果才增加对应标记。

### 十一、需要进行 statistical significance test

在这个基础上，Yinuo建议进一步做 statistical significance testing，而不是只通过均值和 SEM 肉眼判断。

她提到了 ANOVA 一类的 statistical test，并询问我是否知道怎么做。我确认自己可以处理，因此不需要她额外演示。

她举例说，有些结果例如 91.55 和 90.72，即使 mean 有差异，根据当前 SEM 来看，很可能并不 significant。

相反，如果一个方法是 61.7 ± 0.31，而其他方法明显低很多，那么这种差距很可能在统计检验中会达到较高的 significance level。

因此，一个比较合理的做法是把当前多次运行的 raw results 拿出来，进行正式的 significance analysis，再根据结果决定 Table 中哪些 improvement 可以明确强调。

我表示可以同时做两件事：

一方面重新计算 standard deviation；

另一方面也进行 statistical significance test。

两套数字都准备好之后，再决定最终 manuscript 中使用 SD 还是 SEM，以及 Results 中如何描述。

Yinuo同意这种做法。

### 十二、为什么 Yinuo 更倾向于保留 SEM + significance

虽然两种 reporting strategy 都可以，但 Yinuo个人更倾向于保留 SEM，同时加入 significance information。

她的理由主要来自这篇论文真正的 audience 和 story。

如果这篇工作是纯 machine-learning paper，那么模型 performance 的小幅提升以及 standard deviation 可能更加符合 community convention。

但我们的工作本身并不是要证明“我们发明了一个全新的 machine-learning algorithm，在同一个 benchmark 上比所有模型准确率更高”。

实际上，已经有很多团队在 ICBHI 等数据集上尝试不同 ML methods，我们也不是第一个做 respiratory sound classification 的团队。

因此，论文真正应该面对的 audience 更可能是关心：

如何从 respiratory sounds 中识别 abnormal conditions；

不同 respiratory datasets 之间为什么不能直接互换；

不同 annotation structures 如何影响 model training；

以及如何建立一种能够联合不同 datasets 的 evaluation / learning framework。

Yinuo查看了一下 ICASSP 的定位，也认为这个 venue 的读者本身应该更关心 acoustic / speech / signal processing，而不是只关心 machine-learning benchmark leaderboard。

因此在这个 context 下，machine learning 更像是我们解决 respiratory acoustic dataset integration 问题的工具。

模型设计当然仍然是 contribution，但更重要的 contribution 是：这个方法能够把不同 datasets、不同 annotation structures 和不同 supervision 结合起来。

从这个角度看，保留 SEM，并明确告诉读者哪些 improvement significant、哪些只是 mean 上更高，反而可能更符合论文的科学叙事。

我表示会先把 SD 也计算出来，然后结合 significance results 再最终决定。

### 十三、Table 3 中 patient-related component 的 ablation

之后我们继续讨论 Table 3 剩余的 ablation。

其中一个 variant 是把之前参考已有工作的 patient-related component 去掉，但仍然保留我们主方法的其他结构。

这个组件本身来自 prior work，而且 prior work 主要针对 ICBHI。因此从实验结果来看，移除这个 component 后，在 ICBHI 上 performance 会下降，这和它原本针对 ICBHI 优化的设计是吻合的。

但在其他 datasets 上，移除这个 component 并没有造成特别明显的 performance degradation。

甚至在某个较简单的 binary external task 上，去掉该 component 后 mean score 还稍微更高。

我解释，对于 KAUH 之类的数据集，这个现象并不是完全无法解释，因为该 dataset 本身已经进行了比较严格的 patient-level organization / grouping，而这个 component 原本也是针对 patient-related variation 进行处理的。

不过 Yinuo再次提醒：不能因为某个 mean 数字稍微更高，就直接说这个 variant better。

从 SEM 来看，这个 improvement 仍然可能 not significant。

真正可以比较明确地讨论的是 variance。

如果去掉某个组件之后 mean 没有显著变化，但是 SEM 更小，那么至少说明不同 runs 之间的 variation 更小，model 的结果更加稳定。

尤其对于一个比较简单的 binary task，这种 variance reduction 可能比非常小的 mean improvement 更值得讨论。

因此这一部分 Results 也需要在 statistical test 之后重新组织语言，避免把 numerical difference 直接解释成 performance improvement。

### 十四、针对 SPRSound supervision 的 ablation

接下来一个 variant 是专门针对 SPRSound 做的。

这个 variant 保留整体 architecture，但对于 SPRSound training data，移除了 SPRSound 对我们提出的两个 auxiliary supervision heads 的监督。

实验结果显示，移除这些 supervision 后，整体 performance 会明显下降。

我认为这一结果比较直接地支持了我们设计共享 supervision 的必要性：如果不同 datasets 只是被放在一起训练，却不能共同参与这些 supervision signals，那么整个 multi-dataset learning framework 的效果会明显变差。

换句话说，这个 ablation 支撑的是为什么我们需要让多个 datasets 共享同一套 supervision structure，而不是仅仅把不同 datasets 拼接起来训练。

对于这一条结果，Yinuo没有提出明显异议，认为它和当前论文 story 是一致的。

### 十五、将 HF 加入 shared supervision 的最后一个 variant

Table 3 最后一个 variant 是把 HF dataset 也进一步加入 shared supervision。

也就是说，我们尝试让 HF 不只是 external evaluation dataset，而是让其中兼容的部分数据也参与我们提出的 supervision framework。

但是结果显示，这样做并没有很好地改善 performance。

我解释，这和 HF 本身的数据结构有很大关系。

首先，HF 的数据量相对于其他 datasets 非常大。如果直接把它完整加入 training，它的数据量可能会 dominate 整个 learning process，使模型训练被 HF 主导。

其次，HF 的 annotation philosophy 和我们当前主要使用的两个 training datasets 差异很大。

HF 并不是像当前主要 datasets 那样，对完整 respiratory cycle 或完整 event 进行统一 annotation。它更多只标注某个 acoustic event 出现的时间，而且缺少和我们当前 supervision framework 完全对应的 negative samples / complementary labels。

因此，它并不适合作为一个能够直接加入当前 joint training framework 的完整 training dataset。

我们只能从中提取一部分与当前 label system 兼容的 subset。

这也是为什么即使尝试把 HF 的部分信息加入 shared supervision，也没有带来特别明显的 improvement。如果未来真的希望充分利用 HF，可能需要进一步进行 dataset-specific adaptation，而不是直接套用当前 supervision strategy。

### 十六、为什么 HF 和 KAUH 主要作为 external evaluation datasets

Yinuo随后进一步问：为什么这里特别是 HF 会有这种处理方式，以及这些原因是否已经在 Discussion 中解释。

我说明目前已经提到 HF 缺少对应 negative samples，以及为什么我们只能使用它的 subset，但后续还可以把理由写得更清楚。

HF 不适合作为主要 training dataset，核心有两个原因：

第一，数据量异常大，直接加入 joint training 后容易 dominate optimization；

第二，它的 annotation unit 和 annotation philosophy 与当前主要 training datasets 差异过大，无法直接共享完全相同的 supervision。

对于 KAUH，原因又不同。

KAUH 的数据量太少，总共只有大约 336 个 recordings，而且这些 recordings 实际来自大约 112 个 patients，每个 patient 对应三种不同 filtering conditions。

因此，它看起来有 336 recordings，但独立 patient 数量实际上非常有限。

如果再把这个数据集拆成 training / validation / test，数据会变得非常碎，很难形成稳定的 training set。

与此同时，KAUH 中部分 annotations 也和我们当前 target labels 不完全一致，因此甚至不能直接使用全部 recordings，只能使用兼容的 subset。

因此我们最终没有把 KAUH 纳入主要 training datasets，而是把它作为 external evaluation site。

这样，四个 datasets 在论文中的角色就有比较明确的 rationale：

ICBHI 和 SPRSound 是主要 joint-training datasets；

HF 和 KAUH 则因为数据规模、annotation structure、label compatibility 等不同原因，更适合作为 external evaluation datasets。

这些 rationale 需要在 Dataset / Discussion 中明确写出来。

### 十七、最后一个 HF variant 是否应该保留

在理解了 HF 的逻辑后，Yinuo一开始对 Table 3 最后这个 HF-related variant 是否有必要保留表示不确定，并建议我也可以问 Jingping。

我解释，这个 experiment 的意义并不是为了证明 HF 能提升性能，而是回答另一个问题：

我们提出的 shared supervision 是否只能在当前两个主要 training datasets 上工作，还是其他 respiratory dataset 也有可能利用这套 supervision？

也就是说，这个 variant 实际上是在测试我们方法向第三个 heterogeneous dataset 扩展时会发生什么。

即使结果没有明显提升，它仍然提供了有价值的信息：当前 supervision framework 并不能直接无条件扩展到所有 datasets，尤其当 annotation structure 和 sample distribution 差异过大时，需要额外 adaptation。

听完这个解释后，Yinuo认为这个 point 是成立的，因此改变了最初的看法，同意保留这个 variant。

### 十八、Conclusion 和 Abstract 暂时最后再改

Table 3 讨论结束后，我们简单谈到了 Conclusion。

我的想法是，Conclusion 和 Abstract 目前不需要马上重新改，因为接下来主方法、Table 结果、significance reporting 和部分 Method wording 还会发生变化。

因此更合理的顺序是先把 Method 和 Results 稳定下来，最后再回头调整 Abstract 和 Conclusion，使它们准确对应最终版本。

Yinuo同意这个顺序。

### 十九、会议结束前对当前问题的总结

我最后重新总结了一遍目前论文各部分的状态。

Introduction 基本没有大的问题。

Dataset Section 需要 Yinuo 会后快速完整检查一次，尤其是语言、Figure 1C 的 Frequency (Hz) 标注，以及 HF / KAUH subset rationale 的位置是否统一。

Method 中最大的问题是 ICBHI checkpoint selection protocol。尤其需要确认 prior work 是否真的使用 official test set 进行 checkpoint selection，以及他们在论文中是如何描述这一点的。

我会把相关论文和必要的 repo 发给 Yinuo。

Evaluation Setup 本身基本没有大问题，主要需要统一 HF 和 KAUH subset information。

Results 最大的修改则是把新的 configuration 提升为主方法，并同步修改所有 Tables、Figures 和对应文字。

另外需要重新处理 SEM / SD / statistical significance 的问题。

### 二十、Yinuo 建议的具体修改顺序

Yinuo随后给出了一个比较明确的修改顺序。

第一步，不要先大规模修改正文，而是先把和新主方法直接相关的 Figures 和 Tables 更新掉。

尤其需要先更新 Table 1、Table 2 以及相关 method configuration，因为这些结果会直接决定 Section 4.1 和 4.2 应该如何重新写。

也就是说，先确定最终主方法到底是什么、它在主要实验中的数字是多少，然后再根据这些最终结果修改 Results narrative。

第二步，在结果数字稳定之后处理统计报告问题。

这里可以有两条路线：

一种是把目前的 SEM 改成 standard deviation；

另一种是保留 SEM，同时做 formal statistical significance test。

如果选择后者，那么 Section 4.1 和 4.2 中可能需要分别增加两三句话，解释哪些 numerical improvements statistically significant，哪些虽然 mean 更高但 difference not significant。

Yinuo举例说明，如果我们的模型在某一个 setting 下不是绝对最高，但是它与最高方法之间没有 significant difference，而在其他 datasets / conditions 下又明显优于其他方法，那么可以从整体上讨论模型的跨数据集稳定性。

不过这里的文字必须建立在真正的 significance test 结果上，不能只凭均值做判断。

我表示 SD 和 significance test 可以两个都做，因为计算成本都很低。等结果出来后再决定 manuscript 最终采用哪一种 reporting style。

Yinuo同意。

第三步，在 Figures、Tables 和 statistical reporting 都确定之后，再修改 Section 4.1、4.2 的具体 narrative。

这样可以避免先写一版 Results，之后数字变化又重新改一遍。

第四步，Yinuo会去检查前面的 writing 和 logic，尤其是 Method 中我标记出来的那些可能 confusing 的段落。

与此同时，她会阅读我发给她的 ICBHI prior work，重点确认 test-set-based checkpoint selection 到底是如何处理和描述的。

### 二十一、关于 ICBHI protocol 的最终补充讨论

在会议快结束时，我们又回到了 ICBHI checkpoint selection。

Yinuo再次表示，这件事情从标准 evaluation methodology 来看确实很奇怪。

理论上 test set 不应该参与 model selection。如果根据 test score 选择 checkpoint，本质上相当于通过考试结果不断决定自己应该采用哪种学习方法，这会破坏 test set 作为 independent evaluation 的作用。

我说明，我第一次实现时其实采用的是正常 validation 思路，也就是从 training data 中进一步拆 validation。

但是这样做之后有两个现实问题。

第一，原本 training data 就不算很多，再拆 validation 后真正用于 training 的数据进一步减少。

第二，也是更关键的问题，最终 score 无法与大量 existing work 公平比较，因为很多 state-of-the-art work 实际上直接使用 official test score 选择 checkpoint。

因此，如果我们单独使用更严格 protocol，最终数字可能明显低于 prior work，但这种差距并不完全来自模型能力，而是来自 evaluation protocol 不一致。

Yinuo理解这个问题。

她因此没有要求我们立刻改变 protocol，而是让我挑一两篇在这个领域比较 high-impact、最有代表性的工作发给她。

如果这些论文能够非常明确地解释这种做法，并且它已经成为该 benchmark 的公开 evaluation convention，那么我们可以为了 fair comparison 使用相同方式，同时准确披露。

但如果论文自己没有说清楚，只是在代码里偷偷把 test loader 当 validation loader，那么这个问题就需要更加谨慎。

我提到这些工作很多本身也是发表在 ICASSP、Interspeech 等 venue，因此至少从历史实践来看，这种 protocol 在相关 community 中并不是完全罕见。

Yinuo最终表示先看这些论文具体怎么写，再决定我们的文字是否只需要 rephrase，还是需要更大的 protocol adjustment。

### 二十二、本次会议最终确定的待办事项

本次会议结束后，我需要首先确定新的主方法，并把当前表现更好、结构上也更符合论文 contribution 的 configuration 提升为 main method。与新主方法重复的旧 variant 应从 Table 3 中删除。

随后需要同步更新 Figure 2、Table 1、Table 2、Table 3，以及 Method 和 Results 中所有与旧主方法相关的方法名称、结构描述和结果数字。修改时要确保 Figure、Method 和 Evaluation 三部分使用完全一致的方法定义。

在更新实验结果后，我需要重新计算各组实验的 standard deviation，同时基于现有多次运行结果进行 statistical significance test。之后再决定最终论文是采用 mean ± SD，还是继续采用 mean ± SEM 并额外标记 significance。按照 Yinuo 的个人倾向，目前更推荐后者，但最终仍然根据实际计算结果决定。

如果保留 SEM，则 Results 中所有“higher”“better”“improved”等表述都需要重新检查。均值更高不等于 statistically significantly better。对于差异不显著的结果，应明确说明 difference is not significant；对于显著结果，可以考虑在表格中用星号等方式标记。Bold 仍然可以用于表示 numerical best，但不能让 bold 本身暗示 statistical significance。

对于 Table 3 中 patient-related component 的 ablation，需要重新检查当前文字，避免把 external binary task 上很小的 numerical increase 描述成明确 improvement。如果 difference not significant，更合理的讨论重点可能是 variation / stability，而不是 mean performance。

对于 SPRSound supervision ablation，需要继续保留并强调其核心作用：移除 SPRSound 对两个 auxiliary supervision heads 的监督后，整体 performance 明显下降，因此这一实验支持不同 training datasets 共享 supervision structure 的设计。

对于 HF-related variant，最终决定保留。它的意义不是证明 HF 加入训练后能够提高 performance，而是测试我们提出的 shared supervision 能否进一步扩展到 annotation structure 明显不同的第三个 dataset。结果没有明显改善本身也是一个有意义的观察，并说明当前 framework 面对 annotation philosophy 和 sample distribution 差异更大的 dataset 时可能需要额外 adaptation。

Dataset Section 中需要统一 HF 和 KAUH 的 subset rationale。HF 为什么只使用 subset、为什么不作为主要 training dataset，以及 KAUH 为什么主要作为 external evaluation，都应该在数据介绍阶段尽量讲清楚，而不是等到 Results 才第一次解释。

Figure 1C 需要补充纵轴标签 Frequency (Hz)。

Method 中关于 abnormal broad class、不同 label structures 之间关系，以及 optional supervision 的几段文字，需要由 Yinuo 会后重点检查。她会根据情况直接修改或者留下 comments。

最重要的方法学待办事项仍然是 ICBHI checkpoint selection。我需要把一到两篇最有代表性的相关工作以及必要时对应的 GitHub implementation 发给 Yinuo。重点不是单纯证明“别人也这么做”，而是确认 prior work 到底有没有明确说明使用 official test set 进行 checkpoint selection，以及这种做法在该 benchmark 中究竟是公开的 evaluation convention，还是只存在于 implementation 中。

Yinuo会检查这些论文，然后再判断我们的 manuscript 应该如何处理。如果 prior work 对 protocol 说明得很清楚，那么当前实验大概率只需要重新措辞并透明说明；如果 prior work 本身也存在不清楚甚至明显不规范的地方，则需要重新考虑是否应该从 official training set 中建立独立 validation split，并视情况重新运行相关实验。

最后，Abstract 和 Conclusion 暂时不优先修改。应该等 Method、主方法定义、Tables、statistical analysis 和 Results narrative 全部稳定之后，再回头统一调整 Abstract 和 Conclusion。

### 二十三、最终确定的修改优先级

综合最后的讨论，目前最合理的工作顺序可以概括为：

第一，先确定新的 main method，把表现更好且包含完整 supervision design 的 configuration 提升为最终方法，同时删除与之重复的 ablation configuration。

第二，更新所有受主方法变化影响的实验结果和展示内容，尤其是 Figure 2、Table 1、Table 2 和 Table 3。这里要先把“最终数字”确定下来，不要急着先重写 Results。

第三，重新计算 SD，并对多次运行结果进行 significance analysis。比较 SD 和 SEM 两种 reporting 方式，再决定最终 Table 的呈现方式。如果继续使用 SEM，则同时设计 significance notation，并相应调整正文。

第四，根据最终 Tables 重写 Section 4.1 和 4.2，尤其注意 numerical improvement 与 statistically significant improvement 的区别，同时重新组织对 cross-dataset stability 的描述。

第五，统一 Section 2 中 HF 和 KAUH subset selection 的解释，补 Figure 1C 的 Frequency (Hz)，并完善 dataset roles 的 rationale。

第六，由 Yinuo检查 Method 中目前标出的 unclear paragraphs，包括 abnormal label、label mapping、optional supervision 等表达问题。

第七，我把 ICBHI 相关 representative papers / repositories 发给 Yinuo，由她检查 test-set checkpoint selection 的实际做法和论文表述，然后共同决定当前 Method 只需要 rephrase，还是必须修改实验 protocol。

第八，在上述内容稳定后，最后统一修改 Abstract、Introduction 中受影响的少量表述，以及 Conclusion，确保全文从 research question、method、results 到 conclusion 使用同一套最终 story。

### 二十四、整场讨论形成的论文主线

经过这次讨论，我们对论文应该如何讲故事也进一步形成了比较明确的共识。

论文不应该被写成一个单纯追求 ICBHI benchmark accuracy 的 machine-learning model improvement work，也不应该把主要贡献包装成“在一个 base model 上加入几个 regularization 后性能有所提升”。

论文真正关注的是 heterogeneous respiratory acoustic datasets 之间存在不同的 native tasks、annotation structures、label spaces 和 dataset characteristics。在这种情况下，单一 dataset 上优化得到的模型即使 native performance 较好，也未必能够迁移到其他 respiratory datasets。

因此，我们的方法应该作为一个完整 framework 出现：保留适合 native tasks 的 prediction structure，同时利用跨数据集共享的 supervision，使多个具有不同 annotation structures 的 datasets 能够参与统一学习，并进一步在未直接参与主要训练的 external datasets 上评价这种设计的泛化能力。

从这个角度看，Table 1、Table 2 和 Table 3 各自承担不同的证据角色。

Table 1 说明相比简单 frozen encoder / native baseline，完整训练 framework 能够获得更好的整体表现。

Table 2 说明我们的重点不仅是 native benchmark performance，还包括与其他 multi-dataset / label-handling strategies 相比时的 cross-dataset behavior。DCASE-derived method 和 PC-MCL 虽然性质不同，但分别提供了 multi-dataset strategy 和 ICBHI-specific strategy 两种参照。

Table 3 则通过 ablation 回答完整 framework 中不同设计为什么存在。Single-dataset variants 展示只在一个 dataset 上训练的跨数据集局限；shared supervision ablation 展示多个 datasets 共同参与 supervision 的必要性；patient-related component 展示已有设计在不同 datasets 上的作用并不完全一致；HF experiment 则展示当前 supervision framework 向 annotation structure 差异更大的 dataset 扩展时存在的边界。

因此，最终主方法应该直接代表这套完整设计，而不是一个人为定义的较弱中间版本。

### 二十五、统计结果在最终论文中的表达原则

Yinuo在这次讨论中反复强调的另一个核心问题，是不要把 numerical difference 自动写成 scientific conclusion。

尤其如果继续报告 SEM，那么像 91.55 和 90.72 这样的结果，即使前者 numerical value 更高，也必须结合 uncertainty 和 statistical test 判断两者是否真的存在显著差异。

因此最终写作中应该区分三种情况：

如果某个方法 mean 更高且 difference statistically significant，可以明确讨论其 improvement，并在表格中使用 significance notation。

如果 mean 更高但 difference not significant，可以报告 numerical difference，但不能把它写成明确的性能优势。

如果 mean 基本没有显著变化，但 uncertainty / variation 更小，则可以讨论 model stability，但同样要准确说明证据实际支持的是什么。

这对于我们的论文尤其重要，因为真正的 contribution 并不是在每一个 dataset 上都必须取得最高的单点数字，而是希望说明一个完整方法在 heterogeneous respiratory datasets 下能够提供更稳定、更具有 cross-dataset applicability 的表现。

不过，“stable”“better generalization”或者类似表述最终仍然必须由对应实验和 statistical analysis 支撑，而不能只根据某一个 mean 值推导。

### 二十六、会议结束

最后我确认会先按照上述顺序修改论文，并把 ICBHI 相关论文发给 Yinuo。

Yinuo会在会后继续检查 Dataset 和 Method 的 writing / logic，并重点阅读我提供的 prior work，判断 ICBHI checkpoint-selection protocol 应该如何处理。

由于投稿 deadline 已经非常接近，目前目标是优先解决会直接影响科学正确性和全文一致性的问题：主方法定义、Results 数字、statistical reporting，以及 ICBHI evaluation protocol。语言层面的细节修改则在这些内容确定之后继续完成。

如果后续修改过程中还有问题，我会直接通过微信和 Yinuo 沟通。

至此，本次关于论文最终修改、Method、Evaluation、ablation 和 statistical reporting 的讨论结束。
