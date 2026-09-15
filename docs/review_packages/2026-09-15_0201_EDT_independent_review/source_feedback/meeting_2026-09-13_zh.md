# 2026-09-13 Paper Revision Meeting 完整中文会议记录

Notion：[会议全文](https://app.notion.com/p/3da309efda29818fa0f1d0edb27a7ea6)；关联[9/13–9/14新工作计划](https://app.notion.com/p/3da309efda29818d9474cf12d24dfdd9)。

日期：2026-09-13。来源：用户提供的完整版中文会议纪要。下面保留用户提供的全文；本任务未重新听取原始录音。后续分析与行动计划单独记录，不改写为老师的原话。

## 阅读索引

- 中心问题：整篇论文应围绕明确的 research question、gap 和 contribution 组织。
- Introduction：motivation、current gap/limitation、contribution，约2–4个自然段，贡献用一段结构化文字表达。
- Dataset 与 Figure 1：区分主训练、HF辅助条件和外部评测，只保留能服务结果解释的信息。
- Method 与 Figure 2：输入、输出、变量、latent representation、监督及损失应相互对应。
- Evaluation：按问题组织表格，清楚标记来源、baseline、proposed method、增益和trade-off，并控制结论范围。
- 时间与沟通：老师希望第二天晚上看到能用的稿件，并要求给advisor留出反馈时间。

## 用户提供的完整纪要

下面是按你之前偏好的风格整理的完整版中文会议纪要。我尽量保留了老师的原意、批评重点、具体修改建议以及你们来回讨论的逻辑，不做额外引申，也不把它压缩成过度简短的总结。

2026年9月13日 Paper Revision Meeting 完整中文会议纪要

本次会议主要围绕当前 ICASSP paper 的整体完成度、实验结果展示、Introduction 的逻辑组织、Dataset Analysis、Method Figure、Evaluation Table 以及整篇论文的学术写作方式展开。老师认为目前论文最大的障碍已经不只是某一个实验缺失或者某一句话需要修改，而是整篇文章的 high-level story、逻辑组织、figure/table 的信息表达和文字之间缺少足够清晰的对应关系。当前版本更像是把已经完成的工作逐项罗列出来的 lab report，而不是围绕一个核心 research question 和 contribution 组织起来的 conference paper。因此，接下来最重要的工作不是继续往论文里增加更多内容，而是重新梳理现有内容，使每一个实验、每一张图、每一个表和每一段文字都能够明确服务于论文的中心论点。

会议开始时先讨论了实验进度。目前主要还剩下一组消融/对照实验没有完全跑完，具体是 Table R 里的两个 “dataset-only/native-task” setting，即只使用 ICBHI 自身任务或只使用 SPRSound 自身任务训练的模型。你解释，这两组实验主要用于说明：如果模型只围绕单一数据集的原生任务进行训练，它在自身数据集上的表现可能合理，但在另一个数据集上的迁移或泛化性能会明显下降。例如，只在 ICBHI 上训练时，ICBHI 上可以获得大约 57 左右的结果，但是在 SPRSound 上会接近随机水平；反过来，SPRSound-only 的模型在 SPRSound 上表现合理，但在 ICBHI 上性能也会明显下降。你希望通过这组结果说明单一原生任务训练无法自然解决跨数据集泛化问题。目前实验实际上已经能看出这个趋势，但因为其他结果都是统一跑三次，所以仍准备把剩余实验按照同样的设置补完。

老师随后指出，这个例子正好暴露了当前 Table R 最大的问题：你口头解释之后，老师能够理解你想证明什么，但仅仅看表格本身完全无法迅速读出这个结论。对于 reviewer 而言，他们不会预先知道 ICBHI、SPRSound 各自是什么任务，也不会知道某一个数值到底算高还是低。如果表里只是一排方法名称和一排数字，reviewer 需要自己推测哪个是 baseline、哪个是 ours、哪个是 native-task result，以及某个数字为什么重要，这种呈现方式是不合格的。

老师强调，Table R 必须首先把 baseline 和 proposed method 的关系表达清楚。比如 ICBHI-only 对应什么，SPRSound-only 对应什么，你们最终的方法又对应什么，都必须有明确命名和视觉区分。现在像 HFO、某些代码内部缩写直接出现在表中，对 reviewer 完全没有意义。你解释这些名称只是代码阶段临时使用的名字，因为方法名称还没有最终确定，所以暂时直接放进了论文。老师认为，无论最终叫什么，现在都必须把它们改成 reviewer 能理解的表达，不能把内部代码变量直接放进正式论文。

除了名称之外，表格还要通过加粗、下划线、符号、脚注或者 improvement percentage 等方式直接告诉 reviewer “应该看哪里”。例如，如果某个 baseline 是原论文的结果，就应该在 table caption 或脚注中说明；如果某个方法相对于 baseline 提升明显，可以直接标出提升幅度；如果某个结果略低但另一个任务显著提高，也应该通过视觉组织让 reviewer 一眼看出 trade-off，而不是让 reviewer 自己用计算器或者反复对比数值。老师希望 Table R 能清楚表达这样的逻辑：一个统一模型在多个数据集任务之间取得兼容性能，同时在某些任务上显著提高，而单数据集的 native-task model 在跨数据集测试时会明显退化。这个结论必须直接从 table 的视觉设计中呈现出来。

老师认为，这并不是一个单独的 table 问题，而是全文共同存在的问题。目前论文“把所有结果陈述出来了”，但是没有持续围绕论文中心问题组织内容。读者需要不断自己总结“这一段为什么在这里”“这个结果到底支持什么”“这个图跟后面有什么关系”，而正常 reviewer 不会投入这么多精力去替作者完成这些推理。如果 idea 本身没有被主动突出，那么即使实验都做完了，reviewer 仍然可能觉得文章散乱、难以 follow。这也是老师认为目前 paper 最大的 rejection risk。

老师明确指出，现在整篇文章缺少“扣题”。很多段落更像是把之前整理的 bullet points 一个一个扩写成句子，再拼进 paper 中。虽然每一个信息点可能都正确，但它们之间没有明显的承上启下，也没有不断 callback 到 research gap、challenge 和 contribution。老师认为，这种写法看起来像 lab report：把做过的所有事情全部列出来，但是没有说明这些事情如何共同证明论文想表达的 central idea。

因此，在 Introduction 部分，老师建议采用非常标准、甚至略显模板化的学术论文结构。Introduction 不需要写得很花，而应该非常清楚地分成三个核心部分：motivation、current gap/limitation，以及 our contribution。Related work 可以嵌入 motivation 或 current limitation 中，但不能变成单纯的文献罗列。Introduction 总体控制在大约两到四个自然段，最好在一页左右或者稍微超过一页结束，不需要为了长度而塞入太多背景。

第一部分是 motivation，要解释为什么 respiratory acoustic classification / heterogeneous respiratory datasets 这个问题值得研究。已有工作可以在这里被提及，但作用不是单纯告诉 reviewer “有人做过这些事情”，而是用来支持“当前为什么还存在问题”。例如，已有工作可能已经处理某个单数据集任务或者某种 respiratory sound classification，但仍然没有统一面对不同数据集之间 annotation、task definition 和 label availability 的差异。

第二部分要明确写出 current gap 或 limitation。老师强调，关键词应该直接出现，不能让 reviewer 自己推断。比如可以明确地说 “However, existing studies…”、“A key limitation is…”、“Current methods do not…” 等。你们已经总结出的 gap 大致可以分成几类：第一，不同 respiratory datasets 的 annotation taxonomy、task definition、label availability 等高度不统一，而现有方法通常针对单一 dataset 设计，没有真正统一处理这种 heterogeneous supervision；第二，单数据集训练出来的模型直接迁移到其他数据集时，generalizability 很差，你们的 native-task experiment 可以作为具体 evidence；第三，不同 datasets 本身可能在 acoustic feature distribution、class imbalance 或采集条件上存在明显差异，而这些差异可能影响 cross-dataset learning 和模型 performance。

老师认为，哪怕某些现象是你们实验中发现的，也可以在 Introduction 里简洁地作为 motivating evidence 提到，然后明确告诉 reviewer “details are discussed in Section X”。这样能把 Introduction 和后续实验结果建立 cross-reference，而不是每个 section 完全割裂。

第三部分是 contribution。老师不建议在 ICASSP 这种 short paper 里用一串 bullet points 罗列贡献，而是建议用一段完整但高度结构化的文字表达。开头直接写 “To address these gaps…”、“To address the aforementioned limitations…” 或者 “Motivated by these challenges…” 等，然后明确说你们提出了什么样的 architecture/framework、如何利用不同形式的 supervision、解决什么问题，以及最终在什么范围内验证。老师强调，贡献段落不是为了写得漂亮，而是为了让 reviewer 一眼看明白“问题是什么—你解决了什么—怎么解决的—证据是什么”。

老师反复强调，论文写作某种程度上就是需要“模板化”。目的不是文学表达，而是尽量降低 reviewer 的理解成本，避免 rejection。Reviewer 不会前后反复读你的文章，帮你拼接前面提过的概念，也不会花大量时间猜作者真正想表达的东西。因此该 cross-reference 的地方一定要 cross-reference，该把 gap/limitation/contribution 关键词明确写出来的地方就必须写出来。你不能假设 reviewer 拥有你对这个项目的全部 prior knowledge。

老师进一步把这个问题延伸到了你平时做 research update 和未来 PhD communication 的方式。他认为，你现在无论写 paper、做 weekly update 还是开会汇报，都容易一开始直接进入 detail，缺少 high-level picture。对于一个普通学生来说，这可能还可以接受，但如果以后做 PhD、和 advisor 开会或者面试，就需要养成非常强的结构化沟通能力。在有限的二十分钟或三十分钟里，需要先告诉对方“我今天想证明什么、overall picture 是什么”，再进入细节。Advisor 不可能对每个学生的所有项目都拥有完整 prior knowledge，也不可能花大量时间自己整理你的思路。因此，从大图到细节的组织方式是 research communication 非常重要的一项能力。

随后会议转到 Section 2 和 Dataset Analysis。你解释，目前 Figure 1 主要想展示几个 respiratory datasets 的数据量、label distribution 以及部分 acoustic feature difference，以此说明不同 datasets 的异质性，并为后面的统一学习框架提供背景。

老师首先问了一个非常直接的问题：Figure 1 中展示的这些 feature difference，在后面的 evaluation 里有没有被重新使用来解释实验结果？你承认目前基本没有，只有 annotation mismatch 在后面有所讨论，而 feature distribution 并没有真正和结果联系起来。老师认为这是一个严重问题。如果一张图占据了短论文里非常大的版面，那么它必须承担明确功能，不能只是“展示一下数据有差异”。每一张图、每一段话都必须被后文真正用到。

老师指出，annotation mismatch 和 feature distribution 是两个不同的问题。annotation mismatch 可以支持你们为什么需要 label curation、heterogeneous supervision 或 task harmonization，但 feature distribution 如果只是展示 PCA、统计分布或者 outlier，却没有解释模型 performance，那么它的存在意义很弱。你需要明确回答：这个 feature difference 能告诉 reviewer 什么？是否能够解释某些 dataset pair 之间迁移性能差异？是否能和 Table 2、ablation 或 external evaluation 中的趋势对应？如果不能，那么就应该删除、压缩或者重画。

老师建议，Dataset Section 实际上不需要现在这么大，可能 half column 就足够。如果真正需要强调的是各数据集 annotation scheme 不一样，可以使用一个非常紧凑的小表格或 illustration，明确画出不同 datasets 的 label taxonomy、prediction unit、task definition 等。如果 dataset distribution 能够解释后面的结果，那么 distribution figure 才值得保留较大版面。

同时，老师认为目前 Figure 1 的信息密度太低。现在不同 histogram、distribution、feature plot 被拆成很多独立小图，一张图只表达一个信息，非常浪费版面。完全可以通过 overlay、共享坐标轴、上下组合、左右组合、shading、不同 marker 或 annotations，把多个信息压进同一张 figure。比如两个 x/y label 相近的 distribution 没有必要左右分别占一大块，可以合并；class count 和 class imbalance 可以放在同一图；不同 dataset 的 annotation information 也可以和数据分布视觉上融合。

为了说明这个问题，老师让你去看他以前的论文和其他论文里的 dataset figure。他强调，一张优秀的 dataset figure 往往同时表达 dataset size、distribution、window count、error、annotation 或其他 metadata，而不是一张图只展示一个简单 histogram。尤其 short paper 能放的图数量非常有限，所以每张图都应该非常 intense、information-dense。视觉上让 reviewer 一眼看到“这里确实有很多工作”，同时所有信息又围绕同一个 argument 组织起来。

你也意识到现在 Figure 1 的最大问题不是简单“画得不好看”，而是信息量和论文后文的联系都不足。你认为 Section 2 的内容本身都需要重新调整，而不仅是 layout。Dataset analysis 应该和后面的 evaluation result 形成对应关系，否则这一节放太多 dataset statistics 没有实际意义。

老师进一步给出了一个比较具体的思路：如果你们后面发现 HF_Lung、ICBHI、SPRSound 或其他 dataset 的 performance gap 与它们在 acoustic distribution、class imbalance 或 annotation mismatch 上的差异存在一致趋势，那么 Figure 1 就应该直接展示这些相关 difference。之后在 Results Section 再 callback：“如 Figure 1 所示，Dataset A 与 Dataset B 的分布/annotation 差异更大，与我们观察到的跨数据集性能下降趋势一致。”这里应当谨慎表达 association，而不是写成 causal claim。老师强调，他之前要求你做这些 dataset analysis，并不是因为这些图一定必须放进论文，而是希望看看其中是否有任何 pattern 可以解释现有结果；只有真正能帮助解释 results 的内容才值得留下。

接着老师发现 Table 2 主要只有 ICBHI 和 SPRSound，而 HF_Lung 和 KAUH 似乎没有作为同等 training dataset 出现在结果中，于是追问为什么。你解释，在之前实验中发现加入 HF_Lung supervision 后，主模型性能损失比较明显，因此目前主方法实际上只使用 ICBHI 和 SPRSound 作为主要 supervised training datasets，而 HF_Lung 和 KAUH 更多作为 external evaluation datasets。相关实验其实已经存在，只是还没有在论文中清楚写出来。

老师对此非常不满意，认为这就是当前文章无法阅读的重要例子。Figure 1 和 Dataset Section 把四个 dataset 以完全平等的形式并列展示，让 reviewer 自然会认为四个 dataset 全部参与了同一种 training protocol；但后面的结果实际上只有两个 dataset 参与主要训练，另两个主要用于 external evaluation。这样会给 reader 造成错误预期，甚至让人误以为你的实验还没做完。

因此，老师要求必须非常明确地区分 training datasets 和 external evaluation datasets。Figure 1 可以直接通过左右区域、颜色、annotation、grouping 等形式写清楚哪些数据参与 training、哪些只作为 external evaluation。Table 1、Dataset Section、Introduction 和 Method 中的描述也必须保持一致。不能前面泛泛地说 “we use four datasets” 或 “we integrate multiple datasets”，结果后面其实只有其中两个参与联合训练。

你提到 Introduction 最后一两句其实有说明这一点，但老师认为这远远不够，因为前面的整体视觉和 wording 已经让 reader 建立了错误印象。Paper 里不能只是某个地方“提过一句”就算完成说明，重要设计选择必须在 reviewer 最需要知道的位置清楚呈现。

同时，HF_Lung 没被加入主 training pipeline 的原因也不能写成“因为加进去结果不好所以没有用”，而是需要给出真正合理、可解释的 methodological rationale，比如 annotation mismatch 更严重、label granularity 不匹配、sample distribution 差异、数据规模/质量因素等。如果目前只能在 “current stage” 将 HF_Lung 和 KAUH 作为 external evaluation，也应该如实界定 scope，而不能让论文暗示已经完全解决四个 dataset 的 unified supervision。

这也引出了老师反复强调的另一个重点：避免 overclaim。当前论文有些句子把非常局部的实验结果写成了对整个 respiratory acoustic classification 领域的广泛 conclusion。老师认为，claim 必须严格限制在你真正实验覆盖的 setting 中。比如不能笼统写 “our method improves respiratory sound classification” 或 “supports classification inference across datasets”，如果你只在特定 dataset combination、特定 task setup 或 isolated classification setting 中验证了，就必须把这些限定词写清楚。

如果是尚未验证的未来可能性，可以写 “has the potential to…”、“may enable…” 等，但不能把 future potential 当成已经证明的 contribution。老师说，现在审稿人尤其容易对 overclaim 敏感，所以 scope 必须非常干净。

老师随后再次回到 Table R。他认为除了把 baseline 标清楚，Table R 本身还可以按照实验目的进一步分组。例如上半部分可以是 “our method vs. existing/native baselines”，下半部分可以是 “different label/data curation strategies” 或其他 ablation。中间可以加一条横线或者一个 merged column header，明确告诉 reviewer 两部分各自在比较什么。并且表中的 terminology 必须和前文保持完全一致。如果前面 Method 里叫 “label curation”，table 里就也叫 label curation；如果前面叫 heterogeneous supervision，这里就不要突然换另一套词。全文应该有很强的 terminology coherence。

老师指出，现在有些 baseline 是你们自己重新跑的，有些可能来自原论文，但在表里没有标明，reviewer 根本无法知道哪一个数字对应 published baseline、哪一个是你们 reproduction、哪一个是 proposed architecture。所有这些都必须通过 caption、footnote 或 notation 标得非常清楚。

之后讨论 Method Section。老师认为，Method 的文字必须和 Figure 2 几乎一一对应。你正在重新画 Figure 2，老师要求所有正文中提到的重要 variable、loss、input、output、latent representation、branch、supervision source，都必须在 Figure 2 中出现。反过来，Figure 2 中画出的模块在正文里也应该有对应解释。

老师说，一个好的 system/method figure 应该做到：即使 reviewer 不读正文，只看图，也大致知道 input 从哪里来、经过哪些模块、latent space 是什么、不同 supervision 在哪里作用、loss 怎么计算、最终 output 是什么。正文再进一步解释为什么这样设计。现在你的图和文字基本没有形成这种 correspondence，所以 Figure 2 目前看起来像“装饰图”，没有真正承担解释 method 的作用。

老师展示了自己的 proposal 里的一些 architecture figure 作为参考。他强调，公式里的 variable、输入输出关系、latent representation、equation 的位置等完全可以在 architecture figure 中直接标出来。这样 reviewer 看正文的时候可以不断对照图，理解成本会大幅下降。图和文字必须“必须对得起来”，否则这张图就是无效的。

你总结说，Method Section 后续应该围绕 Figure 2 完全重组，先确定整个 model flow，再按照 figure 的顺序写正文，而不是文字一套逻辑、图另一套逻辑。老师认可这一点。

会议中老师多次强调，当前 paper 最大的 high-level 问题可以概括为“信息不是不够，而是组织和密度都不够”。现在已经有很多实验、dataset、analysis 和结果，但它们没有朝论文的中心思想收束。文章看起来“乍一看有很多东西”，但经不起细读，因为 reviewer 无法快速回答：“What can this table tell me?”、“What can this figure tell me?”、“Why is this paragraph here?”。

老师提出一个非常直接的检查标准：对于每一个 table，直接问自己 “What can this table tell you?”。如果仅仅看 table 无法得出一个明确 conclusion，那么这个 table 就没有设计好。Figure 也是一样。Paper 里的每一个 element 必须有 takeaway，而且 takeaway 应该尽可能在视觉设计和正文第一句话里明确告诉 reader。

老师认为目前 paper 最大的问题不是信息量不足，而是 high-level idea 没有清楚传达，同时 figure/table 的 visual density 也不够。你也总结，目前一方面是整个 paper 的 central idea 没有被反复强调，另一方面图表里承载的信息过少，导致短论文宝贵的版面没有得到充分利用。

老师再次强调，不要把这个问题归咎于 Codex、Claude 或任何 AI 工具。AI 可以帮助生成初稿、整理 bullet points、润色句子，但最终如何组织研究故事、哪些内容保留、哪些删掉、每个实验如何服务于 central claim，全部是作者自己的责任。你可以使用 AI 帮助 writing，但 high-level research reasoning 必须自己理清楚。

老师也解释了为什么前一天没有立刻逐句给你很多 feedback，而是要求你先自己总结 research question、gap 和 contributions。他想先确认你自己是否真正理解这篇论文到底在讲什么。如果作者自己都没有先把中心思想提炼出来，那么逐句修改 wording 没有意义，因为整个 structure 都可能需要重写。

会议最后再次讨论时间安排。距离当前 deadline 只剩大约 48 小时，老师表示自己同时还有其他 proposal 和 paper，非常忙，不可能把全部时间都用于这一篇文章。老师希望你不要把 conference deadline 等同于 advisor 的 deadline，因为老师不可能在最后一天突然腾出完整时间帮一个学生从头修改全文。学生需要提前给出足够 ready 的版本，给 advisor 留出 feedback 时间。

老师提到，这也是很多学生常见的问题：只把投稿 deadline 当成自己的 deadline，却没有考虑 advisor 的时间。如果所有学生都在最后两三天才把 paper 发出来，老师不可能全部处理。因此当前阶段只能尽可能修改，不能再期待大规模来回迭代。

你表示今天和明天会把 paper writing 作为最高优先级，其他 revision 暂时往后放，集中完成 ICASSP。老师希望在第二天晚上的 meeting 能够看到一版真正“能用”的稿子，而不是当前这种 “semi-ready 但实际上还非常 semi-ready” 的状态。

综合整场讨论，接下来最优先的修改并不是继续增加新实验，而是完成以下几个核心方向。首先，Introduction 必须彻底重写成 motivation → current gap/limitation → contribution 的结构，并且明确使用 gap、limitation、to address these gaps 等关键词。第二，Dataset Section 和 Figure 1 必须重新设计，明确区分 training datasets 与 external evaluation datasets，同时只保留能够服务于后面结果解释的信息。第三，Figure 1 的信息密度要提高，避免一张子图只表达一个简单统计，把 annotation、distribution、class imbalance、dataset role 等信息尽可能整合。第四，Table R 必须通过 grouping、baseline 标记、加粗、improvement、footnote 等方式让 reviewer 一眼看出 native-task、baseline 和 proposed method 的比较关系。第五，Method Section 要围绕 Figure 2 重写，确保 input、output、variable、loss、latent space、supervision 和文字全部一一对应。第六，全文所有实验结论都需要检查 scope，避免从局部结果推广到整个领域，未来可能性和已验证结果必须严格区分。第七，全文需要显著加强 cross-reference 和 callback，让前面的 dataset observation、method motivation、后面的 result explanation 形成完整闭环。

老师最后对整篇文章的评价可以概括为：现在不是“没有东西”，而是有很多东西但没有被组织成一篇真正的 paper。当前版本更像一份很完整的实验记录或 lab report，读者只有在已经了解所有项目背景和 prior knowledge 的情况下才能理解你真正想贡献什么。接下来 48 小时最重要的目标，是把所有内容重新围绕一个明确的 central idea 组织起来，让一个完全不了解这个项目的 reviewer 能够快速、低成本地理解问题、gap、方法、实验和结论。

这一版我保留了老师关于 paper writing、figure/table、dataset role、method figure、overclaim 和沟通方式的完整讨论，没有额外加入我自己的研究建议。

## 会后补充：老师认可的大方向（9/13用户转述）

来源：用户在会后补充老师认可的limitation与contribution方向。以下保留补充内容；正式论文中比较基线、措辞强度和机制归因的细化记录在关联work plan中。

老师我整理了一下，现有work的limitation主要是

1. SPRSound（Zhang et al., TBCAS 2022）的数据融合实验与LungMix（Ge et al., ICASSP 2025）通过类别合并或重映射构建统一分类任务；Multi-breath（Chua & Cheng, 2024）与PC-MCL（Jeong & Kim, ICASSP 2026）以已有 ICBHI 类别为基础组织多标签监督，但是没有用一个模型监督不同数据集不同标注体系和任务目标
2. 单数据集的SOTA work直接迁移到其他任务上的时候泛化能力很差

所以我们的contribution可以是

1. 提出了一种连接异构呼吸音标注与相异原生任务的联合学习框架。在同一联合模型中连接周期四分类与事件二分类以及迁移到另外两个dataset的测试
2. 分层的多classifier设计以极小的性能损失换取了跨数据集的泛化能力
