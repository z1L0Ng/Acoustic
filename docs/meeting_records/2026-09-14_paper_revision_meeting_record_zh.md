# 2026-09-14 Paper Revision Meeting：完整中文会议记录（姓名校正）

Notion：[会议记录](https://app.notion.com/p/3db309efda29813d8044fbe789225846)。

日期：2026-09-14。来源：用户提供的完整中文会议整理稿。下文保留原有58项讨论和10条总结，只修正转录人名及标题排版；不把会后计划加入老师发言。

人名校正：Honey → Hanlin（4处）；Wit → Wade（5处）。英文变量 W、Wheeze 等保留原意。

时间说明：原记录中的“今晚12点前”“凌晨1点左右”“凌晨3点”“deadline前约6小时”未注明时区，以下按原话保留。本轮工作计划另按用户指定的美东时间安排至2026-09-15 01:00 EDT；此时间不等同于投稿系统的官方截止时间。

会后工作计划：[9/14夜间工作计划](https://app.notion.com/p/3db309efda29817f856ff55a4c82e8e3)。

上一场会议：[9/13会议记录](https://app.notion.com/p/3da309efda29818fa0f1d0edb27a7ea6)。

会议主要围绕论文当前版本的结构、Figure 1/Figure 2、baseline 定义、Table 1 和 Table 2、Method 写法，以及整篇论文的逻辑连贯性展开。老师的核心判断是：目前实验基本够了，真正严重的问题已经转变为论文写作和组织。当前稿件虽然比之前短很多，但仍然非常分散，更像 lab report，而不是一篇已经接近投稿状态的论文。

---

## 一、Figure 2 和论文/方法名称

会议开始时，老师先确认 Figure 2 的内容现在基本已经定下来了。

老师表示，他会尝试直接在自己的 slide deck 里面帮你重新画 Figure 2。

你随后提到，目前论文标题以及方法/model 的名字还没有最终决定，而且你感觉当前使用的名字并不好。

老师回应说，他之前已经让你给他发几个候选名字，但目前还没有收到。

你表示没有问题，会再花一个晚上仔细想一下，整理几个候选方案，之后让老师来决定。

后面会议结束前老师又再次提到，他已经在聊天里发了一些名字，可以从里面选一个，同时也希望你自己再尝试组合几个名字，然后告诉他你认为哪一个更好。

---

## 二、你对目前整篇论文进展的汇报

你向老师说明，除了这些问题之外，论文剩下的大部分内容其实已经基本完成。

你已经把 slide 的副本发给老师，因此老师可以直接在上面进行编辑。

然后你开始按照论文各部分向老师介绍你目前所做的修改。

---

## 三、Introduction 的结构

你说你已经把 Introduction 压缩成了大约三个 paragraph。

现在的组织方式是：

第一段：

介绍这个领域目前大家主要在做什么，也就是 respiratory sound / acoustic disease detection 这一方向的背景和已有研究。

第二段：

讨论目前已有工作的 limitations，也就是现有方法还存在什么不足。

第三段：

讨论：

* 你们准备如何解决前面提到的问题；
* 你们如何填补这个 research gap；
* 你们的方法是什么；
* 以及这个项目的 contributions 是什么。

也就是说，你目前希望 Introduction 的逻辑大致是：

背景 → 现有局限 → Gap → 我们如何解决 → Contributions。

---

## 四、Section 2：Dataset

接着你介绍了 Section 2。

你表示目前 Section 2 已经缩减为两个 subsection。

第一部分：四个 datasets

第一部分主要是之前已经讨论过的内容：

你们一共有四个 datasets，需要说明：

* 四个数据集分别是什么；
* 它们之间有什么区别；
* 在论文中分别如何使用这些数据集。

---

第二部分：Labels、Annotations 和其他信息

第二部分则是讨论：

这些 datasets 里面到底包含哪些：

* labels；
* annotations；
* metadata；
* task information；
* prediction-related information；
* 后续分析会用到的信息。

你认为，这些信息不应该只是通过 Section 2 的正文告诉 reviewer 和 reader。

你希望进一步把这些内容放到 Figure 1 里面。

也就是说，现在你正在考虑如何让 Figure 1 展示更多信息。

Figure 1 不应该只是简单描述 datasets 的一些 characteristics，而应该把后面论文真正会讨论的关键信息提前组织出来。

---

## 五、Section 3：Method

然后你开始介绍 Method。

你说这部分也进行了明显压缩。

你的第一部分主要讨论：

shared representations。

然后你把相关 loss functions 尽可能放到一起，希望能够统一解释它们之间的关系。

接下来再讨论：

如何进行 joint learning。

也就是：

* 不同 representation 如何联合；
* 各种 loss 如何联合；
* 最终模型怎样利用你们设计的结构进行训练；
* 如何从这个 design 产生最终 prediction；
* 以及 additional HF-related loss 如何参与最终 objective。

---

## 六、老师开始检查 Method Figure

老师看了你当前的方法图以后，开始确认图里的 loss relationship。

他说，大致来看：

这一部分是 L-core，对吧？

另一部分应该是类似：

λ × L-HF。

也就是说，老师想确认：

* 哪部分是 core objective；
* 哪部分是 HF loss；
* 它们最后是怎么组合成总 loss 的。

老师随后说，他也可以帮你重新组织这张图。

你回应说可以。

---

## 七、老师指出 Method Figure 现在仍然非常不完整

随后老师强调：

当前这张 figure 实际上还是一张非常 unfinished 的 figure。

他现在的问题之一是：

论文里给 figure 的空间有限。

他说他最终需要把这整张 figure 压缩到一个比较小的尺寸。

现在这张图的信息组织方式，在缩小以后很可能根本无法阅读。

所以他会尽量帮你调整，但是目前图的结构本身还需要重新组织。

---

## 八、先不讨论图，老师要求直接进入 Results

老师接下来表示：

先看看 Results。

他说：

先把所有 results 和 evaluations 的含义解释清楚。

也就是说，老师此时不希望继续纠结 Figure 的细节，而是想知道实验到底在讲什么。

---

## 九、老师让你先解释 Table 1 的“故事”

你说：

Table 1 目前还没有完全重新处理，但是你准备继续修改。

老师直接问你：

你到底希望通过这张 table 告诉 reviewer 什么？

如果还没完全做完也没关系，那你就告诉我：

你最终计划通过这张表讲一个什么 story？

---

## 十、老师打断你：不要先讲实验细节，要讲 High-level Takeaway

你开始解释说：

Table 1 里面有之前跑过的各种 baseline。

但老师很快打断你。

他说：

我现在不想先听 details。

他要你直接从 high level 讲：

这张表的核心 takeaway 是什么？

你随后解释：

这些 baseline 基本上是在不同 datasets 上分别训练的，比如在 SPRSound 上训练，或者在其他 dataset 上训练，因此它们的 score 是分别得到的。

这也是 Table 1 想展示的一个核心内容。

---

## 十一、老师开始追问“Baseline 到底是什么”

老师马上问：

你这里说的 baseline 到底是什么意思？

因为不同的人对 baseline 的理解是不一样的。

老师举例问：

表格里类似：

AST + ICBHI

这样的组合是什么意思？

是不是代表：

你使用 ICBHI dataset，然后用 AST 这个方法？

还是其他含义？

---

## 十二、你解释 Baseline 的实现方式

你解释说：

基本上是使用这些 encoders，然后对 classifier 做训练。

同时 encoder 部分会进行相应设置，例如冻结或者有限 fine-tuning，然后针对不同分类任务训练 classifier。

---

## 十三、老师确认：这些 Baseline 其实是你们自己构造的

老师听完后确认：

那么也就是说，这些 baseline 其实也是你们自己生成的实验配置。

它们不是：

某篇 existing paper 的完整方法，被你们直接复现出来。

而是：

使用已有 pretrained encoder，然后你们自己构建 downstream classifier 来完成任务。

你确认是这样。

---

## 十四、Baseline 不应该错误使用 Citation

老师随后明确指出：

既然这样，那么这些地方就不需要用 citation 来表示“这是某一个 existing method”。

因为如果你把 citation 直接放在 baseline 名称旁边，reviewer 很容易理解成：

你们在复现某一个已有方法。

但实际上不是。

所以这个 citation 会造成误导。

---

## 十五、必须在论文中明确定义 Baseline

老师继续说：

如果你要叫它 baseline，就必须明确写清楚。

尤其需要配一个简单的 figure 或者清楚说明：

* 这里使用哪个 pretrained encoder；
* 哪些 model components 是 frozen；
* 哪些 parts 被训练；
* downstream classifier 怎么做；
* baseline 和你们 proposed method 的区别是什么。

老师举的意思大概是：

直接使用 encoder，然后做 multi-class classification。

如果这是你们定义的 baseline，那就必须明确告诉读者：

This is our baseline.

否则 reviewer 会非常困惑。

---

## 十六、老师强调：不能使用“Baseline”这个词，却不定义 Baseline

老师说：

你不能假设所有人看到 baseline 都和你理解一样。

在很多论文里，baseline 通常意味着：

reproduction of existing methods。

而你这里不是。

所以 baseline 的 definition 一定要明确。

否则会导致 reviewer 对整张表产生错误理解。

---

## 十七、你提到还会有一张关于 Encoder 的 Figure

你回应说：

明白。

你还有另外一张 figure，准备进一步解释 encoders 的设置。

你也想把某些特别重要的结果 highlight 出来，让 reviewer 更容易注意。

---

## 十八、老师要求 Highlight 必须遵循明确格式

老师说：

如果你要 highlight，就要做到清楚且规范。

例如：

* bold；
* underline。

不能只是随意强调。

同时老师也指出：

Table 1 caption 的 grammar 目前需要检查。

现在的 caption 还没有达到 publication-ready 的状态。

---

## 十九、Table 1 和 Table 2 的角色

你随后解释：

Table 1 主要对应 Section 4.2。

它的重点是展示：

你们主要 joint method / proposed method 的 performance。

然后 Table 2 是：

ablation study，以及其他 supervision 和 readout 的实验。

---

## 二十、你对 Table 2 的当前设计解释

你继续解释 Table 2。

其中有一些 ablation setting。

例如 cross-related 的实验中：

你们会 mask 掉某些 design。

然后只使用比较简单的两层结构，或者直接使用现有信息，在 SPRSound 上进行训练。

你希望通过这种设置观察：

当去掉部分 proposed design 以后，performance 是否还能保持在一个合理水平。

---

## 二十一、Direct Setting 和 Native Task

你还提到一个 direct setting。

它大致表示：

直接使用模型去预测 crackle、wheeze 这一类目标。

此外还有 native task。

你解释说：

例如 W 等标记代表使用 ICBHI 或 SPRSound 自己原生的 native task。

然后，Table 2 中有些数字是：

当前 ablation setting 的结果，与 main method 结果之间的 difference。

你希望通过这种方式展示：

你们的方法相对于这些 reduced/alternative settings 有多少 improvement。

---

## 二十二、你自己承认 Table 2 目前还是比较 Confusing

你随后主动承认：

Table 2 目前仍然有一些 confusing。

你也在重新考虑：

* Table 2 应该如何组织；
* 每一行到底应该表达什么；
* 正文应该怎么讨论；
* 如何让 reviewer 更容易理解这个 ablation story。

---

## 二十三、老师问这些 Section 到底是正式正文还是 Notes

老师问：

你现在这些内容已经正式写完了吗？

还是现在仍然只是 notes？

你回答：

目前还没有全部写完。

你已经完成了一部分实验相关文字。

但是 4.2 和 4.3 仍然在调整，因为 Table 1 和 Table 2 本身还需要修改。

---

## 二十四、老师确认实验是否已经做完

老师随后问：

但是论文真正需要的 experiments，现在是不是都已经做完了？

你回答：

是的。

而且实际上你们现在已经有的实验比论文最终需要展示的还要多。

---

## 二十五、老师判断：接下来主要问题已经是 Writing，而不是 Experiment

老师表示：

既然这样，那么剩下的工作应该主要就是 writing。

也就是说，老师现在认为：

实验已经基本不是最紧急的 bottleneck。

真正的问题是：

如何把已经有的结果组织成一篇论文。

---

## 二十六、老师再次要求减少 Subsections

老师再次提醒：

他之前已经告诉你：

不要使用这么多 subsection。

应该更多使用 paragraphs 来组织。

你回应说：

现在实际上每一部分已经基本只剩两三个 subsection。

你不确定还能不能继续减少。

老师明确说：

仍然可以继续减少。

甚至很多地方根本不需要 subtitle。

---

## 二十七、老师认为目前的结构仍然像 Lab Report

老师说：

现在这样的组织还是非常粗糙。

看起来仍然像：

一个 lab report。

而不是一篇完整写出来的 paper。

即使是那些已经填充了正式文字的地方，整体感觉仍然没有脱离 lab report 的风格。

---

## 二十八、你提到 Equation 3 和 Equation 4 也可以进一步合并

你回应说：

例如 Equation 3 和 Equation 4，也可以尝试合并到一起。

这也是你进一步压缩结构的一部分。

---

## 二十九、老师再次提到论文名字

后面老师问了一下投稿相关信息，不过很快表示这不是重点。

然后他说：

他已经在聊天里发了几个名字。

可以从里面选一个。

同时他再次表示：

目前稿件完成度比他预期的要晚很多。

因为他自己当天一直在旅行，甚至没有办法正常打开和编辑。

他已经把文件下载下来了，但在路上也没有办法仔细处理。

所以到现在为止，他实际上还没有进行正式修改或者系统 comments。

---

## 三十、老师再次强调：不要用 Bullet-point 式写法

老师特别指出：

正文要写成 paragraph。

不能像现在这样：

一句一句的 bullet-point language。

他的核心意思是：

即使形式上没有 bullet，也不能把句子写成一个个相互独立的信息点。

需要真正形成段落。

---

## 三十一、老师认为当前 Writing 最大的问题是“Scattered”

老师直接说：

你看看你现在的 writing 有多 scattered。

现在还是非常像 bullet-point language。

不同 paragraph 之间缺少 transition。

不同 section 之间也缺少 transition。

---

## 三十二、你提出准备补 Section 之间的 Transition

你回应说：

你准备开始补：

* subsection 1 到 section 2；
* section 2 到 section 3；

这样的 transition。

你觉得这样可能能改善文章的连贯性。

---

## 三十三、老师解释：Transition 不等于每个地方硬加一句过渡句

老师随后纠正：

Transition 并不是说：

每两个 section 或 paragraph 之间都机械地写一句 “transition sentence”。

真正需要连续的是：

* terms；
* philosophy；
* logic；
* flow。

也就是说：

文章使用的概念体系应该一致。

前后 argument 要连续。

不能突然跳到另一个完全没有铺垫的内容。

老师强调：

整个 flow 必须保持 continuous，而不是被破坏。

---

## 三十四、老师用 Figure 1 的句子举例

老师找到一句类似：

Figure 1 summarizes these rules and the annotation units.

然后直接问：

什么 rules？

什么 annotation units？

问题在于：

这句话虽然表面上是完整句子，但 reader 根本不知道它在指什么。

它没有和前面的 context 建立清晰关系。

---

## 三十五、老师认为很多句子本来应该是 Takeaway，但现在只是孤立信息

老师继续解释：

有些句子实际上应该承担一个 conclusion 或 takeaway 的作用。

但你现在只是把它作为一个 isolated information 放在那里。

所以 reviewer 看完以后不知道：

这句话到底想证明什么？

为什么要告诉我这个？

老师直接问：

What can you tell over there?

也就是：

你从这些 information 里面真正能得出什么？

---

## 三十六、Acoustic Characteristics 的逻辑顺序有问题

老师继续看 acoustic characteristics 部分。

他发现你的 Section 3.1 和 2.2 的关系有问题。

老师问：

为什么先讨论某个东西，然后再讨论 acoustic characteristics？

如果是：

先做 A，然后再分析 acoustic characteristics，

那么这个顺序是否真的 coherent？

老师直接判断：

不是。

---

## 三十七、你回应会重新检查 Detail Language

你表示理解。

你会重新看这部分的具体 language 和顺序。

---

## 三十八、老师提出两种可能：Method 先讲，或者 Dataset 和 Method Merge

老师说：

如果这些内容属于方法，那么可以先讲 methods，再讨论 datasets 如何进入这个方法。

或者：

也可以把 dataset preparation 和 method 的部分内容 merge 到一起。

重点不在于固定模板，而是：

逻辑上要成立。

---

## 三十九、老师再次提醒：之前已经专门花了大约一小时讲 Writing

老师随后说：

他之前已经花了大概一个小时和你详细讨论：

* 论文应该怎么写；
* 每个部分的 contribution 是什么；
* 每个 paragraph 到底告诉读者什么；
* paragraph 的 flow 应该是什么。

但他现在看到的版本是：

虽然你确实把内容缩短了，

但仍然还是非常 scattered。

---

## 四十、老师再次评价：现在看起来像“巨大的 Lab Report”

老师说：

现在整个稿件看起来仍然像一个：

huge lab report。

问题不是缺 information。

而是：

你需要通过 writing 告诉 reviewer：

这些 information 的 meaning 是什么。

这是更重要的事情。

---

## 四十一、老师要求 Logic 必须 Continuous and Smooth

老师强调：

逻辑应该：

continuous and smooth。

例如：

你先讨论 datasets and annotations。

那么接下来的 paragraph 和这个 paragraph 是什么关系？

如果两个 paragraph 是并列讨论 datasets 的不同维度，那么应该形成 parallel structure。

不能让 reader 不知道为什么突然从一个东西跳到另一个东西。

---

## 四十二、老师继续追问 Acoustic Characteristics 为什么放在这里

老师问：

当你开始讲 acoustic characteristics 时，

为什么它出现在这里？

它和上一段之间是什么关系？

你必须能解释。

---

## 四十三、Shared Acoustic Representation / Window Length 可能应该属于 Data Preparation 或 Method

老师进一步指出：

像：

* shared acoustic representations；
* window lengths；

这些内容如果是你们对 datasets 做的处理，那么它们实际上应该是：

data preparation 或 method 的一部分。

否则，如果把它们孤立放在 Dataset Section 中，看起来就很像只是 related work 或 dataset description。

---

## 四十四、Dataset Section 不能只是把所有信息全部堆进去

老师说：

如果你们是在描述：

你们已经处理好的 datasets 的 characteristics，

那它应该和你们的方法、data preparation 联系起来。

而不能只是：

所有信息都展示一遍。

因为现在这种写法仍然非常不可读。

---

## 四十五、老师直接评价当前 Setup：“Such a Mess”

老师说：

现在整个 setup：

is such a mess.

而且：

no one can follow it.

也就是说，现在最大的风险不是 reviewer 不认同方法。

而是：

reviewer 根本无法顺着论文理解你们做了什么。

---

## 四十六、老师需要登机，只能暂停详细 Review

这时老师表示：

他需要准备登机了。

所以没办法继续逐句帮你检查。

他说：

Figure 1 他会帮你处理。

但是他希望：

今天晚上 12 点之前，

你能够整理出一个更加 coherent 的组织。

重点是：

必须 readable。

---

## 四十七、老师询问 Figure 1 是否在 PowerPoint

老师问：

Figure 1 是不是也已经在 PowerPoint 里？

你回答：

目前不在 PowerPoint。

你只是为了展示临时打开了它。

但你会尽快把它放进去。

---

## 四十八、老师说明自己接下来的 Bandwidth 非常有限

老师明确告诉你：

一直到第二天凌晨 3 点之前，

他的 bandwidth 都非常有限。

原因是：

他当天一直在 travel。

而且他明天还有另一个 proposal deadline。

所以他没有办法在这篇论文上持续投入很多时间。

---

## 四十九、如果你不能尽快整理好，老师无法保证还能给 Revision

老师非常明确地说：

如果你不能尽快把这些内容整理完成，

那么他很难保证：

还能给你任何 revision 或者 feedback。

因为他真正能够开始腾出时间的时候，

可能已经是 deadline 前大约六个小时。

---

## 五十、老师给出非常严重的判断：当前 Writing 是“100% Reject”

老师随后直接评价：

基于你现在准备出来的这个版本，

如果按照这种 writing 状态提交：

100% reject。

他的意思不是实验一定不够，

而是：

当前 presentation 和 writing quality 根本没有达到可以投稿的程度。

所以必须 significant improvement。

---

## 五十一、老师再次要求：Readable and Coherent

老师重新强调：

必须确保整个稿件：

* readable；
* coherent。

这两个词是他这次会议后半段反复强调的核心。

---

## 五十二、老师建议让 Hanlin 和 Wade 先读

老师建议：

你可以让 Hanlin 和 Wade 先把整篇稿子读一遍。

然后问他们：

* Does that make sense?
* 有没有 inconsistency？
* 哪些地方看不懂？
* 哪些 section 之间关系不清楚？

老师的想法是：

先进行一轮最基础的 readability test。

---

## 五十三、老师建议也可以用 Claude / GPT 做 Logical Consistency Check

老师还说：

甚至你可以把内容给 Claude 或 GPT。

问它：

* paragraph 之间的关系是什么？
* 这个逻辑是否正确？
* 前后是否 consistent？
* 这个结构 reviewer 能不能理解？

老师认为：

AI 很容易就能指出目前某些地方逻辑上并不成立。

所以这些 basic consistency check 应该提前完成。

---

## 五十四、老师要求不要拖到最后一分钟

老师强调：

这些检查应该现在就做。

不要等到最后一刻。

否则老师只能在凌晨 3 点开始给 feedback。

而那时已经距离 deadline 非常近。

---

## 五十五、老师说明自己真正有时间可能只剩 Deadline 前 6 小时

老师说：

大约 deadline 前六小时，

他才可能真正开始有比较完整的时间。

所以目前不能依赖老师来完成整个 rewrite。

---

## 五十六、老师原本以为这次应该是 Finalization Meeting

老师说：

他原本以为今天这次会议应该已经是：

finalization meeting。

也就是说：

原本预期现在应该是：

* 最后检查；
* 最后 polish；
* figure 调整；
* wording 调整；
* submission preparation。

但实际上目前论文还没有达到 finalization stage。

老师明确说：

Unfortunately, it’s not at that stage yet.

---

## 五十七、老师也要求其他 Co-author 必须真正参与 Read-through

老师说：

现在已经到了必须准备 submission 的阶段。

对于 Hanlin 和 Wade：

如果是 co-author，就应该真正读完整篇论文。

给：

* feedback；
* suggestions；
* readability check。

老师尤其对 Wade 表示：

如果你是 co-author，而且你读完整篇以后仍然无法理解其中的 details，

那就要问一个很现实的问题：

如果 co-author 都看不明白，怎么能期待 reviewers 看明白？

这也是老师希望你们进行内部阅读测试的原因。

---

## 五十八、老师准备登机，会议结束

最后老师表示：

他需要去赶飞机，已经开始 boarding。

他预计当天大约凌晨 1 点左右到达目的地。

你向老师表示感谢。

老师最后再次强调：

希望下一版至少是：

readable format。

然后再次提醒你：

从聊天中选一个名字，或者自己组合新的名字。

选好后告诉他你认为哪个更好。

会议随后结束。

---

## 本次会议老师的核心判断

把整场会议压缩下来，老师真正反复强调的并不是“再做更多实验”，而是下面几点。

### 1. 实验基本已经足够

你明确告诉老师：

论文所需要的 experiments 基本都已经做了，而且结果实际上比最终论文需要展示的更多。

因此当前主要 bottleneck 已经不是 experiment。

---

### 2. 当前最严重的问题是 Writing

老师认为当前稿件：

虽然缩短了很多，

但仍然：

* scattered；
* fragmented；
* bullet-point-like；
* 缺少 paragraph-level argument；
* 缺少 section-level continuity；
* terminology 和 philosophy 不连续。

---

### 3. 当前稿件看起来像 Lab Report，而不是 Paper

这是老师反复使用的评价。

你目前的写法更接近：

我们做了什么 → 又做了什么 → 得到什么数字。

而真正的 paper 应该让 reviewer 始终知道：

为什么现在要讲这件事？

这段的 takeaway 是什么？

它如何支持 research question？

它和下一段是什么关系？

---

### 4. Baseline 必须重新定义

尤其是：

AST + ICBHI 之类的 baseline。

必须明确：

* 这是你们自己构建的 experimental baseline；
* 不是完整 reproduction existing method；
* encoder 怎么使用；
* classifier 怎么训练；
* frozen / fine-tuned parts 是什么。

不能用 citation 让 reviewer 误解。

---

### 5. Table 1 和 Table 2 必须分别讲清楚 Story

Table 1：

应该明确体现 main results 和主要 takeaway。

Table 2：

应该明确体现 ablation / supervision / readout 的作用。

不能只是罗列数字。

---

### 6. Figure 必须服务于理解，而不是增加信息量

Figure 1 和 Figure 2 都需要重新组织。

尤其 Figure 1：

不能只是把 dataset 信息全部放进去。

它应该帮助 reader 理解：

* dataset heterogeneity；
* annotations；
* prediction units；
* preparation；
* 后续 method 为什么需要这样设计。

---

### 7. 今天晚上必须先做出一个“别人能读懂”的版本

老师希望你在当天晚上 12 点前至少整理出一个：

* coherent；
* readable；
* logically consistent

的版本。

因为他接下来 bandwidth 极低。

---

### 8. 当前版本如果直接提交，老师判断是“100% Reject”

这是老师整场会议里最严厉的一句话。

问题主要不是研究 idea 本身，而是：

当前 writing 和 organization 根本没有达到 publication-ready level。

所以老师要求：

significant improvement。

---

### 9. Internal Read-through 必须马上做

Hanlin、Wade 或其他 co-author 应该完整阅读。

如果内部 co-author 都不能顺着文章理解，

reviewer 更不可能理解。

也可以使用 GPT / Claude 检查 paragraph relationship 和 logical consistency。

---

### 10. 老师原本认为这应该已经是 Finalization Stage，但实际上还没有到

这也是老师明显担忧的地方。

目前距离 deadline 已经很近，

但稿件还处于：

重新组织 narrative 和 writing structure

的阶段。

因此接下来最重要的不是继续扩展内容，而是：

把已有内容真正写成一篇 reader 能够顺畅理解的 paper。
