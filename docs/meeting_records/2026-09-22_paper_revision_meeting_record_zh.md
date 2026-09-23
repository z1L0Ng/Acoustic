# 2026-09-22 论文修改讨论：标签体系、图表与论证结构

Notion：[完整会议记录](https://app.notion.com/p/3e4309efda298119b493c71b5617914e)；[今晚分章工作计划](https://app.notion.com/p/3e4309efda298130b73ac43649349b54)。

## 记录说明
会议日期：2026-09-22（周二）。参会人为用户与老师，附件未给出老师姓名及具体会议时刻。本页完整保存用户提供的29节中文纪要，仅调整标题和列表格式；不是重新听录音生成的逐字转录，末尾总结沿用附件原文。
本次讨论的重点是标签定义、章节组织、图表可读性、贡献与证据对应及最终呈现。具体的今晚todo另放工作计划，会议内容与管理拆解分别保存。

## 本次会议要点
- Dataset先介绍原始数据与标注，再定义本文使用的子集、映射及统一标签，最后讨论声学差异。
- Figure 1的原始类别写全称，用统一视觉marker标明本文实际使用的范围；marker不再创造另一套标签名称。
- Figure 2直接展示ICBHI native head＋共享C/W auxiliary heads，以及SPRSound native head＋相同的共享C/W auxiliary heads；区分训练监督、原生输出、推理与optional组件。
- Native-head版本仍可作为最终方法，但必须解释清楚harmonization发生在哪里，以实际整体表现支撑claim。
- Contributions要显眼，Evaluation标题、表格分组及结果论述逐项回应contribution；准确陈述性能，不声称所有数据集都第一。
- 图表优先，caption简明且有信息；完成术语、排版、官方盲审与引用格式核对后，再给老师完整版本。

## 关联材料
今晚工作计划：[2026-09-22 晚间 Working Plan](https://app.notion.com/p/3e4309efda298130b73ac43649349b54)；按章节拆分todo，不安排具体时刻。
上一轮会议记录：[2026-09-21 与Yinuo讨论](https://app.notion.com/p/3e2309efda2981f2b46fc0ea762b320e)。
上一轮计划：[2026-09-21 Working Plan](https://app.notion.com/p/3e2309efda2981d4927cf7476ef18691)。
原始附件本地留存：`docs/meeting_records/source_materials/2026-09-22_paper_revision/user_record.txt`。

## 完整会议纪要（29节）
这次讨论的核心不是继续补实验，而是集中解决论文目前在“表达、结构、图表、定义一致性和 reviewer 可读性”上的问题。老师反复强调：现在最大的问题并不是工作本身完全不成立，而是论文没有把已经做过的事情组织成 reviewer 能快速理解、顺着读下去的形式。做完实验和写成一篇 paper 是两回事，当前需要优先完成的是把工作重新包装成一条清晰、稳定、前后一致的论证链。

---

### 一、Label 定义：全文只能有一套一致的表达体系

一开始老师首先指出，论文里 N、C、W、CW 等 label 的定义和表示方式非常混乱。

如果这些符号已经在前面正式定义，那么后面全文就不应该反复重新定义；反过来，如果后面的 Figure、Method、Evaluation 都依赖这些符号，那么它们必须在 Dataset 部分一次定义清楚。

目前的问题是：

- 有的地方用简称；
- 有的地方用全称；
- 有的地方用数据集原始 label；
- 有的地方又突然出现 harmonized label；
- 图里和正文的表示方式不一致；
- 某些 label 之间本身还有 overlap，因此如果简称体系不清楚，会进一步增加 reviewer 的理解成本。

老师的要求是，全文必须保持 consistent。

#### 1. Dataset 原始标签和本文使用标签必须明确区分

针对 Figure 里展示各个数据集原始 label 的部分，老师认为现在最大的问题是，读者不知道：

“这些只是原始数据集的 background 信息，还是这些 label 真的是本文训练/评测会使用的 label？”

因此必须明确区分两件事：

第一层是：

- original dataset labels；
- 每个数据集本身原来有哪些 annotation。

第二层才是：

- 本文实际使用了其中哪些 label；
- 哪些原始类别可以映射到统一的 N/C/W/CW 等兼容类别；
- 哪些类别只是介绍数据集时出现，但本文并没有使用。

老师建议，不要让 reviewer 自己猜。

#### 2. 图中的数据集标签尽量全部写全称

老师认为现在 Figure 里大量缩写非常难读。

既然每个词本身并不长，那么图里完全可以把：

- Normal
- Crackles
- Wheezes
- Crackles and Wheezes
- 以及其他数据集原始 annotation

直接写成全称。

不要为了“省地方”继续堆叠一套数据集自己的简称。

因为现在最大的冲突是：

- 一套是原始数据集缩写；
- 一套是你们 harmonized 后的缩写；
- 两套缩写还可能有 overlap。

这样 reviewer 根本不知道某个 C、N、W 到底属于哪套体系。

所以老师的建议是：

图里原始 dataset annotation → 全部写全称。

而你们真正使用的统一类别，可以用一套非常明确、统一的 marker 标出来。

---

### 二、用“特殊标签/Marker”表示哪些类别是本文真正 care 的

你们讨论了一个比较具体的视觉表示方法。

老师建议，不需要在每个原始类别旁边再写另一套简称。

可以采用一种类似 code/font-style 的特殊标签，把“本文实际使用的类别”单独标记出来。

也就是说：

如果某个原始 label 能映射到本文统一类别，那么给它加一个特殊 marker。

如果本文根本不用，则不要标。

这样 reviewer 一眼能区分：

- 这是数据集原始 annotation；
- 这是本文真正使用/映射进统一体系的类别。

老师明确说，这个 marker 的作用只是一个视觉 marker，不要把它继续当成另一套新的 label naming system。

重点是“标出来哪些是我们 care 的”，而不是再创造第三套简称。

你提出可以用类似 console/code font 的方式显示这些特殊 label，老师认可这个方向，并要求图里所有这类标记保持统一字体。

---

### 三、Dataset Section 当前的逻辑顺序不对

接下来老师指出，现在 Section 2 的组织本身有逻辑问题。

目前有一些“annotation / mapping”的内容被放进了 Acoustic Characteristics，老师认为这是不对的。

因为：

dataset annotation ≠ acoustic characteristic。

一个是：

“这个数据集有哪些标签，以及我们用了哪些标签。”

另一个是：

“这些数据集在声学特征上有什么差异。”

这两件事不能混在一起。

老师建议 Dataset 部分最好明确分成如下逻辑：

第一层：Existing / Original datasets

介绍：

- 原始数据集；
- 每个数据集本身的任务；
- 每个数据集的原始 annotation。

这里属于 background。

第二层：Our dataset selection / harmonized dataset

这里才正式说明：

- 本文用了哪些数据；
- 每个数据集具体用了哪些子集；
- 哪些原始 label 被保留；
- 哪些被丢弃；
- 如何映射成本文统一使用的类别；
- 最终训练和评测实际使用哪些 harmonized labels。

老师的意思是：

“existing work / original dataset” 和 “our dataset” 必须明确分开。

第三层：Acoustic characteristics

等 dataset 和 mapping 都定义完了之后，再讨论：

- acoustic characteristics；
- dataset differences；
- feature distribution；
- 其他声学层面的差异。

老师认为目前的文字顺序太跳跃。

Figure 的顺序并不能决定论文文字结构，应该根据阅读逻辑来组织文字。

---

### 四、必须明确写一句：本文究竟只关心哪些统一类别

老师建议在 Dataset 部分非常直接地写清楚类似这样的意思：

“Across these datasets, we only consider the harmonized labels N, C, W, and CW.”

具体措辞可以后面再润色，但信息必须非常明确。

不能让 reviewer 看完 dataset annotation 后，仍然不知道：

“你们最后到底用了什么？”

老师特别强调，这一信息不应该到 Method 或 Evaluation 才重新解释。

Method 只应该“使用已经定义好的东西”。

不是到了 framework diagram 里又突然冒出一个别人从没见过的 label。

---

### 五、Abstract：不要浪费篇幅解释业内常见缩写，要把空间留给 contribution

你提到，一诺建议在 Abstract 里面把 ICBHI、SPRSound 之类名字全部写全称。

你觉得这会占篇幅。

老师基本认同你的判断：

如果这些名字本来就是业内常见的数据集名称，就没有必要在 abstract 里花大量空间把全称全部展开。

更重要的问题是：

当前 abstract 信息量还不够。

老师问：

“你的 contributions 不应该在 abstract 里面写出来吗？”

他希望把空间用于：

- dataset harmonization；
- 你们解决的核心问题；
- 方法；
- 主要结果；
- contribution。

而不是过度解释常见缩写。

你答应会把之前列过的 contribution 内容重新加回摘要。

---

### 六、Method Figure 是当前最严重的问题之一

老师随后集中批评了 Method Figure。

他的核心判断是：

现在这张图“看不懂”。

不是说图中每一个 component 都错，而是 reviewer 无法仅通过图本身理解：

- 哪些 head 会被激活；
- 哪些是 native task；
- 哪些是 auxiliary；
- 哪些是 harmonized/shared supervision；
- 不同 dataset 走的是哪条路径；
- loss 是怎么组成的；
- output 是什么。

老师认为目前这张图把不同层级的信息画成了“全部一样的东西”。

---

### 七、Native Head / Auxiliary Head 的关系必须直接画出来

你解释当前方法时提到：

从 encoder 出来以后，存在：

- ICBHI native task head；
- SPRSound native task head；
- Crackles auxiliary head；
- Wheezes auxiliary head。

其中某些 head 不会同时激活。

例如：

对于 ICBHI：

ICBHI native head + Crackles head + Wheezes head

对于 SPRSound：

SPRSound native head + Crackles head + Wheezes head

老师听完之后的反应是：

“那你直接把这种 possibility 画出来。”

不要让 reviewer 通过看箭头、猜 loss、再去读正文才能知道谁和谁一起激活。

他希望图里非常明确地表达：

ICBHI branch

ICBHI native head

- Crackles auxiliary head
- Wheezes auxiliary head

SPRSound branch

SPRSound native head

- Crackles auxiliary head
- Wheezes auxiliary head

同时：

- native head 和 auxiliary head 要通过不同视觉形式区分；
- 可以用虚线框；
- 可以用不同 shape；
- 可以用背景框；
- 可以用 line style；
- 总之要让 reviewer 一眼能区分不同角色。

老师说：

“一张图完全可以表达多层 information。”

重点不是减少信息，而是把信息层次组织清楚。

---

### 八、你们需要重新确认：Native task head 到底是不是主方法的一部分

这里出现了比较关键的方法论讨论。

你解释说：

原始主方法本来是 shared/harmonized representation。

后来做了一个 variant：

在 shared learning 之外，额外加入 native task head。

结果这个 variant 在 overall performance 上最好。

于是你和一诺讨论后，考虑把这个 variant 直接当成最终主方法。

老师对此明显有疑虑。

他问的核心是：

“如果最后还是每个 dataset 有自己的 native head，那 harmonization 的 claim 到底是什么？”

也就是：

你们前面一直讲把不同 dataset 的 supervision harmonize 到一起。

但如果最终结构又回到每个数据集自己的 task head，那么 reviewer 很可能会问：

“你前面费这么大力气统一，最后为什么又重新拆开？”

---

### 九、这个版本仍然可以成立，但 claim 必须非常准确

你解释说：

即使加入 native head，模型仍然：

- 学习共享 acoustic information；
- auxiliary/harmonized supervision 仍然存在；
- native task performance 会提高；
- overall performance 也提高；
- 在多个 dataset 上都能获得更高结果。

老师确认了你的意思：

也就是说：

加入 native task supervision 后，并不是只改善 native score，而是 overall performance 同样提高。

如果是这样，那这个 variant 仍然可以作为 final method。

但老师强调：

你们必须重新把“为什么还叫 harmonization”讲清楚。

不能让 Figure 和正文看起来像：

“harmonization 做完以后又各自训练各自的。”

---

### 十、Figure 应该直接展示不同任务下哪些模块被激活

老师给了一个比较具体的画图原则：

不要让所有模块都平铺在那里，然后让 reviewer 自己猜组合关系。

应该把：

- ICBHI + auxiliary
- SPRSound + auxiliary

这种实际执行路径直接表示出来。

如果需要，可以：

- 用大的 group box；
- 在 group 内再用小 box；
- 用虚线表示 auxiliary；
- 用另一个 style 表示 native；
- 用不同 outline 表示不同信息层次。

总之要把“谁和谁一起工作”画出来。

---

### 十一、图中的格式问题很多，需要统一

除了概念问题，老师还指出很多图形格式细节：

#### 1. 箭头

当前不同箭头：

- 粗细不一致；
- 箭头头部大小不一致；
- 某些曲线箭头几乎看不出方向。

原因并不是“图被压缩”，而是 linewidth 本身设置不一致。

要求：

所有同一语义层级的箭头统一：

- line width；
- arrow head size；
- direction clarity。

#### 2. Head 的 orientation

如果 native task head 是横向放置，那么 auxiliary head 也应该横向放置。

因为它们属于同一个 information level。

不要一个横着，一个竖着。

同一语义层级的信息必须保持 format 一致。

#### 3. Capitalization

像：

- Native Task Head
- Auxiliary Head
- Task-Specific Output

这种标题，首字母大小写需要统一。

#### 4. 括号和空格

图里类似：

“Abnormal(…)”

“Abnormal (…)”

这类格式不一致也需要全部统一。

#### 5. 不要在 Framework 里重新定义 Dataset Label

老师再次强调：

Dataset Section 定义过的 label，在 framework 中直接使用。

不要在方法图里第一次出现一个：

- others
- special class
- unexplained category

之类没有前文定义的词。

---

### 十二、Figure Layout 不能留大块空白

老师很不喜欢现在图中出现的大面积空白。

他的视觉原则是：

整个图应该“铺满”。

不是说一定塞满所有角落，而是不能出现明显的：

- 一边特别挤；
- 一边空一大块；
- component 没有对齐；
- visual center 不稳定。

老师提到自己之前改图时会尽量：

- bottom align；
- 调整 encoder 位置；
- 调整 head 排布；
- 让整体宽度、留白、组件密度合理。

如果某一部分很占空间，可以通过重新排列：

encoder → head → output → loss

而不是简单缩小。

---

### 十三、Optional 模块不要占据主视觉中心

如果某一部分只是 optional，不要把它画得像 main component。

可以：

- 放到末尾；
- 用虚线；
- 标记 optional。

但不要让 optional 组件和核心架构同权重展示。

---

### 十四、Contribution 不能“藏”在一段普通文字里

接下来老师回到 Introduction / Contribution。

你说之前 contribution 本来写成几个点，后来被改成一段话。

老师说：

他不是反对 paragraph form。

但即使是一段话，也必须让 reader 一眼看到：

“我们到底贡献了什么。”

例如可以在 paragraph 里非常明确地写：

“We make two contributions…”

或者其他明显的 transition。

重点是：

即使不用 bullet，也必须 inline emphasize。

不能让 reviewer 扫一眼这一段却完全找不到 contribution。

---

### 十五、每一个 Contribution 都应该能在 Evaluation 里找到对应证据

老师进一步强调：

Introduction 里提出什么 contribution，Evaluation 就必须证明什么。

也就是要形成：

Claim → Evidence

对应关系。

如果 Introduction 说：

“我们证明了 dataset harmonization 可以……”

那么 Evaluation 里应该明确有一部分结果就是：

“Effectiveness of Dataset Harmonization”

如果说：

“我们证明 auxiliary supervision improves generalization”

那么 Evaluation 就应该明确有一部分结果证明这一点。

老师反复说，现在的问题是：

“扫一眼看不到任何重点。”

---

### 十六、Evaluation 各部分的小标题必须告诉 reviewer：这一部分证明什么

老师认为现在类似：

“Joint Learning and Transfer”

这种标题过于 AI-generated，信息价值不够。

问题在于 reviewer 看标题后仍然不知道：

“这一组实验是为了证明什么？”

他希望小节或 table 周围的信息能清楚告诉 reviewer：

- baseline performance；
- effectiveness of XXX；
- ablation of XXX；
- native-task supervision；
- shared supervision；
- transfer performance；
- cross-dataset performance。

也就是：

标题不应该只描述“做了什么实验”。

而应该明确表达“这个实验在验证哪个 claim”。

---

### 十七、Table 中的方法组织要统一

你们讨论了主结果表。

老师指出：

如果你们自己的最终方法全部集中放在 table 底部，那么就应该统一把这部分和 baseline 分开。

可以用：

- horizontal line；
- visual grouping。

不要让 proposed methods 和 baselines 混成一坨。

如果某些 variant 是 ablation：

也要单独标明。

例如：

Main method

——

Ablations

这样 reviewer 会直接知道哪些结果属于：

- baseline；
- proposed model；
- ablation。

---

### 十八、Table 里的 Citation 不要乱放

老师认为：

模型 citation 应该主要在正文首次介绍时 cite。

Table 里面不需要每一行方法名都继续塞 citation。

尤其目前 table 已经很挤。

如果一定要 citation，就必须整体保持一致。

不能有的行 cite，有的行不 cite。

老师倾向于：

正文中 cite。

Table 只保留 clean model names。

---

### 十九、结果不是所有数据集都第一没有问题，但必须准确表述

你说明目前结果：

在大部分数据集上最好；

在 SPRSound 上是 second best。

老师对此没有要求强行包装成“全部第一”。

他的意思是：

要准确说。

不要为了 claim 强行说：

“state of the art everywhere”。

而是把真实 performance 讲清楚。

重点是你们真正要证明的贡献。

---

### 二十、论文不是 Project Report

这段是老师这次最核心的方法论反馈之一。

老师说：

做完实验和写论文是两回事。

Paper 的目的不是：

“把自己做过的所有东西都记录下来。”

那叫：

- lab report；
- project report。

真正的 paper 需要：

基于你做过的事情，重新组织成一个：

- audience 能理解；
- reviewer 能 follow；
- reader 能快速抓住重点；

的表达结构。

你也回应说：

写多了之后容易陷入“自嗨式”的表达，只顾自己知道做了什么，却没有从 reader 视角想：

“别人第一次看这篇文章能不能懂？”

老师确认，这正是当前最大的问题。

---

### 二十一、Reviewer 通常先看什么

老师明确讲了 reviewer 的阅读顺序。

很多 reviewer 第一遍不会逐字逐句认真读全文。

更常见的是先看：

- Abstract；
- Contributions；
- Figure；
- Table；
- Figure/Table caption；
- Section headings。

如果仅通过这些东西 reviewer 就能大致知道：

- 问题是什么；
- 方法是什么；
- 贡献是什么；
- 主要结果是什么；

那他才会继续回到正文里找细节。

老师甚至概括为：

如果 reviewer 看完所有图表和 caption，就已经能大概 get the idea，这篇文章的“第一印象”就基本过关了。

当前版本最大的危险是：

看完图表仍然不知道重点。

---

### 二十二、图表是当前最高优先级

因此最后你和老师形成共识：

现在优先修改：

1. Figure；
2. Table。

因为 reviewer 最先看的很可能就是这两个。

你也说：

这两类内容“最抓眼球”。

老师完全同意。

---

### 二十三、Caption 要 informative，但不要过长

老师拿以前 paper 的格式举例。

他的观点是：

Table/Figure title 和 caption 不能没有信息。

但是也不能像之前一样写得特别冗长。

理想状态是：

caption 自己就应该能告诉 reviewer：

- 这张图/表是什么；
- 比较什么；
- 主要目的是什么。

也就是 informative。

但不要把正文塞进 caption。

---

### 二十四、排版需要处理 orphan / widow 等问题

老师后面还指出了很多非常具体的排版问题。

比如：

一段文字最后只剩下一行；

下一段只单独出现一两个词；

某一列最后只剩很短的一行。

这种视觉效果会显得不 professional。

他要求你检查全文：

- orphan line；
- widow line；
- 单字/短词独占一行；
- 不自然换行；
- paragraph spacing。

可以适当调：

- sentence 长度；
- word choice；
- spacing；
- line break。

目的不是 purely cosmetic，而是让整体页面看起来像 final paper，而不是 rough draft。

---

### 二十五、全文 consistency 要重新检查一遍

老师要求你在改完 definition、marker、Figure 之后，再完整跑一次 consistency check。

因为即使你之前已经查过，现在修改这些核心定义以后，新的 inconsistency 很容易再次产生。

要重点检查：

- 同一个概念是否前后换名字；
- 前面定义、后面没有使用；
- 后面突然出现、前面没定义；
- dataset name 是否一致；
- label name 是否一致；
- capitalization 是否一致；
- hyphen / dash 是否一致；
- native / auxiliary / harmonized terminology 是否一致；
- figure 和正文是否使用同一术语；
- table 和正文是否一致；
- method 名称是否一致；
- contribution claim 与 evaluation 是否一致。

老师明确说：

可以用 AI 帮你查这种：

“有没有前后 conflict、有没有 terminology inconsistency、有没有 definition mismatch。”

---

### 二十六、不要再给老师看 Intermediate Version

这也是老师最后非常明确的要求。

他说：

“不要给我看 intermediate。”

他的意思是：

与其反复给他一个还存在明显结构问题、视觉问题和定义问题的版本，让他继续帮你抓基础错误；

不如你先自己全部整理好。

他宁愿一次花时间看：

“一个能看的版本。”

而不是持续 review 半成品。

后面再给他看的版本至少应该满足：

- 图能看懂；
- 表能看懂；
- label 一致；
- contribution 明确；
- section 顺序合理；
- formatting 基本 clean；
- consistency 已检查。

---

### 二十七、Reference / Citation 方面还有几个最终检查项

会议末尾提到：

#### 1. 作者 ID

Wade 的 ID 你已经拿到了。

老师说：

网上能搜到的信息不要直接去问别人。

应该先搜。

搜不到的再问。

#### 2. Conference blind policy

你们对是否 double blind / single blind 不是完全确定。

老师要求：

重新 double-check 官方要求。

不要靠记忆。

#### 3. Citation format

老师觉得目前 citation format 和他以前发表的 paper 格式不太一样。

要求确认：

- 官方 template；
- reference style；
- citation style。

不要最后格式错。

#### 4. Reference 数量 / 页面空间

如果需要在参考文献中补 relevance，可以适当：

- 删一些不重要的内部相关 paper；
- 补更直接相关的 ICASSP 工作；
- 在不增加新页面的情况下把 reference section 填得更合理。

重点是 relevance，而不是机械增加数量。

---

### 二十八、老师对当前 Paper 的总体判断

老师这次的整体情绪很明显：

他认为现在最大的问题不是“实验没做”，而是：

论文视觉和逻辑表达非常混乱。

尤其是：

- Figure；
- label；
- dataset annotation；
- method structure；
- contribution；
- evaluation organization。

他说现在很多地方自己作为熟悉项目的人都要停下来问：

“这个是什么？”

那 reviewer 更不可能快速理解。

但他的修改方向其实非常明确：

不是要你推倒重做整个研究。

而是要把已有内容彻底重新组织成 reviewer-oriented presentation。

---

### 二十九、这次会议最终明确的修改清单

按优先级归纳，你现在需要完成：

1. 重新整理 Dataset Section 的逻辑：
    original datasets → original annotations → our harmonized subset/mapping → acoustic characteristics。
2. 把每个数据集 Figure 里的原始标签尽量改成全称。
3. 用统一 marker 标出本文实际使用的类别。
4. 只保留一套 harmonized label 系统，避免简称混用。
5. 在 Dataset Section 一次性定义：
    本文到底使用哪些 labels、哪些 subset、如何 mapping。
6. Method 后面不再重复定义 labels。
7. 重新画 Framework Figure。
8. Figure 必须明确画出：
    ICBHI native head + auxiliary heads；
    SPRSound native head + auxiliary heads。
9. 明确 native head / auxiliary head / harmonized supervision 的视觉区别。
10. 统一：
    arrow width、arrow head、font、capitalization、spacing、box style、orientation。
11. 减少图中无意义空白，让整体排版更紧凑饱满。
12. Optional component 用虚线/标记处理，不抢主视觉。
13. Abstract 不必大量展开常见 dataset 全称，把空间留给 contribution。
14. 把 dataset harmonization 等核心 contribution 加回 Abstract。
15. Introduction 中明确标出 contribution，即使是 paragraph 也要清楚。
16. 每个 contribution 要在 Evaluation 有对应实验。
17. Evaluation subsection / table title 要直接告诉 reviewer：
    这部分证明什么。
18. 统一 table formatting。
19. Main method 与 ablation 分开。
20. Baseline 与 proposed method 分开。
21. Table 里的 citation 做统一处理，优先放正文。
22. 对 performance 的描述保持准确，不强行声称所有 dataset 都第一。
23. 修改 figure/table caption，使其 informative。
24. 全文检查 orphan/widow、奇怪换行和排版细节。
25. 改完以后重新跑一次 terminology/definition/format consistency check。
26. 确认 conference blind policy。
27. 确认官方 citation/reference format。
28. 检查 references relevance 和页面利用。
29. 下一次不要给老师看半成品，先整理成一个接近 final presentation quality 的版本再发。

---

如果把老师这次的要求压缩成一句话，就是：

你们现在不缺“做过什么”，而是缺一篇能让陌生 reviewer 在不依赖作者解释的情况下，仅通过 Abstract、Section 结构、Figure、Table 和 Caption 就能快速看懂“问题是什么、方法是什么、贡献是什么、证据在哪里”的 paper。
