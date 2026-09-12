# ICASSP 审稿人视角评审｜2026-09-09 稿件

评审对象：`main.tex` + `Section/1--5`（2026-09-08 四数据集 7 页 chapter-review 版）
评审范围：**story telling / 逻辑 / 方法定位**。图表版式、篇幅压缩、模板与作者信息不在本轮。
立场：模拟 ICASSP 审稿人首轮阅读，不做内部善意补全。

---

## 0. 评分与总体判断

**Recommendation: Reject（当前状态）／ Weak Accept（若补齐 §5 的前两项实验）**

| 维度 | 评分（1--5） | 说明 |
|---|---|---|
| Novelty | 2 | 所有组件在自引文献中已有先例，差异点未被实验隔离 |
| Technical correctness | 4 | 协议描述严谨，边界声明罕见地诚实 |
| Experimental support | 1 | **没有一个匹配对照；主张与证据不对应** |
| Clarity | 3 | 句子清楚，但论文级论证结构缺失 |
| Relevance to ICASSP | 4 | 主题合适 |

**一句话总结：这份稿件把"实验设计"当成了"实验结果"提交。**

摘要最后一句是 "Matched annotation and classification ablations remain necessary
to separate the effects..."；Conclusion 最后一句是
"Matched annotation and classification controls remain necessary to establish
how much of these changes is attributable..."。
论文自己在首尾两处声明：本文的核心问题尚未回答。
审稿人读到这里就可以停止了——这不是一个 caveat，这是 **缺少 finding 的自我认定**。

技术诚实度是真实优点，但诚实不能替代结论。当前形式下，审稿人无法写出
"this paper shows that ___" 这个句子。

---

## 1. 致命问题：论文没有可被证伪的主张

### 1.1 贡献列表里有两条是未来时

> C3: "...with supervision-matched comparisons **designed to** separate their
> effects on task performance and class-specific trade-offs."

`designed to`，不是 `showing that`。审稿人对 contribution list 里的未来时极其敏感。
配合 Table IV 的 caption——"Ablations required by the storyline. **Results are
pending**"——以及 §4.9/§4.10 通篇 "a matched annotation study **will** keep..."，
论文实际提交的是第 4 节的实验计划书。

`\draftnote` 宏在正文出现 4 次。即使这是内部版本，它标记的正是审稿人会攻击的同一批位置。

### 1.2 唯一被"支撑"的主张是恒真的

Conclusion："Existing results **demonstrate the joint core capability**"；
§4.2："the shared model **supplies both requested outputs**"。

模型能同时输出两个 native task 的标签，是解码规则定义出来的，不是实验发现的。
没有任何实验可以让它为假。这在审稿意见里会被写成：
*the only claim the paper supports is true by construction.*

这里存在一个必须在正文分开的概念混淆（贵组自己的 novelty review 已指出，但没有写进稿件）：

- **Task preservation**：保留预测单元/标签/评价定义 —— 已实现，但是设计选择，不是结果；
- **Performance retention**：相对单任务训练保留了多少性能 —— **这才是论文想要的主张，而它没有对照**。

正文目前用 "preserve" / "retain" 同时指这两件事，是当前叙事最大的逻辑滑动。

---

## 2. Table I 目前是**反证**，不是证据

这是我作为审稿人最强烈的反应。Table I 呈现：

| 行 | ICBHI | SPRSound |
|---|---:|---:|
| frozen BEATs（单数据集） | 51.37 | **92.00** |
| PAFA（ICBHI 专用 checkpoint） | **64.14** | -- |
| **本文 joint shared attributes** | **61.17 ± 0.31** | **90.70 ± 0.34** |

**快速扫表的审稿人得到的结论是：本方法在 ICBHI 上输给 PAFA 3 分，在 SPRSound
上输给一个 frozen encoder 1.3 分——两个轴上都不占优。**

紧接着的正文说：

> "These different reference points **do not establish a sharing gain or cost**"

即：论文的主结果表格自己声明它不证明任何事情。这不是免责，这是把
"我们知道这张表不成立" 写进了论文。审稿人会直接引用这句话作为拒稿理由。

`92.00` 是 single-seed、2 s 窗、独立 selection 的 frozen 参考——但它被放进了
主结果表，**没有任何 matched framing**。审稿人不会替作者做这个折扣。

**这不是排版问题，是论证结构问题。** 一张主结果表如果需要正文说明"这些数不能比"，
它就不该以这个形式出现在主结果位置。

---

## 3. 最强的 motivation 被删掉了

Table I 的 `ICBHI source checkpoint references` 行块里，PAFA / SG-SCL / Patch-Mix
的 SPRSound 列全是 `--`。

而 repo 里这三个数是存在的且已核验：
**PAFA 55.82、SG-SCL 59.98、Patch-Mix 59.38，all-Normal floor 50.00。**
（`result/pafa_sprsound_transfer_*`、`sg_scl_sprsound_transfer_*`、
`sprsound_patchmix_frozen_transfer/`；本轮 work plan §4 也列为已有证据。）

把这三个数填进去，同一张表立刻从"我们两边都输"变成：

> ICBHI 上最强的专用模型，迁到第二个 benchmark 上贴着 trivial floor（+5.8 / +10.0 / +9.4）；
> 同一个共享模型在 ICBHI 上落后 3 分，但在 SPRSound 上高出 30 分以上。

这是全篇唯一能回答 *"why should I care"* 的东西，而它被一个破折号替换了。

必要的边界声明（zero-target-tuning frozen-checkpoint 迁移，非 fine-tune；
因此它证明"专用模型不具备跨 benchmark 能力"，**不等于**"联合训练的净收益"）
应该写在脚注，而不是导致整块证据被移除。

**审稿人立场：删掉动机、保留全部 caveat，是这一版最不划算的取舍。**

---

## 4. 方法：与最近邻工作的区别没有被隔离，而论文自己指出了这一点

### 4.1 "两层"实际上不是层级

§3.1：

> "the C/W probabilities are **marginal predictions**, with **no explicit
> parent--child probability-consistency constraint**"

Figure 2 caption：

> "the two-level decision structure **does not make the predictors a cascade**"

所以方法 = 共享 256 维表示上的三个**并行**线性头 + 一个后处理解码规则。

论文自己在 Introduction 引用的 PC-MCL 是
*"explicit multi-label targets and deterministic four-class conversion"*
\cite{jeong2026pcmcl}——**这与上面的描述几乎逐字对应**。
引用了最近邻，然后没有与它区分。审稿人会在 Related Work 段落就标记这一点。

同理，Eq.(2) 的 availability mask 在 DCASE 2024（missing-target masking，
论文自引）与 Schutera / Bevandić 处均有先例；full BEATs fine-tuning 与
PCSL/GPAL 是直接继承的既有组件（论文明确说 "We retain PAFA's..."）。

**逐项拆开后，剩下的差异只有一个：abnormality gate + margin tie-break 的解码规则**
（ICBHI 先由 $\hat a$ 判 Normal，再由属性定四类；两属性都不过阈时按
$p_k-\tau_k$ 较大者取，ties→Crackle），相对 PC-MCL 式的纯 C/W 转换。

### 4.2 而论文恰好没有做那个唯一能证明它的实验

§4.10 提出的第二项对照就是：

> "Two-level vs. C/W-only readout, **fixed scores**"

**这个实验不需要训练。** 固定已保存的 attribute scores，只换解码规则重算即可，
成本是分钟级。它是唯一能把本文与 PC-MCL 分开的对照，也是全篇成本最低的实验。

**没有跑这个实验，是这一版最难向审稿人解释的遗漏。**
审稿意见会写成：the authors identify the decisive control themselves and do not run it.

### 4.3 masking 的实际覆盖率没有报告

叙事重心（"annotation-aware"、"Learning from Available Supervision"、Eq.(2)）
建立在 availability mask 上。但正文从未说明**有多少样本真的被 mask**。

按现有映射：ICBHI 四类全部提供完整 $A,C,W$；SPRSound 的 Normal / Coarse & Fine
Crackle / Wheeze / Wheeze+Crackle 也提供完整三节点；**只有 Rhonchi 与 Stridor
两类的 C/W 被 mask**（协议常量 39 + 15，且还要再分 fitting/validation）。

也就是说，核心训练里被 mask 的样本大约在**百分之一量级**。
审稿人若自己算出这个数，会认为叙事重心与机制的实际作用范围不成比例。

**必须在 Data 或 Method 明确给出 mask 覆盖率。** 如果确实很小，就诚实地缩小
"annotation-aware" 的叙事权重，把重心移到解码设计上；
或者构造一个 mask 真正起作用的条件（即 §4.9 提出的 SPR 只保留 coarse 监督那一组）。
两条路都可以，唯独不能让读者自己发现这个落差。

---

## 5. 若要让我改分，需要什么（按性价比排序）

| # | 实验 | 训练成本 | 它解决什么 |
|---|---|---|---|
| **E1** | 固定 scores，两层解码 vs. C/W-only 解码 | **零训练** | 与 PC-MCL 分开；把 C1 从"系统描述"变成"设计发现"。**最高优先** |
| **E2** | ICBHI-only / SPRSound-only（同配方、同预算、同 selection）→ 各评两个任务 | 2 run | 把 Table I 从反证变成 2×2 retention matrix；这是 performance retention 唯一的合法证据 |
| **E3** | 共享 encoder + native heads（同标签信息） | 1 run | 隔离"分类接口"与"监督信息"；C3 的直接支撑 |
| **E4** | SPR 只保留 coarse 监督 | 1 run | 让 mask 机制真正进入实验；C2 的直接支撑 |

**E1 + E2 是把这篇论文从 Reject 拉到可讨论区间的最小集合。**
E1 尤其没有任何理由不做。

若时间只够一项：做 E1，同时把 §3 的证据填回 Table I。

---

## 6. 结构与篇幅：四页装不下当前内容，且贵的部分是最弱的

当前 7 页，需要压到 4 页。压缩顺序应当由证据强度决定，而不是按比例缩：

| 内容 | 当前占用 | 建议 |
|---|---|---|
| HF（Data §2.3 + Method §3.4/§3.5 + Eval §4.3/§4.4/§4.6） | ≈1 页以上 | **压到 4--5 行 + Table II 两行**。它支撑的只有一个 single-seed 历史实验，而论文自己声明该实验的监督规则不是想测的那个（"annotation-derived negatives prevent interpreting this historical run as a test of a strictly positive-only objective"）——用一页篇幅承载一个被自己否定的实验，是四页论文负担不起的 |
| Table III（frozen-encoder HF/KAUH native references） | 一张表 + 一段 | **删**。正文明说 "Differences in task, aggregation, selection, and training **prevent a direct comparison**"。一张明确不能比较的表在四页里没有位置 |
| KAUH 四分类（47.00 ± 8.30 / Macro-F1 / UAR） | 表 II 三行 + §4.7 | **只留 binary**。scored subset 是 86 patients，其中 Crackle 8、**Both 2**。在 n=2 上报四类聚合分并给标准差，会被直接质疑 |
| KAUH binary（72.17 ± 1.64） | | 保留，一句话："粗粒度 Normal/Abnormal 可迁移到未见数据集，细粒度属性不可"——这是有价值的一句 |
| §4.9 / §4.10 / Table IV（pending） | ≈半页 | 变成真实结果，或删除。计划不进正文 |

腾出的空间给 E1/E2 的结果和 §3 的 motivation 表。

---

## 7. Abstract / Introduction 的具体问题

**Abstract 没有任何比较对象。** 给出 61.17 / 90.70 两个绝对数，读者无法判断好坏。
且最后一句自陈实验缺失。ICASSP 摘要需要一个 finding：
"X 的代价集中在 Y，收益集中在 Z" 这种句式，而不是 "we study" + 两个数 + "ablations remain necessary"。

**Introduction 第三段（related work）目前是一份"这些都被做过了"的清单。**
SPRSound fusion、LungMix、OPERA、Bevandić、DCASE、PC-MCL、OCAD 依次列出，
然后一句 "Our focus is the specific relationship between shared attribute
supervision, annotation conditions, and native respiratory-task readouts."
——这句话没有排除上面任何一项。标准结构应是：既有工作做 X，X 在 Y 上失效（给证据），
本文做 Z。目前缺的正是 "Y"（§3 的迁移证据）。

**Contribution C1 是系统描述，不是贡献。** "An annotation-aware joint learning
framework that shares... and returns... through a two-level decision scheme"
——描述的是做了什么，不是发现了什么。按贵组自己的 novelty review 结论，
组件级创新不成立；那么 C1 就必须改写成经验发现（"我们发现共享属性监督在
两项原生任务上的代价集中在 ICBHI specificity 而非 sensitivity" 这类），
或者与 C2 合并成一条范围具体的方法设计贡献。

**建议保留但需改写的一处**：§4.2 的 per-class 数据里其实藏着一个可写的发现——
Se 47.35 与 PAFA published rows（47.63 / 48.21）实质持平，
而 Sp 74.98 vs 82.05 差 7 分。**ICBHI 的整个差距在 specificity，不在 sensitivity。**
这是一句有机制含义、零成本、且直接呼应 HF 实验方向（HF-on 反向推高 Sp、压低 Se）
的观察。当前正文把它埋在一串数字里，没有点出来。

---

## 8. 需要在正文回答的问题（审稿人 questions）

1. 核心训练中被 availability mask 的样本占比是多少？
2. 固定 attribute scores，仅将两层解码换成 C/W-only 转换，ICBHI/SPRSound 各变化多少？
3. 同配方、同预算、同 selection 下的 ICBHI-only 与 SPRSound-only 结果是多少？
4. 相对 PC-MCL 的 deterministic four-class conversion，本文的 abnormality gate
   在哪一类判断上产生差异？
5. Table I 的 frozen BEATs SPRSound 92.00 与本文 90.70，是否说明共享训练在
   SPRSound 上没有收益？（这个问题一定会被问，必须在正文先行回答）
6. ICBHI test-selected 协议的乐观量级是多少？
   —— repo 里已有可用参考：PAFA BEATs+CE 复现的 clean 54.83 vs test-selected 59.86，
   **约 5 个 ICBHI 点**。把这句写进 §4.1，等于主动量化披露 Table I 所有行共享的偏差，
   比只写 "describe a test-selected benchmark" 强得多。

---

## 9. 值得肯定的部分（不要在修改中丢掉）

- 协议边界声明的密度与准确度高于该领域平均水平：官方 split 非严格 patient-held-out、
  SPR inter/intra 不合并、HF gap 不当 negative、KAUH B/D/E 为同一患者的相关版本、
  不构造 pooled score —— 这些都应保留。
- §3.4 主动指出历史 HF 实现使用了 annotation-derived negatives、
  因此不是 positive-only 目标 —— 这种自我修正在投稿论文中很少见，是加分项。
- 指标定义写得清楚（ICBHI Se 是"分到正确异常子类"，混淆 C/W 会降低 Se；
  SPRSound official Score 的 AS/HS 定义）——很多同类论文不写这些。
- 四数据集角色分工（core / supervision test / external）本身是清晰的组织框架。

**问题不在严谨性，在于严谨性目前被用来说明"什么都还没证明"，
而不是用来支撑一个被证明的、有边界的结论。**

---

## 10. 一句话给作者

这篇稿子把每一条主张都诚实地限定到了不存在。
补上 E1（零训练）和 E2（两个 run），把已有的迁移证据填回 Table I，
它就从"实验计划"变成"有边界的经验发现"——而这正是这个课题真正值得写的东西。
