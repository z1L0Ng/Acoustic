# Prompt for GPT-5.6 Pro

你是一位熟悉 ICASSP、respiratory sound classification、audio foundation
models、cross-dataset learning 和短篇论文组织的资深研究合作者兼严格审稿人。

请完整阅读我上传的所有材料，先复盘项目已有工作，再提出 paper draft 的逻辑方案。
不要直接写完整论文正文；这一轮的目标是确定 research question、gap、contributions、
evidence structure、section organization、figures/tables 和截止日前的最小实验集合。

最重要的是先回答 2026-09-03 advisor meeting 提出的这些问题：

1. 这篇论文究竟回答什么 research question？
2. 除了宽泛的 adaptability 以外，我们解决的具体 gap 是什么？
3. 哪两到四条 contribution 可以由现有结果直接支撑？
4. 为什么 heterogeneous respiratory datasets 难以被同一个模型处理？哪些差异属于
   prediction unit、label availability、class distribution、acoustic characteristics、
   device/filter、grouping 和 native metrics？
5. Acoustic-feature analysis 怎样才能解释 dataset-dependent performance，而不是只展示
   PCA/outliers，或把 association 误写成 causality？
6. 当前主结果之后，论文必须补哪些结果？哪些只是 nice-to-have，哪些应该删除？
7. Hanlin 的 single-dataset acoustic foundation-model baselines 和 strong ICBHI
   task-specific methods 应怎样组织，才能真正支撑 motivation 和 comparison？
8. 怎样解释 HF auxiliary supervision 改善 HF/SPR，却使 ICBHI specificity 上升、
   sensitivity 和 Both recall 下降的现象？
9. 当前 JH2 使用 ICBHI official-test checkpoint selection。考虑到 PAFA 等相关工作
   也使用 test selection，论文应如何透明、准确地报告这一点？是否必须补一套
   prospective validation-selected companion result？
10. 在 ICASSP 四页限制和 2026-09-16 截止日期下，最可信且最可完成的故事是什么？

请避免把论文组织成“我们尝试了 A、B、C、D，最后某个组合最好”。请使用：

observed problem -> concrete gap -> method response -> directly supporting evidence

来建立逻辑。

请严格保持以下证据边界：

- 当前 JH2 三种子结果是 ICBHI-test-selected，不是 clean generalization estimate。
- SPRSound official inter 在每个 selected checkpoint 后只评测一次，但不能消除 ICBHI
  checkpoint-selection 的 test exposure。
- JH4 目前是 single-seed auxiliary extension，不能作为稳定的 universal improvement。
- HF/KAUH 的 fixed-checkpoint 结果是 external diagnostics，不是 native benchmark
  reproduction。
- PCA 和 acoustic feature differences 只能支持 association 或解释性 hypothesis，
  不能直接支持 causality。
- 不得把 HF annotation gap/empty 当作 Normal 或 negative。
- 不得把 KAUH B/D/E 当作独立患者。
- 不得构造四数据集 pooled score。
- 不得声称 SOTA、robust、universal generalization 或所有数据集全面改善。
- 不要把不同 split、prediction unit、label space 或 metric 的文献结果直接并表。
- 如需补充文献，请优先使用 primary paper、official proceedings 或 official repository，
  并明确直接可比与背景参考的边界。

请提供以下输出，按顺序组织：

## A. Executive diagnosis

- 用不超过十条 bullet 总结项目目前最强证据、最大弱点和最关键风险。

## B. Direct answers to the advisor

- 逐项回答上述十个问题。
- 对每个判断标明是 existing evidence、interpretation 还是 proposed next step。

## C. Paper-story alternatives

- 给出两个可行故事：
  1. ICBHI+SPRSound core adaptability；
  2. core model 加 HF positive-only auxiliary 与 KAUH external analysis。
- 比较两者的创新性、证据完整性、四页可写性和截止日前风险。
- 明确推荐其中一个，并说明为什么。

## D. Research question, gap, key phrase, and contributions

- 一句英文 research question；
- 一句英文 gap statement；
- 一个最适合贯穿标题、Introduction 和 Conclusion 的 key phrase；
- 三条英文 contribution bullets；
- 每条 contribution 后列出其 supporting result/figure/table；
- 如果某条贡献现有证据不足，请直接删除或降级，不要替我们补造证据。

## E. Four-page paper organization

- 给出完整 section/subsection 结构；
- 为每段说明目的、核心论点、所需引用和对应证据；
- 给出建议 page budget；
- 说明 Related Work 应压缩在哪里；
- 不要写完整段落，只给可以指导后续写作的详细 outline。

## F. Results package after the main result

请建立一个表格，包含：

- result/experiment；
- reviewer question；
- current status；
- evidence strength；
- required for submission / useful / remove；
- whether new training is required；
- estimated value versus deadline risk；
- intended paper location。

请重点判断以下候选：

- published PAFA BEATs+CE and PAFA references；
- Hanlin 的 four-dataset single-dataset foundation-model baselines；
- one or two strong ICBHI task-specific baselines；
- matched ICBHI-only and SPRSound-only baselines；
- independent task heads versus shared hierarchy；
- eligibility masking ablation；
- PAFA-off ablation；
- JH4 multi-seed；
- HF Lung and KAUH fixed-checkpoint diagnostics；
- MVN and Soft Bridge negative results；
- specificity/sensitivity and per-class error analysis。

## G. Figure and table plan

评估并改进当前计划：

- Figure 1: dataset scale, class composition, acoustic PCA/features；
- Figure 2: high-level input -> harmonization -> model -> output；
- Table I: JH2 three-seed ICBHI/SPRSound main results plus direct references；
- Table II: baselines and controlled ablations。

请说明每个图表必须回答的问题、最少字段、可删除信息和可能的 reviewer concern。

## H. Acoustic-analysis contract

- 明确 Wade 的分析应测试哪些 hypothesis；
- 给出 PCA 之外最值得做的 group-aware quantitative measures；
- 说明如何进行 shared Normal/Abnormal 或 class-matched analysis；
- 指出哪些观察可以进入论文，哪些只能留作诊断；
- 给出一项在 48 小时内可完成的最低交付。

## I. Baseline contract

- 明确 Hanlin 应整理或复现的最小 baseline 集合；
- 强调 paper score、local score、delta、task、split、unit、metric 和 selection；
- 指出哪一个 strong ICBHI method 最值得优先，以及是否值得测试 cross-dataset
  transfer；
- 给出一项在 48 小时内可完成的最低交付。

## J. Friday-to-Sunday execution plan

- 把任务分配给 Zilong、Wade 和 Hanlin；
- 每天给出 deliverable、decision gate 和 backup；
- 不让本科生成为论文关键路径 blocker；
- 不启动没有明确 paper destination 的实验。

## K. Reviewer-style verdict

- 以 ICASSP reviewer 的角度说明当前故事为什么可能被接受、为什么可能被拒绝；
- 列出最可能的五条 reviewer criticisms；
- 说明截止日前哪三项工作最能提高论文质量；
- 给出一个保守的 paper readiness assessment，不要给虚假的精确录用概率。

请用中文完成分析，但标题、research question、gap、contributions、section names 和可直接
用于论文的术语使用英文。保持批判性：如果当前证据不足以支撑预期故事，请明确指出，
不要为了让项目看起来完整而弱化方法学问题。
