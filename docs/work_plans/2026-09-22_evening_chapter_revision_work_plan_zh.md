# 2026-09-22 晚间 Working Plan：图表格式与全文一致性

Notion：[今晚Work Plan](https://app.notion.com/p/3e4309efda298130b73ac43649349b54)；[9/22完整会议记录](https://app.notion.com/p/3e4309efda298119b493c71b5617914e)。本地：[完整纪要](../meeting_records/2026-09-22_paper_revision_meeting_record_zh.md)。

## 依据与今晚重点
会议记录：[9/22完整会议纪要](https://app.notion.com/p/3e4309efda298119b493c71b5617914e)。
工作时间：2026-09-22（周二）晚上，不安排具体时刻或时段。
用户进一步明确：今晚重点是格式与呈现。老师指出的问题主要集中在Figure/Table、标签表示、信息层次、标题/caption和版面一致性；文字只做配合这些问题的必要局部调整。
本计划据此纠正上一版将“从Intro重梳论点、广泛重写Results”作为主线的安排。已有方法、结果和研究方向作为本轮基础；今晚先让读者能看懂并快速定位重点。

## 逐章推进方式
按所属章节逐项讨论和确认，优先处理Section 2/Figure 1 → Section 3/Figure 2 → Section 4/Tables与结果图；再处理Introduction、Abstract中的必要局部呈现，最后核对全文格式。
Figure 2的信息层次与执行路径是重点：老师要求把实际组合画清楚，包括native head和auxiliary heads谁与谁同时工作；这会涉及图的组织，不只是改字体。
本页checkbox均待实际修改与用户核对后勾选。本次只修正计划，没有开始论文或图件修改。

## Section 2 / Figure 1｜标签表示、marker与定义位置
负责人：论文写作二（任务2），由其与论文绘图协调。
- [ ] S2-01｜Figure 1中的原始dataset annotation尽量全部写全称，避免原始缩写和统一缩写混用。
- [ ] S2-02｜用一种清楚、统一的类别使用marker标出本文实际关心/映射的类别；未使用类别不标。marker只承担视觉标识作用，不增加第三套命名。
- [ ] S2-03｜这类marker统一使用约定的code/console字体；统一Figure 1中的字体、大小写、名称、对齐及数据集标识样式。
- [ ] S2-04｜N/C/W/CW等统一表示在Dataset处一次定义清楚，包括实际使用范围；后续正文和图件使用同一套名称，删除重复定义或无前文定义的标签。
- [ ] S2-05｜仅调整必要的段落位置：original datasets/annotations → our subset与mapping → acoustic characteristics；将annotation/mapping移出声学特征段，配套缩短并明确Figure 1 caption。
验收：原始标注、实际使用类别、统一表示一眼可分，图文定义对应。

## Section 3 / Figure 2｜实际执行组合与视觉层次
负责人：论文写作二（任务2），由其统一协调论文绘图。
- [ ] S3-01｜直接画出两种组合：ICBHI native head＋共享C/W辅助头；SPRSound native head＋相同的共享C/W辅助头。让图明确表达哪些head同时激活以及共享监督仍在哪里，必要时配一句简短说明。
- [ ] S3-02｜用group box、不同outline/shape或虚实线区分native与auxiliary、共享监督与任务输出；optional组件放次要位置并清楚标记。
- [ ] S3-03｜同一信息层级保持相同orientation：native和auxiliary heads统一横/纵向安排；encoder、heads、output、loss对齐。
- [ ] S3-04｜同语义箭头统一line width、arrow-head size和线型，保证曲线/转折箭头方向清楚，训练监督和输出路径可辨。
- [ ] S3-05｜通过重排组件消除“一边拥挤、一边空白”，调整bottom alignment与视觉重心，保持文字可读和可编辑源。
- [ ] S3-06｜统一标题capitalization、括号前空格、字体/字号和box style；复用Section 2已定义标签。caption简洁说明图中组合及共享关系，Method仅做与图一致所需的局部修改。
验收：读者仅看图即可辨认不同数据集的执行路径、head角色、output与loss关系。

## Section 4 / Tables与结果图｜分组、标题、caption和显示一致性
负责人：论文写作（任务1），负责结果表、结果图及共享caption。
- [ ] S4-01｜通过横线和清楚的分组把baselines、proposed main、ablations分开；统一方法摆放顺序、完整名称及表格对齐。
- [ ] S4-02｜清理表中citation：优先移到正文首次介绍方法处，表内保留clean model names；如有保留引用，执行统一规则。
- [ ] S4-03｜把Evaluation小标题/表格标题改成可辨认的实验目的，并与已有contribution对应；让读者扫标题就能定位证据，不重新构造一套研究论点。
- [ ] S4-04｜统一Figure/Table caption：简短交代展示什么、比较什么、目的是什么；必要的指标/误差棒说明保留，不把正文塞进caption。
- [ ] S4-05｜统一结果图与表的呈现：数据集颜色/形状、方法名、数值精度、mean±SD和图例说明一致。沿用当前已确认的紧凑结果图样式，同dataset的三个seed点、均值及误差棒严格竖直对齐；核对实际插入Overleaf的资产。
验收：读者能直接分辨比较组并找到各实验目的，图表和caption独立可读。今晚不重新计算ANOVA或全面重写Results；既有排名与显著性表述如发现具体错误，只修正该处。

## Section 1 / Introduction｜让已有Contribution显眼
负责人：论文写作（任务1）。
- [ ] S1-01｜在现有contribution段加入明确引导或inline emphasis，例如“We make two contributions…”，让两项贡献容易找到；与S4-03的标题对应，不把这一项扩大为重写Intro/gap。
验收：扫读时能定位已有贡献及其证据入口。

## Abstract｜补回必要信息，控制篇幅
负责人：论文写作（任务1）。
- [ ] A-01｜按会议要求把已有的核心contribution（包括dataset harmonization）补回摘要；保留常用ICBHI/SPRSound名称，避免机械展开占空间，只做必要局部调整。
验收：贡献信息可见，名称与正文一致。

## Section 5 / Conclusion｜随全文做一致性核对
负责人：论文写作（任务1）。
- [ ] S5-01｜检查方法名、术语、数字和现有结论与修改后的图表/定义一致；只有具体不一致处才改，不单独安排Conclusion重写。

## 全文格式与提交要求
负责人：用户＋论文写作（任务1）；任务2核对Section 2/3及对应图件。
- [ ] F-01｜统一全文的dataset/label/method名称、简称首次定义、native/auxiliary/harmonized指代、capitalization、括号空格、hyphen/dash；同步修正受影响的术语索引位置。
- [ ] F-02｜根据用户Overleaf编译稿处理orphan/widow、单词或短行独占、不自然换行、段间距及图表位置；通过必要的句长或布局微调使四页正文整洁。
- [ ] F-03｜核对实际投稿会议的官方template、citation/reference style与页数要求，统一现有引用格式和reference排版。
- [ ] F-04｜核对官方blind policy与作者信息/ID要求；Wade已提供的ID直接核对，其他公开信息先查询。按相关性检查参考文献及页面利用，不机械增加数量。
- [ ] F-05｜最后只看Abstract、Contributions、section headings、Figures/Tables和caption做一次整体检查，再核对全文；形成图表清晰、标签一致、格式clean的完整版本，再由用户决定发给老师。
验收：老师无需继续指出显而易见的基础格式和呈现问题。

## 协作与保留事项
- 仍使用main下`docs/paper/Overleaf_Sync_final/`。任务1负责Sections 1/4/5、Abstract、结果表和main.tex/citation.bib/共享caption；任务2负责Sections 2/3及相应图件，统一对接论文绘图。
- 各章先讨论具体呈现修改再落实；用户在Overleaf编译，完成并核对后再勾选。保持用户控制的管理同步方式，不恢复DONE标记。
- 已有实验、逐seed统计及分析资产继续使用。结果图源在`result/analysis/2026-09-22_unified_seed_figure/`；论文中的最终资产需与确认稿对齐。
- ICBHI协议材料、Arian/Yinuo未关闭意见及Hanlin后续材料保留在既有记录；发现具体未解决项单独列出，不默认扩成今晚的新研究设计或整篇文字重写。
- 9/21记录与计划继续归档保留原状态。原始9/22会议纪要保持不变；本页仅修正管理对今晚执行重点的理解。没有启动新实验、写作任务、绘图任务或独立reviewer。
