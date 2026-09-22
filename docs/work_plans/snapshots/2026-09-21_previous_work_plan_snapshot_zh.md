# 9/21会后计划切换前快照

归档日期：2026-09-21。以下保留9/18–9/20本地计划在本次归档前的完整内容，包括真实未完成checkbox；不是一次新的完成验收。原Notion页：https://app.notion.com/p/3df309efda2981a68624c85b689f596b。

---

# 2026-09-18–09-20 Working Plan：修稿核对与合作者反馈

Notion：[当前工作计划](https://app.notion.com/p/3df309efda2981a68624c85b689f596b)。

## 9/20集中复核与下一步
用户与写作任务已完成一轮核对，本次逐项检查T01–T32及T18a：33项中22项已完成（本次新增勾选20项），11项仍有具体余项；不将“已入稿”视为完整外部实验验收。页面保持In progress，原9/18–19时间块作为历史安排，当前下一步为合作者反馈。
- [ ] T33｜9/20上午：按已约定时间与合作者meet，展示当前paper和figures并取得修正反馈；会后按“反馈—对应章节/图表—下一步修改”整理下一轮todo。具体时刻与参会人按用户安排，不另行发送邀请。
本轮已完成整体论点/数据角色、Figure 2符号及路径核对、Arian C1/C4与指标定义、现有结果/归因讨论、全文和Abstract/Conclusion核对。表格现组织为Table 1 frozen references、Table 2方法系统比较、Table 3 ablations。
真实剩余项：Figure 1图内标签与最终集成（T03/T08，用户＋论文写作二/绘图）；baseline输入/选模/外评差别说明（T09，论文写作）；成功PC-MCL轻量归档（T12，管理）；统计附注与Arian最终逐点回应（T21/T22，管理＋论文写作）；Hanlin完整原始材料及PAFA（T23/T24，材料到达后核验）；最新版Overleaf四页与最终图表/PDF（T28/T29，用户＋写作）；反馈后最终编辑通知草稿（T31，管理）。
核对依据：当前`docs/paper/Overleaf_Sync_final/`源码、两项写作任务本轮交接、现有Figure 2导出/预览、Hanlin `result(3).xlsx`及已保存的结果预测。Native+C/W三seedHF/KAUH从逐样本预测复核为88.78±1.29% / 77.91±3.02%；Table 1五组四列与表格材料20格一致。最新四页编译稿尚待用户确认。
当前不启动新模型执行，不改paper、不编译、不提交Git、不发送邮件；先完成今早meeting，再由用户反馈决定下一轮修改。

## 9/19阶段决定（历史背景）：结果归档，转入paper work

用户已明确：先暂存并提交当前结果，随后集中修paper与figures；除等待和核验Hanlin已安排的更新外，暂不补其他实验。项目侧Native-only、without-PAFA及Native-only独立HF/KAUH post-hoc均已完成并完整回收到本地，服务器结果不再是写作依赖。

当前优先级：
1. 修Figure 1/2及Section 2/3，落实训练/外评角色、符号/维度、loss、Arian C1/C4与指标定义。
2. 将PC-MCL/DCASE和两组归因消融、独立post-hoc写入适当表格与讨论，保持原生主结果和独立外评来源清楚。
3. 根据实际结果统一贡献与C2/C3回应；允许整理现有预测和逐seed统计，不新增训练、推理、特征提取或搜索。
4. 收拢Abstract/Conclusion、全文衔接和四页稿；Hanlin的PAFA与五组frozen references收到后单独核验、更新。

原时间窗口保留为本周期安排参考，当前按上述paper优先级继续。先前“等卡后启动”“缺反馈由项目补跑”等执行安排已结束，不再自动触发。独立reviewer仍仅由用户明确启动。

结果入口：`docs/result_exports/2026-09-19_lsaa_attribution/README.md`；完整本地资产保留于`result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/`，含694个原始文件及12个best/last checkpoint。本次Git保存682个非权重原始结果文件，约58.03MB；完整权重按既有政策保留本地/服务器。

## 本周期目标

今天9/18（周五）先把图件、协议说明与已有结果整理成连贯稿；明天9/19（周六）核验返回结果，补齐Arian回应、全文一致性与四页检查，形成可以交老师编辑的版本。图件必须由作者先修到正确完整，不把基础重画工作留给老师。

会议依据：<mention-page url="https://app.notion.com/p/3df309efda29810c84bdc20f5462b88f"/>
上一周期：<mention-page url="https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8"/>
组会中英文讲稿：<mention-page url="https://app.notion.com/p/3df309efda2981f5bf1ec12b9eea20eb"/>

**时间统一使用芝加哥CDT；纽约EDT＝芝加哥时间＋1小时。** 以下是项目内部工作窗口，含集中确认与缓冲，不是对Hanlin或服务器完成时间的承诺。若新结果提前返回，可在下一次结果处理窗口整合；写作不等待全部外部回报。

## 已完成基础与仍在等待的结果

- [x] PC-MCL 5 s适配三seed及四数据集固定评测完成，结果已核验。
- [x] DCASE-style三seed及四列评测完成，成功结果已拉回本地并归档。
- [x] 主方法、单源控制、Coarse SPR、Native+C/W与HF-on的现有三seed统计已具备；本轮重点是解释、核对和补写。
- [x] Arian原件与中文译文已保存，C1–C4已识别。
- [x] 组会PPT与中英文讲稿已完成，今天完整会议记录已保存。
- [x] Native-only与LSAA without PAFA两组三seed代码、必要检查与Git同步完成，GPU0/1正式队列已启动。
- [x] Native-only三seed原生结果已完成、核验并拉回：ICBHI 63.87±1.43%、SPR 88.93±3.36%。
- [x] LSAA without PAFA三seed及规定终点评测已完成、核验并拉回：ICBHI 58.60±1.60%、SPR 91.04±0.51%、HF CAS 86.66±1.83%、KAUH BA 78.30±1.95%。
- [x] 已收到Hanlin五组frozen references汇总并完成Table 1数值抄录核对：AST、BEATs、PANNs、OPERA-CT、HeAR，共20格一致；完整实验材料验收仍见T23。
- [x] 用户在写作任务另行批准的Native+C/W独立HF/KAUH三seed外评已完成并入稿：HF CAS 88.78±1.29%、KAUH patient BA 77.91±3.02%。已从保存预测复核指标、支持数和固定KAUH阈值；该新增结果尚未纳入9/19旧Git归档包。
- [ ] 明确并核验Hanlin的PAFA recipe、三seed含义、四列评测与原始产物。

## 今天9/18：图件与不依赖新数字的修订

### 16:00–16:30｜统一整体叙事与数据角色

负责人：用户、论文写作、论文写作二。集中确认后各任务按既有章节分工并行准备。

- [x] T01｜确定这轮overall论点与每章的作用：异构标注问题 → 共享声学监督 → 原生任务读出 → 联合训练与迁移证据 → 适用边界；贡献写研究价值与发现，不写成图表清单。
- [x] T02｜统一从第一页开始的数据角色：ICBHI＋SPR为核心训练，HF-train只在独立LSAA w/HF条件辅助；HF source-test/KAUH用于固定模型评测。Intro、Section 2及Figure 1 caption已明确；HF source-test对核心LSAA是外部数据，对w/HF是HF的held-out测试分区。图内标签余项见T03。

验收：形成一个可供两条写作任务共同使用的简短论点段落与数据角色说明；不出现“四个dataset默认一起训练”的暗示。

### 16:30–18:15｜修Figure 1/2；并行准备协议与指标说明

负责人：论文写作二统筹数据/方法图，论文绘图制作；论文写作并行准备Evaluation及表注。

- [ ] T03｜完成Figure 1图内角色标签和最终图件验收。9/20核对：正文/caption已明确core train、optional auxiliary与fixed evaluation；图内仍是Train + Test等原标签，A/B方案尚待用户选择。随后核对图例、标签关系、单位并接入定稿；描述性声学分析的范围保持不变。
- [x] T04｜修Figure 2：按实现逐项核对encoder、pooling、projection、三个heads、PAFA分支与HF可选分支；统一D与768的写法，定义T_i、H_i及attention输入输出，避免abnormality A与attention记号混淆。
- [x] T05｜核对Figure 2分类、患者正则和HF辅助路径及loss：当前图的分类/PAFA分支与可选HF C/W项清楚，D/d、H_i、Z_i与正文一致；概览图未展开的ICBHI门内fallback由Section 3明确，未发现图文矛盾。已核对现有预览及对应PDF；论文尺寸下的最终版面验收仍见T28。
- [x] T06｜准备Arian C1/C4协议段：清楚区分source内部validation阈值拟合与ICBHI official-test checkpoint选择；列清数据混合/源比例、batch组成、是否类别均衡、batch size、优化器、LR/schedule、epoch、增强、早停和划分。先核对现有配置，不更换评测协议。
- [x] T07｜补dataset-specific指标定义和引用：ICBHI Score与SPR official Score分别说明；明确ICBHI异常子类召回与SPR二分类召回的区别，HF列写CAS AUROC、KAUH列写patient BA。所有比较在同一数据集列内解释。

验收：图件符号/维度与代码、公式及caption一致；指标名和训练细节不再依赖读者猜测。图用可编辑源文件维护，导出后检查裁切、字距、箭头与实际论文尺寸下的可读性。

### 18:15–18:45｜集中看图并接入稿件

负责人：用户、论文写作二、论文绘图；共享main.tex/caption由论文写作执行局部修改。

- [ ] T08｜完成Figure 1/2最后导出稿核对与统一接入。9/20：Figure 2符号版PDF已接入唯一Overleaf目录，Section 3/caption已同步；Figure 1 caption已改，图内标签与最终导出仍依赖T03。本项保持未完成，不恢复DONE标记，不自动编译。

18:45–19:30预留晚餐与修图缓冲；若图仍有内容错误，优先用此窗口修好。

### 19:30–21:00｜把已完成结果写成论文

负责人：论文写作负责Section 4及Tables 1–3；管理提供结果来源。

- [ ] T09｜将PC-MCL/DCASE及LSAA系统比较按用户确认的新组织放入Table 2；Table 1保留五组frozen references。9/20：数字、caption、训练来源和系统比较讨论已入稿；选模差别仅概述，baseline输入构造及固定外评读出差别尚未完整写清，作为剩余协议说明保留。
- [x] T10｜按用户确认的新布局整理Table 3绝对值消融矩阵：单源/联合、Native-only/Native+C/W、without-PAFA、Coarse SPR与HF-on。三seed均值/sample SD已核对，正文保留Coarse的all-Normal run、阈值拟合/损失强度变化，以及HF-on跨任务取舍；旧Table 2三block/配对差值展示已由此布局替代。
- [x] T11｜同步Section 1/4中的比较解释：承认DCASE能够做异构多源masked learning，且在SPR/KAUH均值更高；PC-MCL的KAUH也不应被省略。把“可能学到更广声学结构”保留为讨论假设。
- [ ] T12｜PC-MCL成功结果的来源与轻量归档缺项已定位；成功三seed汇总已用于论文，但本地尚无PC_MCL_ICBHI5s_lr1e4_20260917原始结果目录。保留成功run的config/log/metrics/predictions与汇总归档待办；不接收旧失败文件，不重跑模型。

验收：已完成的两组方法baseline有可追溯表格与连贯结果段；训练来源/微调/正则/读出差异不被写成单因素因果。

### 21:00–22:15｜将Arian意见落入正文和回应清单

负责人：论文写作牵头，论文写作二负责Sections 2/3具体段落。

- [x] T13｜Arian C1/C4已落文：Section/3method.tex末段写明source比例、batch、优化器、增强、划分、阈值拟合与official-test选模；Section/4evaluation.tex开头定义各数据集Score、HF CAS AUROC与KAUH patient BA。具体修改位置已核对。
- [x] T14｜用已有归因消融完成C2的有边界正文回应：Table 1是frozen-reference系统比较，Table 2是多标签系统比较；Native-only/Native+C/W与without-PAFA/Full分别解释匹配因素和实际取舍，注明后端等差别。已明确不能把系统差异归因为单一结构；统计与最终逐点回复仍见T21/T22。
- [x] T15｜回应C3：已区分共享属性监督的价值与特定层级读出的必要性，说明Native+C/W需要共同重训encoder/native heads。新增固定native-head HF/KAUH三seed结果已入稿，承认其四列均值高于LSAA，不把它写成临时换头或纯readout因果。
- [x] T16｜本轮chat/comments已整理成连续正文，完成修改范围的衔接阅读并保存到Overleaf_Sync_final本地源码；本次集中核对记录剩余项。最新编译PDF及交稿版本验收另见T28/T29，不将源码保存等同于Git提交或最终PDF。

22:15–22:30预留收尾缓冲。今晚目标是图件正确、C1/C4落文、已有结果齐全、C2/C3解释形成工作稿；不以等待新结果为理由停住这些内容。

## 明天9/19：反馈、统计与全文收口

### 09:30–10:00｜接收信息，确定今日可用结果

负责人：管理；服务器任务保持待命、监视暂停，本科生对接任务核对用户实际转交材料。

- [x] T17｜核对当前回报状态：项目侧原有实验/Native-only独立外评完整；Hanlin的result(3).xlsx五组四列汇总已收到并入Table 1，逐格抄录一致；完整原始材料未验收，PAFA仅有一列初步汇总且暂不入稿。不再等待服务器或轮询SSH。

### 10:00–11:30｜整合新增消融；未返回时推进已有正文

负责人：论文写作、管理；必要的结果分析由原执行任务承担。

- [x] T18｜Native-only三seedICBHI/SPR及独立HF/KAUH已写入Table 3并与Native+C/W比较。正文写明固定ICBHI原生头外评；保留独立post-hoc来源与原主汇总，未监督C/W头不作属性指标。
- [x] T18a｜Native-only独立zero-target HF/KAUH post-hoc三seed完成并拉回：HF CAS AUROC 81.62±3.69%、KAUH patient BA 75.59±5.62%。固定ICBHI四分类头，HF三窗max[P(W)+P(Both)]，KAUH B/D/E平均[1−P(N)]且>0.5判异常；957条HF/86位KAUH患者，原生主汇总未改。
- [x] T19｜LSAA without PAFA三seed已按用户确认的四列布局写入Table 3及讨论，与Full比较ICBHI/SPR/HF CAS/KAUH。C/W原始统计保留为分析资料，本轮不要求在四列矩阵新增一列；正文说明PAFA的收益依赖任务及CUDA/MPS差别。
- [x] T20｜已核对新旧参照的划分、loss/选模与CUDA/MPS后端差别，保存逐seed值及mean/sample SD；六个主run、三个post-hoc均完整。已纠正早前将末轮Score误报为selected-checkpoint Score的问题，后续以最终原始summary为准。

9/20复核：T18/T19已实际入稿并核对，现勾选完成；Native+C/W新增外评也已用于Table 3。原始结果与统计保持各自来源，不改写原生主汇总。

### 11:30–12:15｜完成Arian统计问题的回应

负责人：管理与原结果分析任务核对，论文写作落文。

- [ ] T21｜统计附注仍未完成：用原始逐seed配对值准备效应和不确定性说明，先确认比较清单与假设再做必要配对统计。现稿有mean/sample SD但无配对检验结果；仅三个seed，不从四舍五入均值反推p值，不将不显著当等效，不将同患者cycle/BDE当独立患者。
- [ ] T22｜最终Arian逐点回复尚未收口。9/20：C1/C4正文完成，C3主要问题已回应，C2已增加匹配对照及适用边界，但完整因果隔离不能据此宣称解决；统计附注及最终“意见—修改位置—回应”仍待整理。

12:15–13:00预留午餐与分析缓冲。

### 13:00–14:30｜Hanlin结果处理窗口与结果缓冲

负责人：本科生单数据集实验对接、管理核验；论文写作更新Table 1。

- [ ] T23｜Hanlin五组frozen references数值更新已收到：result(3).xlsx的20个mean/SD单元格与Table 1一致。仍待核对真实seed、5 s配方、权重、split、checkpoint选择、逐样本/患者输出及原始日志；完成数值回填不等于完整实验验收。
- [ ] T24｜Hanlin PAFA仍待完整核验。目前仅收到BEATs+PAFA (frozen)的ICBHI 50.83±1.33%，另外三列及recipe/seed含义/原始产物未齐；用户决定暂不入稿。收到后再确认frozen/fine-tuned、目标监督和fixed-transfer角色。
- [x] T25｜本轮已按实际完整/部分回报推进paper，Hanlin材料核验保持独立依赖；不设个人硬截止，不因计划到期自动由项目补跑，是否新增执行另由用户决定。

### 14:30–16:00｜全文连贯性与Abstract/Conclusion

负责人：论文写作牵头，论文写作二核对数据与方法接口。

- [x] T26｜完整通读Sections 1–5，检查问题、贡献、数据角色、符号、方法、结果与解释是否互相支撑；删除拼接痕迹和重复论点。
- [x] T27｜根据已落定的正文与实际可用结果更新Abstract/Conclusion；不把尚未完成实验写成发现，不声称所有数据集SOTA或某一readout必需。

### 16:00–17:00｜四页与最终图表检查

负责人：用户在Overleaf编译，写作任务据用户提供的编译稿修订。

- [ ] T28｜用户在Overleaf检查正文四页、图表可读性、caption/引用、浮动位置、符号、作者和内部注释；对篇幅问题优先删冗余文字，不以缩小图中文字掩盖问题。未经用户另行要求，不自动本地LaTeX编译。
- [ ] T29｜源码中的Figure 1/2、Tables 1–3及新增数字已做一致性核对；最终交稿PDF/源码版本说明仍待Figure 1余项与用户最新版Overleaf编译稿。保留最新四页、浮动、caption/引用及版面检查，不以现有图件单页预览代替全文验收。

### 17:00–18:00｜缓冲、集中确认与交稿准备

负责人：用户与管理，相关写作任务完成已确认修订。

- [x] T30｜本轮用户与写作任务的最后修改、全文术语/数字复核及Native+C/W新外评回填已完成；未返回的Hanlin完整材料和最终版面继续单列。本项仅表示本轮修订处理完成，不代表最终投稿版本。
- [ ] T31｜在Paper和figures最终确认后，准备给老师的简短最终编辑通知草稿，附版本与主要修改；目前尚无本轮草稿。下一步先进行9/20上午合作者meeting并收修正反馈，再确定交稿通知内容；不自动发送邮件。
- [x] T32｜9/20完成本轮管理集中复核：根据用户与两项写作任务的核对、当前源码/图件及结果材料更新checkbox，记录真实剩余项和负责人，并加入今早合作者meeting。Hanlin缺项不自动转成项目补跑，独立reviewer保持用户控制。

## 章节所有权与依赖

- 论文写作：Sections 1/4/5、Abstract、Tables 1–3及main.tex/citation.bib等共享文件。
- 论文写作二：Sections 2/3与对应图表，直接协调论文绘图；共享caption/宏/引用改动交论文写作局部整合。
- 论文绘图：按章节负责人确认的brief修图，交付可编辑源和导出图。
- Acoustic服务器：本轮训练和独立post-hoc已完成，保持待命与监视暂停；不启动新训练/推理、不等卡自动开跑、不自动接手Hanlin缺项。按用户需要提供已有结果。
- 本科生单数据集实验对接：处理用户转交的Hanlin材料，核验协议与原始结果；不假设已经联系或已经收到新结果。
- 管理：维护paper优先级、checkbox与结果来源，按用户授权向既有写作任务同步；9/19旧结果Git快照已完成，新稿与Native+C/W新增结果仍需用户另行要求后提交。不新增实验，不自动激活独立reviewer。

## 本周期完成标准

- [ ] 图件内容完整、维度/符号/箭头正确，作者能够解释每个组件，老师无需代为重画。
- [ ] Arian C1/C4已落文；C2/C3与统计回应有清楚依据，待返回结果明确列出。
- [ ] 已完成PC-MCL/DCASE与现有ablation有准确表格、协议和连贯讨论。
- [ ] Hanlin和服务器返回的完整结果经过核验再入稿；所有未完成项有明确后续负责人。
- [ ] 全文及Abstract/Conclusion一致，用户确认四页编译稿后再准备交给老师。
