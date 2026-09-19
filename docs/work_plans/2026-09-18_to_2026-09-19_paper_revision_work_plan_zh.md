# 2026-09-18–09-19 Working Plan：组会后修稿、修图与结果整合

Notion：[当前工作计划](https://app.notion.com/p/3df309efda2981a68624c85b689f596b)。

## 9/19最新决定：结果归档，转入paper work

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
- [ ] 收到并核验Hanlin的五组frozen references更新：AST、BEATs、PANNs、OPERA-CT、HeAR。
- [ ] 明确并核验Hanlin的PAFA recipe、三seed含义、四列评测与原始产物。

## 今天9/18：图件与不依赖新数字的修订

### 16:00–16:30｜统一整体叙事与数据角色

负责人：用户、论文写作、论文写作二。集中确认后各任务按既有章节分工并行准备。

- [ ] T01｜确定这轮overall论点与每章的作用：异构标注问题 → 共享声学监督 → 原生任务读出 → 联合训练与迁移证据 → 适用边界；贡献写研究价值与发现，不写成图表清单。
- [ ] T02｜统一从第一页开始出现的数据角色：ICBHI＋SPR为核心训练；HF-train只在独立HF-on条件辅助；HF-test与KAUH用于固定模型外评。让Intro、Section 2与Figure 1第一次出现数据集时就交代清楚。

验收：形成一个可供两条写作任务共同使用的简短论点段落与数据角色说明；不出现“四个dataset默认一起训练”的暗示。

### 16:30–18:15｜修Figure 1/2；并行准备协议与指标说明

负责人：论文写作二统筹数据/方法图，论文绘图制作；论文写作并行准备Evaluation及表注。

- [ ] T03｜修Figure 1：明确train / optional auxiliary / external evaluation，HF训练与测试角色分开；检查标签关系、图例、单位和caption，保留描述性声学分析的范围。
- [ ] T04｜修Figure 2：按实现逐项核对encoder、pooling、projection、三个heads、PAFA分支与HF可选分支；统一D与768的写法，定义T_i、H_i及attention输入输出，避免abnormality A与attention记号混淆。
- [ ] T05｜补Figure 2各箭头与loss含义：分类路径和患者正则路径清楚；HF只进入其批准的C/W辅助项；ICBHI异常门内C/W均未过阈值时的fallback与正文一致。
- [ ] T06｜准备Arian C1/C4协议段：清楚区分source内部validation阈值拟合与ICBHI official-test checkpoint选择；列清数据混合/源比例、batch组成、是否类别均衡、batch size、优化器、LR/schedule、epoch、增强、早停和划分。先核对现有配置，不更换评测协议。
- [ ] T07｜补dataset-specific指标定义和引用：ICBHI Score与SPR official Score分别说明；明确ICBHI异常子类召回与SPR二分类召回的区别，HF列写CAS AUROC、KAUH列写patient BA。所有比较在同一数据集列内解释。

验收：图件符号/维度与代码、公式及caption一致；指标名和训练细节不再依赖读者猜测。图用可编辑源文件维护，导出后检查裁切、字距、箭头与实际论文尺寸下的可读性。

### 18:15–18:45｜集中看图并接入稿件

负责人：用户、论文写作二、论文绘图；共享main.tex/caption由论文写作执行局部修改。

- [ ] T08｜逐张核对Figure 1/2导出稿，把确认版本接入唯一Overleaf同步目录；同步Section 2/3解释和caption。完成不恢复DONE标记，不自动本地编译。

18:45–19:30预留晚餐与修图缓冲；若图仍有内容错误，优先用此窗口修好。

### 19:30–21:00｜把已完成结果写成论文

负责人：论文写作负责Section 4及Table 1/2；管理提供结果来源。

- [ ] T09｜整合已完成PC-MCL与DCASE结果到Table 1的方法对比分块，更新caption和baseline说明；区分frozen references、fine-tuned方法和LSAA，注明训练来源、选模、输入及固定外评读出的差别。
- [ ] T10｜整理Table 2现有三个block：单源/联合、细粒度监督/读出、HF-on；核对mean、sample SD与配对参照，保留Coarse SPR单seed崩塌与HF-on跨任务取舍，避免把口述简写直接当实验定义。
- [ ] T11｜同步Section 1/4中的比较解释：承认DCASE能够做异构多源masked learning，且在SPR/KAUH均值更高；PC-MCL的KAUH也不应被省略。把“可能学到更广声学结构”保留为讨论假设。
- [ ] T12｜整理PC-MCL成功结果的轻量本地归档需求与原始路径；只接收成功run的config/log/metrics/predictions及汇总，不拉回旧失败文件，不因此重跑模型。

验收：已完成的两组方法baseline有可追溯表格与连贯结果段；训练来源/微调/正则/读出差异不被写成单因素因果。

### 21:00–22:15｜将Arian意见落入正文和回应清单

负责人：论文写作牵头，论文写作二负责Sections 2/3具体段落。

- [ ] T13｜把T06/T07准备好的C1/C4内容真正写入对应章节和表注；清单记录修改位置，做到“反馈 → 修改 → 可核对段落”。
- [ ] T14｜用已返回结果回应C2：Table 1保持系统比较定位；Native-only/Native+C/W与without-PAFA/Full分别解释其匹配因素和实际取舍，不写成所有任务都获益，不新增实验追求某种结论。
- [ ] T15｜回应C3：将共享属性监督的价值与特定层级读出的必要性分开；准确描述Native+C/W的训练方式与现有优势，不宣称临时接一个未训练native head即可恢复性能，不宣称未测得的外部迁移优势。
- [ ] T16｜将零散chat/comments合并为连续、可直接编辑的段落，做一遍今天修改范围的衔接阅读，保存明确版本与未完成项。

22:15–22:30预留收尾缓冲。今晚目标是图件正确、C1/C4落文、已有结果齐全、C2/C3解释形成工作稿；不以等待新结果为理由停住这些内容。

## 明天9/19：反馈、统计与全文收口

### 09:30–10:00｜接收信息，确定今日可用结果

负责人：管理；服务器任务仍自行监视，本科生对接任务核对用户转交材料。

- [ ] T17｜服务器结果已齐，继续处理实际收到的Hanlin材料，明确完整/部分/未返回；不再等待服务器或追加SSH进度轮询。

### 10:00–11:30｜整合新增消融；未返回时推进已有正文

负责人：论文写作、管理；必要的结果分析由原执行任务承担。

- [ ] T18｜将已验收的Native-only原生三seed写入Table 2，与Native+C/W比较ICBHI/SPR；原主结果不变，未监督C/W不作为属性指标；独立HF/KAUH结果注明与主汇总的区别。
- [x] T18a｜Native-only独立zero-target HF/KAUH post-hoc三seed完成并拉回：HF CAS AUROC 81.62±3.69%、KAUH patient BA 75.59±5.62%。固定ICBHI四分类头，HF三窗max[P(W)+P(Both)]，KAUH B/D/E平均[1−P(N)]且>0.5判异常；957条HF/86位KAUH患者，原生主汇总未改。
- [ ] T19｜将已验收的LSAA without PAFA三seed及native、C/W、HF CAS、KAUH结果写入表格和讨论，与正式Full比较；据实际结果完成C2回应，不新增补跑。
- [x] T20｜已核对新旧参照的划分、loss/选模与CUDA/MPS后端差别，保存逐seed值及mean/sample SD；六个主run、三个post-hoc均完整。已纠正早前将末轮Score误报为selected-checkpoint Score的问题，后续以最终原始summary为准。

项目侧结果已完整可用，T18/T19剩余工作是入稿而非等待实验；这两项在实际写回稿件并核对前保持未勾选。

### 11:30–12:15｜完成Arian统计问题的回应

负责人：管理与原结果分析任务核对，论文写作落文。

- [ ] T21｜用原始逐seed配对值准备效应与不确定性说明，优先Native+C/W对Full及本次完整返回的新对照；确认比较清单和假设后计算必要的配对统计。明确只有三个seed的局限，不从四舍五入均值反推p值，不把不显著当等效，也不把同一患者的cycle/BDE视为独立患者。
- [ ] T22｜更新Arian C1–C4及统计附注的逐点回应，给出最终修改位置，区分已完成回应与仍依赖结果的项。

12:15–13:00预留午餐与分析缓冲。

### 13:00–14:30｜Hanlin结果处理窗口与结果缓冲

负责人：本科生单数据集实验对接、管理核验；论文写作更新Table 1。

- [ ] T23｜收到五组frozen references后，核对5 s设置、真实seed、权重来源、split、checkpoint选择、逐样本/患者输出和四列指标，再回填Table 1及受影响比较文字。
- [ ] T24｜收到PAFA后先核对是三个源模型seed还是固定encoder下游seed，并确认fine-tuned/frozen、目标监督与fixed-transfer角色；名称与表格定位随真实协议，不能混作原法三seed复现。
- [ ] T25｜Hanlin未反馈时记录缺项与旧结果的证据范围，将处理窗口用于paper/figure修订；不设个人硬截止，也不因本周期结束自动由项目侧补跑。未来是否新增执行另由用户决定。

### 14:30–16:00｜全文连贯性与Abstract/Conclusion

负责人：论文写作牵头，论文写作二核对数据与方法接口。

- [ ] T26｜完整通读Sections 1–5，检查问题、贡献、数据角色、符号、方法、结果与解释是否互相支撑；删除拼接痕迹和重复论点。
- [ ] T27｜根据已落定的正文与实际可用结果更新Abstract/Conclusion；不把尚未完成实验写成发现，不声称所有数据集SOTA或某一readout必需。

### 16:00–17:00｜四页与最终图表检查

负责人：用户在Overleaf编译，写作任务据用户提供的编译稿修订。

- [ ] T28｜用户在Overleaf检查正文四页、图表可读性、caption/引用、浮动位置、符号、作者和内部注释；对篇幅问题优先删冗余文字，不以缩小图中文字掩盖问题。未经用户另行要求，不自动本地LaTeX编译。
- [ ] T29｜检查Figure 1/2、Table 1/2与所有新增数字的最终版本一致；形成可交老师编辑的PDF/源码版本说明及剩余依赖清单。

### 17:00–18:00｜缓冲、集中确认与交稿准备

负责人：用户与管理，相关写作任务完成已确认修订。

- [ ] T30｜处理最后一轮修改及晚到的完整结果；没有返回的结果保持未完成，并明确其对最终稿的影响，不因为周期结束虚构完成。
- [ ] T31｜Paper和figures经用户确认后，准备给老师的“版本已整理好，请最终编辑”的简短通知草稿，附版本和主要修改说明，由用户确认/发送。本计划不自动发送邮件或联系合作者。
- [ ] T32｜18:00完成内部收口：记录已完成checkbox、真实未完成项、由项目接手的外部缺项及下一步；独立reviewer仅在用户明确要求后启动，不因时间块自动激活。

## 章节所有权与依赖

- 论文写作：Sections 1/4/5、Abstract、Table 1/2及main.tex/citation.bib等共享文件。
- 论文写作二：Sections 2/3与对应图表，直接协调论文绘图；共享caption/宏/引用改动交论文写作局部整合。
- 论文绘图：按章节负责人确认的brief修图，交付可编辑源和导出图。
- Acoustic服务器：本轮训练和独立post-hoc已完成，保持待命与监视暂停；不启动新训练/推理、不等卡自动开跑、不自动接手Hanlin缺项。按用户需要提供已有结果。
- 本科生单数据集实验对接：处理用户转交的Hanlin材料，核验协议与原始结果；不假设已经联系或已经收到新结果。
- 管理：完成结果Git快照，维护paper优先级和完成状态，向既有写作任务同步已验证材料；不新增实验任务，不自动激活独立reviewer，遵守用户控制的写作同步规则。

## 本周期完成标准

- [ ] 图件内容完整、维度/符号/箭头正确，作者能够解释每个组件，老师无需代为重画。
- [ ] Arian C1/C4已落文；C2/C3与统计回应有清楚依据，待返回结果明确列出。
- [ ] 已完成PC-MCL/DCASE与现有ablation有准确表格、协议和连贯讨论。
- [ ] Hanlin和服务器返回的完整结果经过核验再入稿；所有未完成项有明确后续负责人。
- [ ] 全文及Abstract/Conclusion一致，用户确认四页编译稿后再准备交给老师。
