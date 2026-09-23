# 2026-09-21 Notion Working Plan：9/22归档前快照

来源：https://app.notion.com/p/3e2309efda2981d4927cf7476ef18691

## 今日目标与执行顺序
日期：2026-09-21（周一）。本计划取代9/18–9/20周期，以今天与Yinuo的会议为依据；原计划归档保留，未完成项不会因归档自动勾选。
会议记录：<mention-page url="https://app.notion.com/p/3e2309efda2981f2b46fc0ea762b320e"/>。
目标：先确认完整主方法及每组结果究竟比较什么，形成一致的Figure 2、Method和Tables 1–3；核清SD/SEM并完成可支持的统计分析，再修改Results，最后统一Abstract/Conclusion。Introduction只处理受方法/结论变化影响的局部内容。
建议内部窗口为今天17:45–23:30 CDT（纽约EDT加1小时）。时间块用于安排顺序与并行工作，不是对合作者/本科生的硬截止；未能在窗口内解决的科学问题保持明确待定，不以到点视为完成。
本次操作仅建立会议记录、归档旧计划并制定新计划；没有直接改写稿件、自动启动写作/绘图/reviewer或模型执行。后续仍由用户在对应写作任务中讨论后落实。
## 已具备的基础
- [x] B01｜本次三段会议材料已合并为26节完整记录，会议原话与会后核对分开保存。
- [x] B02｜五篇ICBHI论文PDF及原文/官方代码选模核查已完成，材料位于docs/literature/2026-09-21_icbhi_checkpoint_selection/；“资料准备好”不等于已经发送给Yinuo。
- [x] B03｜Native+C/W已有三seed原生与固定HF/KAUH结果，可作为新main候选的现有数据基础；本轮不需要为“改主方法名称”重训这组模型。
## 先处理三个会影响后续修改的决定
1. **主方法及消融对应关系。** 会议方向为Native+C/W提升为完整主方法，但Jingping的确认未记录。先做下方配置映射，再决定名称、旧分支位置与去重；不能只换表中名称。
2. **SD/SEM。** 会议口述当前为SEM，但本地Section 4写sample SD，已有原始汇总也有sample_std/sample_sd。先复算均值、sample SD（ddof=1）与SEM（SD/√n），核对n/单位，再决定正文显示哪一种。不得保留同一“±”数字却只换标签。
3. **ICBHI选模。** 文献核查目前只有released-code层面的test-selection证据，四篇论文未明确写该规则。必须交给Yinuo/Jingping讨论，不能预设“别人也做，所以只改措辞”。如果决定改协议，先确定受影响实验与预算，再由用户另行决定是否运行。
## 新旧配置与结果的对应表（待N01/N02最终确认）
<table fit-page-width="true" header-row="true">
<tr>
<td>现有配置/别名</td>
<td>本轮角色</td>
<td>已知数据/结构</td>
<td>写作限制</td>
</tr>
<tr>
<td>Native+C/W（现名LSAA-N w/C+W）</td>
<td>新main候选：两个native heads＋共享C/W监督</td>
<td>ICBHI 63.67±1.67；SPR 90.86±1.21；HF 88.78±1.29；KAUH 77.91±3.02（当前记录为sample SD）</td>
<td>沿用已完成run；最终主名待N01确认。</td>
</tr>
<tr>
<td>Native-only（现名LSAA-N）</td>
<td>新main的无C/W监督对照候选</td>
<td>同为native heads，保留native CE的1/3系数</td>
<td>结构匹配；仍需核对CUDA/MPS、split、selection与其他recipe差别。</td>
</tr>
<tr>
<td>旧LSAA / Full</td>
<td>explicit A/C/W readout配置</td>
<td>已有三seed，不因主方法改名而失效</td>
<td>保留为框架内另一种实例化；不复制为新main第二行。</td>
</tr>
<tr>
<td>ICBHI-only / SPR-only</td>
<td>旧explicit分支的单源控制</td>
<td>已经完成，但并非新native-head结构的单源run</td>
<td>可以说明该分支的joint/source对比；不能将新main减它们直接称为单因素联合训练收益。</td>
</tr>
<tr>
<td>Coarse SPR / without-PAFA</td>
<td>旧explicit分支的监督/患者正则分析</td>
<td>不等于新native-head main去掉同一组件</td>
<td>保留真实结构和原参照；Coarse另有阈值来源及相对loss强度变化。</td>
</tr>
<tr>
<td>HF-on</td>
<td>第三来源扩展的边界实验</td>
<td>现有结果对应旧explicit分支＋HF辅助</td>
<td>按会议决定保留；说明实际分支，不写成新native-head main的已测HF扩展。</td>
</tr>
</table>
## 17:45–18:20｜先确认主方法、表格组织与协议问题
负责人：用户＋论文写作；论文写作二核对方法结构，管理整理证据。
- [x] N01｜确认最终主方法采用Native+C/W以及全文主名，记录与Jingping确认的实际状态。验收：一句研究价值陈述＋完整配置定义，明确两个native task heads、共享C/W辅助监督及PAFA角色，避免沿用旧“三个A/C/W头”的主模型描述。
- [x] N02｜逐行核对Table 3的真实config、训练源、head、loss、readout、split与选模；依据上表将新main、最接近匹配的消融及旧explicit分支分析分清。删除的是重复展示行，不删除原实验资产；旧单源/Coarse/without-PAFA/HF-on不能直接伪装为新main的单因素消融。
- [x] N03｜把已准备的PAFA/Patch-Mix原文位置与代码证据作为核心材料，附SG-SCL/PC-MCL及独立validation反例，由用户转交Yinuo；记录她的反馈。此项是发送/讨论待办，当前未代发消息。
验收：先有主方法和比较对象，再改表/图；ICBHI协议问题有明确待决项和负责人。
## 18:20–19:15｜确定数字与比较清单，先形成Tables工作稿
负责人：管理核对来源，论文写作负责Section 4/Tables；方法结构由论文写作二核对。
- [x] N04｜建立一次性的逐seed结果矩阵：新main及全部保留对照，分别列native/HF/KAUH的原summary或预测来源、seed、support、指标、单位和选模方式；区分原生结果与独立post-hoc。原始材料不全的Hanlin/PC-MCL项明确缺项，不从汇总小数反推seed值。
- [x] N05｜按用户确认的最终main更新Tables 1/2/3的行名、mean与比较排序工作稿；移除重复Native+C/W展示，保留旧explicit实例化及HF边界实验的真实参照。先不加未经检验的significance stars，不沿用旧主方法的3.39/2.12 pp等差值作为新main结论。
- [x] N06｜确定每张表回答的问题。Table 1是完整系统对frozen references，不能隔离fine-tuning；Table 2分别说明DCASE的多源native-class union＋mask和PC-MCL的ICBHI-only迁移；Table 3按实际匹配程度分析组件，明确后端与recipe差别。
验收：每一行名称能对应真实run；缺失的匹配消融先改组织/限定结论，不自动新增训练。
## 19:15–20:15｜核对SD/SEM并完成统计分析与展示决定
负责人：管理整理分析依据，结果分析由具备原始资料的执行任务承担；论文写作负责最终表注和文字。
- [x] N07｜从真实逐seed值复算mean、sample SD、SEM和n；核对当前稿“sample SD”与会议“SEM”的差异。Hanlin五组frozen references需确认其统计量实际含义，不能假定所有“±”都是同一口径。
- [ ] N08｜先列出核心比较、对应科学问题、统计单位与配对条件，再选择合适检验及多重比较处理。优先新main对Native-only、native/explicit实例化，以及各旧分支内部的有效对照；不直接对整张表机械套ANOVA，不把同患者多条cycle/BDE当独立患者。
- [ ] N09｜在已有原始数据支持的范围计算差值、不确定性和检验结果；报告三seed限制和无法可靠检验的项。没有原始数据或合理统计条件时，保留描述性均值/变异，不虚构p值或把“未检验”写成“不显著”。
- [ ] N10｜与用户决定最终mean±SD或mean±SEM，统一表注、n、比较基准、检验说明及显著性标记。Bold只表示numerical best；是否加星必须来自预先明确的比较和实际检验。误差条重叠不替代检验，不显著不等于等效。
验收：同一表中的统计口径可解释；均值优势、显著差异与观察到的变异被准确区分。
## 18:20–21:00并行｜Figure 2/Method与Section 2/Figure 1
负责人：论文写作二＋论文绘图；共享main.tex/caption由论文写作局部整合。须在用户讨论确认N01/N02后开始具体改写。
- [x] N11｜先由方法负责人给Figure 2一致的brief：新main的shared encoder/256维表示、ICBHI四类及SPR二类native heads、共享C/W训练辅助与患者正则；HF/KAUH的固定native概率读出写清。旧HF-on实验的分支来源明确说明，不能在新main图中暗示已验证一个不存在的native-head HF-on run。
- [x] N12｜按新main实际实现更新Method结构、classification loss与训练/推理接口；旧A门控readout只作为确实保留的variant解释。处理Yinuo标出的abnormal broad class、label relationship、optional supervision段落，确认“可用标注”和“原生输出”之间的关系。
- [x] N13｜Section 2统一说明HF/KAUH的subset与数据角色：HF 957 eligible池及Other不等于Normal；KAUH 258 recordings/86 compatible patients及B/D/E相关性；HF-on的held-out source-test身份单列。体量/负标注缺失/分布差异作为设计理由或解释假设，不写成已验证的性能下降原因。
- [x] N14｜Figure 1C纵轴补Frequency (Hz)，caption明确呈现的是median spectral centroid；一并确认遗留的训练/可选辅助/外评角色标签。接入唯一Overleaf目录并核对图、正文和caption对应，保留可编辑PPT源，不自动编译。
验收：Figure 2、Method、表格中的“main”指同一个已完成配置；Figure 1角色/单位清楚。
## 21:00–22:15｜围绕最终Tables重写Results，并处理合作者意见
负责人：论文写作主导Sections 4.1/4.2；用户处理Overleaf中的Yinuo comments/changes。
- [ ] N15｜重写Section 4.1：先说明完整模型解决的任务覆盖问题，再据最终Table 1/2陈述native与external结果。分别交代DCASE与PC-MCL的参照角色，避免把系统差别说成输出结构的单因素优劣。
- [ ] N16｜重写Section 4.2：按有效对照讨论共享监督、native/explicit实例化、patient regularization和HF扩展边界；区分numerical difference、已检验的显著差异与未能确定的变化。较小SD/SEM只先描述观察到的variation；“stable/generalizes better”需要对应证据。
- [ ] N17｜汇总并处理Yinuo对Dataset/Method的实际修改与comments；accept/reject由用户确认，未收到反馈标待返回。同步更新Arian C1–C4回应位置，使新方法定义、选模说明与归因解释一致。
验收：Results按已经确认的数字和统计写；没有先写结论再挑比较的情况。
## 22:15–23:30｜条件具备后统一首尾、全文及四页稿
负责人：论文写作＋用户；论文写作二核对方法接口。
- [ ] N18｜在main/表格/统计/协议表述稳定后，更新Abstract与Conclusion；Introduction只改受影响的配置/贡献表述。统一“异构标注→共享声学监督＋原生任务结构→联合学习/外评证据→边界”的论点。
- [ ] N19｜完整核对Figures 1/2、Tables 1–3、所有主方法/variant名称、数值/差值及引用；刷新9/20术语索引受影响条目。用户在Overleaf检查正文四页和最终版面，写作任务依据该编译稿修订；没有新版PDF时此项保留待办。
- [ ] N20｜记录ICBHI选模的共同决定：若保留当前benchmark，准确说明released-implementation依据、内部validation用途与test-selection限制；若决定调整，先列范围/现有可用checkpoint/额外成本，再单独获用户执行决定。尚未决定时不把全文标成最终可提交。
- [ ] N21｜今日收口核对实际完成项、待反馈项与下一步，保存可交合作者的版本说明及通知草稿；是否发送由用户决定。本页checkbox按实际交付/确认勾选，不按时间自动完成。
## 独立依赖与不阻塞写作的支线
- [ ] D01｜Hanlin五组frozen references：收到汇总并入表不等于完整验收，继续核对真实seeds、SD/SEM、5 s设置、checkpoint/split、各列目标监督与原始输出。PAFA仅有ICBHI初步列，完整材料未齐；不设个人硬截止、不自动接手补跑。
- [ ] D02｜成功PC-MCL轻量原始材料及最新post-hoc归档：缺少做统计所需的原始文件时只列具体缺项，后续按用户授权取回/保存；不拉失败run，不据此重跑。Git提交/push不由本计划自动触发。
- [ ] D03｜SPR4/HF-DAS探索实验：脚本及必要检查已准备，但正式执行被自动审批按旧只读阶段拒绝，等待用户在拒绝后确认。当前没有新结果；本次会议记录/规划不替代该确认。将来完成后再决定是否入paper；SPR test Both仅1事件/1患者，不能以1–2 pp差距直接宣称稳定适配成功，也不现在安排挤掉Figure/Table。
## 上一计划遗留项如何处理
- 旧T33的合作者meeting实际于9/21完成并形成此纪要；旧日期保留为历史计划，不写成9/20已准时完成。
- 旧T03/T08（Figure 1/集成）→ N13/N14/N19。
- 旧T09（baseline协议说明）→ N04/N06/N15；旧T12（PC-MCL原始归档）→ D02。
- 旧T21/T22（统计/Arian回应）→ N07–N10/N17；旧T23/T24（Hanlin/PAFA）→ D01。
- 旧T28/T29（四页/最终版本）→ N18–N21；旧T31（通知草稿）→ N21。
- 旧计划中已完成的Method/Figure 2/表格/Abstract检查属于旧主方法版本；本轮只重开受主方法切换影响的具体内容，不抹掉此前完成记录。
## 章节所有权与执行边界
- 论文写作：Sections 1/4/5、Abstract、Tables 1–3，以及main.tex/citation.bib/共享caption。
- 论文写作二：Sections 2/3及其对应图表，直接协调论文绘图；只向主写作任务提供所需共享文件的局部改动。
- 管理：会议/计划、结果来源与依赖；文献任务本轮原文核查已完成。Yinuo/Jingping的意见以用户实际转交为准，不自动对外发消息。
- 唯一稿件为main下docs/paper/Overleaf_Sync_final/；旧稿保留为历史资产。依照用户既有流程先讨论后改稿，不恢复DONE标记，不自动编译或唤醒独立reviewer。计划中列实验问题不等于启动许可。
