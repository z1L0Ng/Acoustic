# 2026-09-12（周六）Working Plan：Paper论点复盘、多seed补证与4页稿

Notion：[9/12周六Working Plan](https://app.notion.com/p/3d9309efda2981e0bae7e27f069c5366?pvs=204)。

日期：2026-09-12，周六；时区：America/Chicago。用户已确认本地串行队列可延长到9/13（周日）。本页是新的当前计划，旧日期计划保留为历史。

当前暂停状态（9/13 15:20 America/Chicago）：用户明确暂停“论文绘图”和“论文写作”任务，两项均已确认暂停，等待用户明确恢复。保留当前已落盘稿件、PPT及图件；不继续修改、导出、渲染、集成或自动回填新结果。两任务确认没有遗留的本任务后台作业。本地训练进程、剩余有限队列及既有训练授权保持不动，不因本次暂停改变训练安排。

最新交付（9/13）：Figure2原生可编辑PPT首稿已完成并集成`docs/paper/Overleaf_Sync_final/Figure/figure2_method.pptx/.pdf/.png`，对应caption已同步。图参考旧roadmap，以当前Section2/3与四源角色为准，尺寸178×57mm，文字9–11pt；PPT含原生形状、文字和连接线，不含整图图片。可复算JS位于`figures/paper/figure2_method_ppt.mjs`。后续按作者反馈改版，Figure1和整篇压缩保留为后续待办，训练继续原队列。

最新执行指示（9/13）：用户要求“正常继续跑就行”，剩余SPR-only0→IC-only1→SPR-only1按原有限队列继续完成，原周日窗口改为时间估计，不因预计超过今晚暂停这批已批准任务。用户同时批准写作任务立即将已完成结果写回`docs/paper/Overleaf_Sync_final/`，统一Table2及受影响的协议、结果解释、Introduction、Abstract和Conclusion。本轮完成后一次性交接；不本地编译，不新增实验或Git操作。后文9/12及10:45的窗口限制保留为历史记录，以本条最新指示为准。

执行进度（9/13 10:45 America/Chicago核对）：新增完成5/8，P0完成4/4，P1完成1/4。Coarse0/1、Native0/1及IC-only0均完整结束；SPR-only0由Python5792、continuation64304正常运行，完整epoch10/update3260，epoch11真实update已至3552。IC-only1和SPR-only1未启动。当前Full/Coarse/Native各n=3、IC-only n=2、SPR-only n=1；运行中的结果不计入汇总。

科学解释更新：Native三seed的IC/SPR/CW为63.67±1.67/90.86±1.21/97.27±0.14%，同seed Full配对均值差+2.50/+0.16/−0.05pp；IC三seed均提高，但SPR seed1下降1.49pp，不能继续说两项native任务均稳定提高。Coarse三seed为54.66±4.05/67.23±36.62/88.84±8.59%，包含seed0退化，不能将seed42“二分类几乎不变”推广。继续已批准的P1，不改早停、不删不利结果、不补跑替换。论文Table2/Abstract/Conclusion仍沿用seed42控制描述，尚待用户讨论后更新。

时间预算更新：IC-only0实际408.09min（40epochs），超过最初按seed42估计的257min；后续IC-only1和SPR-only1按既有完整run约需9–12h，另加当前SPR-only0剩余时间。逐run检查周日窗口，不能承诺8/8在周日晚全部结束。本次未改变训练配方或启动额外实验。

授权更新（9/12）：用户已明确批准本地训练任务完成缺失多seed实验，包含必要的runner准备、分区核对及本地正式训练/评测/汇总。先执行P0四项，再按周日剩余窗口推进P1四项；已完成seed42与Full三seed不重跑。若新增运行预计超出9/13窗口，继续前回报时间需求。本次授权不包含新的Full/HF变体、其他模型、图文改动、编译或Git推送。

## 1. 起点与本日目标

唯一稿件：`/Users/zilongzeng/Research/Acoustic/docs/paper/Overleaf_Sync_final/`。所有后续与Overleaf同步的修改以此目录为准；历史稿只用于读取证据、脚本和来源。

9/12要完成：明确论文主张与贡献；冻结最小多seed方案；推进图文到技术内容4页的第一轮稿。9/13回收核心队列，更新结果解释，再完成版面验收。学生/外部GPU未计入必达路径，不给本科生设置个人硬截止时间。

官方当前列出的常规投稿截止日是9/16；技术内容控制在前4页，第5页仅允许参考文献、资助致谢及伦理合规声明。正文含标题、作者、摘要、图表和结论一起计入4页目标。官方要求字号不小于9pt，摘要约100–150词。[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)

| 项目 | 9/12已核对状态 | 本轮含义 |
| --- | --- | --- |
| 唯一稿件 | docs/paper/Overleaf_Sync_final/，5位作者；Wade邮箱已填 | 旧稿只用于查证结果和来源，不回写覆盖 |
| 结构 | Abstract、Sections 1–5、2张图、2张表均存在；Section 2.2已补文字 | 不再沿用“Section2.2/摘要/结论尚未写”的旧待办 |
| 现稿篇幅 | 粗略英文词数：摘要134；Intro355；Data515；Method548；Evaluation1323；Conclusion86 | 叙述加摘要约2961词，不含图表caption、表格数据和参考文献；不是实际页数 |
| 已有主结果 | Full seeds 0/1/fresh42、固定C/W读出、主模型HF/KAUH外评已完成；frozen基线已是3seed | 这些不重复训练或推理 |
| 核心控制 | P0新增4run全部完成，Coarse/Native均已3seed；IC-only0完成、SPR-only0运行中；Coarse42仍只用完整attempt2 | P1剩余IC-only1和SPR-only1；各行只汇总完整run |
| 当前代码 | 9/12已支持seed0/1/42，Full引用与输出目录随seed绑定；6项定向测试通过 | 本地任务核对实际sample IDs后直接进入已批准队列；原seed42目录保留 |
| 作者/版面 | 作者单位仍需确认；现稿有caption、bibliography和float间距调整 | 单位信息补齐；间距改动需对照模板，字号与页数最终看PDF |

尚无这份最新源码的整篇编译PDF可供本次核验，因此不沿用旧稿“7页”的说法，也不把词数估算当成已经达到4页。

## 2. 从完整论文逻辑重新梳理

### 核心问题与建议主张

研究对象是呼吸音的native acoustic tasks，不是直接的疾病诊断。ICBHI的cycle四分类、SPR的event二分类端点与可用细标签并不一致；HF/KAUH的单位和研究角色也不同。共同类别在共同5s处理下仍有来源差异，但这只是问题背景。

建议核心论点：在所研究的ICBHI/SPRSound条件下，保留可用的细粒度属性监督会影响属性判别和原生任务表现；监督信息的价值与最终分类接口的表现需要分开评价。

论证顺序：原生任务与标注信息不同 → 仅看native二分类分数不足以说明细标签的作用 → 用A/C/W共享监督与availability mask保留兼容信息 → 分别改变来源、标注细度和分类接口 → 用native指标及属性指标检验收益和代价。

现有结果不适合支撑“我们的hierarchy更优”或“学到的features不能迁移”。固定source head的失败不等于representation不能被目标任务适配；声学分布差异也没有证明迁移性能差的因果机制。

### 贡献应写成能力与发现

| 候选贡献／研究价值 | 当前证据 | 措辞与补证 |
| --- | --- | --- |
| C1：以共享属性保留不同原生任务中的可用监督，并检验属性在粗标签来源中的复用 | seed42 Coarse SPR未使用SPR C/W训练或校准标签，AUROC为93.67%，同seed IC-only为81.23%；新增Coarse0 AUROC为78.93%，选中模型的native预测退化 | 这是具体设置下的能力，不是首次提出mask、多头或多数据集学习。Coarse接触SPR音频/A标签，不叫zero-shot；相对IC-only的+12.44pp仍有来源暴露差异 |
| C2：原生二分类得分可能掩盖细标注对属性学习的价值（需重议力度） | 三seed Full−Coarse的IC/SPR/CW配对均值差为+6.51/+23.46/+8.48pp；Coarse0退化，Coarse1为57.28/86.48/93.93% | seed42的SPR差仅0.15pp仍是单seed事实；三seed不支持“SPR二分类几乎不变”。保留全部seed并讨论训练/选模敏感性 |
| C3：监督信息的价值与分类接口的表现需要分别判断 | Native三seed已齐：IC/SPR/CW为63.67±1.67/90.86±1.21/97.27±0.14%，同seed配对均值差+2.50/+0.16/−0.05pp；IC均提高，SPR seed1为−1.49pp | 将前两seed的“两项native均提高”收窄为ICBHI方向一致、SPR有取舍。此对照同时改变native objective/head capacity；没有Native-only，不能单独证明辅助loss增益 |
| 声学分析：支撑问题背景 | 共同5s处理后四个兼容类中IC−SPR level +25.13–29.11dB、band centroid −28.95至−68.38Hz；Both的SPR只有10患者 | 不是独立的“首次发现domain gap”贡献；幅值、设备、滤波和人群混杂仍在，不能用它证明迁移失败或疾病机制 |

新颖性判断：mask、多头预测和保留不同任务评价不能单独被称作新原则。DCASE 2024 Task4已研究异域数据、缺失标签和不同标注/评价细度；PC-MCL已有Normal/C/W多标签设计及患者相关学习。本文需要突出呼吸cycle/event原生任务下的具体问题、监督匹配对照和得到的新认识，而不声称首次统一多数据集。[DCASE论文](https://arxiv.org/abs/2406.08056)；[PC-MCL论文](https://arxiv.org/abs/2601.17080)

这三条是本轮建议的贡献表述方向，需在WP1逐条确认。多seed若改变方向，就同步缩小或改写贡献；不为维持初稿故事挑选有利seed。

### 当前证据与缺口

| seed42条件 | ICBHI Score (%) | SPR official Score (%) | SPR C/W AUROC (%) |
| --- | --- | --- | --- |
| Full fresh42 | 61.17 | 90.37 | 97.17 |
| ICBHI-only | 56.17 | 56.86 | 81.23 |
| SPRSound-only | 47.93 | 92.32 | 97.86 |
| Coarse SPR (attempt2) | 56.71 | 90.22 | 93.67 |
| Native+attributes | 62.30 | 91.94 | 97.41 |

Table1主模型已经是3seed：ICBHI Score 61.17±0.31%，SPR official Score 90.70±0.34%；其SPR C/W macro AUROC为97.32±0.15%。上述Table2表格只列fresh42参照及对应控制，不能把3seed均值与单seed控制作配对差。

固定C/W-only读出已覆盖同一组三seed，Score相对Full的配对变化为−0.88±1.60pp，逐seed方向不一致。主模型HF/KAUH外评已完成3seed；独立历史HF-off/on只是一组seed42，保留其不同HF-off参照与不均匀收益。

核心控制三seed已完成，当前缺口是剩余单源重复seed及将完整结果纳入论文解释。多seed不会消除official-test选模、source-only选模目标不同、source exposure不同或Native head容量变化这些比较条件；本轮不另起validation-only协议。

### 逐节审阅任务

| 部分 | 应承担的功能 | 本次复盘与修订重点 |
| --- | --- | --- |
| Title / Abstract | 一句话说清研究对象、问题和主要发现 | 保留shared-attribute/native-task主题；摘要目前约134词，长度合适。避免把单seed方向写成普遍结论；新seed回来后最后统一数字和力度 |
| Introduction / Contribution | 观察→具体问题→方法带来的能力→直接证据 | 现有contribution句仍偏“characterization + benchmarks”的工作清单。改成上述有价值的能力/发现，并明确与DCASE、PC-MCL等先例的区别 |
| Data / Section2 | 交代四源分工、语义与声学观察 | 2.2已补，转为精简和准确性审阅。区分recording/cycle/event/5s、patient/date proxy；保留HF负目标假设、KAUH过滤版本及兼容映射 |
| Method / Figure2 | 使读者能复现共享监督与native决策 | 保留encoder、A/C/W并行头、availability mask、核心loss及阈值规则；层级发生在readout。PAFA细节引用原作，重复实现说明压缩；Full、Coarse、Native的loss系数差异要能查到 |
| Evaluation / Tables | 用对照回答研究问题 | 是压缩主战场。协议只定义一次；Table1是已发表/冻结/联合不同设置的背景，Table2是核心问题。保留负向结果与比较条件，删逐格复述和过长数字串 |
| HF / KAUH | 说明额外监督与外部适用范围 | 主模型3seed外评与历史单seed HF-off/on pair分开。保留不均匀收益和外部代价；不扩成四源联合训练或临床诊断准确率 |
| Conclusion | 回答最初研究问题 | 总结标签信息和分类接口的不同作用；不能只是再列做了哪些模块。当前结论已存在，待多seed完成后调整强度 |

## 3. 多seed实验计划与可行性

| 优先级 | 新增运行 | 对应论点／价值 | 时间预算 |
| --- | --- | --- | --- |
| P0：必须先补 | Coarse SPR seed0/1；Native+attributes seed0/1，共4run | C1/C2/C3的核心重复证据；与已有Full 0/1/fresh42分别配对 | 按已完成run外推18.4h；全到50epoch约32.3h；另留代码准备/汇总及运行波动时间 |
| P1：有窗口再补 | ICBHI-only seed0/1；SPRSound-only seed0/1，共4run | 加强source-sharing和coarse属性复用的比较；把Table2五行都提升到3seed | 额外约18.0h；加P0约36.4h；全部满50epoch约61.4h，不能承诺周日内全部收齐 |
| 本轮不追加 | 新的Full、HF-on多seed、新backbone/新结构/重新做已完成外评 | 避免改变当前问题或重复已有资产 | 仅在发现既有Full确实不可复用时重新估算并决策 |

耗时依据是已完成的本地MPS run：IC-only 256.87min/28epochs；SPR-only 281.63min/34epochs；Coarse 294.88min/30epochs；Native 257.57min/27epochs。18.4h/36.4h按这些实际早停耗时外推；32.3h/61.4h按同一每epoch速度外推到50epochs，是规划情形而非运行保证。工程准备、汇总、负载和故障缓冲另计。

授权顺序：Coarse0 → Native0 → Coarse1 → Native1 → IC-only0 → SPR-only0 → IC-only1 → SPR-only1。9/13核对前五项已完成，第六项运行中。后两项在启动前按实际耗时检查周日窗口；若无法继续，按实际完成seed数报告，不删已完成的单源重复结果。

### 开跑前需要做的最小调整

- 已完成：`baseline/pafa/table2_benchmark_controls.py`的validate、CLI、Full参考路径、输出目录和metadata支持0/1/42，同一seed传入模型、采样和原分组逻辑。新seed输出在`PAFA_BENCHMARK_4COND_multiseed/`，旧seed42目录不覆盖。
- 每个新控制与对应的Full seed配对。ICBHI原分区随seed变化：0为3636/506，1为2880/1262，42为3174/968；SPR为5219/1437。不能把seed42分区给所有新seed后再称与旧Full0/1严格配对。
- 使用保存的config与metadata/validation sample IDs核对对应分区、完整Full参照和实际loss系数。不进行模型前向或smoke来证明代码“能启动”；只做所需的小范围参数/分组功能检查。
- 保留5s、MPS FP32、batch32、326更新/epoch、原epoch cosine、max50/patience10、相同初始化与PAFA系数。Full/Coarse/Native/IC-only仍以ICBHI official-test Score选模；SPR-only仍以本源inter Score选模；阈值只用相应允许来源的validation。
- Coarse不读取SPR细标签用于训练/校准/选模，保留A/3；Native保留native CE与C/W各1/3。既有Full有效节点权重在新增配对seed中也需核对。
- 继续落盘逐样本NPZ，自动汇总所需native Score、C/W AUROC以及逐seed差值。保存partial并如实报告失败；不重定义早停来赶时间、不把部分run写成完成结果。
- P0完整后，对Full/Coarse/Native报告相同3seed的mean±sample SD与配对差值，明示各seed方向。若仅完成2seed就标n=2；未补的单源仍标seed42/n=1，分块或caption说清。

用户随后已明确授权启动上述补seed工作，且不以贡献措辞讨论完成为前提。管理已完成WP2的runner泛化和6项定向检查，本地任务核对saved Full sample IDs后直接进入已批准队列；不重复申请每个run的启动许可。正式启动情况见下一段。

启动核对（9/12）：Coarse SPR seed0于10:40（Chicago）正式开跑，已保存首个完整epoch、checkpoint/NPZ，原训练PID60120持续运行。原zsh调度中的status变量冲突已修复；独立continuation PID64304同时等待原训练及queue shell自然退出，从Native0接续其余7项。每个追加run检查周日窗口，训练失败或预计超窗停止后续；完成/停止时自动更新新结果根下的`multiseed_summary.json/.md`。初始汇总仍是已有Full三seed和control seed42，尚无新增run完成。

## 4. 重画Figure与图表收敛

已只读检查当前副本的两张独立PDF。按约7in通栏显示，Figure1高2.26in，常规标签约6.7–8pt、标题9pt；Figure2高约2.75in，文字约9–11pt。Figure1需要解决小字问题，Figure2需要减少重复框与路径。

- Figure1：核心保留IC/SPR同类别level与band-limited centroid证据。优先尝试两个清晰面板；四源概览可保留为紧凑小面板，但必须满足字号和总高度预算。四源在Data/Method中的角色仍完整保留，不以保留全部箱线图替代清楚的主张。目标高度约1.8–2.0in，实际标签至少9pt。
- Figure1统计口径不变：recording→patient中位数、95%组bootstrap区间、Both的SPR 10患者支持、HF日期代理和KAUH滤波聚合；使用已核对源表，不为画面分离重新选特征/阈值/样本。源幅值不是校准声压。
- Figure2：主视觉围绕共享学习→A/C/W并行分数→native readout。明确SPR只读A；ICBHI的层级仅在决策处。将底部重复的“Same shared model”框压成清晰的HF辅助监督/固定外评说明，目标约1.8–2.0in，保持至少9pt。
- 两图统一符号、配色和线宽；保留黑白可读性。减短caption，在正文交代一次必要定义。图只承载需要看图才能理解的信息。
- 绘图源脚本/小CSV在独立工作区或analysis目录准备，最终资产接入`Overleaf_Sync_final/Figure/`。不得直接运行会覆写历史目录的旧脚本。
- Table1减少正文逐格复述；Table2为多seed预留mean±SD列宽，保留真实现值与n标记，不填虚构占位统计。核心与单源若n不同，必须显式区分。

## 5. 将技术内容压缩到4页整

| 部分 | 当前粗略词数 | 首轮目标词数 | 压缩方式 |
| --- | --- | --- | --- |
| Abstract | 134 | 120–140 | 保持问题、设计、主要发现及范围 |
| Introduction | 355 | 270–310 | 聚焦最相关先例，删除重复动机 |
| Data | 515 | 300–350 | 四源角色、语义、输入和声学结论各写一次；详细支持表留结果材料 |
| Method | 548 | 350–400 | 保留关键定义/公式/决策规则；借用方法引用而非展开 |
| Evaluation | 1323 | 740–820 | 少复述表格；减少重复协议与逐端点数字串；HF案例保留收益和代价的代表端点 |
| Conclusion | 86 | 65–80 | 回答问题并控制范围 |
| 合计（叙述＋摘要） | 2961 | 1845–2100 | 减少约29%–38%；caption另控制约200–220词，最终仍以PDF验收 |

保留：研究问题、四源角色和语义、可用性mask、核心loss/decoder、selection/threshold/seed规则、核心控制与负向结果。压缩：重复的实验说明、PAFA借用细节、Table1逐项差值、同一规则的多处解释、长串次要端点数字。

HF/KAUH继续留在paper，但不平铺所有已算数字；保留主外评及历史HF案例的代表性收益和代价。其余完整数值留在结果材料中，不能只保留有利端点，也不把附加材料默认视为可占用第5页的技术正文。

建议页面功能：第1页问题/核心论点及必要数据背景；第2页精简Data/Method与Figure1；第3页Method/Evaluation与Figure2、核心对照；第4页结果解释、Table1/2剩余部分及Conclusion。实际float位置以PDF调整，不强制沿用旧稿“每页一个通栏float”的放置结果。

验收以用户Overleaf编译的当前PDF为准：技术内容完整落在前4页，结论在第4页结束；第5页只放官方允许内容。不缩小字号/边距来代替内容取舍，不为“4页整”添加无信息文字。现有caption/参考文献/float间距patch逐项核对实际显示和模板要求。本轮不编译整篇。

## 6. 9/12执行顺序与9/13承接

| 工作项 | 负责角色 | 输出／验收 | 依赖与预计投入 |
| --- | --- | --- | --- |
| WP1 论点与贡献复盘 | 管理＋用户，写作落实 | 确认1句核心论点、3条候选贡献、每条证据/反例与范围；决定Full仍作reference的表述 | 9/12首先讨论；约60–90min |
| WP2 多seed最小实现准备 | 管理完成runner修改，本地训练核对 | seed0/1/42、对应Full引用与独立输出路径已实现，6项定向检查通过 | 已获明确start；sample IDs核对后直接执行，不等待贡献措辞讨论 |
| WP3 本地P0队列 | Acoustic本地训练 | 4run全部完整结束，Full/Coarse/Native三seed及配对汇总已核对 | 已完成；P1的IC-only0完成、SPR-only0运行中，剩余两项检查周日窗口 |
| WP4 Figure2 PPT重画 | 论文绘图，管理核对集成，写作衔接 | 已交付并集成单页原生可编辑PPTX及单图PDF/PNG，178×57mm，9–11pt；caption已同步 | 首稿已完成，待作者反馈；未编译整篇或验收4页 |
| WP5 四页正文第一轮 | 论文写作＋用户 | 按章节词数预算删冗余，保留必要定义、负向结果和适用范围；现有真值排版并预留mean±SD列宽 | WP1后与图重画并行；约3–4h |
| WP6 周日结果汇总与解释 | 训练＋管理＋写作 | 逐seed值、mean±sample SD、配对delta及方向；更新Table2/摘要/结论；决定是否纳入P1 | P0实际完成后；约1–2h，不把未完成run凑成n=3 |
| WP7 4页PDF验收 | 用户Overleaf编译，写作/管理核对 | 技术内容全部在前4页；图表、公式和结论未漂到第5页；字号/字体/作者信息/引用一致 | 图与数值冻结后；预留至少一次完整PDF复核 |

9/12的工作可以分成两条并行线：本地训练串行运行；用户与写作/绘图任务处理论点、图和篇幅。训练并行于写作，不并行启动多个本地MPS训练run。

周六收口目标：主张与贡献确定、P0配置/命令准备完毕并在获得start后进入队列、两图重画稿与4页第一轮稿可审。周日收口目标：P0完整结果及配对汇总、Table2/Abstract/Conclusion统一、当前Overleaf PDF四页验收；P1按实际剩余窗口决定。

若P0符号不稳定：降低对应贡献力度，报告各seed；若Native依然高于Full：把层级作为reference，不重新包装成优势。不会为保持初稿结论改变比较对象或只合并有利run。

## 7. 今日检查清单

- [x] 以当前Overleaf_Sync_final全文重新盘点，确认Section2.2及Wade邮箱已经存在。
- [x] 核对已完成seed42控制、Full三seed与当前runner的硬编码限制。
- [x] 根据实际耗时计算核心4run和可选8run预算，纳入用户允许延长到周日的约束。
- [x] 检查当前两张图的内容、大小和字号；核对ICASSP当前页数与字体要求。
- [ ] 与用户逐条确认核心论点及贡献措辞。
- [x] 用户已明确授权本地补seed；runner泛化及6项定向检查完成。
- [x] 对应Full实际sample IDs已匹配，P0四项全部完成，三seed汇总及配对差已核对。
- [ ] 完成P1剩余运行：IC-only0完成，SPR-only0运行中，IC-only1与SPR-only1待运行；按9/13最新指示完成原队列。
- [x] Figure2原生可编辑PPT首稿、单图PDF/PNG已核对并集成，caption已同步；Figure1及正文压缩仍为后续待办。
- [x] 首批完整结果已回填并统一核心文字：Full/Coarse/Native n=3、IC-only n=2、SPR-only n=1，配对差仅用同seed交集；后续新完成单源结果另次更新。
- [ ] 确认作者单位，核对当前PDF的4页、作者、引用和全部图表。

## Sources与历史入口

- 当前论文：`docs/paper/Overleaf_Sync_final/main.tex`及`Section/`、`Figure/`；本计划依据当前磁盘版本，不用历史稿覆盖。
- 已完成实验：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_{0,1,42}/`；`PAFA_BENCHMARK_4COND_seed42/*/seed_42/`与Coarse `seed_42_attempt2/`中的run_summary/terminal资产。
- 声学报告：`/Users/zilongzeng/.codex/worktrees/5513/Acoustic/docs/analysis/wade_acoustic_validation/REPORT_zh.md`及对应结果表；历史Figure1小源表可只读复用。
- 已推送项目快照：[e0bdf3e](https://github.com/z1L0Ng/Acoustic/commit/e0bdf3e)，用于查证历史资产；不替代当前Overleaf副本。
- 前一份Notion计划：[09-08 Working Plan](https://app.notion.com/p/3d4309efda29819aa175ddb25fcfcf94)，仅作历史。本页重新核对已完成事项，不恢复过期clean-suite、旧7页稿或学生seed缺项。
- [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)；[DCASE 2024 Task4](https://arxiv.org/abs/2406.08056)；[PC-MCL](https://arxiv.org/abs/2601.17080)。
