# Acoustic Work Plan

## 当前周期：2026-09-14晚–2026-09-15 01:00 EDT

本轮以9/14最新会议为依据，优先把已有结果组织成连贯、可读的论文。先明确overall论点、baseline定义和Table 1/2各自的takeaway，再整理Section 2/3、段落逻辑、图件与caption；主体稳定后收拢Abstract/Conclusion。23:45 EDT形成可通读稿，约09/15 01:00 EDT收口（芝加哥00:00 CDT）。

当前计划：[9/14夜间工作计划](work_plans/2026-09-14_to_2026-09-15_0100_ET_paper_work_plan_zh.md)；[Notion工作计划](https://app.notion.com/p/3db309efda29817f856ff55a4c82e8e3)。会议：[9/14完整中文记录](meeting_records/2026-09-14_paper_revision_meeting_record_zh.md)；[Notion会议全文](https://app.notion.com/p/3db309efda29813d8044fbe789225846)。Honey与Wit分别校正为Hanlin与Wade；58项讨论及10条总结保留。

最新双写作分工（9/14 20:30 EDT）：线程1“论文写作”（01a08442-92e6-7110-8399-e42eca520ea8）负责Section 1、Section 4、Section 5及Abstract，包括Table 1/2和Evaluation图表；线程2“论文写作二”（01a0a275-3711-73f3-9fec-7dbfd04c1611）负责Section 2、Section 3及对应Figure 1/2和数据/方法图表。两条任务同在main，各改自己负责的章节，保留Section 4已落盘修改后交接给线程1。线程2直接对接论文绘图；main.tex、citation.bib及共享宏仍由线程1唯一实际写入，线程2传递用户已确认的局部图注/引用/全局设置变更。不得整文件回写或覆盖对方章节。

独立reviewer：“论文独立审阅”（01a0a220-f1a7-7973-9778-4a986479c4fb）已创建，独立worktree已就绪，9/14 20:23 EDT状态为idle。它不读取任一写作任务的记忆；只有用户在该任务明确要求review时才开始，一轮结束后回到待命。工作计划时间块、管理消息和训练完成均不触发审稿。本管理回合不执行具体重写或联系合作者。

本轮稿件起点为main提交a4627d6；唯一Overleaf同步目录仍为docs/paper/Overleaf_Sync_final/。后台HF-on按原授权继续，9/14 18:49 EDT最近核对为seed1 epoch23/update7200、尚无完成summary；当前新增seed0和历史seed42已完成。PAFA三seed缺合格权重，不进入今晚关键路径，不新增源域训练。未完成数字不计入三seed汇总。

最新用户规则：DONE与\revisiondone已停用，不添加或恢复任何完成标记；由用户在Overleaf编译，不例行本地编译。Wade/Hanlin的内部read-through由用户安排，不设个人硬截止；本周期未获反馈时由用户和获明确授权的reviewer完成检查。

下方及上一轮文件按历史保留，未完成项目不因换计划被标记完成。

---

## 2026-09-13–2026-09-14 周期记录（历史）

当前执行入口：用户已恢复“论文写作”任务与用户逐章讨论和修订，从Introduction开始，按Introduction→Datasets/Analysis→Method→Evaluation推进。完成本章约定修改并标红DONE后继续下一章。Abstract/Conclusion暂时跳过；需要图件时由写作与论文绘图任务直接协作，写作核对后集成到唯一同步目录。无需管理逐章转发或等待HF训练。

后台HF-on补证：seed0已正式训练，本次同步写作时已见epoch1/update200；seed1串行待运行，新增完整结果0/2，旧seed42复用。训练与收尾由本地任务独立推进，写作不等待这些结果，也不把未完成数字纳入论文。

老师指定红色DONE：每个section完成本轮约定修改后，在标题之后、正文之前加一个红色DONE；图完成并接入后，在对应figure caption开头加红色DONE，标记不画进PPT。未完成内容不标，旧首稿不自动算新一轮完成。该用户指定审阅标记优先于旧的内部标记禁用约定，其他内部注释仍放在同步稿外。

当前计划：[Paper整体思路与逐章重构](work_plans/2026-09-13_to_2026-09-14_paper_restructure_work_plan_zh.md)；[Notion工作计划](https://app.notion.com/p/3da309efda29818d9474cf12d24dfdd9)。

会议记录：[9/13完整中文纪要](meeting_records/2026-09-13_paper_revision_meeting_record_zh.md)；[Notion会议全文](https://app.notion.com/p/3da309efda29818fa0f1d0edb27a7ea6)。

用户已补充老师认可的大方向：提出连接异构呼吸音标注与相异原生任务的联合学习框架，并以原生任务性能与跨数据集可迁移性之间的取舍作为主要价值。现以当前A/C/W联合框架为主线，Native+attributes保留为对照。具体章节和必要配图按用户在写作任务中的讨论执行，继续细化基线、归因及图文对应；本地训练安排不动。

上一轮核心控制已收口：8/8于9/13 19:01（America/Chicago）全部完成，Full、Coarse、Native、IC-only、SPR-only均为3seed。完整run_summary/terminal及当时队列正常退出已核对，finish正确跳过已完成项。最终统计见本轮plan和result内multiseed_summary；新增HF-on0/1为本次独立追加授权。

- [x] 会议全文与两天规划已写入Notion并互相链接，原文48个段落完整保留。
- [x] 用户转述的老师认可大方向已补入会议记录与新计划，四篇直接相关文献的设置已核对。
- [x] 本轮8个新增run与所有条件三seed汇总完成；Full对同配方单源的IC/SPR配对均值差分别+3.39/−2.13pp。
- [x] 已将HF-on0/1追加任务及红色DONE规则同步对应执行任务，并写入新work plan。
- [ ] 完成HF-on0/1及对应native/HF/KAUH评测；不作为写作前置条件。
- [ ] 量化第二条贡献的比较基线、任务代价与归因，细化正式方法名称和贡献措辞。
- [ ] 确认逐章蓝图、Figure1取舍、Figure2图文公式对应和核心表格组织。
- [x] 用户已恢复主体第1–4章逐章讨论与修订，写作/绘图可按需直接协作。
- [ ] 当前先完成Introduction讨论与本章修改，再按顺序继续；Abstract/Conclusion后置。

上一轮计划保留在[会后重排前快照](work_plans/snapshots/2026-09-13_previous_work_plan_snapshot_zh.md)及本地提交16a9296中。旧Notion页已加历史标记，下方管理记录按当时状态保留。

---

## 会后重排前的管理记录（历史）

更新日期：2026-09-13（周日）

会后重排准备（9/13）：上一轮work plan已按当前状态保存为[旧计划快照](work_plans/snapshots/2026-09-13_previous_work_plan_snapshot_zh.md)。等待用户提供刚结束的meet内容后再制定后续计划；写作/绘图暂停及本地训练原安排维持。

当前暂停状态（9/13 15:20 America/Chicago）：用户明确暂停“论文绘图”和“论文写作”任务，两项均已确认暂停，等待用户明确恢复。保留当前已落盘稿件、PPT及图件；不继续修改、导出、渲染、集成或自动回填新结果。两任务确认没有遗留的本任务后台作业。本地训练进程、剩余有限队列及既有训练授权保持不动，不因本次暂停改变训练安排。

最新交付（9/13）：Figure2原生可编辑PPT首稿已完成，管理已核对并集成到`docs/paper/Overleaf_Sync_final/Figure/figure2_method.pptx/.pdf/.png`。图参考旧roadmap，尺寸178×57mm，文字9–11pt；SPR只连A，IC分别接A门控与C/W，HF/KAUH角色独立。main.tex对应caption已同步。可复算JS位于`figures/paper/figure2_method_ppt.mjs`。后续按作者反馈改版；Figure1与整篇压缩仍待推进，本地训练继续原队列。

最新执行指示（9/13）：用户要求“正常继续跑就行”，剩余SPR-only0→IC-only1→SPR-only1按原有限队列继续完成，原周日窗口改为时间估计，不因预计超过今晚暂停这批已批准任务。用户同时批准写作任务立即将已完成结果写回唯一同步目录`docs/paper/Overleaf_Sync_final/`；Table2、协议、相关结果解释、Introduction、Abstract和Conclusion统一修订。本轮允许完成后一次性交接，不进行本地编译、额外实验或Git操作。

训练进度（9/13 10:45 America/Chicago核对）：新增完成5/8，P0完成4/4，P1完成1/4。已完成Coarse0/1、Native0/1和IC-only0；当前SPR-only0由Python5792、continuation64304正常训练，完整epoch10/update3260，epoch11真实update已至3552。剩余IC-only1、SPR-only1尚未启动。IC-only0实际用时408.09min，后续时间预算需结合新耗时复核；周日晚全齐不能保证。Coarse0的退化结果50.00/25.00/78.93%完整保留。

科学解释更新：Full、Coarse、Native均已具有seeds0/1/42完整结果。Native三seed的IC/SPR/CW分别为63.67±1.67/90.86±1.21/97.27±0.14%，与同seed Full的配对均值差为+2.50/+0.16/−0.05pp。IC在三seed均提高；SPR的seed1为−1.49pp，因此撤回将前两个seed“两项native Score均提高”的观察推广到三seed的说法。Coarse三seed为54.66±4.05/67.23±36.62/88.84±8.59%，包含seed0退化；“SPR二分类几乎不变”不是稳定结论。以上仍沿用official-test选模协议，不能解释为纯结构因果。稿件仍为seed42控制描述，尚未自动写入这些结果。

当前计划：[论文复盘、多seed补证与4页稿](work_plans/2026-09-12_paper_review_work_plan_zh.md)；[Notion 9/12 Working Plan](https://app.notion.com/p/3d9309efda2981e0bae7e27f069c5366?pvs=204)。用户9/12明确批准“本地训练线程完成缺的多seed实验”，授权包括必要的runner seed泛化、分区核对及正式训练/评测/汇总。顺序为Coarse0→Native0→Coarse1→Native1，再IC-only0→SPR-only0→IC-only1→SPR-only1；均复用对应Full参照，不重跑已完成seed42。队列可延长到9/13，若追加运行预计超出该窗口则在继续前回报。前5项已完整结束，第6项SPR-only0正在continuation中运行；不包含新Full/HF/其他模型、论文编辑、编译或Git操作。

唯一稿件入口仍为[Overleaf_Sync_final](paper/Overleaf_Sync_final/main.tex)；旧稿目录为历史资产，只读查证、不自动覆盖当前稿。9/12重新读取后，Section2.2已补齐、Wade邮箱已填写，Abstract/Conclusion及2图2表均存在，不再把这些列作未写内容。当前主要工作是论点与贡献、重复seed证据、图的字号/高度及正文取舍；作者单位仍待确认。

- [x] 从当前副本完整复盘核心论点、候选贡献、方法、结果和逐节功能，形成claim/evidence/缺口清单。
- [x] 核对Full三seed及四个seed42控制；管理已完成runner seed0/1/42支持、动态Full引用和独立输出目录，6项定向测试通过。原seed42默认目录与配方保留；执行任务核对最终sample IDs后直接进入已授权队列。
- [x] 估算P0的Coarse/Native seed0/1共4run：既有早停耗时外推18.4h，全部50epoch情形约32.3h；可选全部8run为36.4h，满预算情形约61.4h，不能承诺周日全齐。
- [x] 只读检查两张现有图；Figure1存在6.7–8pt标签，Figure2高约2.75in。正文叙述加摘要粗计2961词，Evaluation占1323词，已制定1845–2100词的首轮叙述预算及4页PDF验收规则。
- [x] 新计划已写入Acoustic的Notion Working Plan库并回读核对；旧9/8页加历史标记，保留原记录。
- [x] 用户已批准本地补齐缺失多seed；9/13进一步确认正常继续原队列，无需逐run重复申请。此授权不以论文贡献措辞讨论完成为前提。
- [x] P0四个新增run全部完成，Full/Coarse/Native三seed汇总及同seed配对差已核对，Coarse0退化结果保留。
- [ ] P1四个新增run完成及汇总：IC-only0完成，SPR-only0运行中，IC-only1和SPR-only1待运行。按9/13指示继续完成原队列；保留时间估计、训练错误停止和自动汇总。
- [ ] 与用户逐条讨论核心论点和候选贡献，再按完成的实验结果确定最终力度。
- [x] 首批已完成结果已写回Table2/相关正文/Abstract/Conclusion；采用Full/Coarse/Native各n=3、IC-only n=2、SPR-only n=1的明确快照，配对差仅用同seed交集。后续新完成单源结果在下一次回填更新。
- [x] Figure2原生可编辑PPT首稿、单图PDF/PNG已交付并集成唯一同步目录，caption已同步；后续按作者反馈调整。
- [ ] 后续处理Figure1和正文压缩，使用用户Overleaf编译PDF验收技术内容4页。
- [ ] 补齐已确认的作者单位，核对最终作者、引用和图表信息。

本轮训练输出：`result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/`。原首项日志为`queue_logs/multiseed_queue_console.log`，接续日志为`queue_logs/continuation_queue_console.log`；自动汇总为`multiseed_summary.json`及`.md`，n=1的sample SD为空，配对差只按同seed交集计算。当前Full/Coarse/Native各n=3，IC-only n=2，SPR-only n=1；运行中的SPR-only0未计入完成结果。

以下为历史工作记录；其中的待办、旧路径和旧完成状态不代表当前Overleaf副本，以9/12计划为准。

当前状态（09-11 11:10核对）：4/4条件全部完成。恢复队列已于今天03:14（芝加哥）正常退出，QUEUE_EXIT=0；实时检查无Acoustic训练进程。Coarse采用seed_42_attempt2完整结果，原中断目录保留且不计入结果；没有启动额外实验。

新增工作（09-11 Wade feedback）：已收到四数据集声学特征箱线图与PCA图，原图保存在[Wade材料目录](student_tasks/received/2026-09-11_wade_acoustic_features/)。独立任务“Wade 声学特征独立验证”已完成正式交付，ID为`01a09336-a12c-7181-9cf2-1f9610d494e8`，工作区为`/Users/zilongzeng/.codex/worktrees/5513/Acoustic`；只读访问main中的四源原始数据及最新方法/结果。管理已核对[最终报告](/Users/zilongzeng/.codex/worktrees/5513/Acoustic/docs/analysis/wade_acoustic_validation/REPORT_zh.md)、关键统计表与候选图。本次仅执行已授权的本地声学特征提取、统计分析和科学绘图，没有新增模型训练或模型推理。原代码/逐样本表仍缺；本次完成独立合理性验证，不宣称精确复现Wade的计算。

- [x] 独立任务完成13,704条真实录音的声学提取，覆盖ICBHI920、SPR BioCAS2022 2683、HF9765、KAUH336，并提取45,618个论文5s输入；`extraction_summary.json`未记录音频问题。此项是声学分析完成，不是模型实验或论文结论验收。
- [x] 独立验证特征定义、幅度特征冗余、DC/滤波影响、PCA缩放和载荷，以及样本/组数量不均的影响；所有固定对照和有限探索性定义核对已交付。
- [x] 核对全录音与论文5s输入视角，完成标签兼容的class-matched及组级描述；HF日期代理和KAUH B/D/E版本按真实依赖关系处理。model5s较原录音少4组，源于SPR四患者的全部录音无event，并非PCA删失。
- [x] 已交付逐项结论、可复算代码/表、六组PNG/SVG/PDF科学图和与当前五行seed42结果的衔接建议。
- [x] 用户已明确授权将验证内容同步写作，完成Figure1并补齐Section2，然后通读全文、完成Abstract与Conclusion；管理已向声学验证任务和写作任务下发本轮工作。

本轮论文收尾（已获执行与同步授权，沿用Overleaf编译流程）：

- [x] 声学验证任务已完成7.00×2.26in通栏三panel Figure1；管理核对PNG、脚本和caption定义后，已将PDF/SVG/PNG/PY及少量源表复制到main的`Figure/figure1_dataset_characteristics.*`和`Figure/figure1_source_data/`，并通知写作接入。
- [ ] 写作任务：在main接入Figure1及caption，补齐Section2声学内容，并核对四源角色、分析单位和标签语义。
- [ ] 写作任务：通读全文，统一贡献、方法、数字和协议；改掉过期validation-selected计划描述，完成基于现有结果的Abstract和Conclusion。
- [ ] 管理：核对最终main图文和静态检查结果，向用户交付本轮完整稿及确实仍需补充的信息；本轮不编译整篇、增加实验或发送合作邮件。

独立验证结论：四源声学异质性有支持，但Wade图中的部分幅值/频谱量级、silence解释和PCA的37.1%/16.0%尚未按其原定义验证，不判定其计算错误。相同5s处理后，ICBHI与SPR兼容四类内部仍有组级差异：ICBHI−SPR的level差25.13–29.11dB，80–2000Hz谱质心差−28.95至−68.38Hz；Both的SPR仅10患者组，保留支持限制。112患者KAUH配对的D−B带内谱质心差+105.84Hz，95%区间96.89–108.51Hz，说明滤波版本会影响该统计量，不解释为疾病差异。建议Figure1保留四源概览，声学部分优先讨论[同类别5s候选图](/Users/zilongzeng/.codex/worktrees/5513/Acoustic/result/wade_acoustic_validation/figures/04_class_matched_5s.png)；PCA作为背景/敏感性材料。声学观察不证明迁移因果或hierarchy优势，现有Native+attributes高于Full的结果继续如实讨论。全部产物保留在独立工作区`result/wade_acoustic_validation/`，无待启动的新分析分支。

备选训练复盘：[2–3天补齐数字与Hanlin分工草案](work_plans/2026-09-09_2to3day_replan_zh.md)。
暂停前计划：[正文同步与最小补证](work_plans/2026-09-09_experiment_work_plan_zh.md)。
实验规格：[四条件benchmark规格与准确命令](paper/ICASSP_2026_acoustic_disease/EXPERIMENT_SPEC_2026-09-09_zh.md)。

用户已对“现在按ICBHI-only→SPRSound-only→Coarse SPR→Native+attributes串行开跑，各seed42”的明确问题回答“可以”。授权包括这四项正式训练、各自规定的选模/评测及结果交接；成功完成一项后继续下一项，失败或用户暂停则停止后续队列。Full/HF、其他seed与旧14/18-run计划均不在本次启动范围。

- [x] 本地训练任务：核对并整理正式JH2三seed主结果、Full/CW-only配对结果、属性指标、已有HF/KAUH及HF扩展结果；管理补齐实际训练eligibility统计并核对原validation样本。交付目录为`docs/paper/ICASSP_2026_acoustic_disease/LOCAL_ASSETS_NO_TRAIN_2026-09-09/`。
- [x] 本地训练任务：可用数字、来源、适用论文位置和比较限制已落盘并回报管理。其跨任务就绪通知被审批拦截，管理已成功通知论文写作从同一main工作区读取修正版。
- [x] 论文写作任务：已接收并核对上批全部统计，更新Data的实际训练覆盖句，完成[资产接收与缺项清单](paper/ICASSP_2026_acoustic_disease/LOCAL_ASSETS_RECEPTION_AND_GAPS_2026-09-09_zh.md)。当前主checkpoint的核心推理已齐全；已有结果可以用于benchmark、固定读出和外部分析。
- [x] 本地训练任务：完成现有checkpoint预测查漏、HF/KAUH端点后处理及精确历史HF-off的SPR inter补推理；历史HF-on用已有NPZ。主JH2与JH3.3的已完成核心推理未重复，无训练或优化器更新。
- [x] 论文写作任务：已将benchmark读出/属性/外部结果及独立历史HF案例写入§4.2；管理核对并补上最后的历史SPR C/W AUROC/AUPRC配对，更新接收与缺项清单。Abstract/Conclusion最后处理，本轮未编译；后续日常写作仍不自动汇报。
- [x] 管理与用户：已确认延续benchmark的最小方向，顺序为两个单源、Coarse SPR、Native+attributes，各seed42一次；Full/HF不默认重跑。
- [x] 模型设计任务：已更新当前协议，新增`table2_benchmark_controls.py`及定向测试，提供四条命令；metadata/loss/schedule检查通过，未执行模型前向或训练。
- [x] 本地训练任务：已只读核对预训练初始化、旧Full参照、原配方/分组与独立输出路径，未发现相关活动训练进程；建议根目录`PAFA_BENCHMARK_4COND_seed42`当前不存在。四个新variant不含Full，包含Native+attributes；命令就绪也不自动启动。
- [x] 管理：完成静态审阅，修复学习率调度与旧Full不一致的问题，按原JH2每epoch cosine更新、epoch内保持常数；新增epoch1/25/50纯函数检查通过。
- [x] 用户已明确授权启动四项串行训练；本地任务已收到启动指令，不再重复申请。
- [x] ICBHI-only seed42：完成28 epochs，selected18，共9128 updates，用时256.87分钟；terminal预测与run_summary已落盘。
- [x] SPRSound-only seed42：完成34 epochs，selected24，共11084 updates，用时281.63分钟；terminal预测与run_summary已落盘。
- [x] Coarse SPR seed42：工程重试seed_42_attempt2已完成30 epochs，selected20，9780 updates，294.88分钟；原partial保留，不用于最终表格。
- [x] Native+attributes seed42：已完成27 epochs，selected17，8802 updates，257.57分钟；今天03:14最终评测和队列成功结束。

本科生新材料（09-10接收）：已保存[25组配置三seed汇总与核对说明](student_tasks/received/2026-09-10_baseline_3seed/REVIEW_zh.md)。来稿caption声明seeds0/1/42的mean±sample SD，覆盖五个backbone的ICBHI、SPR binary、SPR event7、KAUH nine-class及HF四标签时间任务。用户已明确确认本次结果为frozen encoder、5s输入，input方法与主方法一致；此前2s/1s说法不适用于这份新汇总。写作已将Table1五行旧seed42背景值升级为三seed mean±SD，并同步caption、分块标题、5s说明和正文举例；管理已核对30个mean/SD单元与来稿一致。逐seed原始metrics/config尚未随文件提供，按学生汇总值记录，不宣称已独立复算每个run；已确认的输入设置不再列为待确认。HF/KAUH指标不与本论文的外部兼容评测混用；该文件未包含Figure1声学分析或PAFA/SG-SCL等强方法的新多seed结果。本地四条件训练继续，不受此次材料接收影响。

合作沟通：已为`arian_azarang@med.unc.edu`准备[英文进度邮件草稿](communications/2026-09-10_arian_progress_email_draft.md)，说明当前paper story、联合模型/基线与两个单源结果、正在补齐的标签/分类接口分析和后续临床合作反馈。邮件尚未发送；发送时应按当时的实际实验状态更新进度措辞。

中断与恢复：17:48的实时ps已无原carrier/queue/caffeinate/训练Python；原session72539在管理与执行任务均返回Unknown process id。现有日志无traceback或退出码，不能归因为模型异常、OOM或已知tee问题；epoch25的no-improvement counter为8，未达到patience10。最后可确定阶段是epoch26 validation prediction已保存。旧last checkpoint无optimizer/RNG，不能无损续跑。恢复使用独立tmux与直接stdout/stderr日志：保留原Coarse目录，重试输出为`coarse_spr/seed_42_attempt2`，Native仍用原未创建的`native_attributes/seed_42`；只完成原批准的两项剩余条件，不重跑两项已完成单源，不改科学规格。

恢复实证（18:05）：tmux `acoustic_benchmark_recovery_20260910`存活，server PID19730的PPID为1，pane/queue PID19731、caffeinate PID19734、Coarse Python PID19736均存活。新console直接落盘于`result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/recovery_logs/recovery_queue_console.log`，已保存实际update1至160。配置确认seed42、MPS FP32、原BEATs预训练初始化与epoch cosine，未加载旧Coarse checkpoint；旧目录保留。恢复脚本记录每项exit code，失败停后续；不依赖原Codex PTY会话。退出根因仍未知，不能把更换carrier写成已证实的根因修复。

下一步（不自动增加实验）：

- [x] 写作任务已补齐Table2最后两行、signed差值和结果解释；管理已回读Evaluation并核对Coarse/Native数字，本批训练数字无xxx。
- [ ] 与用户讨论Native+attributes高于Full对主方法定位及贡献措辞的影响；不自行更换主方法。
- [ ] 完成Figure1与§2.2的声学说明，随后收尾Abstract/Conclusion、全篇一致性和投稿版面。
- [ ] 根据最终讨论更新Arian邮件并按用户发送指令处理；当前仍是草稿。

本次已完成结果（均为seed42，Score为百分数）：

| 条件 | ICBHI Score | SPR official Score |
|---|---:|---:|
| 既有Full fresh42 | 61.17 | 90.37 |
| ICBHI-only | 56.17 | 56.86 |
| SPRSound-only | 47.93 | 92.32 |
| Coarse SPR（attempt2） | 56.71 | 90.22 |
| Native+attributes | 62.30 | 91.94 |

四项完整run_summary、selected terminal NPZ及native指标均存在；恢复console的QUEUE_COMPLETE为2026-09-11T08:14:09Z，QUEUE_EXIT=0。Coarse工程重试与原attempt的config逐字段仅output_dir不同。管理已从Coarse/Native最终NPZ独立复算ICBHI Score、SPR official Score及C/W macro AUROC，均与保存汇总一致。

关键已完成对照（均为seed42）：Full相对Coarse的ICBHI/SPR/CW-AUROC分别高4.46/0.15/3.50pp；Native+attributes相对Full分别高1.13/1.58/0.24pp。不能宣称hierarchy优于native heads，也不自动将Native替换为主方法；主线与最终contribution需与用户讨论。单源选择目标/暴露差异仍按原协议披露。

写作收尾已下发：最后两行和对应数值语句使用完整Coarse attempt2和Native结果，更新来源与缺项清单。Table1本科生frozen encoder、同5s输入、三seed背景已完成并核对30个mean/SD单元。接下来是结果解释、Figure1、Abstract/Conclusion及投稿版整理；Arian邮件仍为本地未发送草稿。

08:55观察记录：总queue_console在昨晚首epoch后未继续同步，本地任务也未按原约定及时回报两项完成；当时训练本身未停。管理已用各run的train_log/run_summary与实时进程核对，并要求本地任务恢复session72539输出接收、持续追加日志及每项完成交接；不重启队列。两个已完成结果已通知写作读取，按实际benchmark协议更新Table2与对应正文。

原队列启动记录（这些进程现已结束）：queue session `72539`，carrier PID86111/86112，caffeinate PID86114，首项Python PID86118；UTC启动时间2026-09-10T03:01:49（芝加哥09-09 22:01）。首个update1实际loss=1.721060，后续update224=0.687375，仅作运行证据，不作实验结论。console为`result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/queue_console.log`；初始tee先于目录建立导致日志文件未打开，已从持久exec session恢复保存，训练未中断或重启。后续由执行任务继续追加日志。

首个完整epoch已核对落盘：`train_log.jsonl`、`validation/epoch_001.npz`、`selection_test/epoch_001.npz`及best/last checkpoint均存在。epoch1学习率与原JH2调度一致；选模分数属于训练过程，不写入论文表格。暂不依据单个epoch重估整批完成时间，后续以实际早停和完整run耗时为准。

已获启动授权的最小方案：先ICBHI-only和SPRSound-only各seed42，再Coarse SPR与Native+attributes各seed42。配方可比时使用现有fresh42 Full配对，不拿三seed均值替代；若执行时发现不能复用的实质差异，先报告，不自动增加Full训练。历史HF案例已完整，本轮不追加HF训练。四个常规run按历史MPS耗时约18–21小时估算，另留汇总和写作时间；正常早停下以2–3天为项目目标，不构成时限承诺或学生个人截止时间。Hanlin资源未确认，不计入必达路径。

本轮实施口径：ICBHI-only/Coarse SPR/Native+attributes按ICBHI official-test native Score选模，SPRSound-only按本源SPR official inter native Score选模；单源仅用本源validation拟C/W阈值，另一目标在选模冻结后评估。Full保留其原ICBHI-selected checkpoint，所以SPR-only与Full的selection目标不同，按原生benchmark参照解释，不把差值全部因果归于来源共享。复用原fresh42两分数据（ICBHI3174/968、SPR5219/1437），不采用新calibration/selection三分；每epoch326更新、max50/patience10。四个新条件均从原BEATs预训练初始化开始，不从finetuned Full续训；Coarse在SPR只保留A/3，隐藏SPR细标签及其validation用途。Native匹配C/W辅助标签和1/3系数，使用native-head argmax。详细实现由模型设计写入当前规格后核对。

Full复用核对补充：仅重建fresh42原metadata与确定性batch schedule（27 epochs、每源4401个batch），ICBHI/SPR所有batch的A/C/W均非空，SPR每batch至少28个C/W eligible样本。故该已完成Full的available-node平均损失实际每节点均为1/3；不能只因公式写法与固定1/3不同就判定必须重跑。该观察不消除数据划分、选模与预算差异；Coarse变体移除C/W后仍需明确保持A系数，不能自动归一为1。无音频、模型前向或训练执行。

推理执行采用对应完整checkpoint原来的输入预处理、readout、既有validation阈值与评测单位，冻结权重，不重选epoch、不用test调阈值；不新增特征研究、训练缓存或服务器任务。已有test-selected结果按benchmark协议如实使用，不一律废弃，也不冒充validation-selected结果。查漏若确认无新推理需要，记录无需重复执行，继续后处理和写作。

本轮具体推理已完成：本地补齐HF/KAUH的BA、Macro-F1、UAR等端点，并对精确历史`PAFA_JH2_test_selected_seed42_attempt2/best_checkpoint.pt`（epoch19）完成1429个SPR inter events推理。重算native SPR official Score为0.8920104437477139，与原存档一致。历史HF-off/on的SPR C/W macro AUROC为96.806%→96.950%（+0.144pp），macro AUPRC为76.243%→75.122%（−1.121pp）；HF-on仅读取已有NPZ。新资产位于`PAFA_JH4_JH2_HFaux_seed42_attempt2/historical_jh2_spr_cw_posthoc/`，同一交付目录已更新并通知写作读取，未新增训练。

此前存量数字落文已完成：§4.2包含三seed benchmark读出/属性/外评及独立历史HF案例。新增四项训练现已全部完成，Table2最后两行与结果解释正在同步；后续聚焦主线讨论和论文收尾，不再把原四条件列为待跑。旧clean队列及原Coarse partial不恢复。

上批接收完成：修正版JSON、CSV及中文说明已完成并获写作接收；SPRSound录音/事件单位、SPR选模标注和clean-suite条件性缺项已修正，逐seed C/W平均AUROC已补齐。管理从已有三seed SPR NPZ独立复算C/W平均AUROC为97.318±0.148%（sample SD），并从原config/metadata补齐旧JH2实际训练eligibility，三个seed的重建validation样本均与已存NPZ完全匹配。旧ICBHI训练cycles依seed为3636/2880/3174，C/W均100%eligible；旧SPR各seed训练5219events，C/W各5170eligible、49masked（0.939%）。该批无新增训练或管理侧模型前向；当前新增推理授权及数字落文按上面的最新清单执行。

用户另在本地任务直接授权“补充计算HF lung和KAUH的test”：JH3.3既有checkpoint的固定外部推理已于09-09约19:00完成，无新增训练，独立保存在`result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_HF_KAUH_external_seed42/`。已通知写作读取；该资产是JH3.3外部诊断，不替代JH2 Full或HF-on配对，也不混入原CPU-only交付的执行范围。

旧clean队列暂停记录：Full42在update448附近被Ctrl-C中断，完整保存至validation point1/update326，官方test未访问；Python59399与旧queue controller59395均已结束，后续不会自动继续。best/last checkpoint、NPZ与日志保留；旧clean协议的两个单源从未启动，当前benchmark队列状态以上方最新核对为准。

- 旧备选两案：A本地六条件seed42；B Hanlin同CUDA环境核心九项＋本地带独立Full参照的机制四项。两案均待现有资产盘点后重审，不是已确定的缺失训练清单；B还取决于其资源确认。
- 暂停前14/18-run预算不再作为自动执行清单；新seed分配和恢复需用户确认。E1/属性指标从已有输出派生，不额外增加训练。
- 旧E1已存在：Full/CW-only平均Score61.17%/60.29%，逐seed差值方向不一致；仅test-selected诊断，不填新Table2。
- 本机历史同类运行4.4–5.2小时/run；HF旧run约7小时。新Full42已实际训练并保存首个validation point，但目前停止，不能把partial当完成结果。
- 写作只补仍缺内容，Abstract与Conclusion留到最后。日常改稿仍由用户在Overleaf编译；本轮指定PDF比较的特批已完成。

工作目录：`/Users/zilongzeng/Research/Acoustic`，分支`main`。当前纸稿、计划和实现修改尚未新增Git提交。

执行任务：论文写作`01a08442-92e6-7110-8399-e42eca520ea8`；模型设计`019fb42d-11d9-7b53-a6ae-d0ab010609c5`；Acoustic本地训练`01a03625-ebcd-7433-8e0e-042abd365d1e`；管理`01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae`。

[此前写作进度记录](work_plans/2026-09-08_work_plan_zh.md)保留已完成交付与摘要确认；[上次同步的Notion计划](https://app.notion.com/p/3d4309efda29819aa175ddb25fcfcf94)与09-08 Structured时段作为历史安排。本轮不自动设置新完成期限、提交Git、启动服务器或联系外部人员。Hanlin不设具体期限，seed0/1回报仍按原规则处理。
