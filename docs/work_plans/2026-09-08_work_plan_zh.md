# Acoustic Work Plan｜2026-09-08 周二写作

更新日期：2026-09-09（原09-08计划的进度同步）  
状态：历史写作进度快照；当前安排见[09-09正文同步与最小补证计划](2026-09-09_experiment_work_plan_zh.md)。本页保留已有交付和确认记录，不再据此派发旧待办。

Notion：[周二 Working Plan](https://app.notion.com/p/3d4309efda29819aa175ddb25fcfcf94?pvs=204)。

依据：09-03项目会议、09-04 story/contribution短会及用户09-08顺延指令。因时间安排，09-07未完成的paper写作与审阅事项移至09-08（周二）；原9项已完成交付保持勾选，不重置为待办。

## 1. 用户最新决定

- 用户09-08要求论文写作任务直接使用主目录`/Users/zilongzeng/Research/Acoustic`和`main`分支。任务迁移及当时的论文版本已提交；随后四数据集补稿在main工作区更新，本轮补稿尚未Git提交。
- 本页在原09-08计划上继续记录09-09进度：原9项工作稿已交付，新增英文摘要确认与落实；其余待办按实际逐章审阅推进。
- 前期各任务已并行准备工作稿。用户09-08要求按“共享属性监督—标注条件—两层原生任务读出”主线重写，并完整覆盖四数据集：ICBHI/SPR主训练，HF单独的辅助监督测试与评测，KAUH外部评测。四数据集完整稿已补齐并编译，下一步从Introduction逐章审阅。
- 用户09-09已认可英文摘要并要求写回与中文翻译；上述交付已完成，下一步继续Introduction精修。其余章节与最终贡献仍待确认。
- 用户09-09更新写作方式：编译由用户在Overleaf完成，日常修改只更新源文件；本地编译及向管理同步均需用户明确允许，文字确认或章节完成不自动触发管理同步。
- 邮件与联系Jingping放在今天研究、写作安排完成之后；当前不发送、不联系，不在研究工作之前处理外发。
- 用户已反馈让Hanlin补跑seed0和seed1。已有报告只记录seed42，新增0/1仍未收到完成结果；只有相同任务/配置/split/selection的结果才可合并统计。
- 本科生仍不设具体完成期限。Hanlin已经提供反馈，不触发“无反馈则接手”；缺少的实验与材料按实际情况协同。
- 本轮不启动项目侧训练、validation/test、feature/cache或服务器任务；需要新增实验时列明最小需求并待用户明确start。

## 2. 已完成内容（按实际产物勾选）

以下勾选表示对应工作稿、图表或核对工作已完成；不表示用户已最终认可贡献措辞、实验结果已补齐或论文已完成投稿验收。

- [x] 最接近文献与研究定位调研：原8项primary-source定位已完成；09-08写作任务进一步交付比较15项相关工作、汇总23项来源的深度调研稿，含当前JH2实现与损失分析。报告已收到，贡献定位与建议仍待用户讨论确认。
- [x] 论文缺失数据/证据清单：完成15项矩阵、必须/可后补/可删优先级、最小补齐方式及R1/R2监督匹配限制。
- [x] 结果→候选发现讨论稿：已有对应数值、artifact、研究价值与证据范围，作为后续写作输入。
- [x] Story / contribution / skeleton工作稿：09-08共享属性主线与三条候选贡献已形成，并按用户要求用于整稿重写；最终措辞与贡献力度仍待逐章审阅，C2/C3所需发现尚未由实验建立。
- [x] 09-08四数据集完整稿：摘要、Introduction、Data、Method、Evaluation、Conclusion均已补齐ICBHI、SPRSound、HF、KAUH的角色、监督或评测设置及已有结果。
- [x] 四数据集图表工作稿：新增数据集角色表；Figure2更新为core、HF辅助监督、外部评测三条输入路径及四组读出，保留音频/标注路径与A/C/W并行预测的区分。
- [x] 结果与消融表已补齐：两源主任务结果、正式JH2三seed的HF/KAUH固定外部评测、历史单seed HF-off/on比较分别呈现；学生HF时序与KAUH九分类另列为不同任务的背景。匹配的属性标注/分类设计消融仍标明pending。
- [x] 09-08四数据集审阅PDF已编译，共7页；这是完整内容逐章审阅版，尚未压缩到投稿篇幅。写作任务已逐页视觉检查，14条引用及交叉引用解析；管理核对当前main分支稿件、PDF和7页编译日志。正式ICASSP模板与作者信息尚未完成。
- [x] 集中审阅包已汇总，今日Work Plan与Notion已同步；Hanlin补seed0/1的用户指令已记录并同步本科生任务。
- [x] Abstract英文确认与落实（09-09）：用户已认可的英文摘要已写回`main.tex`；`ABSTRACT_EN_ZH.md`已交付英文及对应中文，英文与LaTeX经文本规范化后一致。`review_2026-09-09.pdf`已编译为7页并由写作任务逐页检查；中文译文交付不等于整篇稿件已验收。

当前逐章审阅入口：[LaTeX源稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/main.tex)及用户Overleaf编译结果。[09-09七页PDF](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-09.pdf)是已编译快照；[摘要中英对照](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/ABSTRACT_EN_ZH.md)已交付，早期PDF与[09-07集中审阅包](../paper/ICASSP_2026_acoustic_disease/SPRINT_REVIEW_2026-09-07_zh.md)保留。

## 3. 逐章审阅协作

- 用户在「论文写作」任务逐章讨论并确认文字，管理维护当前计划和验收状态。
- 文献调研与模型设计已完成本轮工作稿，周二按写作中出现的具体问题提供定向支持，不另起全面检索或实验。
- 后续审阅以main分支的LaTeX源稿及用户Overleaf编译结果为准；现有09-09 PDF是已编译快照，不随每次修改自动更新。英文摘要已确认并落实，下一步Introduction，再Data/Method、Evaluation与Conclusion。
- 仅在用户明确允许同步时，写作任务向管理回报，管理再收集进度并更新本地计划和Notion；不逐次轮询写作改动，也不将文字确认自动视作同步授权。
- 学生反馈不作为周二写作完成的前置条件；没有新完成回报不等于学生没有在执行。

## 4. 当前证据与必须保留的边界

### 已有证据

- JH2正式0/1/fresh42已完成：ICBHI Score61.17±0.31%，SPRSound official Score90.70±0.34%；ICBHI official-test选择checkpoint，属于test-selected benchmark。
- 三份ICBHI-specialized checkpoint到SPRSound的既有transfer AS：PAFA55.82、SG-SCL59.98、Patch-Mix59.38；all-Normal AS50.00。JH2另有AS90.95±0.31%。AS与official Score分开。
- 正式JH2三seed的HF/KAUH固定外部评测已入稿：HF为source-test区间覆盖与recording属性存在性，KAUH为B/D/E概率聚合后的86名兼容患者。它们与学生HF时序任务、KAUH九分类结果分开呈现。
- 历史单seed HF-off/on辅助监督条件已单独入稿，不与正式JH2三seed汇总混合，也不替代尚未完成的匹配标注/分类设计消融。
- Todo1结果讨论稿已有，用户随后要求paper-level innovation表述。C1已有定向文献重叠提示，整体贡献尚未最终确认。
- Hanlin已提交25组单seed42背景材料：ICBHI5、SPRSound两个tasks共10、KAUH5、HF5。没有当前所需strong-method multi-seed或固定ICBHI checkpoint的跨数据集transfer；数值缺少独立核验产物。

### 创新与写作

- 09-08定位调研及共享属性主线讨论稿均已交付。用户已要求依该故事重写审阅稿；C1呈现完整方法设计，C2标注变化下属性复用与C3监督匹配的分类/读出作用仍是待验证的候选贡献，最终表述与补证范围尚未冻结。
- DCASE2024 Task4已包含异质数据、缺失标签masking、BEATs和不同数据集评价；Bevandic、Schutera、LungMix等也需纳入最接近先例。不能把宽泛“异质监督共享/保留原生评价”直接当新原则。
- 需要具体说明本研究在respiratory cycle-flat4 / event-binary、层级属性与标注可用性上的问题、设计及受控证据增量。换一个应用领域本身不自动形成方法创新。
- 固定source head受限不等于learned features不可迁移；冻结encoder后训练target-native head是target-supervised adaptation。
- 当前论文完整覆盖四数据集，主模型训练仍只有ICBHI/SPR；HF source-train只用于单列的辅助监督条件，HF source-test与KAUH不进入共享模型训练、选优或阈值拟合。四数据集覆盖不等于四源联合训练、恢复全部native标签或四数据集性能可比。
- HF监督表述已按保存config与实际_make_hf_windows/_hf_loss修正：窗口含任一D/W/R/S时C/W同时eligible，对应标签存在设1、未出现设0；gap/empty/phase-only整体mask。应称含标注派生负目标的辅助BCE，说明标注完整性假设；旧“纯positive-only”简称不再作为当前依据。HF不监督A，也不参与阈值拟合、模型选择或早停。
- 写作任务另报告HF旧helper含继承的epoch19/threshold来源字段，而JH4顶层selection为epoch9；本稿未沿用这些继承字段。本轮没有新增模型执行核验，不能把旧helper文字当作已核准的模型选择来源。
- C3若要讨论joint supervision与对齐interface的作用，仍需matched controls；R1/R2按当前定义同时改变了SPR属性监督与分类结构，未经监督匹配不能称纯hierarchy结构因果。缺少结果时限制结论，不用历史变体冒充。

## 5. 尚未完成，继续保留的证据与后续事项

下列实验数据、学生回报和外部沟通仍按原依赖处理，不因周二排期而被视为已授权运行或承诺当日拿到结果；本次Structured排入的是第6节的写作与审阅工作。

- [ ] 与主张对应的最小补证方案及结果：既有R1–R4（joint hierarchy、joint independent heads、ICBHI-only hierarchy、SPRSound-only hierarchy）仍无匹配结果；09-08稿另列属性标注变化、信息匹配分类设计与固定分数gate比较问题，尚未冻结或启动实验矩阵。需先明确问题、split/selection/budget、属性指标及保留损失权重；旧JH2或LocalCleanQueue不替代新结果。
- [ ] Figure1的group-aware/class-matched定量结果。若保留相应定量claim则需补；5s coverage/uncertainty只在相应覆盖或统计推断claim下必需，不阻塞写已核实的5s处理事实。
- [ ] Hanlin seed0/1完成回报与原始产物核对。已有材料仅documented seed42；相同task/config/split/selection才可汇总，不设学生具体期限。
- [ ] 合作者进度邮件与Jingping汇报。按用户要求在paper工作和集中讨论后处理，当前未发送、未联系。

完整清单：[15项缺失数据与证据矩阵](../paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)。这份矩阵的编写已完成，上述实验或统计数据本身尚未完成。

## 6. 当前主线：paper逐章审阅（09-09进度）

- [ ] 摘要确认后继续Introduction精修：审阅共享属性主线、四数据集分工、gap/merit与候选贡献文字；C2/C3的待验证状态须与实际证据相符。
- [ ] 逐章审阅后压缩并适配ICASSP官方模板：当前为7页IEEEtran完整内容审阅稿；以官方spconf/Paper Kit重排，技术内容含图表最多4页，可选第5页仅放参考文献及官方允许的资助致谢/伦理合规声明。
- [ ] 逐章审阅Data/Method、Evaluation与Conclusion：四数据集补稿已完成；Introduction确认后依次检查core与HF辅助条件、外部读出、标注派生负目标假设、不同seed/任务的比较条件及结果解释。
- [ ] 审阅并精修四数据集角色表、Figure1/2及各结果/消融表caption：核对AS/official Score、ICBHI Se、HF区间/recording读出、KAUH患者聚合、学生native任务差异，以及单seed/三seed和pending结果的区分。
- [ ] 补齐用户确认的作者/单位信息，并按用户提供的Overleaf编译反馈修正文稿、版面和引用问题；编译由用户完成，只有明确要求时才本地编译。
- [ ] 在用户允许同步的节点汇总稿件版本与剩余问题，管理再更新验收状态；对外沟通继续遵循原有用户授权安排。

当前输入：
- 写作工作区：/Users/zilongzeng/Research/Acoustic。
- [09-08研究定位与贡献深度调研稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/DEEP_NOVELTY_GAP_REVIEW_2026-09-08_zh.md)：已完成的文献与代码分析，作为重写及审阅输入；没有新增实验结果。
- [09-08共享属性主线与贡献讨论稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/SHARED_ATTRIBUTE_STORY_NOVELTY_CONTRIBUTIONS_2026-09-08_zh.md)。
- [09-09摘要更新后7页PDF快照](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-09.pdf)；后续源稿修改不自动更新该PDF，09-08与09-07 PDF保留。
- [摘要英文与中文对照](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/ABSTRACT_EN_ZH.md)：英文已确认并写回，中文为对应译文交付。
- [ICASSP2027官方Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)。

本区保留尚未完成的逐章确认、投稿压缩、模板、作者信息与最终审阅。四数据集7页补稿、图表和编译由写作任务按用户要求完成；本轮补稿及管理计划更新尚未Git提交。没有新增实验、模型重评或外发。

## 7. 集中确认与后续沟通

Abstract英文已确认并落实；接下来从Introduction继续逐章审阅，最后集中判断：

1. 当前story与候选contributions能否准确表达innovation和merit；
2. 相对最接近文献具体新增什么，哪些claim应收窄；
3. 论文仍缺哪些必须数据，哪些结果不足时可以删减；
4. 当前稿件与图表还需哪些修改，是否已可用于合作者/老师讨论。

稿件逐章审阅完成并经过集中确认后，再进入合作者进度邮件与Jingping汇报的准备/发送环节。用户目前要求先不发邮件；收件人、渠道、正文及实际发送在该环节处理，本轮不联系任何人。

## 8. 工作位置与回报

- 管理：01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae。
- 论文写作（已迁移至主目录main）：01a08442-92e6-7110-8399-e42eca520ea8；替代原独立工作区任务01a06df4-2663-7971-9b8e-ad8b6276b2f8。
- 文献调研：01a01d5b-0631-7981-a5b3-8ee4be98d98f。
- 模型设计：019fb42d-11d9-7b53-a6ae-d0ab010609c5。
- 本科生对接：019ff4a1-9b11-7d83-b370-0772f41d5163。

计划：/Users/zilongzeng/Research/Acoustic/docs/work_plans/2026-09-08_work_plan_zh.md。  
结果讨论底稿：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/RESULT_FINDINGS_DISCUSSION_2026-09-06_zh.md。

所有回报区分草稿、用户验收和实验结果。不使用hash/checksum，不跑smoke/probe/preflight，不干预GPU用户，不因本计划启动新实验或服务器监控，不自动发送消息或Git提交。

## 9. Structured 排期

复用现有未完成任务「Acoustic写作」，从09-07顺延至09-08（周二）21:00–23:00，America/Chicago；保留原120分钟写作时段。第6节六项写作待办作为子任务加入，均保持未完成。本次只是安排日程，不执行论文修改或新实验。
