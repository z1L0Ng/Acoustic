# Acoustic Work Plan｜2026-09-08 周二写作

更新日期：2026-09-08  
状态：ACTIVE / CHAPTER REVIEW / INTRODUCTION NEXT

Notion：[周二 Working Plan](https://app.notion.com/p/3d4309efda29819aa175ddb25fcfcf94?pvs=204)。

依据：09-03项目会议、09-04 story/contribution短会及用户09-08顺延指令。因时间安排，09-07未完成的paper写作与审阅事项移至09-08（周二）；原9项已完成交付保持勾选，不重置为待办。

## 1. 用户最新决定

- 用户09-08要求论文写作任务直接使用主目录`/Users/zilongzeng/Research/Acoustic`和`main`分支。最新正文、图表、调研稿与审阅PDF已合入该目录，后续章节修改以此为准。
- 用户将剩余paper写作与审阅工作安排到09-08（周二）完成；本计划只顺延现有任务，保留已核对的9项工作稿交付。
- 前期各任务已并行准备工作稿。用户09-08进一步要求按“共享属性监督—标注条件—两层原生任务读出”主线重写整稿，再从Introduction开始逐章审阅；本轮重写与编译已完成，逐章确认尚未完成。
- 并行产物都先是工作稿，不自动成为用户认可的最终contribution或科学结论。
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
- [x] 09-08整稿重写：标题、摘要、Introduction、Data and Shared Supervision、Method、Evaluation、Conclusion已围绕共享属性主线形成连续英文审阅稿。
- [x] 图表工作稿：Figure1保留任务/标签可用性矩阵；09-08 Figure2已更新可编辑SVG与矢量PDF，分清音频和标注路径、A/C/W并行预测与原生任务读出。
- [x] 09-08 TableI已按原生任务呈现学生五种冻结编码器背景、本地ICBHI专家和JH2，明确训练与选择条件；TableII改为主线对应的消融问题并标明结果pending。固定head的SPR迁移分数不再主导正文主线，仍保留为历史背景。
- [x] 09-08五页审阅PDF已编译；技术正文位于前4页、参考文献延续第5页。写作任务已逐页视觉检查并确认14条引用解析；管理核对新源文件、PDF和编译日志。当前仍是IEEEtran章节审阅稿，正式ICASSP模板与作者信息尚未完成。
- [x] 集中审阅包已汇总，今日Work Plan与Notion已同步；Hanlin补seed0/1的用户指令已记录并同步本科生任务。

当前逐章审阅入口：[09-08五页审阅PDF](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-08.pdf)。[09-07集中审阅包](../paper/ICASSP_2026_acoustic_disease/SPRINT_REVIEW_2026-09-07_zh.md)保留为先前证据汇总，不代表当前稿件版本。

## 3. 周二协作方式

- 用户周二主要在「论文写作」任务推进稿件，管理维护当前计划和验收状态。
- 文献调研与模型设计已完成本轮工作稿，周二按写作中出现的具体问题提供定向支持，不另起全面检索或实验。
- 当前以review_2026-09-08.pdf及对应LaTeX为逐章审阅基线，先Introduction，再Data/Method、Evaluation与Conclusion；已有定位与证据分析作为输入，整稿重写完成与逐章验收分开。
- 用户集中确认后的决定由管理同步至本地计划和Notion；无需重复要求已经给出的确认。
- 学生反馈不作为周二写作完成的前置条件；没有新完成回报不等于学生没有在执行。

## 4. 当前证据与必须保留的边界

### 已有证据

- JH2正式0/1/fresh42已完成：ICBHI Score61.17±0.31%，SPRSound official Score90.70±0.34%；ICBHI official-test选择checkpoint，属于test-selected benchmark。
- 三份ICBHI-specialized checkpoint到SPRSound的既有transfer AS：PAFA55.82、SG-SCL59.98、Patch-Mix59.38；all-Normal AS50.00。JH2另有AS90.95±0.31%。AS与official Score分开。
- HF/KAUH是fixed-checkpoint post-hoc diagnostics，不等同native reproduction或四数据集性能可比。
- Todo1结果讨论稿已有，用户随后要求paper-level innovation表述。C1已有定向文献重叠提示，整体贡献尚未最终确认。
- Hanlin已提交25组单seed42背景材料：ICBHI5、SPRSound两个tasks共10、KAUH5、HF5。没有当前所需strong-method multi-seed或固定ICBHI checkpoint的跨数据集transfer；数值缺少独立核验产物。

### 创新与写作

- 09-08定位调研及共享属性主线讨论稿均已交付。用户已要求依该故事重写审阅稿；C1呈现完整方法设计，C2标注变化下属性复用与C3监督匹配的分类/读出作用仍是待验证的候选贡献，最终表述与补证范围尚未冻结。
- DCASE2024 Task4已包含异质数据、缺失标签masking、BEATs和不同数据集评价；Bevandic、Schutera、LungMix等也需纳入最接近先例。不能把宽泛“异质监督共享/保留原生评价”直接当新原则。
- 需要具体说明本研究在respiratory cycle-flat4 / event-binary、层级属性与标注可用性上的问题、设计及受控证据增量。换一个应用领域本身不自动形成方法创新。
- 固定source head受限不等于learned features不可迁移；冻结encoder后训练target-native head是target-supervised adaptation。
- 当前主文工作范围仍为ICBHI+SPRSound；老师提到四数据集comparable performance是有条件的方向，不能提前写成已实现。
- C3若要讨论joint supervision与对齐interface的作用，仍需matched controls；R1/R2按当前定义同时改变了SPR属性监督与分类结构，未经监督匹配不能称纯hierarchy结构因果。缺少结果时限制结论，不用历史变体冒充。

## 5. 尚未完成，继续保留的证据与后续事项

下列实验数据、学生回报和外部沟通仍按原依赖处理，不因周二排期而被视为已授权运行或承诺当日拿到结果；本次Structured排入的是第6节的写作与审阅工作。

- [ ] 与主张对应的最小补证方案及结果：既有R1–R4（joint hierarchy、joint independent heads、ICBHI-only hierarchy、SPRSound-only hierarchy）仍无匹配结果；09-08稿另列属性标注变化、信息匹配分类设计与固定分数gate比较问题，尚未冻结或启动实验矩阵。需先明确问题、split/selection/budget、属性指标及保留损失权重；旧JH2或LocalCleanQueue不替代新结果。
- [ ] Figure1的group-aware/class-matched定量结果。若保留相应定量claim则需补；5s coverage/uncertainty只在相应覆盖或统计推断claim下必需，不阻塞写已核实的5s处理事实。
- [ ] Hanlin seed0/1完成回报与原始产物核对。已有材料仅documented seed42；相同task/config/split/selection才可汇总，不设学生具体期限。
- [ ] 合作者进度邮件与Jingping汇报。按用户要求在paper工作和集中讨论后处理，当前未发送、未联系。

完整清单：[15项缺失数据与证据矩阵](../paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)。这份矩阵的编写已完成，上述实验或统计数据本身尚未完成。

## 6. 周二主线：paper写作（09-08）

- [ ] 从Introduction开始逐章确认：共享属性主线已用于用户要求的本轮重写；接下来审阅gap、merit与三条候选贡献的具体文字，C2/C3的待验证状态须与结果相符。
- [ ] 适配ICASSP官方模板与篇幅：当前为IEEEtran审阅稿，下一版以官方spconf/Paper Kit为依据重排；技术内容含图表最多4页，可选第5页仅放参考文献及官方允许的资助致谢/伦理合规声明，不能写5页正文。
- [ ] 逐章审阅Data/Method、Evaluation与Conclusion：整稿重写已完成；Introduction确认后依次检查标签与监督、两层语义/决策组织、比较条件和结果解释，再按用户反馈修改。
- [ ] 审阅并精修09-08版Figure1/2、TableI/II及caption：方法图与表格工作稿已更新，下一步随章节确认其论证作用、AS/official Score、ICBHI Se、不同训练条件及待完成消融的表述。
- [ ] 补齐用户确认的作者/单位信息，并在实际修改后重新编译、检查版面、图表浮动和引用；缺少信息不自行编造。
- [ ] 汇总周二稿件版本与剩余问题，交用户集中确认；确认后更新验收状态，再进入对外沟通环节。

当前输入：
- 写作工作区：/Users/zilongzeng/Research/Acoustic。
- [09-08研究定位与贡献深度调研稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/DEEP_NOVELTY_GAP_REVIEW_2026-09-08_zh.md)：已完成的文献与代码分析，作为重写及审阅输入；没有新增实验结果。
- [09-08共享属性主线与贡献讨论稿](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/SHARED_ATTRIBUTE_STORY_NOVELTY_CONTRIBUTIONS_2026-09-08_zh.md)。
- [当前09-08五页章节审阅PDF](/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-08.pdf)；09-07四页稿保留为旧版。
- [ICASSP2027官方Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)。

本区保留尚未完成的逐章确认、模板、作者信息与最终审阅。09-08整稿重写、方法图更新和编译由写作任务按用户要求完成；本次管理只核对交付并同步计划，没有新增实验或外发。

## 7. 集中确认与后续沟通

用户先从Introduction开始逐章审阅，最后集中判断：

1. 当前story与候选contributions能否准确表达innovation和merit；
2. 相对最接近文献具体新增什么，哪些claim应收窄；
3. 论文仍缺哪些必须数据，哪些结果不足时可以删减；
4. 当前稿件与图表还需哪些修改，是否已可用于合作者/老师讨论。

稿件逐章审阅完成并经过集中确认后，再进入合作者进度邮件与Jingping汇报的准备/发送环节。用户目前要求先不发邮件；收件人、渠道、正文及实际发送在该环节处理，本轮不联系任何人。

## 8. 工作位置与回报

- 管理：01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae。
- 论文写作：01a06df4-2663-7971-9b8e-ad8b6276b2f8。
- 文献调研：01a01d5b-0631-7981-a5b3-8ee4be98d98f。
- 模型设计：019fb42d-11d9-7b53-a6ae-d0ab010609c5。
- 本科生对接：019ff4a1-9b11-7d83-b370-0772f41d5163。

计划：/Users/zilongzeng/Research/Acoustic/docs/work_plans/2026-09-08_work_plan_zh.md。  
结果讨论底稿：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/RESULT_FINDINGS_DISCUSSION_2026-09-06_zh.md。

所有回报区分草稿、用户验收和实验结果。不使用hash/checksum，不跑smoke/probe/preflight，不干预GPU用户，不因本计划启动新实验或服务器监控，不自动发送消息或Git提交。

## 9. Structured 排期

复用现有未完成任务「Acoustic写作」，从09-07顺延至09-08（周二）21:00–23:00，America/Chicago；保留原120分钟写作时段。第6节六项写作待办作为子任务加入，均保持未完成。本次只是安排日程，不执行论文修改或新实验。
