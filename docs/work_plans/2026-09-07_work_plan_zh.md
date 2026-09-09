# Acoustic Work Plan｜2026-09-07 当日冲刺

本页已于09-08转为历史。因时间安排，未完成写作事项顺延至[09-08周二计划](2026-09-08_work_plan_zh.md)；已完成勾选保留。下方是09-07核对状态，不再作为当前排期。

更新日期：2026-09-07  
状态：SUPERSEDED / HISTORICAL PLAN

Notion：[今日 Working Plan](https://app.notion.com/p/3d4309efda29819aa175ddb25fcfcf94?pvs=204)。

依据：09-03 项目会议、09-04 story/contribution 短会，以及用户09-07最新指令。本计划替代09-05至09-06的执行顺序；原Todo 1–4未验收内容转入今天，不视为已完成。

## 1. 用户最新决定

- 已核对原Todo 1–4的工作稿、缺失证据清单与论文审阅版；用户明确下午主要放在paper写作，上午/此前已产出的准备稿按交付完成勾选。
- 执行方式改为各任务并行准备工作稿，最后集中与用户确认；不再按旧的1→2→3→4逐项等待。
- 并行产物都先是工作稿，不自动成为用户认可的最终contribution或科学结论。
- 邮件与联系Jingping放在今天研究、写作安排完成之后；当前不发送、不联系，不在研究工作之前处理外发。
- 用户已反馈让Hanlin补跑seed0和seed1。已有报告只记录seed42，新增0/1仍未收到完成结果；只有相同任务/配置/split/selection的结果才可合并统计。
- 本科生仍不设具体完成期限。Hanlin已经提供反馈，不触发“无反馈则接手”；缺少的实验与材料按实际情况协同。
- 本轮不启动项目侧训练、validation/test、feature/cache或服务器任务；需要新增实验时列明最小需求并待用户明确start。

## 2. 已完成内容（按实际产物勾选）

以下勾选表示对应工作稿、图表或核对工作已完成；不表示用户已最终认可贡献措辞、实验结果已补齐或论文已完成投稿验收。

- [x] 最接近文献定位：完成8项primary-source核对、C1/C2/C3重叠与增量判断；已修正SPR训练监督误述。
- [x] 论文缺失数据/证据清单：完成15项矩阵、必须/可后补/可删优先级、最小补齐方式及R1/R2监督匹配限制。
- [x] 结果→候选发现讨论稿：已有对应数值、artifact、研究价值与证据范围，作为后续写作输入。
- [x] Story / contribution / skeleton工作稿：已整合文献和证据缺口；最终贡献选择仍待用户确认。
- [x] main.tex及Section1–5已从placeholder推进为连续正文，含Data/Method、公式、设置、现有结果与限制。
- [x] 图表初稿：Figure1已采用最小任务/标签可用性矩阵；Figure2已生成可编辑SVG和矢量PDF。
- [x] TableI已按published/local/transfer/JH2分块，AS与official Score分列；TableII结构已完成并如实注明四项结果未有。
- [x] 四页审阅PDF已实际编译并逐页检查，未见裁切、重叠或未解析引用。这是通用IEEEtran审阅稿，尚非正式ICASSP模板验收。
- [x] 集中审阅包已汇总，今日Work Plan与Notion已同步；Hanlin补seed0/1的用户指令已记录并同步本科生任务。

集中审阅入口：[今日审阅包](../paper/ICASSP_2026_acoustic_disease/SPRINT_REVIEW_2026-09-07_zh.md)。文献、缺失证据和story文件均已有交付，不再重复排为“等待工作稿”。

## 3. 下午协作方式

- 用户下午主要在「论文写作」任务推进稿件，管理维护当前计划和验收状态。
- 文献调研与模型设计已完成本轮工作稿，下午按写作中出现的具体问题提供定向支持，不另起全面检索或实验。
- 论文修改以当前审阅稿和已有两份审计为输入；工作稿完成与最终贡献/稿件验收仍分开。
- 用户集中确认后的决定由管理同步至本地计划和Notion；无需重复要求已经给出的确认。
- 学生反馈不作为下午写作完成的前置条件；没有新完成回报不等于学生没有在执行。

## 4. 当前证据与必须保留的边界

### 已有证据

- JH2正式0/1/fresh42已完成：ICBHI Score61.17±0.31%，SPRSound official Score90.70±0.34%；ICBHI official-test选择checkpoint，属于test-selected benchmark。
- 三份ICBHI-specialized checkpoint到SPRSound的既有transfer AS：PAFA55.82、SG-SCL59.98、Patch-Mix59.38；all-Normal AS50.00。JH2另有AS90.95±0.31%。AS与official Score分开。
- HF/KAUH是fixed-checkpoint post-hoc diagnostics，不等同native reproduction或四数据集性能可比。
- Todo1结果讨论稿已有，用户随后要求paper-level innovation表述。C1已有定向文献重叠提示，整体贡献尚未最终确认。
- Hanlin已提交25组单seed42背景材料：ICBHI5、SPRSound两个tasks共10、KAUH5、HF5。没有当前所需strong-method multi-seed或固定ICBHI checkpoint的跨数据集transfer；数值缺少独立核验产物。

### 创新与写作

- DCASE2024 Task4已包含异质数据、缺失标签masking、BEATs和不同数据集评价；Bevandic、Schutera、LungMix等也需纳入最接近先例。不能把宽泛“异质监督共享/保留原生评价”直接当新原则。
- 需要具体说明本研究在respiratory cycle-flat4 / event-binary、层级属性与标注可用性上的问题、设计及受控证据增量。换一个应用领域本身不自动形成方法创新。
- 固定source head受限不等于learned features不可迁移；冻结encoder后训练target-native head是target-supervised adaptation。
- 当前主文工作范围仍为ICBHI+SPRSound；老师提到四数据集comparable performance是有条件的方向，不能提前写成已实现。
- C3若要讨论joint supervision与对齐interface的作用，仍需matched controls；R1/R2按当前定义同时改变了SPR属性监督与分类结构，未经监督匹配不能称纯hierarchy结构因果。缺少结果时限制结论，不用历史变体冒充。

## 5. 尚未完成，继续保留的证据与后续事项

- [ ] 四项prospective clean对照R1–R4：joint hierarchy、joint independent heads、ICBHI-only hierarchy、SPRSound-only hierarchy。当前无匹配结果，split/selection/budget及监督匹配尚需明确；不得用旧JH2或LocalCleanQueue替代。
- [ ] Figure1的group-aware/class-matched定量结果。若保留相应定量claim则需补；5s coverage/uncertainty只在相应覆盖或统计推断claim下必需，不阻塞写已核实的5s处理事实。
- [ ] Hanlin seed0/1完成回报与原始产物核对。已有材料仅documented seed42；相同task/config/split/selection才可汇总，不设学生具体期限。
- [ ] 合作者进度邮件与Jingping汇报。按用户要求在paper工作和集中讨论后处理，当前未发送、未联系。

完整清单：[15项缺失数据与证据矩阵](../paper/ICASSP_2026_acoustic_disease/PAPER_MISSING_EVIDENCE_2026-09-07_zh.md)。这份矩阵的编写已完成，上述实验或统计数据本身尚未完成。

## 6. 下午主线：paper写作

- [ ] 明确story与贡献定位：围绕innovation/merit完善表述，集中讨论C1/C2是否合并、C3当前可以承载的证据强度。
- [ ] 适配ICASSP官方模板与篇幅：当前为IEEEtran审阅稿，下一版以官方spconf/Paper Kit为依据重排；技术内容含图表最多4页，可选第5页仅放参考文献及官方允许的资助致谢/伦理合规声明，不能写5页正文。
- [ ] 深化正文：打磨Introduction的gap/merit衔接，检查Data/Method的标签、节点、loss与native readout说明，完善Results/Discussion的结果解释；以已有证据支持的范围写作。
- [ ] 精修Figure1/2、TableI/II及caption：让图表服务论证，核对AS/official Score、ICBHI Se、source vs local及未完成对照状态，不填虚构数字。
- [ ] 补齐用户确认的作者/单位信息，并在实际修改后重新编译、检查版面、图表浮动和引用；缺少信息不自行编造。
- [ ] 汇总下午稿件版本与剩余问题，交用户集中确认；确认后更新验收状态，再进入对外沟通环节。

当前输入：
- 写作工作区：/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic。
- [Story/contribution稿](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/STORY_CONTRIBUTIONS_DRAFT_2026-09-07_zh.md)。
- [当前四页审阅PDF](/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/review_2026-09-07.pdf)。
- [ICASSP2027官方Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)。

本区是下午排期，尚未完成项保持未勾选。本次计划核对没有直接修改论文、转换模板、重新编译、启动新实验或外发。

## 7. 集中确认与后续沟通

集中确认包需要让用户一次判断：

1. 当前story与候选contributions能否准确表达innovation和merit；
2. 相对最接近文献具体新增什么，哪些claim应收窄；
3. 论文仍缺哪些必须数据，哪些结果不足时可以删减；
4. 当前稿件与图表还需哪些修改，是否已可用于合作者/老师讨论。

只有今天研究和写作内容完成并经过集中讨论后，才进入合作者进度邮件与Jingping汇报的准备/发送环节。用户目前要求先不发邮件；收件人、渠道、正文及实际发送在该环节处理，本轮不联系任何人。

## 8. 工作位置与回报

- 管理：01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae。
- 论文写作：01a06df4-2663-7971-9b8e-ad8b6276b2f8。
- 文献调研：01a01d5b-0631-7981-a5b3-8ee4be98d98f。
- 模型设计：019fb42d-11d9-7b53-a6ae-d0ab010609c5。
- 本科生对接：019ff4a1-9b11-7d83-b370-0772f41d5163。

计划：/Users/zilongzeng/Research/Acoustic/docs/work_plans/2026-09-07_work_plan_zh.md。  
结果讨论底稿：/Users/zilongzeng/.codex/worktrees/8ad2/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/RESULT_FINDINGS_DISCUSSION_2026-09-06_zh.md。

所有回报区分草稿、用户验收和实验结果。不使用hash/checksum，不跑smoke/probe/preflight，不干预GPU用户，不因本计划启动新实验或服务器监控，不自动发送消息或Git提交。
