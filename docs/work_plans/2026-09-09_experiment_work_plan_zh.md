# Acoustic Work Plan｜2026-09-09 正文对齐与最小补证

更新日期：2026-09-09  
状态：按用户最新要求暂停全部本地训练及旧自动队列；2–3天方案与Hanlin分工重新讨论中。此前启动授权暂停，新方案确认前不恢复或新开run。

暂停时Full42已运行到update448附近，完整持久保存至validation point1/update326；Python进程与旧队列控制器均已结束。官方test未访问，后续两个single-source未启动。现有checkpoint不包含完整optimizer/RNG续跑状态，不宣称可无损接续。

## 1. 本轮目标与授权

用户要求核对本地稿与Overleaf导出，只处理当前仍缺失的正文内容；Abstract和Conclusion留到最后。正文版本一致后，再开始所需补证。此次明确允许写作任务为比较渲染/编译PDF，并向管理返回比较结论；不恢复日常改稿后的自动编译或逐项回报。

- Overleaf输入：`/Users/zilongzeng/Downloads/ICASSP_2026_acoustic_disease (4).pdf`。
- 正文来源：`docs/paper/ICASSP_2026_acoustic_disease/`当前main工作区。
- 本轮实验先走CPU E1，再走新的validation-selected核心对照。旧R0、J-Clean和LocalCleanQueue不恢复，也不将旧test-selected结果填入新Table 2。
- 启动条件：正文内容对齐，或用户明确允许以main与已冻结规格为准；对应阶段输入/实现已准备后，再由管理向本地任务发送具体启动命令。当前授权不扩展到服务器或自动排满18个run。
- [PDF比较报告](../paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/COMPARISON_REPORT_zh.md)确认实验数字、主要方法、四数据集分工及表格结构一致。Overleaf尚有Evaluation引用/读出术语和Figure1 caption同步差异；用户随后明确回复“可以开跑”，本批以main和冻结规格为准，文字同步不再阻塞该批。

## 2. 已完成的准备与真实缺口

- [x] 七行Table 2及Claude建议已核对。历史E1不仅有NPZ，三个selected metrics JSON已保存Full/CW-only对照；single-source原先并非已有配置开关，现已接入新runner。
- [x] 三项设计选择已有完整文本：共同validation selection、Native+attributes/Coarse损失与信息范围、预先分配的种子预算。见[实验规格](../paper/ICASSP_2026_acoustic_disease/EXPERIMENT_SPEC_2026-09-09_zh.md)。
- [x] 本地环境/数据盘点完成：48 GiB内存、约534 GiB可用磁盘；核心数据和BEATs初始化文件存在。沙箱外只读查询为torch 2.12.1、MPS built/available均True；未通过模型forward或性能测试测可用性。
- [x] [CPU E1脚本](../../scripts/analysis/jh2_cw_readout_pilot.py)已准备；[已有指标汇总](../../result/reproduce/pafa_joint_hierarchy/PAFA_JH2_CW_READOUT_PILOT_EXISTING/cw_readout_existing_pilot.md)已交付。只读取旧JSON，没有从NPZ独立重建或新增模型推理。
- [x] Overleaf与本地正文已渲染并比较，实质文字差异与排版差异已分开列出。
- [x] canonical official-train清单与必要support已核对；ICBHI held-out患者池已按预定metadata规则平衡分配，未使用模型或test结果。
- [x] [新clean runner](../../baseline/pafa/table2_clean_controls.py)已实现active-source、固定权重及六个训练变体的core路径；5项定向metadata/loss/selection测试在Beats环境通过。管理另核对实际Full/I-only/SPR-only loader与manifest的ID、partition及target/eligibility，mismatch=0。
- [x] 此前用户允许的seed42批次已启动Full42；随后按用户新指令停止，已保留partial产物。此项记录历史执行，不授权恢复。
- [ ] 按冻结协议启动并完成正式核心对照；当前没有用smoke或model forward预跑。
- [ ] HF/KAUH新selected checkpoint的独立external evaluator接线，在相应阶段运行前完成；不改动正在使用的core协议。

## 3. 开跑前固定的三项规格

### Selection与数据划分

采用现有canonical official-train外层record assignments；model seed不改变split。SPR保留固定seed42的inner分组。ICBHI原inner selection只有31个异常cycle，因此在固定16-patient held-out池内，按预先写定的患者支持约束与标签/样本/患者平衡objective做一次确定性分配；outer subtrain不变，不根据性能更换划分。

最终ICBHI：subtrain3055 cycles/63 patients，calibration700/5，selection387/11；selection N243/C72/W30/Both42，异常144。最终SPR：subtrain5219/194，calibration694/26，selection743/23。所有必要native class与C/W正负支持均非零，calibration/selection互不共享患者。ICBHI calibration中patient130占507/700 cycles（72.4%），作为固定划分的限制保留，不再根据结果修改。

每326个core optimizer updates进行一次validation。C/W阈值只在calibration groups拟合；checkpoint最大化active训练源的native validation Score等权均值。ICBHI使用(Sp+异常细类Se)/2，SPR使用其official Score；单源自然退化为本源Score。这个均值仅作选择效用，不作为合并测试指标。最大50个validation点，patience10，分数相同保留更早点。HF及source-only的目标域不进入选择或校准；official test仅在模型和阈值固定后评测。

### Native+attributes与Coarse

Full各节点固定1/3权重，某节点无eligible样本时置零、不将其权重分给其他节点。Coarse SPR只保留1/3的A损失，C/W细标签不进入训练、校准、选择或早停；PAFA权重保持。

Native+attributes使用ICBHI四分类头、SPR二分类头及共享C/W辅助头，损失为1/3 native CE + 1/3 C BCE + 1/3 W BCE。辅助属性targets、eligibility与有效权重匹配Full；native分类不使用辅助头，但保留其概率用于SPR属性AUROC。该比较仍改变native目标结构，结论属于分类/监督接口，不包装成纯head形状效应。

### 种子与预算

暂停前的预算为14个训练run：Full、ICBHI-only、SPR-only、Full+HF各用0/1/42；Coarse SPR和Native+attributes各固定seed42。该预算与以下队列现均暂停，2–3天窗口下的新种子数量与分工待用户确认。既有split、loss与selection定义暂不改变。

18个run是扩展预算：为Coarse和Native各增加0/1。是否扩展只能由事先的时间/资源预算决定，不能按测试结果选择要补的seed。当前不自动启动扩展预算。

## 4. 本地执行顺序

| 阶段 | 工作 | 新训练run数 | 输出与解释 |
|---|---|---:|---|
| 0 | 读取并汇总现有JH2的E1，正式0/1/fresh42 | 0 | 已完成。直接复用旧JSON中的fixed-score/fixed-threshold结果；不重复推理。仍是test-selected诊断，不填Table 2。 |
| 1 | 新Full、ICBHI-only、SPR-only，各0/1/42 | 9 | 新Table 2的共同reference及source-sharing核心结果；同seed配对，额外用Full NPZ生成正式E1。 |
| 2 | Coarse SPR，seed42 | 1 | 细属性信息移除的单seed结果，C2证据范围据此限定。 |
| 3 | Native+attributes，seed42 | 1 | 属性信息匹配下分类/监督接口的单seed结果。 |
| 4 | Full+HF，0/1/42 | 3 | 与相同seed的Full配对，HF仅独立辅助条件；额外HF计算不称compute-matched。 |

核心九项内部顺序：先seed42的Full→ICBHI-only→SPR-only，再seed0三项，再seed1三项，以尽早获得一组完整配对。种子分配不会因第一组结果改变。每个run独立输出，MPS顺序运行，不并发争用同一设备。

旧seed42自动串行授权已被用户的暂停指令覆盖；任何恢复、重启或新分工运行均需新的明确指令。旧队列不能自行继续。

当前论文的E1是固定阈值分析。Claude提出的validation重新拟合属于另一种校准敏感性分析；若做，需单独定义并对齐双方的校准目标，不冒充固定阈值行。0.2或2–3 pp都不是自动判定贡献有无的阈值；需看逐seed一致性、类别变化和证据范围。

已有E1：Full/CW-only ICBHI Score为seed0 61.48/60.89、seed1 60.85/58.25、fresh42 61.17/61.73；CW−Full分别为−0.60、−2.60、+0.56 pp。三次run的Full均值61.17%，CW-only均值60.29%，配对差值−0.88±1.60 pp。CW-only的Sp平均低6.80 pp、异常细类Se高5.04 pp，Both recall每对相同。这支持旧模型中的读出取舍描述，不能证明gate稳定提升，也不是新clean结果。历史校准划分随seed变化，checkpoint按Full的ICBHI test Score选取，需保留该范围。

## 5. 时间与资源可行性

本机历史5 s full-FT JH2：seed0为312.97分钟/25轮，seed1为264.93分钟/21轮，fresh42为274.01分钟/27轮，均为MPS、batch32、cpu_threads4。即约4.4–5.2小时/run、10.1–12.6分钟/validation间隔；不是75分钟/run。

- 核心9个训练仅以历史早停范围作预算参照，约40–47小时；若都达到50点，上述速度约需76–95小时。
- 新selection会改变早停位置；single-source及HF额外计算也会改变耗时。因此不把上述乘法当作完成时间承诺。
- 14/18-run全套需要数日串行计算，HF成本单独计入；先取得E1和核心对照比一次性排满全套更有决策价值。
- 第一批结果的目标截止时间已向用户询问，准备工作继续进行。

## 6. 写作配合

对八项旧建议的核对表明：Intro数字槽位不再适用（用户已要求Intro不放实验数字），contribution已融入问题/工作概述，Table1破折号含义已写明，frozen背景限定已有。上述不重做。

真正剩余的是：同步Evaluation最新引用/术语和Figure1 caption；Figure1与§2.2的实际内容；三项已冻结实验规格落文；对应划分的mask数量；有必要时补Resp-Agent的精炼primary-source定位。ICBHI-only是新Table2统一validation协议中的单源参照，不与Table1文献/test-selected行强称受控“协议桥梁”，也不为这个措辞新增实验。Abstract/Conclusion暂不修改。

已核对的Table2 metadata：ICBHI subtrain3055个cycle均支持C/W；SPR subtrain5219个event中5170支持C/W、49个mask（约0.94%，分母是SPR独立event，不是batch exposure）。SPR calibration694均支持，selection743中738支持、5个mask。写作应标明这是新canonical划分，不能套到旧JH2每seed的不同划分。

## 7. 分工与当前交付

- 论文比较与正文：`01a08442-92e6-7110-8399-e42eca520ea8`（论文写作）。
- 规格、metadata split与实现：`019fb42d-11d9-7b53-a6ae-d0ab010609c5`（模型设计）。
- CPU E1及后续本地运行：`01a03625-ebcd-7433-8e0e-042abd365d1e`（Acoustic本地训练）。
- 管理：`01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae`。

本轮不运行smoke/probe，不计算hash，不自动提交Git或联系外部人员。设计准备、CPU后处理结果与正式训练结果分别记录；是否启动下一阶段按本轮用户授权和已满足的具体条件执行。

此前纸稿、摘要确认、Figure2等交付见[09-08计划的09-09进度记录](2026-09-08_work_plan_zh.md)。Hanlin seed0/1反馈、作者信息、最终4+1压缩以及合作者/Jingping沟通仍保留原依赖；学生不设具体期限，不在本轮擅自联系。
