# ICASSP 论文缺失数据与证据清单｜2026-09-07

状态：**DRAFT_READY_FOR_CENTRAL_REVIEW**  
范围：基于现有 artifact 的有界审计与最小补证安排；不是用户验收、实验结果、贡献冻结或实验启动许可。  
主文当前范围：ICBHI cycle flat4 + SPRSound BioCAS2022 inter event binary。HF Lung 与 KAUH 仅可作为 supporting diagnostics。

## 1. 结论先行

今天仅靠现有材料，可以完成问题动机、JH2 test-selected benchmark、原生任务与 eligibility 机制、ICBHI 的 Sp/Se/per-class 分解、以及 HF/KAUH 的有边界 supporting discussion。不能完成的核心主张是：

1. hierarchy 相对 matched independent heads 有增益；
2. joint training 相对相同配方的两个 single-source models 有增益或更好 retention；
3. 当前 joint 模型达到 clean validation-selected 性能。

这三项都必须等待一套新的 prospective clean seed42 四行对照。旧 `LocalCleanQueue`、2 s frozen-feature attribution 和 JH2 official-test-selected 三种子不能替代该结果。

Figure 1 当前已有 native-unit composition、13,704 recordings 的声学特征表、每数据集 112 groups 的描述性 PCA 和若干 signal-level descriptors；若要保留定量 acoustic-separation 或用5 s处理解释性能，还缺 group-aware quantitative separability、class-matched 复核、完整 5 s repeat/truncate coverage 和相应的 group-level uncertainty。Table I 的 published/local/transfer/JH2 行已有大部分数字，但必须按 task、unit、selection 和 metric 分块，尤其不得把 SPRSound `AS` 与 official `Score` 混用。

## 2. 证据口径

- **Verified existing artifact**：只表示本审计确认文件及其中记录存在；本轮没有重新评分、重新统计或独立复现实验。
- **Candidate claim**：写作候选，需由用户集中确认；不因写入本文而冻结。
- **Proposed minimum**：建议的最小补齐方式；不自动授权实现、训练或分析。
- **HOLD**：缺少结果、协议或可比语义；不可把“不可评价”解释为“性能差”。

ICBHI 指标必须写准确：`Sp` 是 Normal recall；`Se` 是异常 cycles 中 **Crackle、Wheeze、Both 细类被正确分类**的比例。Crackle 被判成 Wheeze 仍计错，因此 ICBHI Se 不是二分类 abnormal detection sensitivity。ICBHI `Score=(Sp+Se)/2`。

SPRSound Task1-1 中 `AS=(Se+Sp)/2`，`HS` 是 Se 与 Sp 的调和平均，official `Score=(AS+HS)/2`。现有固定 ICBHI checkpoint 到 SPRSound 的 55.82、59.98、59.38 是 binary AS；JH2 的 90.70±0.34 是 official Score，另有 AS 90.95±0.31。二者不得放入同一无标注数值列。

## 3. Claim—证据—缺口总表

| ID | Paper 问题 / candidate claim | 已有 artifact | 具体缺失 | 优先级 | 最小补齐方式 | 负责人 | 工作类型 | 今天能完成 | 缺结果时必须如何收窄 |
|---|---|---|---|---|---|---|---|---|---|
| E1 | **现有 JH2 是否证明一个模型可同时输出两项 native task？** | `PAFA_JH2_main_multiseed/multiseed_summary.{md,json}`：seed 0/1/fresh42；ICBHI 61.17±0.31，SPRSound official Score 90.70±0.34；每 seed 同一 selected checkpoint。 | 缺 clean validation-selected companion；三 seed checkpoint 均由 ICBHI official-test Score 选取。还需在正文明确 ICBHI Se 的细类定义、每类 support/recall、SPRSound AS/HS/Score。 | 必须整理；clean result 必须另补 | 现有三 seed只放 benchmark stream，醒目标 `ICBHI-test-selected`；从现有 summary 转写 Sp/Se、per-class recall/support 与 SPR AS/HS/Score，不重新评分。 | 论文写作；模型设计核对口径 | 仅整理 | 可完成 benchmark 表行与边界文字 | 只能称“test-selected benchmark 下展示两项输出”；不能称 clean generalization、unbiased estimate 或 joint/hierarchy gain。 |
| E2 | **Prospective clean joint hierarchy 是否成立？** | JH2 hierarchy、BEATs/PAFA 训练主体、loader、native decoder 已存在。旧 `LocalCleanQueue` 已被用户排除为本suite的reference，不能提供现行split或结果。 | 新四-run suite 的 canonical group lists、selection 公式、共同 update budget/validation cadence、output schema与 runner 均未冻结/实现；没有 clean result。 | 必须 | 见第4节：从 canonical official-train records 独立预冻结 group-safe subtrain/calibration/selection；rows E2/E3 共用 validation native composite；test 在 epoch/threshold 冻结后各一次。 | 模型设计；管理/用户批准 | 新实现 + 新训练 | 只能完成合同候选与缺口说明 | 没有结果时，JH2 留在 Table I benchmark；Table II joint-clean 行标 `Not run`，不得把 JH2 或旧 queue 填入。 |
| E3 | **Eligibility-aware hierarchy 是否优于 matched joint independent heads？** | 旧 frozen-encoder `joint_native.py` 和 AST `shared_encoder_native_heads` 含 independent native heads 思路；不能作为当前 BEATs full-finetune/PAFA 对照。当前hierarchy还利用SPR raw7中有支持的Crackle/Wheeze event supervision；Rhonchi/Stridor只监督Level1。 | 缺 matched BEATs full-finetune joint run：ICBHI direct flat4 head、SPR direct binary head；其余 preprocessing、encoder、PAFA branch、batching、updates、optimizer、selection 必须与 E2 一致。更关键的是，SPR-binary direct head会丢弃hierarchy所用的attribute supervision，因此R1/R2同时改变监督信息与classification结构。 | 必须，claim-critical | 当前四行不自动增加新实验轴。运行前二选一并预注册：要么另行定义matched-supervision comparator；要么保留native binary independent head，并把R1−R2结论限定为整个“eligibility-aware supervision/classification interface”效果，不能称纯hierarchy结构因果。两行仍共用同一 validation native composite。 | 模型设计；管理/用户批准 | 新实现 + 新训练 | 可写matching限制和待决项，不能写结果 | 删除“纯hierarchy结构带来增益/不可替代”；未闭合监督匹配时只能把hierarchy作为已实现的整体interface，Contribution保持hypothesis。 |
| E4 | **Joint hierarchy 对 ICBHI 的作用是什么？** | `beats_icbhi_attribution.py` B1 是 ICBHI-only、2 s、frozen-feature diagnostic；历史 PAFA/BEATs+CE single-source rows存在。 | 缺与 E2 完全 matched 的 ICBHI-only hierarchy：相同 5 s package、full fine-tune、PAFA branch、seed42和update schedule；缺 source-only calibration/selection及 frozen transfer 到 SPR 的结果。 | 必须 | 只移除 SPR training batches；checkpoint 仅由 ICBHI internal selection 选，Crackle/Wheeze threshold 仅由 ICBHI calibration 选；selected checkpoint 后各访问 ICBHI test和 SPR inter一次，SPR不参与训练/selection/calibration。 | 模型设计 | 新实现 + 新训练 | 只能记录现有近邻为何不匹配 | 没有此行不能把 joint 与 ICBHI retention 的差异归因给 joint training；2 s B1最多脚注，不进主对照。 |
| E5 | **Joint hierarchy 对 SPRSound 的作用是什么？** | SPR loader、Task1-1 native readout与 hierarchy eligibility mapping存在。 | 缺 SPR-only hierarchy runner/result；缺 source-only selection、SPR-only threshold fitting以及 selected fixed checkpoint 到 ICBHI 的 transfer。 | 必须 | 只用 SPR subtrain；checkpoint 仅由 SPR Task1-1 internal selection 选。若报告 Crackle/Wheeze transfer阈值，只能在 SPR calibration 上冻结；ICBHI不能参与任何选择。selected checkpoint 后各 terminal access 一次。 | 模型设计 | 新实现 + 新训练 | 可写规格，不能写结果 | 没有此行不能声称 joint 比 SPR-only 更好，也不能把 ICBHI→SPR 专家 transfer 当反方向的替代。 |
| E6 | **Figure 1 是否准确说明 native task、label availability 与不可用标签？** | 当前 Figure 1 script 有四数据集 native-unit mapped composition；Data skeleton 已预留 task/availability matrix。四数据集 raw/native contracts 已存在。 | 当前 stacked classes 不是完整的“availability matrix”：HF 无 Normal negative，gap/empty不是 negative；KAUH raw9 与 compatible overlay需分开；SPRSound Rhonchi/Stridor对属性节点不可用。缺每格 annotation state、prediction unit、primary partition、support来源。 | 必须 | Figure 1(a) 改为或并列一张明确矩阵：dataset、unit、native task、Level1/Crackle/Wheeze availability、unknown/masked状态、grouping与primary split；composition仅作为描述，不伪装统一标签。 | Wade提供数值；论文写作作图；模型设计核对语义 | 仅整理 + 作图 | 可从既有 contracts 填字段和已核实 support，未核实格留空 | 若矩阵未闭合，主文只写 ICBHI+SPR 两行；HF/KAUH移到 discussion，不展示伪统一四类。 |
| E7 | **Figure 1 的 acoustic separation 是否超出视觉 PCA？** | `result/acoustic_distribution/recording_features.csv` 共13,704 recordings；draft按 `patient_id` 聚合并每数据集采112 groups做标准化 PCA；另有dataset-ID probe与20-track信号描述。 | 缺预注册的 group-aware定量 separability、held-out group评估、uncertainty；缺 class-matched Normal/Abnormal复核。HF没有合法 Normal标签，使四数据集完全 class-matched 比较不可直接成立。 | 必须，若保留acoustic claim | Primary 先做 ICBHI+SPR 的 group-held-out domain classifier balanced accuracy/AUROC或单一距离量，并在同数目Normal/Abnormal、按group抽样后复核；group bootstrap或repeated grouped folds给 uncertainty。HF/KAUH只做global/descriptive并明确proxy/mapping。 | Wade；模型设计只审计边界 | 新分析，不训练论文模型 | 可冻结输入字段、group定义、metric和fallback文字；本轮不计算 | 若数值缺失，Figure 1只保留 task/availability + 描述性 distribution；删除“class-matched separability”和任何显著/因果语言。 |
| E8 | **5 s repeat-padding/front-truncation对各native unit覆盖多少？** | JH2 config和实现已足以核实“使用5 s repeat-padding/front-truncation”这一preprocessing事实；20-track panel有选择性duration/level，部分数据文档有duration摘要。 | 若要进一步声称coverage充分、事件没有因front-truncation丢失，或用5 s coverage解释性能，仍缺dataset-wide、按paper partition统计的短/长unit比例、重复占比、舍弃秒数及group分布。20-track selection不能代表全体。 | 可后补；仅在coverage/无丢失/性能解释claim下必须 | 对 ICBHI cycle与 SPR event canonical records做一次只读全量 duration/coverage表；按split和group报告，不读test用于调参。HF/KAUH非主文可删。 | Wade或数据审计；模型设计核口径 | 新分析 | preprocessing事实今天可直接写；可先定义后续coverage字段 | 无全量表仍可事实性写“使用5 s repeat/truncate”；只需删除“coverage充分”“事件未丢失”及以此解释性能的句子。 |
| E9 | **group定义与uncertainty是否可复核？** | ICBHI patient ID、SPR patient ID、KAUH P-number、HF date proxy等已有合同；PCA draft笼统称112 groups/dataset。 | **必须补的metadata**：Figure/caption中的各数据集group authority；HF date proxy不能称patient；KAUH B/D/E不可当独立样本。**仅在统计推断时缺的证据**：group-level CI或重复grouped folds。 | group定义必须；uncertainty仅在报告推断性统计时必须 | 无论是否做显著性推断，都在Figure 1 caption/appendix列出四种group单位。若只呈描述性PCA/distribution，可不补CI但必须明确descriptive；若报告可泛化的separability数值，使用held-out groups并以group为重采样/折分单位给uncertainty。 | Wade + 论文写作 | 仅整理；推断性claim另需新分析 | group定义今天可写；CI按是否保留推断性claim决定 | group定义不闭合时删除“patient-balanced”总称；没有CI时可保留描述性图，但不得作显著性或总体推断。 |
| E10 | **Table I 的 published/local rows是否task-compatible？** | 现有primary-source audit、`icbhi_strong_method_reproduction/metrics.json`及paper values：PAFA、BEATs+CE、Patch-Mix、SG-SCL；JH2三种子summary。 | 缺最终registry字段：paper/repo、task、unit、split、seed数、encoder scope、input、selection source、metric、support、paper/local/delta。MVST support不同、ADD-RSC protocol冲突，不宜direct。 | 必须整理 | Table I分块：protocol-compatible ICBHI literature；local author-checkpoint alignment；本文JH2 test-selected benchmark。每行显式selection；不兼容行放Related Work或删。 | 管理/论文写作整理；文献任务核引用 | 仅整理 | 可用现有artifact填大部分；未给原始日志的学生行不可核验 | 缺registry字段的row不进主表；不按数值高低给不兼容rows着色或排名。 |
| E11 | **固定 ICBHI checkpoint 到 SPRSound 的结果能支持什么？** | PAFA/SG-SCL/Patch-Mix → SPRSound official inter，1,429 events；binary AS 55.82/59.98/59.38；all-Normal AS50；无target tuning。 | 缺与target-trained matched architecture的对照；三source methods协议不同；当前数字不能给joint净收益。Table列名若写Score会误导。 | 必须整理；target-trained reference可后补 | 作为motivation独立block：source ICBHI Score、target SPR binary AS、floor、mapping、support、target-label use=none。若要量化transfer gap，另需同架构SPR target-trained reference。 | 模型设计核边界；论文写作 | 仅整理；可选新训练 | 现有motivation表今天可完成 | 只声称“这些固定source checkpoints在该target decision rule下接近floor”；不能说所有expert models失败、representation不可迁移或joint提升约30点。 |
| E12 | **Hanlin 25组背景材料的验收字段是否齐全？** | 用户/计划记录：ICBHI 5、SPRSound 10（两个tasks）、KAUH 5、HF 5，目前只报seed42；补seed0/1已有用户授权。 | 验收仍缺独立train log、selected checkpoint/predictions、support/confusion、完整config、selection/threshold来源和artifact path；seed0/1执行与结果尚未核实。 | 可后补；不阻塞lead-owned clean controls | Hanlin按已有授权补0/1并提供原始产物；管理/论文写作按固定验收字段核对和整理registry。seed42先记为“student-reported/background, verification pending”；只有0/1/42任务、split、unit、selection、trainable scope完全一致才汇总mean/std。 | Hanlin补授权范围内运行/产物；管理/论文写作验收与整理 | 新训练 + 验收整理 | 今天可准备验收字段与HOLD标记；不能生成多seed值 | 未收到可核验0/1时不写“多seed完成”；seed42不升格为正式baseline aggregate，也不再把是否跑0/1写成待用户选择。 |
| E13 | **Strong ICBHI方法的跨数据集证据是否足够？** | 三个local author-style checkpoint有ICBHI对齐及 ICBHI→SPR fixed-head transfer；published LungMix/BTS-CARD等设计线索已在既有comparison表。 | 到HF/KAUH没有等价native task：HF是positive intervals且gap/empty未知，KAUH是patient/recording raw9；固定flat4 source head只能做compatible overlay diagnostic。还缺最接近literature的协议核对及可能的same-architecture target reference。 | Core SPR动机必须；HF/KAUH可删 | Core claim收窄为 ICBHI→SPRSound Task1-1。HF/KAUH若保留，只报告固定checkpoint、无target tuning、可评价subset与excluded support；不把N/A计为差。target-native head训练必须另标target-supervised adaptation。 | 模型设计 + 文献任务 | 仅整理；扩展需新分析/训练 | 可明确mapping与N/A边界 | 没有严格mapping或native comparator时，删除“对HF/KAUH transfer失败”；写“该source head不能回答其native task”。 |
| E14 | **本方法与最接近literature相比到底新增什么？** | 文献工作稿已到：`PAPER_NOVELTY_POSITIONING_2026-09-07_zh.md`。其定位为：DCASE已覆盖异质数据、missing-label mask、BEATs与separate metrics；Schutera/Bevandić已覆盖availability/overlapping labels；LungMix与BTS-CARD已覆盖respiratory cross-dataset gap；PC-MCL已有Normal/Crackle/Wheeze及flat4重建。当前安全差异是同一jointly-trained checkpoint对cycle-flat4/event-binary两个非等价native tasks的scoped formulation，以及待clean matched controls验证的经验分解。 | 仍需管理/用户集中确认candidate claim，并由论文写作将最少必要primary citations和不可声称内容准确落入正文；matched R1–R4未完成前，经验贡献仍不成立。 | 必须，决定contribution wording | 直接采用文献稿的closest-work定位，不重做搜索：主文优先DCASE、SPRSound fusion、LungMix、PC-MCL、Schutera；将“mask/hierarchy/native evaluation新颖”删除，保留respiratory non-equivalent native-task formulation与待验证matched attribution。 | 文献调研已交工作稿；管理/用户定claim；论文写作整合 | 文献整理 | 定位结论已可进入集中审阅 | R1–R4未完成前不写“hierarchy有效”或因果分解；不写“first”“novel general framework”，也不把cross-dataset gap说成首次发现。 |
| E15 | **Figure 2 是否证明方法有效？** | Method skeleton已定义 native inputs→task alignment→shared BEATs→eligibility-aware hierarchy→native outputs；代码支持该forward/loss/readout事实。 | 缺最终claim wording及matched E2/E3结果。图本身不构成增益证据。 | 必须作图；效果证据依赖E2/E3 | 图只画数据流、availability mask、shared encoder、three nodes、两个native readouts；PAFA inherited loss可小标，optimizer/experiment IDs/HF/KAUH不进入core图。 | 论文写作；模型设计审语义 | 仅整理/作图 | 可以完成结构草图 | E2/E3缺失时caption只能描述机制，不写“improves”“outperforms”。 |

## 4. Prospective clean 四行：最小科学合同待决项与建议

以下是供集中确认的 **Proposed Method**，不是冻结协议。

### 4.1 四行与比较对象

| Row | 训练数据 | Classification interface | Selection/threshold可使用的数据 | 它检验什么 |
|---|---|---|---|---|
| R1 Joint hierarchy | ICBHI+SPR | Level1 CE + eligible Crackle/Wheeze BCE；SPR有支持的raw7 events同时提供attribute supervision | 两数据集internal calibration/selection | clean条件下joint hierarchy能否同时保留两项native task |
| R2 Joint independent | ICBHI+SPR | ICBHI direct flat4 CE + SPR direct binary CE | 与R1完全相同的两数据集internal partitions和joint selection公式 | 整个eligibility-aware supervision/classification interface相对native independent heads的作用；监督未匹配时不是纯结构效应 |
| R3 ICBHI-only hierarchy | ICBHI | 与R1相同hierarchy | 仅ICBHI internal calibration/selection；SPR不可参与 | joint对ICBHI native retention的作用；selected source model到SPR的固定transfer |
| R4 SPR-only hierarchy | SPR | 与R1相同hierarchy | 仅SPR internal calibration/selection；ICBHI不可参与 | joint对SPR native retention的作用；selected source model到ICBHI的固定transfer |

### 4.2 待决选项

1. **Internal split**  
   新suite只从两个official-train的canonical sample/group records，以seed42一次性独立确定 subtrain/calibration/selection manifests；不看历史模型prediction。四行全部复用。ICBHI official test与SPR official inter保持terminal-only。旧 `LocalCleanQueue` group lists明确不作现行reference。

2. **Rows R1/R2共同 checkpoint criterion**  
   - 选项A：两个数据集eligible-node raw validation loss等权均值；不需threshold，但与paper headline native metrics存在错配。  
   - 选项B（建议）：calibration groups先冻结 hierarchy 的shared Crackle/Wheeze thresholds；selection groups计算 ICBHI flat4 Score和 SPR Task1-1 official Score，使用 `0.5 × ICBHI validation Score + 0.5 × SPR validation Score` 选epoch，tie取更早epoch。R2 direct heads用各自softmax argmax，不调threshold，但使用同一个native composite。不得按样本数pool。

3. **Threshold policy**  
   - R1：每个attribute一个core-shared threshold，只由两数据集calibration上F1的等权均值选择，tie取更高threshold。  
   - R3/R4：只从本source calibration拟合；target validation绝不参与。  
   - R2：两个direct softmax heads均argmax，不引入dataset-specific test threshold。  
   - epoch冻结后是否用full internal validation按相同规则refit thresholds需在运行前明确；若采用，四行必须一致记录，且不得改变selected epoch。

4. **Matched training scope**  
   建议四行共同继承 JH2 的 BEATs iter3+ AS2M、mono16 kHz、native unit→5 s repeat-pad/front-truncate、full encoder fine-tuning、batch32、Adam 5e-5/wd1e-6、cosine、EMA0.5、PAFA attention projector/PCSL/GPAL、无SpecAugment。R1/R2均保留PAFA projector和patient loss，避免同时改变representation objective。若用户决定去掉PAFA，必须四行全部同步去掉并视为另一套suite。

5. **SPR supervision matching limitation**  
   当前hierarchy不是只从SPR binary标签训练：`m_unified.py:114–158` 将 Normal、Coarse/Fine Crackle、Wheeze、Wheeze+Crackle映射到Level1及有支持的Crackle/Wheeze节点；Rhonchi/Stridor只给Level1=Abnormal，attribute保持unknown。R2的SPR-binary head只使用Normal/Adventitious，因此R1−R2同时比较了监督信息与classification interface。建议当前不擅自增加新轴：若维持R2 native-binary定义，论文只解释为**整个eligibility-aware supervision/classification interface**的对照；若用户要求纯hierarchy结构因果，则必须先另行决定matched-supervision设计。

6. **Compute matching**  
   - 选项A：每行固定50 native-data epochs；source-only updates更少，compute成为混杂。  
   - 选项B（建议）：冻结相同最大optimizer updates、相同update-based cosine和validation cadence；joint保持dataset-homogeneous equal-batch schedule，source-only重复本source batches到相同预算。early stopping使用各行预注册criterion和同一patience。数值预算在实现前从共同runner明确写死，不从test调整。

7. **Source-only selection**  
   R3按ICBHI internal native Score选；R4按SPR internal Task1-1 official Score选；tie均取earliest。它们不能使用joint composite，因为target dataset必须完全不参与训练、calibration、selection或early stopping。

8. **Terminal access与输出**  
   每行只在selected epoch及threshold冻结后读取每个official test一次。保存stable sample ID、group、raw/native GT、logits/probabilities/prediction、eligibility、confusion、support、Sp/Se/AS/HS/Score、macro-F1、UAR和per-class recall。两个数据集分别报告，不构造pooled score。

### 4.3 最小因果对照

- **Supervision/classification interface effect**：R1 − R2；encoder、PAFA branch、input、sampling、budget与selection必须匹配。除非另行闭合supervision matching，不称纯hierarchy structure effect。
- **Joint retention on ICBHI**：R1的ICBHI − R3的ICBHI。
- **Joint retention on SPRSound**：R1的SPR − R4的SPR。
- **Frozen source transfer**：R3→SPR、R4→ICBHI，只回答source-selected checkpoint能否直接服务另一个native task；target不适用的节点标N/A。
- 单seed42只能提供prospective local control，不支持稳定性或显著性主张。

## 5. 图表与证据的对应关系

| 论文对象 | 它应检验/传达什么 | 已有 | 缺失/最小必要对照 | 不能承担的claim |
|---|---|---|---|---|
| Figure 1 | 为什么不能直接pool：unit、label availability、group及acoustic distribution不同 | mapped composition、13,704-recording features、112-group/dataset PCA、描述性feature distributions | 必须补native-task/availability matrix与准确group定义；若保留定量separability再补ICBHI/SPR class-matched复核及group uncertainty；5s coverage仅在coverage/性能解释claim下必须 | 不能证明这些差异导致模型失败，不能证明方法更优 |
| Figure 2 | 方法如何保留native units并通过eligibility mask共享BEATs/hierarchy | 实现与Method skeleton | 最终core-only视觉稿、准确loss/readout标注 | 图不能代替R1/R2效果证据 |
| Table I | 文献背景、local checkpoint alignment、zero-target SPR motivation、JH2 benchmark | published/local rows、三份SPR transfer、JH2 3-seed | 完整registry与metric/selection分栏；Hanlin artifacts核验 | 不能从异协议数值构造leaderboard或joint gain |
| Table II | clean回答“joint有无作用、整个eligibility-aware supervision/classification interface有无作用” | 只有占位符 | R1–R4四行prospective clean seed42；R1/R2需披露SPR attribute-supervision不匹配限制 | 旧JH2、LocalCleanQueue、2s frozen diagnostic不能填入；supervision未匹配时不能写纯hierarchy结构效果 |
| Error analysis | joint benchmark在哪些类/指标受益或受损 | JH2 Sp/Se、per-class recall/support | matched R1–R4相同分解 | 当前不能把差异因果归于joint/hierarchy |
| Supporting diagnostics | fixed checkpoint在HF正属性、KAUH compatible patient overlay上的边界行为 | JH2 3-seed external summary | 若保留，补清楚evaluable/excluded support与mapping | 不能称native reproduction、四数据集comparable performance或clean transfer |

## 6. 今天只靠现有材料能写什么

| 今天可完成 | 证据状态 | 可以写 | 必须等待新结果后才能写 |
|---|---|---|---|
| Research motivation | existing local artifacts | 三个固定ICBHI checkpoints在SPR binary AS下只比all-Normal floor高5.82–9.98点；限定到这三个checkpoint与该mapping | “所有single-dataset experts都失败”“表征不可迁移” |
| Method facts | code + frozen setup | eligibility-aware Level1/Crackle/Wheeze、native-unit batching、BEATs+PAFA、ICBHI/SPR native readouts | hierarchy优于independent heads |
| JH2 benchmark | existing 3-seed test-selected result | ICBHI 61.17±0.31、SPR official Score90.70±0.34及Sp/Se/per-class；醒目标test-selected | clean generalization、joint净收益、unbiased paper main estimate |
| Input/data limitations | contracts + descriptive artifacts | 5s repeat/front-truncate是事实；units/labels/groups不同；PCA/feature distributions是描述性 | 5s覆盖充分、domain差异显著、acoustic差异导致性能差 |
| Table I draft | existing literature/local/transfer artifacts | 分块填task-compatible paper rows、local alignment、SPR transfer和JH2 | 混用AS/official Score；加入缺artifact的student multiseed行 |
| Figure 2 draft | implementation facts | core-only高层数据流和masked loss | 用图暗示outperformance |
| HF/KAUH discussion | fixed-checkpoint post-hoc artifacts | 属性/读出依赖的supporting diagnostic及N/A边界 | native benchmark reproduction、universal four-dataset retention |
| Table II structure | approved paper question | R1–R4问题、列、metric和HOLD | 任意performance value或hierarchy/joint gain结论 |

## 7. 建议给集中审阅的决策顺序

1. 确认主文是否只保留 ICBHI+SPRSound core；建议是，HF/KAUH仅supporting。
2. 确认 Table I 使用“published context / local alignment / zero-target motivation / test-selected JH2”分块，并保留AS与official Score的独立列。
3. 确认 prospective clean split从canonical official-train records独立确定，以及R1/R2是否共同使用equal native composite；这只是后续预注册选择，本稿不冻结。
4. 确认四行都保留PAFA branch并采用相同update budget；并决定R1/R2是补matched supervision，还是把结论限定为整个supervision/classification interface而非纯hierarchy结构。
5. 决定Figure 1是否等待group-aware/class-matched量化；若今天拿不到，采用task/availability matrix + descriptive distributions的fallback，删除定量separability claim。5s preprocessing事实可直接写，coverage统计仅随更强claim补充。
6. 收到Hanlin seed0/1及原始artifacts后再决定哪些学生baseline能进入表；目前只登记seed42背景，不汇总。
7. 使用已到达的closest-work定位稿集中确认contribution wording；在R1–R4缺失时仍不写first、general framework novelty或hierarchy净收益。

## 8. 关键现有 artifacts

- 最新计划：`docs/work_plans/2026-09-07_work_plan_zh.md`
- 09-04短会：`docs/meeting_records/2026-09-04_paper_story_short_meeting_record_zh.md`
- 结果讨论稿：`/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/RESULT_FINDINGS_DISCUSSION_2026-09-06_zh.md`
- 文献定位工作稿：`/Users/zilongzeng/.codex/worktrees/4e78/Acoustic/docs/literature/PAPER_NOVELTY_POSITIONING_2026-09-07_zh.md`
- JH2三种子：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md`
- JH2 HF/KAUH supporting：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.md`
- Strong ICBHI local：`result/icbhi_strong_method_reproduction/metrics.json`
- Fixed-checkpoint SPR transfer：`result/pafa_sprsound_transfer_20260722_235659/metrics.json`、`result/sg_scl_sprsound_transfer_20260722_235659/metrics.json`、`result/sprsound_patchmix_frozen_transfer/metrics.json`
- Figure 1 draft与features：`docs/paper/ICASSP_2026_acoustic_disease/Figure/figure1_dataset_composition_acoustic.py`、`result/acoustic_distribution/recording_features.csv`
- 20-track描述性证据：`result/audacity_four_dataset_panel_2026-08-28/analysis_report_2026-08-29.md`
- Clean selection候选：`reproduce/pafa/CHECKPOINT_SELECTION_SENSITIVITY.md`
- 历史排除项（不作新suite reference）：`baseline/pafa/joint_clean_queue.py`、`result/reproduce/pafa_joint_hierarchy/LocalCleanQueue_reference_seed42/`
- 不匹配的2s source-only diagnostic：`baseline/multidataset_pipeline/beats_icbhi_attribution.py`

## 9. 本轮执行边界

本轮只新增本文件。没有运行训练、validation/test、cache、server、notebook、smoke/probe/preflight或新的统计重评；没有修改模型代码、管理计划、Notion、Git、Overleaf、figure、slides或其他用户文件；没有联系合作者、Jingping或学生；没有计算或记录新的hash/checksum。
