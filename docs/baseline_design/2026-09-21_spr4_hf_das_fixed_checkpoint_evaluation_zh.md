# SPR四类事件与HF DAS：固定checkpoint补充实验

日期：2026-09-21。用户要求由“Acoustic本地训练”独立补做，结果回来后再决定是否入稿；正常paper修订不等待本项。

执行状态更新（9/21）：脚本与直接检查已准备，尚无模型forward或linear-head训练。统一执行器的自动审批拒绝了正式`--run`命令，理由是审批上下文仍判为早先的只读资产审计、禁止forward/head training；拒绝说明明确要求拒绝后的用户确认。返回值不匹配与预测复用问题已修复，HF新属性预测/旧native预测各自保留录音ID和window index；py_compile及直接相关纯函数检查已通过，现等待用户确认启动。不在其他任务或载体执行被拒命令。独立结果目录尚未创建。下文是拟执行的固定范围，不代表已经运行。

## 本地资产核对（9/21，执行任务回报）

六个selected checkpoints在本地：LSAA seeds0/1/42的epoch为15/11/17，Native+C/W为15/13/17。两种模型的SPR selected validation与test属性概率均可复用；SPR文件中Native+C/W的native概率是SPR二类padding，不能当ICBHI四类。LSAA的HF C属性概率及Native+C/W的HF native四类概率也已具备；需要补Native+C/W在SPR上的ICBHI-head输出、HF C属性输出，以及LSAA对应256维h_i。

| SPR分区 | Normal | Crackle | Wheeze | Both | 排除Rhonchi/Stridor | 总事件 | 患者 |
|---|---:|---:|---:|---:|---|---:|---:|
| subtrain | 4114 | 639 | 396 | 21 | 34 / 15 | 5219 | 194 |
| validation | 1045 | 322 | 56 | 9 | 5 / 0 | 1437 | 49 |
| official inter-test | 1040 | 83 | 305 | 1 | 0 / 0 | 1429 | 41 |

三seed的这些支持数相同；仍以各run实际split IDs确定归属。测试四类各自的患者支持为39/13/13/1，类别患者数可重叠。**Both仅1个test事件/1位患者，四类结果只能作为探索性证据；保留逐类结果及预先指定的non-Both敏感性分析。**

HF同一957条eligible池为D-positive 368、Other 589；1956条完整source-test中另有phase-only 995、empty 4。D/CAS是可重叠标注目标。执行任务据旧日志估算固定模型推理合计约25–45分钟，新linear heads CPU训练预计数分钟；脚本实现、读取/提取/汇总另计。此时只是资产核对与实现准备，不代表模型推理已启动或本轮结果已完成。

## 授权与研究问题

本次是9/19实验收口之后的新有限授权：允许核对/复用现有预测，完成必要的固定checkpoint推理与冻结表示提取，并仅为SPR有监督参照训练新的四类linear head。不更新原encoder、共享投影、既有属性头或native heads，不重选主checkpoint，不恢复旧队列。HF linear probe不列入本轮。

研究问题是：已有共享属性监督能否通过固定模型上的新任务读出，支持本研究此前未直接训练的SPR四类输出，以及HF外部D标注存在性排序？这不是未见类别学习；SPR的细类别信息已参与原联合训练，新的阈值又使用SPR validation标签。前三种固定模型方案称“no parameter updates”，并明确是否进行validation calibration。

Native+C/W同时拥有受训C/W头和两个native heads。其SPR binary head本身无法产生四类，但整个模型可用属性组合或借用ICBHI head。因此Native-only仅在“受训C/W属性读出”这一接口上记N/A，不能宣称整个模型不具备四类输出能力。新训练linear head是有监督参照，不是保证性能更高的上界。

## 共同资产和划分

- 复用LSAA reference与Native+C/W原selected checkpoints，seeds 0/1/42；不按本次SPR/HF分数重新选择epoch。Native+C/W已核对的原selected epochs为15/13/17，LSAA epochs以原summary核对为准。
- 使用每个实际run的SPR subtrain/validation患者与样本ID，以及原official inter-test。先核对原split文件，不能按现稿一句seed说明重造划分。新head的训练集必须排除同一比较使用的SPR validation患者；test患者不参与训练、阈值或新head选择。
- 输入沿用原5 s、mono 16 kHz前处理。只有同checkpoint、同样本、同预处理及同输出语义的既有概率/表示可复用；缺失部分才推理或提取表示。
- 保留每种结果使用的checkpoint路径/epoch、split来源、阈值、原始预测和逐类支持数。不改旧run summary、selection、threshold或论文表格。

## A. SPR四类事件任务

类别顺序固定为Normal / Crackle / Wheeze / Both。Normal不变，Fine Crackle与Coarse Crackle合并为Crackle，Wheeze不变，Wheeze+Crackle映射为Both；Rhonchi与Stridor从本次subtrain/validation/test四类池排除。报告三个split的事件数和患者数，包括每一类及排除类别。

### 固定模型的三个主比较

| 条件 | 异常门/四类来源 | C/W阈值 | 参数更新 |
|---|---|---|---|
| LSAA attribute readout | 原A二分类头，复用现有ICBHI层级解码 | 本seed SPR validation重新拟合 | 无 |
| Native+C/W attribute readout | 已训练SPR二分类head给P(Adventitious)，配合受训C/W heads；不存在独立A头 | 同一SPR validation规则重新拟合 | 无 |
| Native+C/W borrowed ICBHI head | 已训练ICBHI四类softmax直接argmax | 无 | 无 |

异常门Normal/Abnormal argmax，同分Normal；C/W以概率≥各自阈值为阳性。异常门内C/W均阳性为Both，单阳性为该类，均未达阈值用较大的p−threshold，平局Crackle。借用ICBHI head按固定Normal/C/W/Both顺序argmax，不加SPR head门或新温度。

阈值使用现有`fit_attribute_thresholds`规则，但attribute_datasets只取SPRSound且仅兼容四类validation事件：每个C/W属性分别最大化validation F1，候选为0、1及对应validation概率，平局取较高阈值，比较为≥。A门不再调阈值。不使用SPR test或HF选阈值、decode、epoch或比较变体。另可直接计算继承原selected thresholds的结果，作为预先指定的无新增校准补充；主结果固定为SPR-validation-calibrated版本，不能按test表现二选一。

### 冻结表示＋新四类linear head参照

只在每个LSAA reference checkpoint的固定h_i（256维，原encoder和原768→256共享投影均冻结）上训练一个新`Linear(256,4)`，每个原seed一个head，共三个。原有heads及患者投影不训练；不引入额外MLP或新的encoder初始化。这个比较直接回答在同一已有表示上，为SPR四类单独学习输出层的收益和成本。

固定配方：缓存/复用SPR subtrain、validation、test的h_i；新head在CPU训练，初始化seed与对应主模型seed一致；Adam lr=1e-3、weight_decay=1e-4、batch size=128，最多50 epochs，patience=10、min_delta=0。用subtrain四类计数得到逆频率class-weighted cross-entropy（每类权重n/(4*n_class)），训练按固定seed shuffle，无augmentation、EMA、PAFA或超参搜索。每epoch仅用SPR validation的四类Score选head，严格改善才替换、平局保留较早epoch；最终head固定后才计算test结果。若subtrain缺一类，报告这个参照的支持不足，不换split或补标签来凑四类。记录特征提取和head训练的实际用时。

### 指标和少样本处理

- 主指标：SPR事件池上的ICBHI-style Score=(Sp+Se)/2，Sp为Normal recall，Se为全部异常事件中正确分到C/W/Both的比例。名称中保留“ICBHI-style on SPR compatible four-class events”，不写成SPR原生official Score。
- 同时报四类macro-F1、逐类recall、完整confusion、每类事件/患者支持数、每seed值与三seedmean/sample SD。宏F1固定四个label、zero_division=0；若某真实类test support为0，明确其recall不可估，不能用汇总掩盖该类证据缺失。
- 预先指定：如果test Both少于10个事件，补“不含真实Both事件的敏感性分析”。保留原四类预测，预测为Both仍计错；对真实N/C/W子集计算Score与三类macro-F1，不重新训练/调阈值、不用更好的补充结果替代四类主结果。
- 比较差值按相同seed配对。1–2 pp只作为实用差距的描述，不作为已证实等效或普适成功的判据；必须同时看Both支持、逐类recall与seed间波动。

## B. HF录音级DAS（D标注存在性）

沿用现有CAS评测的同一957条eligible source-test录音：至少含D/Wheeze/Rhonchi/Stridor之一的标注；排除phase-only和无相关标注录音。目标为含D标注=1，其余eligible=0，并报告D-positive/Other数量。Other表示该标注协议下无D记录，不称为临床正常或保证Crackle不存在。D与CAS可共存，不把DAS/CAS设为互斥类别。

固定三个主读出，均为每条录音原三个5 s窗口分数取max：

1. LSAA：max p_C。
2. Native+C/W属性头：max p_C。
3. Native+C/W ICBHI四类头：max[P(Crackle)+P(Both)]。

主报D-presence/DAS AUROC，另附AUPRC、每seed和mean/sample SD及支持数。保持957条录音ID一致，不用HF训练/validation/test选择任何阈值、head或checkpoint。已有`recording_presence['D'].curve`和逐窗概率若条件匹配可直接复用，不强制重新推理。名称是录音级D标注存在性评测，不称HF原生时序DAS检测完整复现。

Native-only若已有匹配的HF native四类逐窗概率，可用同一P(C)+P(Both)在附录结果文件中追加纯后处理对照；这不能被写成“没有属性监督就无法零参数产生输出”。本次不为可选Native-only SPR行或HF probe另开额外模型执行。

## 执行顺序、资源与交付

顺序：资产/label支持核对 → 复用现有概率的SPR与HF结果 → 补缺失的固定模型推理 → LSAA h_i提取与三个新linear heads。优先把无需参数更新的结果整理出来，不以它们是否达到期待分数决定是否隐藏或重做参照。

由Acoustic本地训练实施必要的独立脚本和直接相关纯函数检查，保持一个本地MPS推理队列，默认inference batch≤16、CPU threads≤2、data-loader workers=0；新head尽量CPU执行。不要启动smoke/probe/profile、hash校验、服务器作业、旧训练队列、环境大规模改装或自动重试。不修改共享paper/figure/Notion，不调用写作任务，不提交或push Git。写作可以照常继续。

独立结果目录：`result/reproduce/pafa_joint_hierarchy/LSAA_TASK_ADAPTATION_20260921/`。保存简明config、逐样本/逐窗预测、支持数、固定阈值、confusion、逐seed指标与完整汇总/说明。完整实验记录保留在本地，当前不决定Table 4或压缩哪张图。

先向管理回报资产支持数与基于旧日志的耗时估算。以上规格具备时直接实施并正式运行本次有限任务；只在真正的标签/划分/资产问题或完成时报告管理，不要求写作等待。若某一臂缺数据，只暂停该臂并说明，已可行臂可继续；不自行改科学定义。最终给出全部主结果、少样本限制、参数更新/校准成本，以及可支持和不可支持的论文表述，供用户决定是否入稿。
