# PC-MCL ICBHI-only 5-s Source Transfer规格｜2026-09-16

状态（2026-09-17更新）：**数值失效后已停止；源码核查与非有限值处理已完成，未重跑**。三个seed从35/45/30轮起NaN，旧产物完整保留。当前只允许已批准的源码核查、数值检查修复与直接单元检查；新运行须另获明确启动指令。详见`2026-09-17_pcmcl_numerical_failure_and_source_audit_zh.md`。
本文件是当前PC-MCL权威候选规格，替代此前joint-core、frozen-head和pooled-cache路线。
执行边界：原两卡400轮任务已经停止，监控已暂停；不恢复其他历史队列，不执行smoke/profile。下方保留本次失败运行的原始方案，不构成重启授权。

官方来源：

- 论文：<https://arxiv.org/abs/2601.17080>
- 代码：<https://github.com/wa976/PC-MCL>

## 1. 研究问题与主比较

研究问题固定为：

> PC-MCL将ICBHI任务重构为显式Normal/Crackle/Wheeze多标签、raw multi-cycle输入和patient-matching辅助任务后，ICBHI-only源模型能否不经过任何目标适配，直接迁移到SPRSound、HF Lung和KAUH；与使用ICBHI+SPRSound联合训练的LSAA主模型相比，其本域和跨数据集表现及边界是什么？

主比较对象是**LSAA联合主方法**，不是LSAA ICBHI-only消融。训练源不同是既定system comparison：

- PC-MCL：ICBHI-only source training；
- LSAA：ICBHI+SPRSound joint training；
- PC-MCL选定后在SPR/HF/KAUH保持全模型固定；
- 不为“匹配训练源”新增PC-MCL joint、LSAA single-source或其他消融。

结果不能单独归因于N/C/W head或patient loss；两系统还同时不同于训练源、输入构造和优化协议。但这些差异是需要如实披露的系统定义，不是新增实验任务。

## 2. Verified source/code facts

### 2.1 本地资产

- generic初始化存在：`.cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt`；
- 本地没有PC-MCL训练完成的`best_model.pth`、`best_epoch_*.pth`或等价checkpoint；
- 官方README只指向通用BEATs iter3+ AS2M初始化，没有提供可直接复用的PC-MCL训练后权重；
- generic BEATs只能作为source initialization，不能填成PC-MCL结果。

因此三seed必须是三个独立的ICBHI source trainings。

### 2.2 官方代码入口

官方`main.py`已核对：

- `ConcatenationMultilabelDataset`提供raw concat和N/C/W BCE；
- `--enable_ssl --ssl_method coherence`启用same/different-patient任务；
- BEATs、N/C/W classifier和patient classifier都在optimizer参数中，encoder并未冻结；
- 默认CLI为`desired_length=8`、batch32、Adam、400 epochs、lr `1e-3`、wd `1e-4`、milestones 120/160、threshold 0.5；
- `enable_ssl`默认关闭且`ssl_weight=0.5`，不能直接当论文`L_main+0.1L_patient`；
- official split下`train_flag=True`读取official train，`False`读取official test；
- `validate()`逐epoch在official test计算ICBHI Score，严格变好且`Se>0.1`时更新best。

这是一条test-selected source protocol，不是clean validation-only evidence。

### 2.3 5-s入口需要的实际修订

只传`--desired_length 5`还不满足用户合同。官方dataset当前先拼接两个原长cycle，最后才把整体裁/补到5 s；其`half_target_samples`主要用于筛选短cycle，不保证每个constituent恰为2.5 s。

本轮要求固定为：

```text
cycle 1 --repeat-pad / center-crop--> 2.5 s
cycle 2 --repeat-pad / center-crop--> 2.5 s
waveform concat --------------------> 5.0 s
```

单unit validation/test/transfer则直接repeat-pad或center-crop到5 s。该接口已准备在`pcmcl_source_transfer.py`；正式runner必须调用它，而不能只改官方CLI。

## 3. ICBHI source training合同

### 3.1 数据与split

- Dataset：ICBHI 2017；
- prediction unit：annotated respiratory cycle；
- sample rate：mono 16 kHz；
- optimization pool：official train全部4142 cycles；
- selection/evaluation pool：official test 2756 cycles；
- 不创建新的内部validation；
- official recording split不是strict patient-held-out，本结果不得如此命名。

每个训练seed保留官方数据组成：

- 原始single-cycle examples：约`1.0N`；
- multi-cycle classification examples：`mixing_prob=0.5`，所有双cycle输入遵守`2.5+2.5=5 s`；
- classification pair保留官方class-conditioned组合（同类、Normal+Crackle、Normal+Wheeze、Crackle+Wheeze及Both相关组合），不是从全体cycles任意均匀配对；
- 与官方实现的必要差异：本轮每个constituent都会规范到2.5 s，因此不再只从“原始时长短于half target”的子池选择；
- patient-matching examples：候选保持源码`ssl_prob=0.3`；
- positive为same-patient pair；hard negative为不同patient且病理profile匹配的pair；
- additive N/C/W target为两个constituent labels的逻辑OR。

### 3.2 模型与梯度

初始化：AudioSet BEATs iter3+ AS2M。

Trainable：

- BEATs waveform frontend与全部encoder blocks；
- mean-pooled feature后的`Linear(768,3)` N/C/W classifier；
- mean-pooled feature后的`Linear(768,2)` patient classifier。

```text
5-s waveform -> trainable BEATs -> pooled 768
                                |-> N/C/W BCE
                                `-> patient CE
```

由于encoder可训练，patient loss通过共享BEATs表示影响主分类路径。这与旧“冻结BEATs后只训练约0.20M adapter/head”的方案完全不同；旧pooled cache不能用于source training。

### 3.3 Loss与优化候选

源loss固定为论文形式：

```text
L_source = BCEWithLogits(N,C,W) + 0.1 * CE(patient_match)
```

需要最小修正官方代码当前的convex mixture，不能把`ssl_weight=0.1`误当完全相同公式。

候选source-code配置：

- batch32；FP32；
- Adam，lr `1e-3`，weight decay `1e-4`；
- 完整400 epochs，关闭patience早停；min_delta0的strict-improvement best选模和tie保留较早best仍保持；
- milestones 120/160各乘0.1，恢复公开源码默认的400轮日程；
- mixing probability 0.5；patient sample probability 0.3；
- N/C/W与所有迁移readout threshold固定0.5；
- SpecAugment默认开启，固定`icbhi_ast_sup`频率/时间mask和mean填充值；依据是官方BEATs代码路径默认启用transform。该字段已显式写入运行config，不能从目标结果选择。

其中优化字段来自官方代码而非论文正文，不将公开默认值当作论文完整运行命令。400轮预算已由用户批准。三个seed使用新目录`result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_400epoch`；旧`PC_MCL_ICBHI5s`目录不覆盖、不改写checkpoint状态。

### 3.4 Checkpoint selection

不引入SPR validation composite，也不使用任何target数据选模。

- 每epoch读取ICBHI official test；
- 最大化ICBHI Score；
- 保留源码`Se>0.1`gate；
- 只接受strict improvement，因此tie保留更早epoch；
- 不因连续无提升而提前停止，完整运行400轮；HF/SPR/KAUH不参与monitor；
- selected checkpoint必须包含encoder、N/C/W classifier与patient classifier；
- evidence label：`ICBHI-official-test-selected 5-s PC-MCL source diagnostic`。

这与现有ICBHI-only LSAA同为test-selected evidence，但不是matched recipe：后者使用5-s单cycle hierarchy/PAFA、50 epochs、最多16300 updates、内部threshold split和不同优化配置。已有I-only结果继续留在原消融中，不是本baseline的主比较或必要控制。

## 4. Source model freeze

每个seed选定source checkpoint后一次性固定：

- encoder/frontend；
- N/C/W classifier；
- patient classifier；
- 0.5 threshold；
- input preparation和readout rules。

SPR/HF/KAUH不得训练或替换head、更新模型、重新选checkpoint、拟合threshold、根据目标效果挑readout，或使用目标validation决定任何字段。

## 5. 预注册固定迁移readout

所有目标先输出并保存`[p_N,p_C,p_W]`，随后使用以下固定规则。

### 5.1 ICBHI本域flat4

阈值固定0.5：`00->Normal`、`10->Crackle`、`01->Wheeze`、`11->Both`。`p_N`保留用于属性诊断，但不在结果后改变原始C/W conversion。

### 5.2 SPRSound official inter

- unit：official inter event，单unit规范到5 s；
- 先用固定C/W bits得到四类预测；
- `Normal -> Normal`；`Crackle/Wheeze/Both -> Adventitious`；
- 所有1429个inter events使用同一规则；
- primary：Task1-1 Sensitivity、Specificity、AS、HS、official Score；
- secondary：SPR C/W AUROC与N/C/W score distribution，只用于解释属性迁移，不参与选择。

### 5.3 HF source-test CAS

- 每个15 s recording使用既有三个固定5 s窗口；
- recording score为三个窗口`p_W`的maximum；
- evaluation pool固定为任一D/Wheeze/Rhonchi/Stridor annotation的957 recordings；
- Wheeze/Rhonchi/Stridor为661 positives，D-only为296 negatives；
- primary为CAS AUROC；
- 这是Wheeze-head对broad CAS union的ranking proxy，不是完整CAS detector；Crackle不得进入CAS主列。

### 5.4 KAUH fixed external

- 每个B/D/E recording view使用同一single-unit 5-s policy；
- view abnormal probability固定为`max(p_C,p_W)`；
- 先对同一P-number的B/D/E三个view概率取mean；
- patient score `>=0.5`预测Abnormal；
- 只评86位compatible patients：N为Normal，E W/I E W/C/I C/I C E W为Abnormal；
- Crep、Bronchial、I C B排除；
- primary为patient balanced accuracy；另报filter-view consistency，但不称device robustness。

固定顺序为“view inference → probability aggregation → one patient decision”，不能先离散分类后多数投票。

## 6. 输出要求

每个seed保存source config、train log、epochwise ICBHI selection、best/last、selected epoch/update；并为ICBHI、SPR、HF、KAUH保存stable ID、N/C/W logits/probabilities、固定readout、GT、support/confusion/metric。汇总三次独立source trainings的mean与sample SD。

不得把同一source checkpoint上的三个target head seeds写成PC-MCL三seed，因为本方案不训练target heads。

## 7. 资源与预算

训练不能使用frozen embedding cache，因为BEATs encoder会更新。

- official train：4142 cycles；
- originals + 0.5 mixed + 0.3 patient约为`1.8N` examples/epoch；
- batch32、drop-last约232 updates/epoch；
- 最多50 epochs约11600 updates/seed，三seed满跑上界约34800 updates；
- 早停可能缩短实际epochs，但预算不得预设必在某个epoch停止；
- best/last model state为BEATs量级；若保存Adam resume state，单份显著大于纯model checkpoint；
- source training是主要成本，固定模型的三个target inference相对较小；
- 当前没有profile，不能承诺精确小时数。

执行资源仍需另行批准；不能预设服务器可用，也不能在无性能证据时排除本地执行。

## 8. 解释边界

- 原文主结果使用10 s总输入；论文65.37%不能替代本轮5 s source结果；
- 主比较是ICBHI-only PC-MCL与ICBHI+SPRSound LSAA，训练源差异必须披露；
- 迁移较差可能来自单源范围、5-s适配、整体表示、标签覆盖或固定readout，不能直接归因N/C/W失败；
- 迁移较好也不能声称patient loss单独导致，需要结合本域Score、SPR属性判别、HF p_W和KAUH fixed readout解释；
- 不预设最低成功指标，不用target结果反选规则，不新增I-only matched control。

## 9. 当前实现与HOLD

新增：

- `baseline/frozen_method_baselines/pcmcl_source_transfer.py`：2.5 s constituent、5 s concat/single、source heads/loss和四个固定readouts；
- `baseline/frozen_method_baselines/pcmcl_icbhi5s_source_transfer.json`：source、selection、freeze、transfer、output和预算合同。
- `baseline/frozen_method_baselines/pcmcl_source_run.json`：max50、patience10及15/20 milestones配置；
- `baseline/frozen_method_baselines/pcmcl_source_runner.py`：canonical ICBHI读取、class-conditioned pair、full BEATs训练、test-selected checkpoint、固定三目标评测和resumable best/last；
- `baseline/frozen_method_baselines/source_transfer_queue.py`与`source_transfer_summary.py`：三seed串行和仅完整seed汇总。

运行时直接复用本地通用BEATs源码和初始化checkpoint；不需要把官方PC-MCL repo复制进项目。完整脚本已准备但未经model forward或训练验证。

- **DESIGN READY：** 研究问题、输入、source gradient、selection、freeze、迁移映射、三seed和预算；
- **CODE READY, UNEXECUTED：** source data/model/train/checkpoint/target evaluation/queue/summary入口已完成；
- **ASSET HOLD：** 无PC-MCL-trained checkpoint；generic初始化存在；
- **EXECUTION HOLD：** 正式启动和执行资源仍需批准；
- **NO RESULT：** 未forward、cache、训练、validation/test、target inference、smoke/profile或server。
