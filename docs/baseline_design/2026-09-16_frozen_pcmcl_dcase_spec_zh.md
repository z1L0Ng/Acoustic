# PC-MCL / DCASE Frozen-Encoder候选规格｜2026-09-16

本文件为旧共同候选。用户现已要求两组ICBHI-only源迁移脚本：PC-MCL见最新source-transfer规格；DCASE使用原生四分类softmax＋CE的DCASE-inspired CRNN，不采用本文件的联合事件头或A-gate。完整实现以[实现brief](2026-09-16_source_baseline_implementation_brief_zh.md)为准，未启动实验。

PC-MCL范围已被用户最新决定替代：ICBHI-only源训练、2×2.5 s→5 s，训练后固定全模型直接评测SPR/HF/KAUH。DCASE也已改为ICBHI-only flat4 softmax source model：frozen BEATs frames + trainable CNN/fusion/BiGRU/attention。以下joint-core/frozen-PC及A/C/W/Other DCASE方案全部只作历史，不能执行。当前入口见[PC-MCL Source Transfer规格](2026-09-16_pcmcl_icbhi5s_source_transfer_spec_zh.md)和`baseline/frozen_method_baselines/SOURCE_TRANSFER_README.md`。

状态：**PC-MCL AND DCASE SECTIONS SUPERSEDED / NO EXECUTION**  
当前责任：项目侧设计PC-MCL与DCASE；PAFA既有资产只保留给Hanlin交接。  
边界：本文是候选规格，不是已批准实验合同，不是实验结果，也不授权forward、feature/cache提取、训练、验证、测试或服务器任务。

官方来源：

- PC-MCL论文：<https://arxiv.org/abs/2601.17080>
- PC-MCL官方代码：<https://github.com/wa976/PC-MCL>
- DCASE 2024 Task 4官方baseline：<https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline>
- DCASE官方技术报告：<https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cornell_baseline_t4.pdf>

## 1. 推荐结论

以下PC-MCL推荐已被用户决定替代，只保留为历史；DCASE候选仍待讨论：

| Baseline | 推荐名称 | 保留的方法核心 | 当前状态 |
|---|---|---|---|
| PC-MCL | `SUPERSEDED` | 见当前5-s source-transfer规格 | 不执行本文件的joint/frozen路线 |
| DCASE | `DCASE-style masked CRNN respiratory adaptation` | frozen BEATs frame embedding、log-Mel CNN late fusion、BiGRU、missing-class loss/attention mask | 设计与frame-fusion/loss接口已准备；官方frame cache和完整CNN接线未执行 |

以下“共同推荐训练信息”只属于已废弃PC候选与仍待讨论的DCASE背景，不适用于当前PC-MCL source-transfer规格：

- 有监督core：ICBHI official-train cycles + SPRSound official-train BioCAS events；
- HF：只做selected fixed-source model的source-test external readout；当前主列沿用maximum-window `p_W` ranking proxy；
- KAUH：只做selected fixed-source model的86位compatible patient external readout；
- HF/KAUH不训练head、不参与选模、不调threshold；
- 三次seed为`0,1,42`，代表完整下游方法训练，而不是同一checkpoint重复评测；
- terminal ICBHI/SPRSound/HF/KAUH只在core checkpoint固定后各访问一次。

这一口径最接近当前LSAA的“ICBHI+SPRSound core，HF/KAUH external”研究问题。它不等价于Table 1中现有五个“分别训练ICBHI和SPRSound模型”的frozen references，因此表格分组和caption必须由用户讨论后确定。

## 2. PC-MCL：推荐raw-concat frozen adaptation

> **SUPERSEDED：** 当前PC-MCL是ICBHI-only full source training，两个2.5 s cycles拼成5 s；selected whole model固定后直接迁移。以下第2节不是执行规格。

### 2.1 它回答的paper问题

在同一个AudioSet BEATs保持冻结时，保留PC-MCL的多周期上下文、显式Normal标签和patient-matching正则，能否比普通pooled frozen head更好地利用ICBHI与SPRSound的患者级重复观测，并向HF/KAUH固定外评迁移？

该问题评估的是**PC-MCL机制在frozen encoder约束下的适配价值**，不是PC-MCL论文65.37%结果的复现，也不是PC-MCL-trained encoder的表示评估。

### 2.2 Verified official mechanism

论文明确包含三项协同组件：

1. 每个训练unit生成一个双cycle组合；既有same-class也有cross-class，既有intra-patient也有cross-patient；
2. 每个cycle先通过repeat padding或center crop规范到`T/2`，两个waveform在encoder之前拼接为固定`T=10 s`；
3. composite target为`[Normal, Crackle, Wheeze]`逐元素逻辑OR；
4. patient head判断两段是否来自同一patient，hard negative来自病理profile相同但patient不同的pair；
5. 论文损失为`L_main + 0.1 L_patient`；
6. 测试时使用单cycle pad/truncate到目标长度，C/W以0.5阈值重建ICBHI flat4。

官方repo只提供通用BEATs iter3+ AS2M来源说明，没有可直接冻结的PC-MCL训练后checkpoint。因此“真实PC-MCL-trained encoder再冻结”仍是独立资产HOLD，但这不阻止下面的generic-frozen机制适配。

### 2.3 推荐输入与pair manifest

每个训练seed生成一份固定pair manifest；pair只在同一dataset内产生，不能把ICBHI patient和SPRSound patient配成一对。

- ICBHI constituent：native annotated respiratory cycle；
- SPR constituent：native BioCAS respiratory event；event不是cycle，因此这一段属于跨数据集机制适配；
- 每个constituent使用论文规则规范到5 s；短unit repeat-pad，长unit center-crop；然后按有序`first || second`拼成10 s waveform；
- same-patient pair的patient target为1；different-patient为0；
- hard negative优先选择另一个patient且supported N/C/W profile相同的unit；
- 主任务允许same-class与cross-class组合；每个subtrain unit每个seed生成一个composite，避免构造无界pair全集；
- validation、official test与external readout不拼pair，按论文单unit inference规则规范到10 s。

这一10 s package是方法特有输入，不与当前LSAA 5 s输入宣称严格matched。

### 2.4 N/C/W target与unknown传播

单unit label order固定为`[Normal, Crackle, Wheeze]`：

| Source label | N | C | W | eligibility说明 |
|---|---:|---:|---:|---|
| ICBHI Normal | 1 | 0 | 0 | 三项observed |
| ICBHI Crackle | 0 | 1 | 0 | 三项observed |
| ICBHI Wheeze | 0 | 0 | 1 | 三项observed |
| ICBHI Both | 0 | 1 | 1 | 三项observed |
| SPR Normal | 1 | 0 | 0 | 三项observed |
| SPR Fine/Coarse Crackle | 0 | 1 | 0 | 三项observed |
| SPR Wheeze | 0 | 0 | 1 | 三项observed |
| SPR Wheeze+Crackle | 0 | 1 | 1 | 三项observed |
| SPR Rhonchi/Stridor | 0 | unknown | unknown | N observed；C/W不得写negative |

pair级规则不是简单把unknown填0：

- 任一constituent对某node有observed positive，则composite为positive且eligible；
- 两个constituent都是observed negative，composite才是negative且eligible；
- 其他情况保持unknown并mask。

接口由`compose_pcmcl_pair_targets`与现有`masked_multilabel_bce`表达。

### 2.5 Frozen与trainable模块、gradient path

Frozen：

- `.cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt`；
- BEATs waveform frontend与全部encoder参数。

Trainable：

```text
10-s raw concat -> frozen BEATs -> pooled 768
                               -> shared LN+Linear(768,256)+ReLU+Dropout
                                  |-> Linear(256,3): N/C/W
                                  `-> Linear(256,2): same/different patient
```

损失：

```text
L = masked_BCE(N,C,W) + 0.1 * CE(patient_match)
```

两个loss都更新同一个约0.20M参数shared adapter。若直接把两个linear heads接在完全冻结的BEATs embedding上，patient loss无法改变pathology representation；那条实现不具备PC-MCL正则意义，本方案不采用。

### 2.6 Candidate core selection与readout

以下只是推荐候选，**尚未获得用户批准，也不能强制替换历史test-selected benchmark规则**：复用项目现有ICBHI/SPR patient-grouped subtrain/validation分区，不新建split；每个epoch计算两个validation native Score，使用

```text
selection = 0.5 * ICBHI_validation_Score
          + 0.5 * SPR_validation_official_Score
```

选最大值，tie取最早epoch；不能按样本数pool。PC-MCL paper的N/C/W阈值0.5保持固定，不用HF/KAUH调参。

Readout：

- ICBHI：按论文，C/W过0.5分别组成Crackle/Wheeze/Both；均不过则Normal；
- SPRSound：若N过0.5且C/W均不过则Normal，否则Adventitious；这是事件任务适配，不是论文原readout；
- HF CAS主列：15 s recording的三个固定5 s窗口分别按单unit规则输入；每个窗口使用N/C/W head的`p_W`，再在三个窗口取maximum。评测阳性为Wheeze/Rhonchi/Stridor、D-only为阴性。该分数必须标为`maximum-window p_W ranking proxy against the broad CAS union`，不是语义完整的CAS detector；`p_C`不得进入当前CAS主列；
- 泛异常或`max(p_C,p_W)`只能另列diagnostic，不能默认占用CAS列；
- KAUH：B/D/E三view先分别推理，patient abnormal score取三view的`1-p_N`均值；只评86位compatible patient，阈值固定0.5。

### 2.7 为什么不推荐旧pooled-pair作为主行

旧`PC-MCL-inspired frozen embedding adapter`只对单unit 5 s pooled embeddings做pair运算。它丢失：

- waveform拼接后由BEATs共同建模的边界与上下文；
- Normal+abnormal混合输入造成的label-bias问题本身；
- 训练concat、测试single-unit的原方法domain shift；
- 10 s encoder感受野内的交互。

它可作为raw路线无法按时完成时的低成本备选，但不应与raw路线同时进入主表，也不能命名PC-MCL。

## 3. DCASE：推荐masked CRNN respiratory adaptation

### 3.1 它回答的paper问题

在保持BEATs冻结的情况下，DCASE式frame fusion、时序CRNN和显式missing-class masking，能否用一个模型吸收ICBHI与SPRSound不完全一致的呼吸音标签，而不把unsupported class当negative，并向HF/KAUH固定外评？

它是LSAA之外一个有真实先例的异构标签baseline。其价值不是环境声音类别本身，而是**frame representation + heterogeneous-label mask + temporal attention**的组合。

### 3.2 Verified official mechanism

DCASE 2024 Task 4官方baseline：

- 16 kHz waveform同时进入128-bin log-Mel CNN与预提取的frozen BEATs frame embeddings；
- BEATs frame序列经adaptive average pooling对齐CNN时间分辨率；
- 两路frame feature拼接、linear fusion后进入BiGRU与MLP；
- 输出frame-wise strong posterior与attention-pooled clip-wise weak posterior；
- 对当前dataset没有标注的classes同时mask loss和attention；
- supervised部分使用BCE；Mean Teacher用EMA teacher和MSE consistency处理weak/unlabeled数据；
- mixup只在同一dataset内进行。

官方模型的BEATs保持冻结；CNN、fusion、BiGRU、classifier与attention均可训练。

### 3.3 推荐保留和省略

保留：

- frozen BEATs frame embeddings；
- 7-layer log-Mel CNN；
- adaptive temporal alignment与late fusion；
- BiGRU；
- frame logits与attention-pooled weak output；
- sample/class eligibility mask同时进入loss和attention。

省略：

- **Mean Teacher**：当前推荐core没有经批准的unlabeled respiratory pool。强行保留只会变成额外consistency regularizer，不能代表官方利用unlabeled data的角色；
- **mixup**：首条baseline不新增augmentation轴，避免与LSAA差异同时包含架构、missing-label和augmentation；
- **class-wise median filter/overlap-add SED后处理**：主结果是native unit classification与external recording/patient ranking，不是DCASE event-onset PSDS。

因此合法名称必须是`DCASE-style masked CRNN respiratory adaptation`，不能写DCASE reproduction。

### 3.4 输入、输出和标签空间

输入沿用当前项目core package：mono 16 kHz、5 s unit；这与DCASE官方10 s环境声音输入不同，属于呼吸音适配。

- log-Mel支路候选保持官方参数：128 mels、2048-sample window、256-sample hop、amplitude dB、instance min-max normalization；
- BEATs支路输出`[B,L,768]` frame embeddings；
- fusion后输出`strong_probability [B,T,4]`与`weak_probability [B,4]`；
- class order固定`[Abnormal, Crackle, Wheeze, Other]`。

Eligibility：

| Dataset/raw label | A | C | W | Other |
|---|---:|---:|---:|---:|
| ICBHI Normal/Crackle/Wheeze/Both | observed | observed | observed | unknown |
| SPR Normal/Fine-Coarse Crackle/Wheeze/Both | observed | observed | observed | observed negative for Other |
| SPR Rhonchi/Stridor | positive A | unknown | unknown | positive |

训练只用unit-level weak BCE；ICBHI cycle和SPR event标签不能虚构成frame onset/offset。frame posterior通过masked attention获得间接监督。

### 3.5 Candidate selection与readout

以下selection同样只是待用户确认的候选，尚未批准替换历史规则：与PC-MCL使用同一validation-only equal-dataset composite，保持固定0.5决策阈值：

- ICBHI：A<0.5为Normal；否则C/W组合为Crackle/Wheeze/Both；若A为异常但C/W均不过阈值，选择C/W较大者；
- SPRSound：A>=0.5为Adventitious；
- HF CAS当前主列：使用每个固定5 s窗口的**weak attention-pooled** `p_W`，再对三个窗口取maximum。它与现稿其他行相同，是Wheeze-head对Wheeze/Rhonchi/Stridor union的ranking proxy；Crackle输出不进入CAS分数；
- DCASE-specific secondary diagnostic：只有在`Other`始终严格定义为Rhonchi/Stridor时，才可另报每窗口`max(p_W,p_Other)`再做maximum-window的AUROC。它比当前`p_W` proxy包含更多目标类别，必须单独命名和列表，不能与主列直接当作同一readout比较；
- KAUH：每个B/D/E view先对frame A取attention weak score，三view概率均值为patient abnormal score；只评86位compatible patient。

HF推理mask固定为所有四个输出通道均开启，所有HF recording相同；只用输入padding得到的frame-valid mask。HF annotation只能在预测落盘后构造CAS evaluation pool和正负target，绝不能用于选择输出class mask。frame-level maximum可作为未来temporal diagnostic，但当前Table主列固定使用weak attention-pooled window score。

HF/KAUH没有训练、validation、threshold或selection角色。该固定源external readout和“在HF训练dedicated CAS head”“在KAUH做目标监督五折OOF”是不同研究问题，本方案不采用后两者。

## 4. 四列训练信息与claim boundary

| Column | PC-MCL/DCASE推荐数据角色 | 允许表述 |
|---|---|---|
| ICBHI Score | joint-core target-supervised；官方test terminal | joint-core native result |
| SPRSound official Score | joint-core target-supervised；inter terminal | joint-core native result |
| HF CAS AUROC | fixed selected source model；无HF训练/选模 | current-table maximum-window weak `p_W` proxy；DCASE W/Other另作diagnostic |
| KAUH patient BA | fixed selected source model；无KAUH训练/选模 | external 86-patient readout |

两条baseline均不能称zero-shot“疾病诊断”；它们只是在没有HF/KAUH目标监督时做固定readout。KAUH B/D/E是同一patient的filter views，不是三个patients。

## 5. 三seed真实含义

PC-MCL seeds `0,1,42`共同改变：

- deterministic pair manifest与hard-negative采样；
- shared adapter与两个heads初始化；
- batch order与dropout。

由于raw pair输入随seed变化，PC-MCL core concat cache也按seed分别准备。固定validation/test/external single-unit cache可共享。

DCASE seeds `0,1,42`改变：

- CNN/fusion/BiGRU/classifier/attention初始化；
- batch order与dropout。

三seed共享同一frozen BEATs frame cache。它们不是三份BEATs pretraining。

## 6. 资产、cache和预算

### 6.1 已有

- Generic checkpoint：`.cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt`，约345 MB；
- 本地BEATs implementation：`.cache/multidataset_pipeline/assets/P2/source/repo/BEATs/`；
- 现有5 s pooled cache约68 MB，但只适合作为旧pooled-inspired备选，不能用于raw PC-MCL或DCASE frame fusion；
- canonical ICBHI、SPRSound、HF与KAUH manifests/splits/group IDs已由现有providers维护。

### 6.2 缺失

- PC-MCL：每seed raw pair manifest、10 s concat waveform adapter及其pooled frozen embeddings；
- DCASE：官方frame extractor contract、frozen BEATs frame cache、完整7-layer log-Mel CNN接线；
- 两者：用户确认后的selection/evidence label与正式run entrypoint；
- 真正PC-MCL-trained encoder checkpoint仍缺，但不是推荐raw frozen adaptation的依赖。

### 6.3 静态规模估计，不是profile

按当前core分区，subtrain约`3174 ICBHI + 5219 SPR = 8393` units；validation约`968 + 1437 = 2405` units。

- PC-MCL：每seed约8393个10 s composite encoder inputs，三seed约25179；validation、两个official terminal与HF/KAUH external single-unit/window inputs可共享。只缓存float32 pooled 768-d时，三seed core pair加共享readout约0.12 GB量级，metadata另计。主要成本是约2.5万个10 s frozen-BEATs forward，而不是adapter训练。
- DCASE：约2.1万个5 s core/terminal/external windows共享一份frame cache。若每个clip有`L`个frame tokens，float32主体为`21187 × L × 768 × 4 bytes`；`L=50–100`时约3–7 GB量级。真实`L`必须在获批extractor运行后记录，本轮没有forward/profile。
- 下游PC-MCL约0.20M trainable parameters；DCASE官方规模的CNN+fusion+单层BiGRU+双输出head约1M量级。冻结cache完成后，三seed下游训练远小于feature preparation；本轮不承诺未经实测的精确小时数。

当前主计划是在用户明确启动后进入**本地顺序队列**：先分别完成PC-MCL concat cache或DCASE frame cache，再从冻结cache训练对应的三个下游seed。PC-MCL cache需要约2.5万个seed-specific 10 s encoder inputs；DCASE cache需要约2.1万个共享5 s frame inputs；后续训练分别只更新约0.20M和约1M参数。Mac是否承担某个cache阶段应在真实授权时根据当前队列、可用磁盘和可接受时长决定，本轮没有性能证据支持提前排除。

若未来确有可用accelerator且另获服务器授权，可以并行特征准备作为**可选加速方案**；当前没有核实两张L40可用，也没有服务器启动授权，因此它不是执行前提或默认排程。精确cache与训练小时数均未评估。

## 7. 最小实现改动

已准备：

- `baseline/frozen_method_baselines/pcmcl_dcase.py`
  - ternary N/C/W pair composition；
  - raw-concat shared adapter + pathology/patient heads；
  - PC-MCL masked loss；
  - DCASE frame alignment + BiGRU + masked attention接口；
  - DCASE masked weak BCE。
  - current-table maximum-window `p_W`与DCASE-specific W/Other diagnostic的独立readout函数；
- `baseline/frozen_method_baselines/pcmcl_dcase_candidates.json`
  - 当前候选数据角色、模块、cache和decision points。
- `baseline/frozen_method_baselines/contracts.py`
  - 新增两条design-only method plan。

正式执行前最少还需：

1. `pcmcl_pair_provider.py`：只复用canonical core samples/group IDs，生成三seed固定pair manifests；
2. `pcmcl_concat_cache.py`：10 s raw concat frozen-BEATs pooled extraction；
3. `dcase_frame_cache.py`：官方frame embeddings和padding/time masks；
4. `dcase_respiratory_model.py`：接入官方式7-layer CNN；
5. 一个共享train/evaluate entrypoint，明确joint core与HF/KAUH external隔离。

本轮没有提前实现这些执行文件，以免在selection和Table 1口径未批准时形成错误runner。

## 8. 用户必须先决定的四点

1. 是否接受两行都是`ICBHI+SPRSound joint-core`，而不是仿照现有五个frozen references分别训练两个source模型。
2. 是否批准validation-only equal-dataset composite selection。若当前LSAA仍保留test-selected历史结果，必须在表格中披露protocol mismatch，或另行统一重选；不能静默混表。
3. 是否接受HF/KAUH只做fixed-source external readout；当前CAS主列统一maximum-window `p_W` proxy，DCASE W/Other只能另列secondary diagnostic。
4. 是否批准先构建PC-MCL raw-concat cache与DCASE frame cache；批准后才冻结epoch/batch/LR等正式训练预算。

## 9. HOLD与未执行事项

- **READY FOR DISCUSSION：** 两条推荐科学路线、label/mask、gradient path、readout、seed和资源规格；
- **IMPLEMENTATION PARTIAL：** head/loss/frame-fusion接口；尚无data/cache/train runner；
- **HOLD：** selection/Table 1分组、feature extraction和正式运行等待用户批准；
- **ASSET HOLD：** 真正PC-MCL-trained encoder checkpoint不存在，但不阻塞推荐generic-frozen raw机制路线；
- **NO RESULT：** 没有模型forward、训练、推理、feature extraction、cache、profile、smoke、validation、test或服务器任务；没有改主稿、Notion或Git；没有计算hash/checksum。
