# PAFA / PC-MCL Frozen-Encoder Baseline 设计规格｜2026-09-16

最新分工：PAFA运行交由Hanlin，本文件保留为交接与此前设计资产；项目侧当前重点为[PC-MCL/DCASE重新设计](2026-09-16_frozen_pcmcl_dcase_design_brief_zh.md)。下文候选方案与资产事实保留，负责人和执行优先级以最新Work Plan为准。

状态：**DESIGN COMPLETE / INTERFACES PREPARED / NO EXPERIMENT STARTED**  
用途：为当前Table 1补充方法来源清楚的frozen-encoder对照。  
边界：本文件不是实验结果，也不授权模型forward、feature/cache构建、训练或推理。

管理核对（9/16）：设计与接口交付不代表实验方案已经由用户批准。冻结路线、下游选模、HF/KAUH是否使用目标标签及其聚合方式仍为候选设置；当前Table 1的比较口径保持原状，需先与已有五行的实际配置核对。

## 1. 决策摘要

### PAFA：推荐立即采用真实method-trained encoder路线

使用已经由PAFA在ICBHI上训练过的BEATs encoder，冻结它，并为每个Table 1任务训练新的target-supervised adapter/head。方法对象是**PAFA训练得到的表示**；downstream不重新声称PCSL/GPAL正在优化表示。

论文/表格名称应为：**PAFA-trained BEATs (frozen)**，不能简称“PAFA reproduction”。该source checkpoint由ICBHI official test选出，必须披露。

### PC-MCL：真实method-trained encoder路线HOLD

官方[论文](https://arxiv.org/abs/2601.17080)与[代码仓库](https://github.com/wa976/PC-MCL)说明PC-MCL依赖三件事：raw multi-cycle concatenation、Normal/Crackle/Wheeze三标签、patient-matching auxiliary task。当前本地没有训练完成的PC-MCL checkpoint，也没有与其10 s raw concatenation对应的四数据集cache。因此不能把AudioSet-only BEATs或普通frozen head命名为PC-MCL。

已准备一个**PC-MCL-inspired frozen embedding adapter**接口，保证native、3-label和patient-matching losses共享同一trainable adapter；但pooled embedding pair不是raw waveform concatenation，HF也没有真实patient ID。是否将它作为明确标注的inspired baseline，需要用户决定。推荐不把它直接填成四列“PC-MCL”主表行。

## 2. 两种冻结方式不能混称

| 路线 | 冻结对象 | 方法信息在哪里 | downstream训练 | 合法名称 |
|---|---|---|---|---|
| Method-trained encoder | 经PAFA/PC-MCL源任务训练后的encoder | frozen encoder weights | target adapter/head | `PAFA-trained frozen encoder`；若有权重则可类比PC-MCL |
| Generic encoder + method-style adapter | AudioSet-only BEATs | 新训练adapter、patient loss或auxiliary heads | method-inspired adaptation | `PAFA-inspired` / `PC-MCL-inspired`，不是原方法复现 |

若generic encoder完全冻结，而PAFA/PC-MCL辅助loss只连接一个与classifier分离的head，loss无法改变分类路径。这种实现没有方法意义。本轮prepared modules强制让classification与patient auxiliary共享256维adapter。

## 3. PAFA frozen representation

### 3.1 权重与cache

- PAFA task checkpoint：`.cache/checkpoints/pafa/server_epoch27/best.pth`
- BEATs architecture/source asset：`.cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt`
- 四数据集pooled embeddings：`.cache/four_dataset_pafa_frozen_encoder/embeddings.npz`
- per-dataset shards：`.cache/four_dataset_pafa_frozen_encoder/embedding_shards/`
- source repository：`result/pafa_sprsound_transfer_20260722_235659/source/repo`

这些资产当前都存在。encoder由PAFA在ICBHI上full-fine-tune；接受状态是epoch-100 container内的selected epoch-27 model state，selection使用ICBHI official test。cache是5 s输入下的768维mean-pooled pre-classifier frame representation。source classifier和PAFA projection head不进入downstream。

源run的随机seed没有在接受artifact中闭合。它是一份固定encoder状态，不是三份PAFA源模型。下游seeds 0/1/42只表示adapter/head初始化、batch order与dropout不同。

### 3.2 冻结与可训练模块

Frozen：

- PAFA-trained BEATs encoder全部参数；
- 现有768维embedding cache；
- source classifier与source PAFA projector不加载到预测路径。

Trainable，按每个task分别训练：

`LayerNorm(768) → Linear(768,256) → ReLU → Dropout(0.2) → native head`

Native heads：ICBHI 4类、SPR 2类、HF CAS 1 logit、KAUH binary 2类。

downstream不使用PAFA PCSL/GPAL。准确解释是：PCSL/GPAL已经参与历史encoder训练；当前评估该方法产生的固定表示能否被目标监督head使用。

### 3.3 不推荐的PAFA替代设计

冻结generic BEATs后重新训练PAFA-style regularizer只能命名`PAFA-inspired adapter`。若未来采用，PAFA loss必须直接作用于与classifier共享的adapter output：

`frozen embedding → shared trainable adapter g → native classifier`  
`                                      ↘ PCSL/GPAL`

这样PCSL/GPAL才能通过`g`影响分类表示。HF date proxy不是patient ID，该loss不能在HF上合法开启。此路线不是当前推荐Table 1 PAFA行。

## 4. PC-MCL frozen design

### 4.1 Official method facts

官方论文使用：

- ICBHI official 60/40；
- multi-cycle waveform concatenation；
-固定10 s network input；
- `[Normal, Crackle, Wheeze]` additive multi-label BCE；
- same/different patient auxiliary task，hard negatives来自相同病理profile的不同patients；
-论文combined-loss patient weight `alpha=0.1`；
- 单cycle test input经pad/truncate后推理。

官方CLI与论文并非完全一致：CLI默认`desired_length=8`、`epochs=400`、`ssl_weight=0.5`，且`enable_ssl`默认关闭。若未来复现，必须显式使用论文设置，不能直接运行默认CLI。

### 4.2 True PC-MCL-trained encoder route

所需资产：包含PC-MCL训练后BEATs encoder state的checkpoint，以及与10 s输入定义匹配的下游cache。当前官方repo README只指向通用BEATs权重，本地也没有PC-MCL成品checkpoint。

状态：`HOLD_MISSING_TRAINED_PCMCL_CHECKPOINT_AND_MATCHED_CACHE`。

要关闭HOLD，只能：

1. 用户提供可追溯的PC-MCL trained checkpoint；或
2. 另行授权项目侧执行一次真实PC-MCL source training，再冻结encoder并生成cache。

选项2是新的full-fine-tuning实验，不在本轮设计授权内，不能隐式插入。

### 4.3 Prepared PC-MCL-inspired adapter

现有可用generic资产：

- AudioSet-only BEATs：`.cache/multidataset_pipeline/assets/P2/checkpoints/BEATs_iter3_plus_AS2M.pt`
- 四数据集5 s pooled embeddings：`.cache/four_dataset_representation_attribution/r1_beats_as2m_audioset_only/embeddings.npz`

Trainable modules：

- shared `LayerNorm(768) → Linear(768,256) → ReLU → Dropout(0.2)` adapter；
- dataset-native heads；
- 3-label N/C/W auxiliary head；
- patient-pair MLP，输入`[z1,z2,|z1-z2|,z1*z2]`。

Loss path：

`L_native → native head + shared adapter`  
`L_3label → 3-label head + shared adapter`  
`L_patient → pair head + shared adapter`

因此patient loss不是dead branch。建议以paper的`alpha=0.1`作为patient-loss weight；官方CLI默认0.5不作为论文设置。native与3-label主loss各占主任务的0.5是本项目提出的adapter定义，不是PC-MCL paper fact。

核心限制：现有cache只含单个5 s native unit的pooled embedding。两个embedding的mean、concat或MLP fusion都不等价于`waveform1 || waveform2 → frozen BEATs`。所以prepared route必须写`PC-MCL-inspired frozen embedding adapter`，不能写PC-MCL reproduction或PC-MCL-trained encoder。

### 4.4 各数据集机制可用性

| Dataset | Native task | 3-label辅助 | patient matching | 科学边界 |
|---|---|---|---|---|
| ICBHI | flat4 | exact N/C/W映射 | true patient cycles，可用 | 最接近PC-MCL语义，但无raw concat |
| SPRSound | binary | Normal/Fine-Coarse Crackle/Wheeze/Both可用；Rhonchi/Stridor C/W unknown | true patient events，可用 | event不是breathing cycle；仍属inspired |
| HF | CAS | 无Normal target；不能构造完整3-label | date proxy不是patient，禁用 | 只剩普通CAS adapter/head，不能代表PC-MCL |
| KAUH | compatible patient binary | raw labels可映射有限subset | P-number是真patient，但B/D/E是filter views而非cycles | patient matching语义改变，不能冒充multi-cycle |

上述限制描述的是在各目标数据集上重新训练全部辅助机制的条件，不意味着缺少patient-matching训练标签就无法在HF/KAUH推理。已训练好的源模型仍可通过语义适用的固定readout做外部评测；其标签映射、输入长度和分数构造须另行明确。

只报ICBHI/SPR、HF/KAUH暂留N/A，是当前inspired目标监督适配方案的一种建议，并非已批准的Table 1安排。另一个候选是固定源模型后做HF/KAUH外部readout：前者研究目标监督适配，后者研究固定模型的迁移能力。不能用普通目标头的结果声称原始PC-MCL全部训练机制已被复现，也不能把目标监督结果写成零目标监督。

SPR多标签接口现已显式要求eligibility mask：Rhonchi/Stridor的C/W unknown不参与该标签的BCE。接口定义在 `models.py` 的 `masked_multilabel_bce`；实际pair采样、数据接线与完整PC-MCL训练runner尚待路线确认后实现。

## 5. 四个endpoint的候选下游协议（尚待确认）

本节是PAFA method-trained encoder目标监督适配的设计提案，不是已批准的Table 1统一协议，也未证明与Hanlin已有五行一致。validation-only selection、HF dedicated CAS head、KAUH五折OOF以及embedding级B/D/E平均均为新提出的设置。它们不能自动替换论文当前选模和固定模型外部评测；PC-MCL采用哪种路线与哪些列也仍待讨论。

| Task | Unit与target | Train/selection | Terminal metric |
|---|---|---|---|
| ICBHI | cycle flat4 | official train内patient-grouped validation；validation ICBHI Score选模 | official test ICBHI Score |
| SPR | event Normal/Adventitious | official train内patient-grouped validation；validation official Score选模 | inter official Score `(AS+HS)/2` |
| HF CAS | 15 s recording的3个5 s frozen embeddings均值；eligible pool为任一D/W/R/S标注；W/R/S=positive，D-only=negative | source-train内date-proxy grouped validation；CAS AUROC选模 | source-test dedicated CAS-head AUROC |
| KAUH | 先mean B/D/E为patient embedding；86位compatible patients；N=negative，其余compatible sound=positive | fixed 5-fold patient OOF，每fold内部validation BA选模 | 86-patient aggregate balanced accuracy |

所有头都是target-supervised；不能写zero-target-tuning。HF使用dedicated CAS head，与LSAA现有`p_W`对CAS union的proxy排名不同，表格/正文必须披露。KAUH frozen baseline使用目标标签训练，与LSAA的external fixed-model evaluation也不是相同target-label条件。

## 6. 三seed、优化与输出

Seeds：`0,1,42`。encoder与cache完全相同；变化的是downstream adapter/head初始化、batch order、dropout，以及inspired方案的pair sampling。因此只能称三次downstream training seeds。

候选统一训练配置：

- max50 epochs；batch128；
- Adam lr `1e-3`、weight decay `1e-6`；
- patience10，strict improvement，tie取更早epoch；
- 每个task按第5节自己的validation metric选模；
- terminal只在selected state固定后读取一次；
- 每seed保存config、train log、best/last、逐样本或逐patient logits/probabilities/predictions、support/confusion与metric；汇总mean/sample SD，不构造跨数据集pooled score。

该配置是本项目frozen-adapter proposal，不机械继承PAFA 100 epochs或PC-MCL 400 epochs。若用户要求与Hanlin五行使用完全相同的head/optimizer预算，需先取得其可核验原始配置；当前来稿未闭合这些字段。

## 7. 已准备的代码与配置

- `baseline/frozen_method_baselines/models.py`：target adapter、shared patient adapter、patient/multilabel loss接口、可选PAFA-inspired regularizer；
- `baseline/frozen_method_baselines/data.py`：既有cache对齐、HF CAS target、KAUH patient aggregation；
- `baseline/frozen_method_baselines/contracts.py`：三种科学身份与资产状态；
- `baseline/frozen_method_baselines/pafa_method_encoder.json`；
- `baseline/frozen_method_baselines/pcmcl_options.json`；
- `baseline/frozen_method_baselines/task_protocol.json`；
- `baseline/frozen_method_baselines/run.py`：设计/资产交接CLI，不执行训练；
- `baseline/frozen_method_baselines/train.py`：PAFA method-trained encoder cache的target-head trainer，已准备但未执行。

设计检查入口：

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.run \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --method pafa_trained_encoder_frozen_target_adaptation --seed 0
```

PAFA后续执行入口示例：

```bash
/opt/anaconda3/envs/Beats/bin/python -u -m baseline.frozen_method_baselines.train \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --task icbhi_flat4 --seed 0 --device mps --run
```

任务可选`icbhi_flat4`、`spr_binary`、`hf_cas`、`kauh_binary`；seeds为0/1/42。KAUH每seed还需依次运行`--kauh-fold 0..4`并在后续评测阶段聚合OOF。该runner当前只服务PAFA method-trained encoder route；PC-MCL true-HOLD与inspired路线未获用户选择，因此没有训练入口。不要调用旧hash/profile/smoke入口。

## 8. 推荐方案

1. **PAFA：设计与代码准备完成，运行前环境HOLD。** 使用真实PAFA-trained encoder cache，三seed训练四个target-supervised任务头。表名写清方法训练表示与test-selection history。当前Mac的`Beats`和`acoustic-pafa`环境在导入既有data loader所需的SciPy PROPACK extension时失败；本轮未修改依赖。需要另行批准一个可用环境后才能运行。
2. **PC-MCL true frozen encoder：HOLD。** 无checkpoint，不得填表。
3. **PC-MCL-inspired adapter：设计候选，尚未批准运行。** 当前已准备共享adapter、多标签mask和patient-matching接口。HF/KAUH是否采用固定源模型readout、目标监督适配或暂不报告，取决于Table 1要回答的问题；缺目标patient-matching标签本身不构成无法推理的理由。

## 9. 下一步需讨论的科学选择

1. 首先确认冻结路线：评价真实method-trained表示，还是采用明确标注机制改动的frozen-encoder适配版本。真实PC-MCL表示目前缺checkpoint；若选择源任务训练，需要另行安排资源和启动。现有inspired方案的缓存便利不构成替代原始机制的充分理由。
2. 再确认Table 1的比较范围：目标监督适配与固定源模型外部评测回答不同问题。HF/KAUH的标签使用、readout和聚合方式必须随路线一起确定，不能把候选OOF或专用CAS头自动当作已有协议。
3. 最后与已有五行核对selection和训练预算。validation-only是本设计候选，不是强制替代论文已有test-selected benchmark的规则；应按所比较的问题选择并披露实际选模数据。Hanlin原始配置未闭合前，不宣称两组协议已匹配。

## 10. 未执行事项

没有运行模型forward、训练、推理、feature extraction、cache构建、smoke/probe/profile或服务器任务；没有计算hash/checksum；没有修改active论文、图、Notion或Git；没有联系Hanlin；DCASE未启动。

静态CLI检查仅执行了`--run`缺省的describe路径，未加载模型或数据。PAFA trainer代码可解析，但实际`--run`会进入旧canonical data loader；当前两个本地Python环境的SciPy二进制导入均失败，因此正式运行保持HOLD，不安装、不修复环境，也不改用未经批准的数据实现。
