# Table 2 seed42 benchmark controls｜2026-09-09

状态：**ALL_FOUR_CONTROLS_COMPLETE / QUEUE_EXIT_0**  
当前目标：延续正式JH2 benchmark，以已有fresh42 Full为reference，只补四个seed42 controls。  
证据边界：全部属于official-test-selected benchmark extension，不是prospective clean、unbiased generalization或纯因果消融。

## 1. 当前五行与执行范围

| 角色 | 条件 | 是否训练 | 当前状态 |
|---|---|---:|---|
| Reference | Full HF-off | 否 | 复用正式JH2 fresh42 |
| Control 1 | ICBHI-only hierarchy | 是，seed42一次 | complete：28 epochs，selected18 |
| Control 2 | SPRSound-only hierarchy | 是，seed42一次 | complete：34 epochs，selected24 |
| Control 3 | Coarse SPR labels | 是，seed42一次完整结果 | complete：attempt2，30 epochs，selected20 |
| Control 4 | Native + attributes | 是，seed42一次 | complete：27 epochs，selected17 |

本轮不默认重跑Full，不运行Full+HF，不补seed0/1，不恢复旧R0、`LocalCleanQueue`、`J-Clean`或新clean split。HF/KAUH external evaluator另行安排，不属于当前四项启动范围。

## 2. Full reference及可复用性

Reference目录：

`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_42/`

已落盘事实：

- seed42；BEATs iter3+ AS2M full fine-tuning；mono16 kHz；5 s repeat-padding/front-truncation；无MVN、无SpecAugment；
- physical batch32；ICBHI与SPR dataset-homogeneous batches；每epoch各163 batches，共326 updates；
- Adam `5e-5`、weight decay `1e-6`、50-epoch cosine、EMA `0.5`；
- A two-class CE、C/W BCE、PAFA PCSL/GPAL；
- completed epoch27，selected epoch17；checkpoint由ICBHI official-test native Score选择；
- ICBHI subtrain/validation为3,174/968 cycles、63/16 patients；SPR为5,219/1,437 events、194/49 patients。

Full可作为当前四项reference，不需重跑。旧Full分类loss按batch中active nodes平均；管理对fresh42实际27轮batch metadata的只读核对表明，每个core batch均有A/C/W eligible rows，因此其实际A/C/W系数均为`1/3`。新runner显式固定`1/3`只是在Coarse空节点时防止A被放大，不改变Full已运行批次的有效权重解释。

Full仍是ICBHI-test-selected benchmark；复用它不把本表升级为validation-selected evidence。

## 3. 共同训练合同

四个新条件全部从同一个pretrained BEATs checkpoint重新初始化，不加载Full的best checkpoint作为训练起点。

- model seed：42；
- device：本地MPS；FP32；`cpu_threads=4`；
- input：native ICBHI cycle / SPR event，mono16 kHz，5 s repeat-padding或front-truncation；
- physical native batch：32；不得以gradient accumulation替代PAFA物理batch；
- encoder：BEATs iter3+ AS2M，full fine-tuning；
- shared classification projection：Linear `768→256`；
- PAFA：attention projection、PCSL系数50、GPAL系数0.0005；patient IDs按dataset限定；
- optimizer：Adam，lr `5e-5`，weight decay `1e-6`；
- schedule：沿用原JH2每epoch cosine，epoch内保持同一learning rate、无warmup；第`e`轮使用`5e-5 × [0.001 + 0.999 × (1 + cos(πe/50))/2]`；EMA `0.5`；
- maximum：50 epochs × 326 core updates；
- validation threshold fitting：每epoch结束后；
- early stopping：patience10；selection Score必须严格变大才更新；完全相同保留更早epoch；
- best与last checkpoint、train log、每epochvalidation predictions、test-selection predictions、selected terminal predictions及native metrics全部保存。

Joint条件每epoch为ICBHI163 + SPR163 updates。Single-source条件每epoch为本source 326 updates。因此总update预算相同，但single-source的本源暴露次数约为joint对应source的两倍；不能称source exposure matched。历史同类MPS full-finetuning约4.4–5.2 h/run只是旧run经验，不是当前运行承诺。

## 4. 原JH2 seed42数据划分

当前benchmark controls直接调用原JH2 seed42 official-train loader：

| Dataset | Subtrain | Validation | Grouping |
|---|---:|---:|---|
| ICBHI | 3,174 cycles / 63 patients | 968 cycles / 16 patients | official-train内patient-grouped split，seed42 |
| SPRSound | 5,219 events / 194 patients | 1,437 events / 49 patients | official-train内patient-grouped split，固定existing SPR split |

运行时`split_reference.json`内嵌本run active sources的实际group lists与逐record assignments，不依赖后来被编辑的外部manifest；不增加hash或新gate。

此前准备的`table2_clean_split_manifest.json`、canonical calibration/selection二分、active-source validation composite及14/18-run预算全部**暂停为历史备选**，不被当前benchmark runner读取或引用为现行split。

## 5. Selection与threshold规则

### 5.1 ICBHI-only

- training：仅ICBHI subtrain；
- C/W thresholds：每epoch只用ICBHI validation，分别最大化对应binary F1；tie取更高threshold；
- checkpoint/early stopping：最大化ICBHI official-test flat4 Score；
- selected checkpoint后才访问SPR official inter一次；SPR不参与训练、threshold、selection或patience。

### 5.2 SPRSound-only

- training：仅SPR subtrain；
- C/W thresholds：每epoch只用SPR validation中合法C/W rows，分别最大化binary F1；tie取更高threshold；
- checkpoint/early stopping：最大化SPR official-inter Task1-1 Score；
- selected checkpoint后才访问ICBHI official test一次；ICBHI不参与训练、threshold、selection或patience。

该条件与Full/I-only/Coarse/Native使用不同的source-native selection dataset。其结果可回答“SPR-selected source model的retention/transfer”，但不能与Full差值无条件解释为joint training的纯效应。

### 5.3 Coarse SPR labels

- training sources仍为ICBHI+SPR；
- ICBHI保留A/C/W全部监督；SPR只保留Normal/Adventitious A target；
- SPR raw7 fine C/W真值不进入training loss、threshold fitting、checkpoint selection或early stopping；
- C/W thresholds只用ICBHI validation；
- checkpoint/early stopping仍最大化ICBHI official-test flat4 Score；
- selected checkpoint后才读取SPR fine labels，报告SPR native Task1-1及evaluation-only C/W AUROC。

PAFA criterion只接收projected features、dataset-qualified patient IDs与PCSL/GPAL系数，不读取raw7 label、classification target或eligibility，因此隐藏的SPR C/W不会经PAFA旁路进入训练。

### 5.4 Native + attributes

- training sources为ICBHI+SPR；
- native inference heads：ICBHI `Linear(256,4)` flat4；SPR `Linear(256,2)` binary；
- training auxiliary heads：共享C/W各`Linear(256,1)`；使用与Full一致的合法attribute target、eligibility与有效权重；
- checkpoint/early stopping：ICBHI native flat4 head在official test上的Score；
- native prediction只用flat4/binary heads，C/W辅助头不参与native类别决定；
- selected terminal仍导出C/W probabilities，以合法SPR fine rows计算Table 2 C/W AUROC。

Native+attributes中的A信息由native CE承载：SPR binary CE等同于A任务；ICBHI flat4 CE同时携带Normal/Abnormal及异常细类结构。它与Full比较的是classification/supervision interface，不是只改变head shape的纯因果对照。

## 6. 固定节点权重

所有当前新训练条件使用固定名义权重：

\[
L_{cls}=\tfrac13L_A+\tfrac13L_C+\tfrac13L_W.
\]

每个node先在其eligible rows内mean；整个batch无eligible row时该node项为0，其`1/3`不转移给其他node。

- ICBHI-only：所有ICBHI rows有A/C/W目标；
- SPR-only：使用原raw7中合法A/C/W；Rhonchi/Stridor的C/W masked；
- Coarse SPR：SPR batch为`1/3 L_A + 0 + 0`，PAFA保持；
- Native+attributes：`1/3 L_native + 1/3 L_C + 1/3 L_W`；C/W辅助信息与Full匹配。

## 7. E1 Direct C/W readout

E1不增加训练。正式JH2 fresh42及seed0/1已存在Full hierarchy和C/W-only同scores、同thresholds结果：seed0 C/W-only较Full低0.596 pp，seed1低2.600 pp，seed42高0.562 pp，方向不一致。

当前benchmark extension可引用该历史test-selected sensitivity，不需重跑脚本。若未来对四个新conditions做readout，仍固定各自selected scores和saved thresholds，只换ICBHI decoder。任何threshold-refit必须另标calibration sensitivity，不能混入fixed-readout效果。

## 8. 实现入口、目录与命令

实现入口：

`baseline/pafa/table2_benchmark_controls.py`

独立输出根：

`result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/`

当前执行恢复说明（2026-09-10）：原Coarse执行中断，完整记录止于epoch25，第26轮仅保存validation。原目录完整保留，不用于warm start；相同协议/seed/预训练初始化的工程重试写入`coarse_spr/seed_42_attempt2/`。Native仍使用下列原目录。当前队列由独立tmux `acoustic_benchmark_recovery_20260910`承载，脚本与直接日志位于输出根的`recovery_logs/`。这不是新增科学条件或独立seed，最终Coarse仅采用完整重试结果。

四个新目录：

- `icbhi_only/seed_42/`
- `sprsound_only/seed_42/`
- `coarse_spr/seed_42/`
- `native_attributes/seed_42/`

用户已明确同意按以下顺序串行执行，由本地任务统一启动；其他任务不重复启动：

```bash
/opt/anaconda3/envs/Beats/bin/python -u -m baseline.pafa.table2_benchmark_controls --repo-root /Users/zilongzeng/Research/Acoustic --variant icbhi_only --seed 42 --run

/opt/anaconda3/envs/Beats/bin/python -u -m baseline.pafa.table2_benchmark_controls --repo-root /Users/zilongzeng/Research/Acoustic --variant sprsound_only --seed 42 --run

/opt/anaconda3/envs/Beats/bin/python -u -m baseline.pafa.table2_benchmark_controls --repo-root /Users/zilongzeng/Research/Acoustic --variant coarse_spr --seed 42 --run

/opt/anaconda3/envs/Beats/bin/python -u -m baseline.pafa.table2_benchmark_controls --repo-root /Users/zilongzeng/Research/Acoustic --variant native_attributes --seed 42 --run
```

正式carrier可在命令外层使用`caffeinate`。runner每epoch开始输出状态，第1及之后每32 core updates输出update/dataset/loss并flush；total loss非有限时立即停止。

## 9. 可回答的问题与不能回答的问题

| 对比 | 可回答 | 不能回答 |
|---|---|---|
| Full vs ICBHI-only | 在相同JH2-style benchmark中加入SPR训练后，ICBHI与SPR endpoint如何变化 | 纯joint因果；两者每源exposure不同 |
| Full vs SPR-only | joint Full与SPR-selected single-source在两个endpoint的差异 | 纯joint因果；checkpoint selection dataset不同 |
| Full vs Coarse | 使用SPR fine C/W supervision的整体差异 | clean generalization；不能排除training dynamics交互 |
| Full vs Native+attributes | shared hierarchy与native+attribute interface的整体差异 | 纯head结构或纯监督信息效应 |
| Full vs E1 C/W-only | 同一分数和threshold下最终ICBHI conversion rule的影响 | 表示学习、训练objective或PC-MCL复现 |

所有结果必须标`seed42 / official-test-selected benchmark control`。四项完成也不能计算跨环境、多seed SD或声称统计稳定性。

## 10. 当前准备状态与HOLD

已准备：

- 原JH2 seed42 split复用；
- 四variant data routing、fixed-weight loss、native heads、selection source和selected-only other-target路径；
- split snapshot、progress logging、best/last及sample-level prediction输出；
- 不含模型forward的定向metadata/loss测试。

管理静态审阅已完成。入口的学习率已修正为与原JH2相同的epoch cosine，而非逐update衰减；epoch1/25/50的纯函数数值检查通过。未运行模型forward、smoke或训练。

仍HOLD：

1. 用户已批准四项实际启动，本地任务负责统一执行；后续以实际运行日志记录状态，不把启动授权当作完成结果；
2. 未做额外BEATs最小forward或smoke；四项均已完成正式训练及终端评测，Coarse使用完整attempt2，旧partial不作为完成结果；
3. HF/KAUH external evaluator未接新benchmark outputs，且不属于当前四项；
4. 任何执行bug若需要修改共用协议，必须先暂停并与管理协调；
5. 旧clean partial目录保持原状，不继续、不删除、不改写。

## 11. 实现准备阶段边界（历史记录）

实现准备阶段只更新规格、独立benchmark-controls入口与定向纯函数/metadata测试，未运行模型或训练。此后用户已明确批准四项串行执行，当前首项已实际启动；实时阶段见Work Plan、console与各run_summary。未完成的条件不能当作已完成实验结果，旧clean队列不恢复。
