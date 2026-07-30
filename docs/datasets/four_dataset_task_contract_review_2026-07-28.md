# 四数据集任务合同审批冻结

日期：2026-07-28

状态：**`approved_for_phase1_baseline_implementation`**；Tail support：**`tail_eligibility_management_accepted`**

Machine-readable contract：`docs/datasets/four_dataset_task_contract_draft_2026-07-28.json`

批准范围：

- Phase-1 shared surface：ICBHI 2017 respiratory cycles + SPRSound BioCAS2022 respiratory events。
- Shared binary acoustic event 为 primary；narrow four 为 secondary。
- ICBHI cycle four-class 与 SPRSound event seven-class 作为 dataset-native heads；指标按 dataset 和 prediction unit 分层报告。
- HF_Lung_V1 与 KAUH/Fraiwan v3 按已批准 safe defaults 保持 dataset-native，不进入 phase-1 shared surface。
- 本批准只开放 canonical schema preparation、schema validation、adapter dry-run preparation 与 baseline implementation preparation。
- 本批准**不**自动开放 real manifest、real split assignment、HF negative sampling、KAUH unresolved normalization、复杂 MoE、大规模训练、processed audio 或 training input。

## 1. 审批结论

- 当前项目的 canonical 人类可读文档位于 `docs/`；本合同不恢复旧的日期目录。
- ICBHI 2017 与 SPRSound BioCAS2022 的 respiratory cycle/event 标签能形成最干净的 shared acoustic surface，但 prediction unit 不同，必须分数据集报告。
- HF_Lung_V1 是 15 秒 recording 输入上的 temporal detection 数据集。raw label 只有正向 interval token，没有显式 `Normal` 或 `Negative`；未标注时间不能自动作为 shared normal。
- KAUH/Fraiwan v3 是 recording-level 数据集。B/D/E 是三种 stethoscope filter，不是三个独立 patient/session；`I C B` 中的 `B` 未被一手 ontology 定义。
- `source_official_facts`、raw-package measurement、inference 和 `proposed_benchmark_policy` 继续分列。用户批准的是 benchmark policy，不会因此改写为 official source fact。
- ICBHI/SPRSound 的批准项标为 `approved_benchmark_policy`；HF/KAUH 的 shared exclusion 标为 `approved_blocked_from_shared_phase1`；已批准但尚未授权物化的 split policy 标为 `approved_policy_not_authorized_to_materialize`。
- 现有 `dataset/processed/schema/label_mapping.csv` 中 HF/KAUH 的若干乐观 mapping 不应直接实施。本次冻结不修改该历史 schema；后续实现必须另过 implementation gate。
- 本次只修改本报告及其 machine-readable contract；没有处理音频、生成 manifest、训练模型或改动 `dataset/processed`。

## 2. 证据与状态规则

| 状态 | 含义 | 禁止解释 |
|---|---|---|
| `observed` | raw 文件、一手 README/paper 直接给出，或由 header/annotation 确定性测得 | 不得称为推断 |
| `inferred` | 由明确规则推断，保留 rule 与 confidence | 不得称为 official |
| `unknown` | 已查一手来源但含义仍不确定，值为 `null` | 不等于 `not_provided` |
| `not_provided` | 当前 release 未提供，值为 `null` | 不等于 negative |
| `not_annotated` | 适用时间/单位没有 annotation | 不等于 normal 或 negative |
| `missing` | contract 预期存在但文件/值缺失，是 QC 异常 | 不得作为标签 |
| `not_applicable` | 字段对该 unit 无语义 | 不应计入 missing rate |
| `blocked_pending_decision` | source facts 已知，但 mapping/split/policy 未批准 | 不得物化 |
| `approved_benchmark_policy` | 用户批准的 benchmark policy，仍非 official fact | 不得改写 source evidence |
| `approved_policy_not_authorized_to_materialize` | policy 已批准，但未开放真实 split/manifest | 不得生成真实 artifact |
| `approved_blocked_from_shared_phase1` | safe default 已批准为不进入 phase-1 shared surface | 不得强制 shared mapping |

JSON 中所有 `null` 均须由同层 `status_code` 解释。

## 3. 四数据集合同总表

| Dataset / release | Native prediction unit | Native task | Identity | Source split | Shared binary | Shared narrow four |
|---|---|---|---|---|---|---|
| ICBHI 2017 official package | respiratory cycle；另有 patient diagnosis | cycle `normal/crackle/wheeze/both` | direct `patient_id` | official recording split；patient 156、218 overlap | **approved primary**，flags 精确折叠 | **approved secondary**，flags 精确映射 |
| HF_Lung_V1, README 2022-01-18 | 15 s recording 输入，interval/frame sequence 输出 | I/E/CAS/DAS temporal detection | patient unavailable；date/session 仅 proxy | source train/test folders；date-proxy overlap=0 | **approved blocked**：无显式 negative/normal | **approved blocked**：无 normal、interval overlap、other labels |
| SPRSound BioCAS2022, commit `874eeb...` | event；recording | official event binary/7-class；record ternary/5-class | direct `patient_id` | train/inter/intra；intra intentionally repeats train patients | **approved primary**，official event binary | **approved subset secondary**；Rhonchi/Stridor 不静默映射 |
| KAUH/Fraiwan v3 | recording；patient diagnosis | raw 9-way sound type；raw diagnosis | direct P-number | no official split | **approved blocked**，dataset-native only | **approved blocked**，dataset-native only |

### Compatibility matrix

| Dataset | Native cycle/event | Native temporal multilabel | Shared binary | Shared narrow four | Native recording sound | Disease |
|---|---|---|---|---|---|---|
| ICBHI | yes | N/A | approved primary | approved secondary | N/A | dataset-specific secondary，当前 phase 未授权 |
| HF | positive intervals only | approved dataset-native | approved blocked | approved blocked | approved 15 s sequence input | not provided |
| SPRSound | yes | N/A | approved primary | approved subset secondary | yes | not provided |
| KAUH | no boundaries | no boundaries | approved blocked | approved blocked | approved dataset-native；patient-group dry-run only | dataset-specific secondary，当前 phase 未授权 |

## 4. ICBHI 2017

### Source / official facts

- Raw cycle annotation row：`start_s end_s crackle_flag wheeze_flag`。
- 两个 binary flags 组成四个 cycle state：`00 normal`、`10 crackle`、`01 wheeze`、`11 both`。
- Raw package measured：920 recordings、126 patients、6,898 cycles、5.492508 h。
- Cycle distribution：normal 3,642；crackle 1,864；wheeze 886；both 506。
- Official challenge recording split：539 train recordings / 381 test recordings；4,142 / 2,756 cycles。
- 该 split **不是 patient-independent**。Patient 156、218 同时出现在 train/test；完整 recording IDs 已保存在 `dataset/processed/schema/evidence.csv`。
- Diagnosis 是 patient-level side file，必须与 cycle event target 分离。
- 音频格式由 raw header 实测为 mono，采样率 4/10/44.1 kHz，bit depth 16/24 bit。

证据：

- `dataset/raw/icbhi_2017/source_original/ICBHI_final_database/ICBHI_final_database/filename_format.txt`
- `dataset/raw/icbhi_2017/ICBHI_challenge_train_test.txt`
- `dataset/processed/manifests/icbhi_2017_cycles.csv`
- `dataset/processed/schema/evidence.csv`

### Approved benchmark policy（非 source fact）

- Shared binary：`00 -> normal`，其余 flags -> `abnormal`。
- Shared narrow four：按四种 flag combination 精确映射。
- 双协议：official split 只用于 literature comparability；另建 strict patient-grouped protocol 用于 robustness。
- 双协议 policy 已批准，但本次不生成 strict split；real split assignment 仍需单独授权。

建议指标：保留 official sensitivity/specificity/average score；同时报告 macro F1、UAR、per-class recall 和 support。

## 5. HF_Lung_V1 完整语义核验

### 5.1 Recording 与 annotation schema

- Release：public GitLab HF_Lung_V1；README 更新 2022-01-18；CC BY 4.0。
- 9,765 WAV，全部为 15.0 s、4 kHz、mono、16-bit，总时长 40.6875 h。
- `steth_...`：Littmann 3200；`trunc_...`：HF-Type-1。
- `trunc_yyyy-mm-dd-HH-MM-ss-LX_N` 的 `LX` 是 location，`N` 是 original recording 的第 N 个 15 s truncation。
- `steth_` 文件不能由文件顺序推断 auscultation location。
- Label row：`<token> <start_hh:mm:ss> <end_hh:mm:ss>`。

Raw tokens 和一手定义：

| Family | Raw token | Source semantics | Rows |
|---|---|---|---:|
| respiratory phase | `I` | inhalation | 34,095 |
| respiratory phase | `E` | exhalation | 18,349 |
| adventitious | `D` | discontinuous adventitious sound；论文说明全部是 crackles，不分 coarse/fine | 15,606 |
| adventitious / CAS | `Wheeze` | wheeze | 8,457 |
| adventitious / CAS | `Rhonchi` | rhonchus/rhonchi | 4,740 |
| adventitious / CAS | `Stridor` | stridor | 686 |

没有 `Normal`、`Negative`、`N` 或 equivalent raw token。Disease label 未在 release 中提供。

### 5.2 Overlap、gap 与 annotation state

Raw interval union 审计口径：每个 15 s recording 内裁到 `[0,15]` 后计算 interval union。

| 项目 | Raw-package measured |
|---|---:|
| Total recording time | 146,475.000 s |
| Any annotation union | 52,051.639 s，35.536% |
| Unannotated gap | 94,423.361 s，64.464% |
| Recordings with any gap | 9,765 / 9,765 |
| Empty label files | 58 |
| Phase annotation union | 49,182.331 s |
| Adventitious-positive union | 25,185.405 s |
| Recordings with adventitious positive | 5,956 |
| Explicit negative intervals | 0 |

主要 overlap 证明 phase 和 adventitious label 不能放入 single-class ontology：

- `D|I` 8,163.396 s；`D|E` 5,269.245 s。
- `Wheeze|I` 3,078.489 s；`Wheeze|E` 2,234.120 s。
- Sound-sound overlap 也存在：`D|Wheeze` 63.783 s、`D|Rhonchi` 20.917 s、`Wheeze|Rhonchi` 11.509 s。

一手论文说明：

- 三位临床 annotator 标注 I/E/W/S/R/D 的 start/end；每个 recording 只由一位 annotator 处理。
- 无法清晰辨认，或位于文件首尾且不完整的 event，被要求**不标注**。
- Source benchmark 将 event intervals rasterize 为 one-vs-rest frame target，并在该任务中操作性地定义 TN。

因此需要区分：

1. `observed_positive`：明确落在 D/Wheeze/Rhonchi/Stridor interval 内。
2. `explicit_negative`：raw package 中不存在。
3. `not_annotated`：所有 gap，以及未被目标 event 覆盖的时间；不能自动升级为 shared normal。
4. `source_task_negative`：复现论文 one-vs-rest detection 时，由 rasterization 规则构造的 zero frame；这是 source task policy，不是原始 normal annotation。

Primary sources：

- `dataset/raw/hf_lung_v1/README.md`
- `dataset/raw/hf_lung_v1/source_original/{train,test}/`
- [HF_Lung_V1 PLOS ONE paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254134)

### 5.3 Prediction-unit compatibility

| Unit | Safe native use | Shared use |
|---|---|---|
| 15 s recording | 输入完整 recording，输出 I/E/CAS/DAS sequence | 不能直接当单一 normal/abnormal recording label |
| Raw interval | 正向 label interval analysis / event detection receipt | 无 negative class，不能直接形成 closed binary classifier |
| Fixed STFT frame | 可按 source paper rasterization 复现 four one-vs-rest detections | 只能标为 source-task constructed negative |
| Gap/background | annotation coverage/QC | `not_annotated`，禁止当 normal |

Source metrics：segment accuracy/PPV/sensitivity/specificity/F1/ROC-AUC；event Jaccard matching、event F1、MAPE。

### 5.4 Identity 与 split

- README-reported population：261 TSECC patients + 18 RCW/RCC residents。
- Package 没有可验证 patient ID，因此 `patient_id=null, status=not_provided`。
- Date 被随机 shift；README 只说 same-date files **very likely** 来自同一 subject。
- Date 可以作为 conservative grouping proxy，不能写入 `patient_id`，也不能证明 patient independence。
- Source folders：7,809 train / 1,956 test；118 / 39 unique date proxies；date-proxy overlap=0。

Approved benchmark policy：按 deidentified date proxy 分组，timestamp/truncation stem 作为次级 session proxy，并审计 device/location。该 safe default 只用于 dataset-native adapter/split dry-run；date proxy 仍不是 `patient_id`，且本次不授权真实 split assignment 或 negative sampling。

## 6. SPRSound BioCAS2022

### Source / official facts

- Scope 固定为 BioCAS2022 classification release，local source pin：commit `874eeb8736ddb78937c2fb5332fc7e7293d0f0ca`。
- Raw package：2,683 recordings、292 patients、9,089 events、8.162338 h；8 kHz mono 16-bit。
- Event raw labels：`Normal`、`Rhonchi`、`Wheeze`、`Stridor`、`Coarse Crackle`、`Fine Crackle`、`Wheeze+Crackle`。
- Record raw labels：`Normal`、`CAS`、`DAS`、`CAS & DAS`、`Poor Quality`。
- Official tasks：event binary / event 7-class；record ternary / record 5-class。
- Source split：
  - train：1,949 recordings / 251 patients。
  - inter test：355 recordings / 41 patients；train overlap=0。
  - intra test：379 recordings / 162 patients；train overlap=162，是官方 repeated-subject design。
- Poor Quality 是 recording quality class，不是 acoustic pathology class。

Evidence：

- `dataset/raw/sprsound/source_original/SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca/BioCAS2022/README.md`
- `dataset/processed/split_statistics.csv`
- [official repository](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)

### Approved benchmark policy（非 source fact）

- Shared binary event：采用 official `Normal vs Adventitious`。
- Narrow four subset：Normal；Fine/Coarse Crackle -> crackle；Wheeze -> wheeze；Wheeze+Crackle -> both。
- Rhonchi 和 Stridor 保留 raw label，从 narrow four 排除，不静默映射为 wheeze。
- Inter 作为 primary cross-subject test；intra 单独作为 repeated-subject diagnostic；两者禁止 pooled metric。
- Poor Quality 从 shared acoustic target 排除，保留为 dataset-native class/QC status。

Official metrics：SE、SP、AS、HS。建议补充 macro F1、UAR、per-class recall/support。

## 7. KAUH/Fraiwan v3 完整语义核验

### 7.1 Source identity 与文件结构

- Release：Mendeley Data v3，DOI `10.17632/jwyy9np4gv.3`。
- Raw：336 WAV、112 patients、每位 patient 恰好 3 个文件；4 kHz mono 16-bit，总时长 1.623918 h。
- Filename：

```text
<filter>P<patient>_<diagnosis>,<sound_type>,<location>,<age>,<gender>.wav
```

- B/D/E filename prefix 的一手定义：
  - `B` = Bell mode。
  - `D` = Diaphragm mode。
  - `E` = Extended mode。
- 每个 P-number 恰有 B/D/E 各一条；三条文件的 diagnosis、sound type、location、age、gender 完全一致。
- 因而 336 是 filtered recordings，不是 336 independent subjects。
- Workbook fields：`Age`、`Gender`、`Location`、`Sound type`、`Diagnosis`，另附 code legend。

Primary sources：

- `dataset/raw/kauh_fraiwan/Data annotation.xlsx`
- `dataset/raw/kauh_fraiwan/source_original/audio_files/`
- [Mendeley Data v3](https://data.mendeley.com/datasets/jwyy9np4gv/3)
- [Data in Brief dataset paper](https://doi.org/10.1016/j.dib.2021.106913)

### 7.2 Raw sound ontology

一手来源定义：`I=Inspiratory`、`E=Expiratory`、`W=Wheezes`、`C=Crackles`、`N=Normal`、`Crep=Crepitations`；论文另列 `Bronchial` sound type。

| Raw sound string | Patients | Recordings | Accepted source semantics | Shared status |
|---|---:|---:|---|---|
| `N` | 35 | 105 | Normal | proposed normal |
| `E W` | 39 | 117 | Expiratory + Wheezes | lossy wheeze candidate |
| `I E W` | 2 | 6 | Inspiratory + Expiratory + Wheezes | lossy wheeze candidate |
| `C` | 7 | 21 | Crackles | crackle candidate |
| `I C` | 1 | 3 | Inspiratory + Crackles | lossy crackle candidate |
| `I C E W` | 2 | 6 | I/C/E/W composition | lossy both candidate |
| `Crep` | 23 | 69 | Crepitations，source 与 C 分列 | blocked |
| `Bronchial` | 1 | 3 | Bronchial raw sound | blocked |
| `I C B` | 2 | 6 | I、C 已定义；B 未定义 | blocked |

`I C B` 的 candidate interpretation：论文 Table 3 恰好报告 2 位 `Bronchial & Crackles`，与 workbook 的 2 位 `I C B` 数量吻合，因此 `B=Bronchial` 有推断依据；但 source ontology 没有明确定义 sound-type 内的 `B`，所以该解释只能记录为 candidate，不能成为 accepted semantics。

`Crep` 与 `C` 被一手来源分别写成 Crepitations 和 Crackles。即使临床上可能相关，本合同不把 `Crep` 自动归一为 crackle；解除阻塞需要 source-author clarification 或 Jingping 对临床 ontology 的明确批准。

### 7.3 Disease 与 split

- Workbook diagnosis raw strings必须原样保留，包括 capitalization、拼写和组合差异：`Heart Failure`/`heart failure`、`COPD`/`copd`、`Asthma`/`asthma`、`Plueral Effusion` 等。
- Sound type 与 Diagnosis 是两列、两个 target family，永久分离。
- Source package 和一手 dataset paper 未提供 official split。
- 安全的 dataset-native task：raw 9-way recording sound-type head；但 evaluation 必须按 P-number grouping。
- Disease head 只能作为 dataset-specific secondary task，且 normalization/scope 仍待审批。
- Shared binary/four-class 只能使用已定义 token 的 recording-level subset；由于 unit 不同及 unresolved labels，当前整体标为 blocked。

Approved benchmark policy：patient-grouped 5-fold CV，或固定 patient-grouped train/validation/test；B/D/E siblings 永远不能分到不同 partition。该批准只开放 grouped-split dry-run preparation，不授权生成真实 split assignment，也不解除 `I C B`、`Crep`、`Bronchial` 的 normalization 阻塞。

## 8. 用户审批结果

以下 safe defaults 已于 2026-07-28 批准。Alternatives 和 tradeoff 保留为决策依据，不代表仍待用户决定。

| Item | Source fact | Safe default | Alternatives | Tradeoff | Approval outcome |
|---|---|---|---|---|---|
| HF negative interval | 无 explicit normal/negative；gap 64.464%；source paper 构造 frame zero | Shared HF binary/four blocked；source reproduction 单独命名 | 所有非正类 frame 当 negative；另做 verified-negative curation | 安全方案不能立即加入 shared task | **已批准：保持 shared blocked** |
| HF proxy grouping | patient_id 不提供；same date only “very likely” same subject | date 仅作 group proxy，patient_id 保持 null | 只用 source folder；完全排除 strict-patient claims | proxy 降低 leakage 但不能证明 independence | **已批准：date proxy only；不称 strict-patient** |
| HF phase vs adventitious | I/E 与 D/W/R/S 是重叠维度 | 分离 phase head 与 adventitious head/mask | joint multilabel head | joint 可建模 overlap，但跨数据集更复杂 | **已批准：分离 dataset-native heads** |
| KAUH B/D/E | 三种 filter，每 patient 各一条 | 同 patient 分组，报告 filter | 只用一个 filter；per-filter study | 三条不是独立样本 | **已批准：group all + per-filter audit** |
| KAUH `I C B` | B 未定义；2 rows 与 Bronchial & Crackles count吻合 | 保留 raw、shared blocked | 推断 B=Bronchial；询问作者 | 推断可多用 6 recordings，但有 ontology 风险 | **已批准：保持 blocked，寻求作者确认** |
| KAUH `Crep/C` | source 分别定义 Crepitations / Crackles | 分开；只有 C 进入 crackle candidate | 专家批准后合并 | 分开 support 更碎；合并引入临床等价假设 | **已批准：保持分离** |
| Disease scope | ICBHI/KAUH 有 patient-level disease；HF/SPR 无匹配 target | dataset-specific secondary heads | phase 1 排除；后续建 disease ontology | secondary 保留信息但增加范围 | **已批准：dataset-specific secondary；当前 phase 未授权** |
| Shared binary/four | ICBHI/SPR event最干净；HF缺 normal；KAUH unit/labels不齐 | binary primary：ICBHI + SPR，unit-stratified；narrow four secondary | 强行四数据集；four-class primary | safe default规模小但可审计 | **已批准：两数据集 shared surface** |

## 9. Approval gate

本次批准允许：

- canonical schema preparation 与 schema-only validation；
- ICBHI/SPRSound adapter dry-run preparation；
- ICBHI cycles + SPRSound events 的 shared encoder baseline implementation preparation；
- shared binary primary、narrow four secondary，以及 ICBHI cycle four-class / SPRSound event seven-class dataset-native head 的代码准备；
- dataset/unit-stratified reporting contract 的实现准备。

以下输出仍保持禁止：

- real split assignment；
- real recording/event manifest 或 training input；
- HF negative sampling；
- KAUH unresolved label normalization；
- 复杂 MoE；
- 大规模训练或任何 training execution；
- resampling、segmentation、feature extraction 的实际执行；
- processed audio 或 training input；
- benchmark/model run。

后续 implementation gate 必须先准备 schema/adapter，并再次申请 real manifest、split materialization 或训练授权。本批准不自动跨越这些 gate。

## 10. Verification receipt

- 合同状态：`approved_for_phase1_baseline_implementation`；approval date：2026-07-28。
- 四个 dataset rows：完整。
- Approved shared scope：ICBHI cycles + SPRSound BioCAS2022 events；binary primary；narrow four secondary；dataset/unit-stratified reporting。
- HF/KAUH safe default：dataset-native only，shared phase-1 blocked。
- Native labels：均保留；KAUH diagnosis 原始字符串与 `I C B`/`Crep`/`Bronchial` 未被覆盖。
- Missing label != negative：通过。
- HF proxy 未写入 patient_id：通过。
- Source split 与 proposed split 分列：通过。
- Source facts 与 proposed policy 分列：通过。
- 用户批准的 policy 与 source official facts 分列；未解决的 source semantics 仍为 `unknown`/blocked，不因批准而升级为 official。
- Real manifest、HF negative sampling、KAUH unresolved normalization、复杂 MoE 和大规模训练：未授权。
- Raw/processed/baseline/model/result/experiments：未修改。
- Notion/Git stage/commit/push：未执行。

## 11. Four-Dataset Tail Eligibility / Support Contract

检查日期：2026-07-29

本节是 Baseline T1 的硬前置门，只冻结 support eligibility，不评价任何新方法。当前 T1 状态为：

> `blocked_pending_hf_proxy_fix_and_regression_verification`

管理于 2026-07-29 接受 `fixed_outer_inter_v1`、`kauh_patient_oof_v1` 及 16 primary / 5 diagnostic / 7 not-evaluable frozen assignments。该接受不解除下述 HF implementation blocker。

本次没有读取 `result/` 中的未来 T1 结果，没有训练、生成 split assignment、重提 embedding 或修改 Baseline。计数来自 raw annotation/workbook、既有 ICBHI manifest，以及现有 deterministic split protocol 的只读复算。

### 11.1 固定任务与证据边界

| Task | Prediction unit | Group identity | Outer evaluation | Negative / missing policy |
|---|---|---|---|---|
| ICBHI `flat4` | respiratory cycle | direct `patient_id` | official challenge recording test | peer class 是 observed negative；official test 非 patient-independent |
| SPR event binary | official event | direct `patient_id` | inter-subject test | peer class 是 observed negative；intra 不进入 primary |
| SPR event seven | official event | direct `patient_id` | inter-subject test | peer class是 observed negative；inter 零支持不能由 intra 补齐 |
| HF phase presence | 15 s recording | canonical `YYYYMMDD` date proxy，非 patient | source test | 只在至少有 I/E annotation 的 recording pool 内，peer-label absence 才是 observed negative |
| HF adventitious presence | 15 s recording | canonical `YYYYMMDD` date proxy，非 patient | source test | 只在至少有 D/Wheeze/Rhonchi/Stridor annotation 的 pool 内定义 peer-label absence；gap 不是 negative |
| KAUH raw9 | recording | direct P-number | five-fold aggregate OOF | raw peer class 是 observed negative；B/D/E siblings 同组 |

HF 每条 15 s recording 的三个 5 s windows 只是 encoder input aggregation，不增加 recording label support。KAUH `Crep`、`I C B` 等只按 raw dataset-native string 审计；本节不批准任何 shared normalization。

## 12. 冻结阈值

阈值在查看最终 eligibility assignment 前提出，并机械应用，无类别例外。

### 12.1 Fixed outer/inter rule `fixed_outer_inter_v1`

`primary_evaluable` 同时要求：

- subtrain positive ≥ 20 samples / 10 groups；
- inner validation positive ≥ 5 / 3；
- outer/inter positive ≥ 20 / 5；
- inner validation observed negative ≥ 5 / 3；
- outer/inter observed negative ≥ 20 / 5。

`diagnostic_only` 同时要求：

- subtrain positive ≥ 5 / 3；
- validation positive、test positive、validation negative、test negative均至少 1 sample / 1 group。

低于 diagnostic 下限，或 outer/inter positive=0，固定为 `not_evaluable`。

### 12.2 KAUH OOF rule `kauh_patient_oof_v1`

`primary_evaluable` 要求：

- aggregate OOF positive ≥ 30 recordings / 10 patients；
- 5/5 folds 的 subtrain/validation/test 都有正例；
- 每折 subtrain ≥ 15 recordings / 5 patients，validation/test 各 ≥ 3 / 1。

`diagnostic_only` 要求 aggregate ≥ 9 / 3，subtrain coverage=5/5，validation/test coverage 均 ≥3/5。其余固定为 `not_evaluable`。

理由：固定测试任务需要独立 group 支撑 selection 与 test interpretation；KAUH 每 patient 有三个相关 B/D/E recordings，因此 raw recording 数不能代替 patient support。

## 13. Support 结果

下表 `samples/groups` 中 ICBHI、SPR、KAUH 的 group 是 patient；HF 是 date proxy，不能称 patient。完整 observed-negative、unknown 与 not-annotated counts 位于 machine-readable contract 的 `tail_eligibility_contract.label_assignments`。

### 13.1 ICBHI cycle flat4

| Label | Full train | Subtrain | Inner val | Official test | Frozen state |
|---|---:|---:|---:|---:|---|
| normal | 2063/77 | 1578/61 | 485/16 | 1579/49 | `primary_evaluable` |
| crackle | 1215/47 | 805/37 | 410/10 | 649/29 | `primary_evaluable` |
| wheeze | 501/39 | 408/34 | 93/5 | 385/26 | `primary_evaluable` |
| both | 363/21 | 264/17 | 99/4 | 143/16 | `primary_evaluable` |

`both` 虽是最小类，但 validation 有 99 cycles/4 patients，official test 有 143/16，机械通过 primary。该结论不消除 official train/test 中 patient 156、218 overlap 的 protocol 限制。

### 13.2 SPRSound BioCAS2022 event tasks

| Task / label | Full train | Subtrain | Inner val | Inter test | Frozen state |
|---|---:|---:|---:|---:|---|
| binary normal | 5159/235 | 4114/188 | 1045/47 | 1040/39 | `primary_evaluable` |
| binary adventitious | 1497/98 | 1105/74 | 392/24 | 389/20 | `primary_evaluable` |
| seven Normal | 5159/235 | 4114/188 | 1045/47 | 1040/39 | `primary_evaluable` |
| seven Rhonchi | 39/11 | 34/10 | 5/1 | 0/0 | `not_evaluable` |
| seven Wheeze | 452/50 | 396/40 | 56/10 | 305/13 | `primary_evaluable` |
| seven Stridor | 15/2 | 15/2 | 0/0 | 0/0 | `not_evaluable` |
| seven Coarse Crackle | 49/15 | 46/13 | 3/2 | 3/3 | `diagnostic_only` |
| seven Fine Crackle | 912/72 | 593/53 | 319/19 | 80/12 | `primary_evaluable` |
| seven Wheeze+Crackle | 30/9 | 21/6 | 9/3 | 1/1 | `diagnostic_only` |

Intra 仅保留为 repeated-subject audit：Rhonchi 14/5、Stridor 2/1、Coarse Crackle 14/5、Wheeze+Crackle 3/3。它不能补充 inter 的 primary support，也不得与 inter 合并。

### 13.3 HF_Lung recording presence

HF support 使用修正后的 canonical date proxy。表中 test `P/N/NA` 分别是 observed positive、eligible-pool observed negative、not annotated recordings；均为 `samples/date-proxies`。

| Task / label | Full-train positive | Subtrain positive | Inner-val positive | Test P / N / NA | Frozen state |
|---|---:|---:|---:|---:|---|
| phase I | 7510/116 | 5132/92 | 2378/24 | 1935/39 · 5/3 · 16/8 | `diagnostic_only` |
| phase E | 4593/102 | 2721/79 | 1872/23 | 950/37 · 990/37 · 16/8 | `primary_evaluable` |
| adventitious D | 3076/102 | 1898/79 | 1178/23 | 368/32 · 589/23 · 999/37 | `primary_evaluable` |
| adventitious Wheeze | 2253/88 | 1265/66 | 988/22 | 405/20 · 552/33 · 999/37 | `primary_evaluable` |
| adventitious Rhonchi | 944/83 | 522/62 | 422/21 | 296/22 · 661/34 · 999/37 | `primary_evaluable` |
| adventitious Stridor | 253/30 | 224/22 | 29/8 | 12/7 · 945/35 · 999/37 | `diagnostic_only` |

`I` 的问题不是 positive 少，而是 source test 只有 5 个 observed negatives/3 proxies，不能提供稳定 specificity 或 balanced binary claim。Stridor 的 test positive 只有 12/7。所有 NA recording 和 recording 内 unannotated gap 都不得转成 negative。

### 13.4 KAUH raw9 aggregate OOF

| Raw label | OOF recordings/patients | Subtrain/val/test fold coverage | Frozen state |
|---|---:|---:|---|
| N | 105/35 | 5/5 · 5/5 · 5/5 | `primary_evaluable` |
| E W | 117/39 | 5/5 · 5/5 · 5/5 | `primary_evaluable` |
| I E W | 6/2 | 4/5 · 3/5 · 2/5 | `not_evaluable` |
| C | 21/7 | 5/5 · 5/5 · 4/5 | `diagnostic_only` |
| I C | 3/1 | 4/5 · 0/5 · 1/5 | `not_evaluable` |
| I C E W | 6/2 | 5/5 · 0/5 · 2/5 | `not_evaluable` |
| Crep | 69/23 | 5/5 · 5/5 · 5/5 | `primary_evaluable` |
| Bronchial | 3/1 | 4/5 · 0/5 · 1/5 | `not_evaluable` |
| I C B | 6/2 | 5/5 · 2/5 · 2/5 | `not_evaluable` |

每个 cell 为 `subtrain recordings/patients · validation recordings/patients · test recordings/patients`：

| Raw label | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| N | 51/17 · 27/9 · 27/9 | 57/19 · 33/11 · 15/5 | 66/22 · 18/6 · 21/7 | 60/20 · 27/9 · 18/6 | 54/18 · 27/9 · 24/8 |
| E W | 69/23 · 24/8 · 24/8 | 69/23 · 12/4 · 36/12 | 81/27 · 27/9 · 9/3 | 69/23 · 24/8 · 24/8 | 72/24 · 21/7 · 24/8 |
| I E W | 3/1 · 0/0 · 3/1 | 0/0 · 6/2 · 0/0 | 3/1 · 3/1 · 0/0 | 3/1 · 3/1 · 0/0 | 3/1 · 0/0 · 3/1 |
| C | 15/5 · 6/2 · 0/0 | 15/5 · 3/1 · 3/1 | 9/3 · 6/2 · 6/2 | 12/4 · 3/1 · 6/2 | 12/4 · 3/1 · 6/2 |
| I C | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 0/0 · 0/0 · 3/1 |
| I C E W | 6/2 · 0/0 · 0/0 | 3/1 · 0/0 · 3/1 | 6/2 · 0/0 · 0/0 | 3/1 · 0/0 · 3/1 | 6/2 · 0/0 · 0/0 |
| Crep | 42/14 · 12/4 · 15/5 | 45/15 · 15/5 · 9/3 | 27/9 · 12/4 · 30/10 | 45/15 · 12/4 · 12/4 | 51/17 · 15/5 · 3/1 |
| Bronchial | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 3/1 · 0/0 · 0/0 | 0/0 · 0/0 · 3/1 |
| I C B | 6/2 · 0/0 · 0/0 | 3/1 · 0/0 · 3/1 | 3/1 · 3/1 · 0/0 | 3/1 · 0/0 · 3/1 | 3/1 · 3/1 · 0/0 |

## 14. Frozen Gate Sets

### Primary gate eligible

- ICBHI flat4：normal、crackle、wheeze、both。
- SPR binary：normal、adventitious。
- SPR seven：Normal、Wheeze、Fine Crackle。
- HF phase：E。
- HF adventitious：D、Wheeze、Rhonchi。
- KAUH raw9：N、E W、Crep。

### Diagnostic only

- SPR seven：Coarse Crackle、Wheeze+Crackle。
- HF phase：I。
- HF adventitious：Stridor。
- KAUH raw9：C。

### Not evaluable

- SPR seven：Rhonchi、Stridor。
- KAUH raw9：I E W、I C、I C E W、Bronchial、I C B。

只有 `primary_evaluable` 可提供下一阶段 go/no-go success vote。`diagnostic_only` 必须报告但不能满足或否决 primary gate；`not_evaluable` 只能报告 support failure。禁止把各数据集 raw Score 合并成一个成功票。

## 15. Provenance 与阻塞项

| Receipt | Project-relative path / identity | SHA256 |
|---|---|---|
| ICBHI cycle manifest | `dataset/processed/manifests/icbhi_2017_cycles.csv` | `d3e57e625a40579db5551dd34e07b85a2d4d3b273361499508aed1c579bee40c` |
| ICBHI official split | `dataset/raw/icbhi_2017/ICBHI_challenge_train_test.txt` | `5afa11096c3988d8aaefa5164c053c873f718e730fbb286d482f516b01f05c52` |
| SPR annotation tree | BioCAS2022 commit `874eeb8736ddb78937c2fb5332fc7e7293d0f0ca` | `431a8b68f476296d4da1e574b12f7b8f965be4676c0059b68bbafebb9b3f2db7` |
| HF label tree | `dataset/raw/hf_lung_v1/source_original/**/*_label.txt` | `31997bf3d5b43f3c959e681b7bee5b3f5c0bd1f320cac2943c9f99ac26861c1b` |
| HF README | `dataset/raw/hf_lung_v1/README.md` | `330f15668edad3b0805a57a922cad9563db14224de2169d6c01c6868bc9cb1ce` |
| KAUH workbook | `dataset/raw/kauh_fraiwan/Data annotation.xlsx` | `a54f46a5fb6b2cf290ad7edc6c97ad1524be109dfe71054922c5693b918bf4cf` |
| KAUH ordered WAV names | `dataset/raw/kauh_fraiwan/source_original/audio_files` | `e846682237aa5969064f1a9ad85a561f2b1f303e04ef8ea3aad6f05e44c7ced8` |

HF verifier 发现：`baseline/four_dataset_frozen_encoder/data.py` 当前对 `steth_YYYYMMDD` 与 `trunc_YYYY-MM-DD` 使用不同 date string，得到错误的 130/41 groups。旧 `130/41` grouping **不能继续作为 accepted split evidence**。唯一 canonical assignment 是 source train/test `118/39`、subtrain/validation/test `94/24/39`，assignment SHA256=`33387aa62ebcb8adbc1fba626e6d27f27a3121d3a686cda6fc075a7da106943e`。Baseline 仍须先修复 HF proxy parser，并通过 regression verifier，T1 才能启动。

## 16. Tail Contract Verification

- 六个 native tasks、28 个 label rows：完整且 `(dataset, task, label)` 无重复。
- Frozen states：16 `primary_evaluable`、5 `diagnostic_only`、7 `not_evaluable`。
- ICBHI manifest/split counts独立复算；`both` group support 明列。
- SPR train/validation/inter 分开；intra 只作 audit，未与 inter 合并。
- HF `patient_id` 保持 unavailable；date proxy 明确不是 patient；missing/not-annotated 未转成 negative。
- KAUH aggregate OOF 恰好 336 unique recordings；五折 patient overlap=0；B/D/E siblings 同组。
- Source facts、approved benchmark policy、proposed threshold、frozen assignment 在 JSON 中分字段。
- Management acceptance：通过；阈值与 16/5/7 assignments 已冻结。
- Baseline T1：仍阻塞，等待 HF parser fix 与 regression verification。
- Raw/processed/baseline/model/experiments/result/Notion：未修改；未训练；未 stage/commit/push。
