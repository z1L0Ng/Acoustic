# 现有本地资产写作交付（无新增训练）

日期：2026-09-09

本文件主体仍是原先的 CPU-only 存量资产交付；后来用户直接授权产生的 JH3.3 HF/KAUH 固定 checkpoint 外评另列为附加资产，不并入下述原始 CPU-only 汇总或 clean Table 2。

## 结论摘要

本批只消化已有本地结果、NPZ、日志、manifest 与 `recording_features.csv`；没有启动训练、模型前向、特征提取、缓存构建或新的 official test。结果应分成三层：

1. **Test-selected benchmark**：JH2 正式 seed0/1/fresh42。三 seed 的 ICBHI Score 为 $61.168\pm0.314\%$，SPRSound official Score 为 $90.699\pm0.342\%$。这是可描述的 test-selected benchmark，不是 validation-selected clean estimate。
2. **Post-hoc external/readout diagnostics**：C/W-only、SPR C/W AUROC、HF Lung、KAUH，以及历史 JH4 HF extension。它们不能填充新的 clean Table 2 primary row。
3. **Prospective clean suite**：`table2_clean_split_manifest.json` 已冻结并验证 support，但 Full42 只完成 point1/update326，不能当作完成的 clean training run。

## 1. JH2 主结果可支持的数字

来源：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.json`。

| 指标 | mean ± sample SD，n=3 |
|---|---:|
| ICBHI Sp | 74.984 ± 5.690% |
| ICBHI Se | 47.352 ± 5.353% |
| ICBHI Score | 61.168 ± 0.314% |
| ICBHI Macro-F1 | 49.732 ± 1.059% |
| ICBHI UAR | 50.405 ± 3.028% |
| SPRSound AS | 90.948 ± 0.311% |
| SPRSound HS | 90.450 ± 0.393% |
| SPRSound official Score | 90.699 ± 0.342% |
| SPRSound Macro-F1 | 86.257 ± 0.684% |
| SPRSound UAR | 90.948 ± 0.311% |

正式 selected epoch 为 seed0=15、seed1=11、fresh seed42=17。ICBHI 每个 seed 在已完成 points 逐点访问 test；SPRSound official inter 每个 selected checkpoint 访问一次。

ICBHI per-class recall mean ± sample SD：Normal 74.984 ± 5.690%，Crackle 60.195 ± 8.482%，Wheeze 29.610 ± 2.078%，Both 36.830 ± 11.369%。三个 seed 的完整 confusion matrices 保存在上述 JSON 的 `per_seed[].icbhi.confusion`，不要重新从稿件数字反推。

## 2. Direct C/W-only readout pilot

来源：`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_CW_READOUT_PILOT_EXISTING/cw_readout_existing_pilot.json`。

该结果直接汇总了三个 selected run 已保存的 `icbhi_flat4_bits_only_ablation`，未重算 probabilities。C/W-only 相对 Full 的 mean paired delta 为：Sp −6.798 pp，Se +5.041 pp，Score −0.878 pp，Macro-F1 −0.488 pp，UAR +0.436 pp，Both recall 0.000 pp。seed方向不完全一致：seed0 −0.596 pp，seed1 −2.600 pp，fresh seed42 +0.562 pp。

写作边界：这是 test-selected post-hoc readout pilot；不要把它填入 validation-selected Table 2，也不要写成一般性“C/W decoder提升/损害”的因果结论。

脚本已准备但尚未执行真实重建：[jh2_cw_readout_pilot.py](/Users/zilongzeng/Research/Acoustic/scripts/analysis/jh2_cw_readout_pilot.py)。

## 3. SPR C/W attribute readout

来源：三个 selected SPR NPZ 的保存 `attribute_probabilities`、targets 与 eligibility 的 CPU 后处理；结果已核对但未写入正式 Table 2。

- Crackle AUROC：96.079 ± 0.405%，AUPRC 60.325 ± 3.861%。
- Wheeze AUROC：98.557 ± 0.114%，AUPRC 94.512 ± 0.564%。
- C/W macro AUROC：97.318 ± 0.148%；定义为每个 seed 先取 Crackle/Wheeze 两个 AUROC 的均值，再跨三个 seed 取 mean/sample SD。对应 macro AUPRC 为 77.418 ± 2.032%。
- 每个 seed 的 SPR inter support 为 1,429 rows；Crackle eligible positives/negatives=84/1,345，Wheeze=306/1,123。

| Seed | Crackle AUROC | Wheeze AUROC | C/W macro AUROC | C/W macro AUPRC | 已保存来源 |
|---:|---:|---:|---:|---:|---|
| 0 | 0.960258 | 0.986030 | 0.973144 | 0.779021 | `seed_0/terminal/selected_sprsound_predictions_scored.npz` |
| 1 | 0.965082 | 0.984277 | 0.974680 | 0.791652 | `seed_1/terminal/selected_sprsound_predictions_scored.npz` |
| 42 | 0.957028 | 0.986410 | 0.971719 | 0.751882 | `seed_42/terminal/selected_sprsound_predictions_scored.npz` |

这些是 selected test probabilities 上的 attribute diagnostic，不改变 native Score selection。

### 旧 JH2 选中 epoch 的实际 C/W mask

以下是旧 JH2 各 seed 的实际 validation NPZ mask，不是新 clean manifest：

| Seed / epoch | Dataset | C eligible / positive / negative | W eligible / positive / negative |
|---|---|---:|---:|
| 0 / 15 | ICBHI | 506 / 179 / 327 | 506 / 190 / 316 |
| 0 / 15 | SPRSound | 1432 / 331 / 1101 | 1432 / 65 / 1367 |
| 1 / 11 | ICBHI | 1262 / 664 / 598 | 1262 / 218 / 1044 |
| 1 / 11 | SPRSound | 1432 / 331 / 1101 | 1432 / 65 / 1367 |
| 42 / 17 | ICBHI | 968 / 283 / 685 | 968 / 244 / 724 |
| 42 / 17 | SPRSound | 1432 / 331 / 1101 | 1432 / 65 / 1367 |

来源为 `PAFA_JH2_main_multiseed/seed_{0,1,42}/validation/epoch_{selected}.npz`；这些旧 partition support 不能替代新 clean manifest。

### 旧 JH2 实际训练样本的 eligibility 覆盖

管理补核：使用每个 run 已保存的 `config.json` 和原 metadata loader 重建 subtrain/validation；三个 seed 重建出的 validation sample IDs 均与该 seed 已保存的 selected-epoch NPZ 完全一致。此处仅统计已有标签和分组，没有读取模型权重或运行模型前向。

| Seed | Dataset | 实际训练 units / groups | C eligible / masked | W eligible / masked |
|---|---|---:|---:|---:|
| 0 | ICBHI cycles | 3636 / 63 | 3636 / 0 | 3636 / 0 |
| 1 | ICBHI cycles | 2880 / 63 | 2880 / 0 | 2880 / 0 |
| 42 | ICBHI cycles | 3174 / 63 | 3174 / 0 | 3174 / 0 |
| 0、1、42 各自相同 | SPRSound events | 5219 / 194 | 5170 / 49 | 5170 / 49 |

因此，旧 JH2 的 ICBHI 训练 C/W mask 比例均为 0；SPRSound 训练每个属性均屏蔽 49/5219=0.939% 的 events（两属性屏蔽同一批49个event，不是98个不同event）。SPRSound validation 共1437 events，其中每属性eligible1432、masked5。完整逐seed、逐partition的 positive/negative/eligible/masked 计数及原文件路径已补入汇总 JSON 的 `jh2_main_test_selected.actual_partition_support`。这些是旧 JH2 的实际划分，与 prospective clean manifest 分开报告。

## 4. HF/KAUH external 与历史 JH4 extension

当前 JH2 三 selected checkpoint 的 post-hoc external 汇总：
`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/external_multiseed_summary.json`。

- HF source-test：D/Crackle recording AUROC 67.823 ± 4.425%，AUPRC 59.108 ± 2.934%，Sp/Se 53.990 ± 22.028% / 69.293 ± 13.382%，BA/Score 61.642 ± 5.373%，Macro-F1 58.921 ± 8.128%，UAR 61.642 ± 5.373%；Wheeze AUROC 86.360 ± 1.376%，AUPRC 84.338 ± 2.261%，Sp/Se 61.775 ± 2.437% / 88.477 ± 1.918%，BA/Score 75.126 ± 0.617%，Macro-F1 73.058 ± 0.833%，UAR 75.126 ± 0.617%。D positive-interval recall 58.278 ± 12.631%，Wheeze 85.431 ± 1.068%。HF 使用 1,956 个 15-s recordings 的三个 5-s windows，其中 957 个 eligible；eligible、gap-unknown 与 unsupported label 定义以代码和 JSON 为准，不要简写成未核对的 positive-only 结论。
- KAUH：336 recordings/112 patients，258 compatible overlay recordings；recording Level1 BA/Score 70.987 ± 3.532%，Macro-F1 70.034 ± 3.732%，UAR 70.987 ± 3.532%；recording flat4 BA/Score 47.784 ± 9.140%，Macro-F1 34.034 ± 2.213%，UAR 45.444 ± 0.786%；patient Level1 BA/Score 72.166 ± 1.641%，Macro-F1 71.293 ± 2.874%，UAR 72.166 ± 1.641%；patient flat4 BA/Score 47.003 ± 8.296%，Macro-F1 33.830 ± 3.168%，UAR 50.241 ± 6.099%。`Crep`、`Bronchial`、`I C B` unresolved，不计入 compatible overlay。

历史 JH4 HF extension：`PAFA_JH4_JH2_HFaux_seed42_attempt2/hf_on_off_comparison.json`。它是 historical single-seed JH4-vs-historical-JH2 pair，不是当前 JH2 0/1/fresh42 matched comparison。实际 support 为 eligible_presence 957、empty 4、phase_only_or_not_annotated 995，D/W interval support 1812/1430；HF-on D AUROC 69.804%、interval recall 93.102%，Wheeze AUROC 89.831%、interval recall 76.923%。不要把该 pair 直接替代当前 Table 2 的 clean HF row。

该历史 pair 的精确 HF-off JH2 epoch19 checkpoint 已按原 5 s/no-MVN Hard Hierarchy 配方补做 SPR inter 1429-event 推理；HF-on 直接读取已保存 NPZ。C/W macro AUROC/AUPRC 为 HF-off `0.968063/0.762432`，HF-on `0.969505/0.751222`，差值分别为 `+0.001442/-0.011210`。Crackle AUROC/AUPRC 差值为 `-0.001106/-0.042612`，Wheeze 为 `+0.003990/+0.020193`。native SPR official Score 重算为 `0.892010`，与原保存值完全一致。

### 新增用户直接授权的 JH3.3 外部资产

用户后来直接要求“补充计算HF lung和KAUH的test”，因此对现有 `PAFA_JH3_3_soft_bridge_mvn_seed42` 的 selected epoch 21 固定 checkpoint 做了独立外部推理。该资产不属于上面的原 CPU-only 汇总，也不改变原 JH3.3 checkpoint selection。

- HF Lung：eligible recording 957/1956；D/Crackle AUROC 57.156%、AUPRC 48.469%、BA/Score 56.711%、Macro-F1 55.431%、interval recall 19.536%；Wheeze AUROC 86.568%、AUPRC 84.645%、BA/Score 75.587%、Macro-F1 73.331%、interval recall 85.944%。
- KAUH：258/336 compatible overlay recordings；recording Level1/flat4 Score=69.048%/53.035%，Macro-F1=68.257%/37.397%，UAR=69.048%/44.940%；patient Level1/flat4 Score=71.989%/51.401%，Macro-F1=71.539%/37.291%，UAR=71.989%/49.564%。`Crep`、`Bronchial`、`I C B` 保持 unresolved。
- 产物：`result/reproduce/pafa_joint_hierarchy/PAFA_JH3_3_HF_KAUH_external_seed42/`；证据标签为 `posthoc_fixed_JH3_3_selected_checkpoint_HF_KAUH_external_diagnostic`。

该结果是 fixed-checkpoint external-transfer diagnostic，不是 clean estimate、HF/KAUH native reproduction 或 paper primary result。另需注意：仓库中该旧 JH3.3 run 的实际 selection artifact 记录为 composite selection；它不能替代后来要求的 ICBHI-only 新 run。

### 四条件 seed42 benchmark controls（已完成）

四项均为 `seed42 / official-test-selected benchmark control`，不是 clean estimate 或纯因果消融。Full 只引用已完成的 fresh42 参照，未与 Full 三 seed aggregate 混合。

| Variant | 状态 / selected epoch | ICBHI Score | SPR official Score | SPR C/W mean AUROC |
|---|---|---:|---:|---:|
| Full_reference_seed42 | reference / epoch 17 | 0.611692 | 0.903674 | 0.971719 |
| ICBHI_only | complete / epoch 18 | 0.561654 | 0.568637 | 0.812306 |
| SPRSound_only | complete / epoch 24 | 0.479290 | 0.923200 | 0.978580 |
| Coarse_SPR | complete / epoch 20 | 0.567058 | 0.902187 | 0.936714 |
| Native_attributes | complete / epoch 17 | 0.623000 | 0.919438 | 0.974070 |

| Variant | ICBHI Sp / Se / Macro-F1 / UAR | ICBHI per-class recall (Normal / Crackle / Wheeze / Both) | SPR Sp / Se / Macro-F1 / UAR | SPR per-class recall (Normal / Abnormal) |
|---|---|---|---|---|
| Full_reference_seed42 | 80.367 / 41.971 / 48.509 / 47.114 | 80.367 / 51.926 / 31.688 / 24.476 | 84.038 / 97.172 / 85.917 / 90.605 | 84.038 / 97.172 |
| ICBHI_only | 67.131 / 45.200 / 44.993 / 48.530 | 67.131 / 60.555 / 18.182 / 48.252 | 63.846 / 50.643 / 55.499 / 57.244 | 63.846 / 50.643 |
| SPRSound_only | 82.774 / 13.084 / 27.923 / 29.399 | 82.774 / 7.550 / 27.273 / 0.000 | 87.019 / 97.943 / 88.438 / 92.481 | 87.019 / 97.943 |
| Coarse_SPR | 74.414 / 38.997 / 45.861 / 50.006 | 74.414 / 44.530 / 22.338 / 58.741 | 82.885 / 98.201 / 85.407 / 90.543 | 82.885 / 98.201 |
| Native_attributes | 78.721 / 45.879 / 50.963 / 50.387 | 78.721 / 57.473 / 30.390 / 34.965 | 86.538 / 97.686 / 87.982 / 92.112 | 86.538 / 97.686 |

SPR C/W attribute support 对五个 row 相同：Crackle 84 positive / 1,345 negative，Wheeze 306 positive / 1,123 negative；per-attribute AUROC/AUPRC 与 confusion 完整保存在各 run 的 terminal metrics/selected NPZ。

| Variant | Crackle AUROC / AUPRC | Wheeze AUROC / AUPRC | 相对 Full 的 ICBHI / SPR / C/W-AUROC 差值 |
|---|---:|---:|---:|
| Full_reference_seed42 | 0.957028 / 0.563082 | 0.986410 / 0.940681 | — |
| ICBHI_only | 0.804072 / 0.197465 | 0.820541 / 0.575086 | -0.050038 / -0.335036 / -0.159413 |
| SPRSound_only | 0.970163 / 0.716408 | 0.986996 / 0.958710 | -0.132402 / +0.019527 / +0.006861 |
| Coarse_SPR | 0.909126 / 0.334770 | 0.964303 / 0.851330 | -0.044634 / -0.001487 / -0.035005 |
| Native_attributes | 0.960781 / 0.638952 | 0.987359 / 0.946198 | +0.011308 / +0.015764 / +0.002351 |

Coarse 的最终结果必须使用 `coarse_spr/seed_42_attempt2`（30 epochs，selected epoch 20）；原 `coarse_spr/seed_42` 中断 partial 保留但不计入五行。Native+attributes 完成27 epochs，selected epoch17。

完整五行结构化汇总、native per-class 指标、confusion、thresholds、signed deltas 与 source paths 已写入 `LOCAL_ASSETS_NO_TRAIN_SUMMARY.json` 和 `LOCAL_ASSETS_NO_TRAIN_TABLE.csv`。

## 5. 数据构成与 Figure 1 / Section 2.2

来源：`result/acoustic_distribution/recording_features.csv`；Figure 1 代码：`Figure/figure1_dataset_composition_acoustic.py`。统计是已有 feature table 的描述，不是本轮重新提特征或 separability 实验。

| Dataset | 表格 rows | 实际 group 定义/数量 | split | duration 简述 |
|---|---:|---|---|---|
| ICBHI | 920 | `patient_id`，126 groups | train 538 / test 381 / NA 1 | mean 21.492 s，median 20.0 s，range 7.856–86.2 s |
| SPRSound | 2,683 | `patient_id`，292 groups；原始 recording-level table rows，为 BioCAS audio recordings；model event units separate | train 1,949 / inter_test 355 / intra_test 379 | mean 10.952 s，median 9.216 s，range 0.304–15.36 s |
| HF Lung | 9,765 | recording-date proxy，157 groups | train 7,809 / test 1,956 | all 15.0 s |
| KAUH | 336 | `patient_id`，112 groups；B/D/E each 112 | prefix_B/D/E | mean 17.399 s，median 15.984 s，range 5–30 s |

Figure 1 的 panel B/C 先按 `dataset × group identifier` 对13个 acoustic features取group-level median，再按最小group数平衡采样到每dataset 112 groups；ICBHI/SPRSound/KAUH 的 group identifier 为 `patient_id`，HF 为 recording-date proxy。该定义应在 Section 2.2 明确为 feature-table group 描述，不等同于训练 split 或 model evaluation unit。

## 6. Clean suite 与条件性缺项

`baseline/pafa/table2_clean_split_manifest.json` 的冻结 prospective support 为：ICBHI subtrain/calibration/selection=3055/700/387 cycles，groups=63/5/11；SPRSound=5219/694/743 events，groups=194/26/23。它不是旧 JH2 的实际训练统计，也没有被 Full42 partial 转化为完成实验。

若保留当前稿 validation-selected Table2 定义，则当前尚无对应完整结果；是否必须新训练、seed budget由用户/管理讨论：

- clean validation-selected Full / ICBHI-only / SPRSound-only 完整三 seed runs；
- clean Full+HF、Coarse SPR、Native+attributes 条件；
- 当前稿件 Table 2 的正式数值填入与协议措辞最终对齐；
- Figure 1 discussion/Overleaf assembly remains writing review；本轮未在本地编译。

不要把 Full42 partial point1/update326、历史 JH4、旧 JH2 test-selected checkpoint或LocalClean partial冒充完成的 clean Table 2 结果。
