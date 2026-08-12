# 多数据集 Pipeline Reproduce Notebooks（2026-08-12 主键）

本目录是 Working Plan / Pipeline 组合实验主表的零执行 orchestration 骨架。每个 active Pipeline 恰好对应一个 notebook；notebook 只声明冻结合同、参数、审批门和 receipt schema，不在单元中启动训练、推理、下载、服务器、Notion 或 Git 操作。

## Active P1–P13

| ID | Notebook | 新语义 | 状态 |
|---|---|---|---|
| P1 | `P1_joint_native_reference.ipynb` | AST joint-native reference；HF 固定 native temporal reference | Design / Not Ready |
| P2 | `P2_beats_native_encoder.ipynb` | 仅替换 non-HF pooled/native encoder 为 BEATs | Design / Not Ready |
| P3 | `P3_panns_cnn14_native_encoder.ipynb` | PANNs Cnn14 non-HF encoder/reference lane | Design / Not Ready |
| P4 | `P4_hear_native_encoder.ipynb` | HeAR non-HF encoder/reference lane | Design / Not Ready |
| P5 | `P5_opera_overlap_reference.ipynb` | OPERA overlap-aware encoder/reference lane | Scientific HOLD |
| P6 | `P6_beats_shared_temporal.ipynb` | BEATs shared temporal package：tokens/mask/time-map/head | Scientific HOLD |
| P7 | `P7_pooling_comparator.ipynb` | matched pooling-block comparator | Design / Not Ready |
| P8 | `P8_eligibility_masked_objective.ipynb` | eligibility-aware compatible supervision | Design / Not Ready |
| P9 | `P9_dataset_balanced_sampler.ipynb` | dataset-balanced sampler comparator | Design / Not Ready |
| P10 | `P10_pafa_projector_only.ipynb` | patient-ID eligible projector-only | Design / Not Ready |
| P11 | `P11_pafa_projector_plus_loss.ipynb` | matched projector + eligible PCSL/GPAL | Design / Not Ready |
| P12 | `P12_mvst_optional_nonhf.ipynb` | optional eligible non-HF MVST lane | Design / Not Ready |
| P13 | `P13_lodo_transfer_ast.ipynb` | target-supervised LODO transfer/adaptation | Design / Not Ready |

8/20 core 为 **P1、P2、P6、P8**。simple baseline 与其余组合行 deferred；`Design / Not Ready` 和 `Scientific HOLD` 均不构成执行批准。服务器任务、完整训练、读取 outer/test 结果和状态升级必须另获管理批准，并先冻结 question、split、metrics、budget、selection 与 go/no-go。

## 旧主键到新主键

| 旧条目 | 新位置 |
|---|---|
| old P1 joint native | new P1 |
| old P2 BEATs native | new P2 |
| old P3 eligibility | new P8 |
| old P6 sampler | new P9 |
| old P7 LODO | new P13 |
| old P9 projector-only | new P10 |
| old P10 projector+loss | new P11 |
| old P11 MVST | new P12 |
| old P12 OPERA | new P5 |
| old P13 shared temporal | new P6 |
| old P4 BEATs+eligibility | `archive/legacy_old_p4_beats_eligibility_combination_deferred.ipynb` |
| old P5 negative diagnostic | `archive/legacy_old_p5_negative_stress_diagnostic_deferred.ipynb` |
| old P8 zero-target gate | `archive/legacy_old_p8_lodo_zero_target_gate_deferred.ipynb` |

`archive/` 只保存旧语义、legacy/deferred notebook；它们不属于 active P 表，也不得被执行器按当前 P ID 发现。

## 共同四数据集 lane 边界

- ICBHI：annotated respiratory cycle，native flat4 `[B,4]`；固定 split，不重分 outer/test。
- SPRSound：annotated event，native binary `[B,2]` 与 raw-seven `[B,7]` 分开报告；terminal join 后才评分。
- HF_Lung_V1：15 秒时间标注；missing、unknown、空标注和 gap 都不是 negative。P1–P5 的 HF native temporal reference 固定；P6 才审计共享时序 package。
- KAUH/Fraiwan：recording-level raw-9 `[B,9]`；B/D/E 同一 P-number 必须同 patient group；shared mapping/diagnosis HOLD。
- 指标按 dataset、prediction unit、native task 分开；不得汇总成跨数据集 pooled Score。

## External prerequisite 与执行门

单数据集任务由本科生对接任务提供已验收 immutable receipt，本目录不实现或运行。最小 receipt 必须含数据/manifest hash、prediction unit、native head、split/grouping、missing-gap 语义、checkpoint provenance、source-time lineage、outer prediction policy 和独立 verifier 状态。

**PROTO-SINGLE-SOURCE-NATIVE** 同时 **referenced_by [P1,P2]**：ICBHI、SPRSound、KAUH 分别提供 AST 与 BEATs 的 single-source runs，共 3 datasets × 2 encoders = 6 runs；每个 run 使用与对应 joint 条件相同架构的 projector 768→256 + dataset-native head。HF fixed temporal reference 仅需 1 个独立 run，由 P1/P2 共用，不进入 AST/BEATs shared projector。

P1 首轮固定为 frozen pretrained AST + trainable shared projector 768→256 + dataset-native heads，sampler=source-proportional；P2 除 non-HF encoder 换为 frozen pretrained BEATs 外完全匹配。joint 只表示 ICBHI/SPRSound/KAUH 共用同一 projector，不表示共享 encoder 覆盖 HF。

所有 active notebook 固定使用单个训练 seed `20260728`；不得改已冻结 split。update budget、selection、checkpoint/input adapter、审批 receipt 未冻结时必须 fail closed。代码单元只能装配配置、读取环境变量/相对路径、验证合同和形成 dry-run plan；当前所有 `Test Result = Not run`，结果表只是 placeholder。
