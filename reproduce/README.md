# 多数据集 Pipeline Reproduce Notebooks（2026-08-12 主键）

本目录是 Working Plan / Pipeline 组合实验主表的零执行 orchestration 骨架。每个 active Pipeline 恰好对应一个 notebook；notebook 只声明冻结合同、参数、审批门和 receipt schema，不在单元中启动训练、推理、下载、服务器、Notion 或 Git 操作。

## Active P1–P13

| ID | Notebook | 新语义 | 状态 |
|---|---|---|---|
| P1 | `P1_joint_native_reference.ipynb` | AST shared 2-s window encoder/projector，四数据集 native routes | Preflight / Adapter HOLD |
| P2 | `P2_beats_native_encoder.ipynb` | matched P1，四数据集 shared encoder package 换为 BEATs | Preflight / Adapter HOLD |
| P3 | `P3_panns_cnn14_native_encoder.ipynb` | PANNs Cnn14 + declared 2048→768 adapter，四数据集 shared-window | Preflight / Adapter HOLD |
| P4 | `P4_hear_native_encoder.ipynb` | HeAR + declared 512→768 adapter，四数据集 shared-window | Preflight / Adapter HOLD |
| P5 | `P5_opera_overlap_reference.ipynb` | OPERA-CT 四数据集 shared-window overlap-aware reference | Scientific HOLD |
| P6 | `P6_beats_shared_temporal.ipynb` | deferred BEATs token-level HF temporal refinement | Deferred / Scientific HOLD |
| P7 | `P7_pooling_comparator.ipynb` | matched pooling-block comparator | Design / Not Ready |
| P8 | `P8_eligibility_masked_objective.ipynb` | eligibility-aware compatible supervision | Design / Not Ready |
| P9 | `P9_dataset_balanced_sampler.ipynb` | dataset-balanced sampler comparator | Design / Not Ready |
| P10 | `P10_pafa_projector_only.ipynb` | patient-ID eligible projector-only | Design / Not Ready |
| P11 | `P11_pafa_projector_plus_loss.ipynb` | matched projector + eligible PCSL/GPAL | Design / Not Ready |
| P12 | `P12_mvst_optional_nonhf.ipynb` | optional eligible non-HF MVST lane | Design / Not Ready |
| P13 | `P13_lodo_transfer_ast.ipynb` | target-supervised LODO transfer/adaptation | Design / Not Ready |

首轮顺序为 **G0 window/HF interface → P1/P2 → P3/P4 → P5 provenance gate**。P6 token refinement 与 P7 pooling、P8 loss、P9 sampler、P10–P13 全部 deferred 到 encoder shortlist 之后；任何 HOLD 都不构成执行批准。服务器、完整训练、outer/test 与状态升级必须另获管理批准。

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
- HF_Lung_V1：15 秒 native recording 进入与其他数据集相同的 candidate encoder/projector；保留 window sequence `[B,K,256]` 与 I/E/CAS/DAS native head。missing、unknown 和 raw gap 不是 raw negative；paper-native rasterization 的零必须标为 source-task constructed negative。
- KAUH/Fraiwan：recording-level raw-9 `[B,9]`；B/D/E 同一 P-number 必须同 patient group；shared mapping/diagnosis HOLD。
- 指标按 dataset、prediction unit、native task 分开；不得汇总成跨数据集 pooled Score。

## External prerequisite 与执行门

单数据集任务由本科生对接任务提供已验收 immutable receipt，本目录不实现或运行。最小 receipt 必须含数据/manifest hash、prediction unit、native head、split/grouping、missing-gap 语义、checkpoint provenance、source-time lineage、outer prediction policy 和独立 verifier 状态。

共同输入合同是 **Proposed Benchmark Policy**：16 kHz、2.0 s source-time window、1.0 s stride；长 unit 在 stride 未覆盖尾部时只追加一个唯一 end-aligned window，短 unit 只 zero-pad并保留 valid-sample mask、window mask与source-time map。该选择基于候选 input compatibility，不是论文/官方事实，也不得用 outer/test 选择。

P1–P5 都要求一个 frozen candidate encoder 处理四数据集 windows，并让同一个 trainable Linear 768→256 接收四个 lanes。ICBHI/SPRSound/KAUH 只对 valid windows masked mean；HF 不聚合时间维，使用 projected sequence与native temporal4 head。PANNs/HeAR 原生维度分别为2048/512，P3/P4 package额外包含跨四数据集共享的 trainable D→768 adapter，因此只能作 package-level comparator。四数据集从未被统一成一个标签任务。

所有 active notebook 固定 seed `20260728`，不得改已冻结 split。P1–P5 的四数据集 subtrain units 为 13,794，batch size=8，冻结为每 reference epoch 1,725 updates、总 86,250 updates、每1,725 updates只用 validation native losses选checkpoint；selection scalar不是pooled performance。production adapter、checkpoint/provenance、CUDA smoke、approval或新 independent verifier未闭合时必须 fail closed。Notebook保持零执行，结果表均为 `Not run`。
