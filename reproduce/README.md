# 多数据集 Pipeline Reproduce Notebooks

本目录保存冻结 Pipeline ID 对应的轻量 Jupyter orchestration entrypoint。Notebook 只组织已冻结合同、参数、执行门和 receipt schema；算法实现属于可复用 Python 模块，实验定义属于显式 config，生成证据属于 `result/`。本轮仅创建骨架，**没有执行 Notebook、训练、推理、服务器任务或外部写入**。

## 冻结清单与状态

| ID | Notebook | 角色 | 当前状态 |
|---|---|---|---|
| P1 | `P1_joint_native_reference.ipynb` | 四数据集 joint native reference | Design / Not Ready |
| P2 | `P2_beats_native_encoder.ipynb` | BEATs AudioSet-only encoder comparator | Design / Not Ready |
| P3 | `P3_eligibility_masked_objective.ipynb` | eligibility-masked objective | Design / Not Ready |
| P4 | `P4_beats_eligibility_combination.ipynb` | BEATs + eligibility combination | Design / Not Ready |
| P5 | `P5_negative_stress_diagnostic.ipynb` | unknown/gap-as-negative stress diagnostic | Design / Not Ready |
| P6 | `P6_dataset_balanced_sampler.ipynb` | dataset-balanced source sampler | Design / Not Ready |
| P7 | `P7_lodo_transfer_ast.ipynb` | AST LODO target-head adaptation | Design / Not Ready |
| P8 | `P8_lodo_zero_target_gate.ipynb` | 合法 zero-target LODO gate | Scientific HOLD |
| P9 | `P9_pafa_projector_only.ipynb` | PAFA projector-only capacity control | Design / Not Ready |
| P10 | `P10_pafa_projector_plus_loss.ipynb` | PAFA projector + eligible patient-aware loss | Design / Not Ready |
| P11 | `P11_mvst_optional_nonhf.ipynb` | MVST optional non-HF comparator | Design / Not Ready |
| P12 | `P12_opera_overlap_reference.ipynb` | OPERA overlap-aware reference | Scientific HOLD |
| P13 | `P13_shared_temporal_hold.ipynb` | shared temporal interface gate | Scientific HOLD |

`Design / Not Ready` 不是可运行或可训练状态；`Scientific HOLD` 表示当前科学合同本身尚不允许执行或支持目标 claim。任何状态变化必须由管理侧冻结新的 preregistration、config、预算、selection、go/no-go、执行授权与独立 verifier。

## 共同数据合同

- ICBHI：annotated respiratory cycle，native flat4 head `[B,4]`；官方 recording split 的 patient 156/218 overlap 必须披露，内部 validation 按 patient group。
- SPRSound BioCAS2022：annotated event，native binary `[B,2]` 与 raw seven-class `[B,7]` 独立；inter 为主，intra 仅 repeated-subject diagnostic；outer/inter 标签只在固定预测完成后进入 terminal scoring。
- HF_Lung_V1：15-second recording 上的合法 observed-positive native presence heads；date proxy 只用于 grouping，不是 patient ID；unknown、空标注与时间 gap 不是 negative/normal。
- KAUH/Fraiwan：recording，raw sound-type head `[B,9]`；B/D/E 是同一 P-number patient 的 replicas，必须同组；shared mapping 与 diagnosis head 保持 HOLD。
- 所有指标按 dataset、prediction unit、native task 分开；禁止 pooled cross-dataset Score。

## 单数据集 external prerequisites

单数据集 reproduction 由本科生对接任务负责，本目录不实现或运行。多数据集 Notebook 只能消费已验收、不可变的 prerequisite receipt，最小接口为：

```text
prerequisite_receipt.json
  status
  dataset_id / release_or_commit / license
  prediction_unit / native_labels / native_head_shape
  manifest_path / manifest_sha256 / ordered_id_sha256
  split_name / grouping_key / leakage_audit
  missing_gap_semantics
  preprocessing_contract / source_time_lineage
  checkpoint_origin / revision / sha256 / selection_caveat
  label_free_outer_prediction_policy
  verifier_status / verifier_warnings
```

缺少任何字段、hash 不匹配、receipt 未被管理验收、或 prerequisite 使用了与 Pipeline 不一致的 split/label/unit 时，Notebook 必须 fail closed。

## Notebook 执行合同

每个 Notebook 固定包含：Pipeline ID/研究问题/role；Verified Contract、Proposed Method、HOLD；组件和唯一 comparator 变化；四数据集 native contract；checkpoint provenance、trainable scope、预算、seed、selection；execution gates；输出/receipt schema；claim boundary；`Test Result=Not run` 与 Decision。

代码 cell 只允许：

1. 定义轻量参数与相对路径或环境变量；
2. 检查冻结 receipt/config/hash/授权是否齐全；
3. 生成但不执行 dry-run orchestration plan。

代码 cell 禁止直接调用训练、推理、Notebook 执行、服务器、下载、Notion 或 Git。正式运行必须改由冻结的 CLI/config 包完成，并在独立 verifier 通过后才产生 Decision。
