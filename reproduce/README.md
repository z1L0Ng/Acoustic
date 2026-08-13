# 多数据集 Pipeline Reproduce Notebooks（2026-08-12 主键）

本目录是 Working Plan / Pipeline 组合实验主表的零执行 orchestration 骨架。每个 active Pipeline 恰好对应一个 notebook；notebook 只声明冻结合同、参数、审批门和 receipt schema，不在单元中启动训练、推理、下载、服务器、Notion 或 Git 操作。

## Active P1–P13

| ID | Notebook | 新语义 | 状态 |
|---|---|---|---|
| P1 | `P1_joint_native_reference.ipynb` | AST shared 2-s window encoder/projector，四数据集 native routes | Local Code/Asset/CPU READY；CUDA/执行 HOLD |
| P2 | `P2_beats_native_encoder.ipynb` | matched P1，四数据集 shared encoder package 换为 BEATs | Local Code READY；canonical Asset provision/CUDA/执行 HOLD |
| P3 | `P3_panns_cnn14_native_encoder.ipynb` | PANNs Cnn14_16k + declared 2048→768 adapter，四数据集 shared-window | Code/合成 CPU READY；Asset/CUDA/执行 HOLD |
| P4 | `P4_hear_native_encoder.ipynb` | HeAR + declared 512→768 adapter，四数据集 shared-window | Code/合成 CPU READY；gated Asset/CUDA/执行 HOLD |
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

所有 active notebook 固定 seed `20260728`，不得改已冻结 split。P1–P5 的四数据集 subtrain units 为 13,794，batch size=8，冻结为每 reference epoch 1,725 updates、总 86,250 updates、每1,725 updates只用 validation native losses选checkpoint；selection scalar不是pooled performance。优化器冻结为 Proposed Benchmark Policy：Adam、learning rate `5e-5`、weight decay `1e-6`、per-update cosine schedule；来源是已有 joint-native reference contract，未使用 outer/test 选择。production adapter、checkpoint/provenance、CUDA smoke、phase-specific approval或新 independent verifier未闭合时必须 fail closed。Notebook保持零执行，结果表均为 `Not run`。

## Production adapter 与 L40 零更新入口

- 真实 frozen provider：`baseline/multidataset_pipeline/real_subtrain_provider.py`，schema=`real_frozen_provider_identity_v2`。它只复用 `baseline/four_dataset_frozen_encoder/data.py::build_samples` 的已接受 split，不重分；data identity 同时绑定 ordered IDs、canonical split authority（ICBHI manifest、SPRSound revision/split、HF assignment/date proxy、KAUH fold/group/test-ID identity）及 HF annotation/tree/interval semantics。`identity bound` 不等于 `independently verified`；HF own label-tree SHA 与 accepted reference 不同，明确保持 `not_verified_equivalent` 和 verifier HOLD。实际 inventory 已核对 subtrain ICBHI=3055、SPRSound=5219、HF=5322、KAUH=198；本地 CPU loader smoke 只解码每 lane 一条 subtrain unit，四 lane 均通过 16 kHz / 2 s / 1 s window 与 lineage gate，outer/test waveform/scoring 均为 0。
- 统一训练装配：`baseline/multidataset_pipeline/train_shared_window.py`，schema=`shared_window_training_v5`。科学比较合同的 `config_sha256` 不含 phase；approval 验证后，smoke/full 的 config、approval、scope、optimizer、cache receipt、logs、checkpoints、selection 与 run receipt 必须分别写入 `result/reproduce/<P>_shared_window_seed20260728/<phase>/<approval_receipt_sha256>/`。fresh exact execution root 若已有任何 artifact 必须拒绝；resume 仅允许 full，且必须绑定同一 approval/execution root、最新 checkpoint receipt 及不超前的 train/validation log。共享 frozen embedding cache 仍位于 `.cache/multidataset_pipeline/embeddings/<P>/`，不按 phase 复制。P1/P2 `full` 在 optimizer update 1 前必须 build/load/verify subtrain+validation×四 lane 共8项 frozen embedding cache，随后训练与 validation 只读取 cache并训练 projector/native heads；每 lane/partition immutable receipt 写入该 execution root 的 `embedding_cache_receipt.json` 与最终 run receipt。`smoke` 明确采用 uncached engineering gate，不生成性能结论。checkpoint 原子写入后记录 path/size/byte SHA、component-state hashes、update、config/data identity 与原 full-approval SHA；resume 必须同时提供 `--resume` 与 `--resume-sha256`，先核对外部 byte SHA 才允许反序列化。每个 validation checkpoint 都持久化 artifact receipt，full 结束生成带候选与 exact winner 的 `validation_selection_receipt.json`；terminal approval 必须绑定该 receipt SHA，且只能评分其中选中的 exact checkpoint。`preflight` 不创建 model/optimizer/run root。
- Production terminal scorer：`baseline/multidataset_pipeline/terminal_scoring.py`，schema=`shared_window_terminal_scorer_v1`。只接受 `ICBHI_flat4`、`SPRSound_binary`、`SPRSound_raw7`、`HF_temporal4`、`KAUH_raw9` 五个精确任务；HF 使用 `[B,Nw,4]` 的 window∩annotation∩valid mask 与 validation-frozen threshold receipt，constructed negatives 明确不是 raw/shared normal。generic runner 的 `terminal-score` CLI 必须显式给 selection receipt/SHA、selected checkpoint/SHA、terminal approval、`module:function` provider 与 provider identity SHA；terminal approval 同时绑定 scorer schema 和 provider identity，provider 只在 immutable gate 全部通过后才可读取 outer/test。当前 tracked `terminal_provider_manifest.json` 不存在，因此 preflight 机器可读状态为 `HOLD_no_registered_production_terminal_provider`，end-to-end terminal 尚不 READY；不得用测试 provider 冒充。输出禁止 pooled/global/cross-dataset score 或 ranking。
- Frozen embedding cache：`embedding_cache.py` + production runner bridge `runner_embedding_cache.py`，schemas=`frozen_window_embedding_cache_v1` / `shared_window_runner_cache_set_v1`。只允许 frozen+eval encoder、deterministic frontend、无 augmentation，并只缓存 subtrain/validation；identity 绑定 dataset/release/ordered IDs/data identity、16 kHz resample、2 s/1 s window/tail/padding/masks、encoder asset、frontend/adapter 与 code/config/schema SHA。cache 使用原子目录写入和逐 artifact size/SHA/shape/dtype 验证；outer/test、partial/stale/corrupt/duplicate/missing 全部 fail closed。P1/P2 full 必须在 update 1 前闭合8项 cache，之后 source-proportional training 与 validation 不再 decode waveform或调用 encoder；smoke 显式保持 uncached engineering gate。
- Tracked asset manifest：`baseline/multidataset_pipeline/adapter_assets.json`。P1/P2 factory 与 inventory/L40 preflight 都读取并交叉核对 manifest；P2 canonical source/checkpoint 改为 `.cache/multidataset_pipeline/assets/P2/...`，旧 `result/pafa...` 只保留历史审计价值，绝不作为 silent fallback，也不会由本代码移动、复制或下载。
- 新 independent verifier receipt schema 升为 `shared_window_first_queue_verifier_v2`，必须同时核对 provider data identity、manifest/split hashes、runner schema、phase approval、optimizer/trainable scope、resume external SHA、checkpoint artifact/component hashes、validation-selection artifact SHA、terminal exact-checkpoint binding、terminal scorer receipt、asset manifest、embedding cache、window/adapter receipts、per-dataset/native-task metrics 与 outer/test access；`archived_baseline_verifier` 不可用于本队列，任何 pooled/global score 均非法。
- 统一 B×K/lineage/receipt：`baseline/multidataset_pipeline/window_encoder.py`；factory：`adapter_factory.py`。
- P1：`ast_window_encoder.py`，2 s fbank 尾部 zero-pad 到 798 frames；本地 checkpoint SHA256 `bc9fe72b1a38b7071db8b606c63f8f2e41bf2cccaf3e80fc0ba5c33094877cb1`。
- P2：`beats_window_encoder.py` 复用 `beats_temporal.py` exact patch mask/pooling；冻结 checkpoint SHA256 `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`。当前 canonical `.cache/multidataset_pipeline/assets/P2/...` 尚待服务器确定性 provision，因此 Asset HOLD；旧 orphan 路径不回退。
- P3：`panns_window_encoder.py` 只绑定官方 16 kHz Cnn14 recipe；当前本地 source/checkpoint/torchlibrosa 缺失，不下载，Asset HOLD。
- P4：`hear_window_encoder.py` 只绑定 accepted local HeAR 1.0.0 SavedModel；当前 gated asset、TensorFlow 与模型条款接受未闭合，Asset HOLD。
- Inventory：`python -m baseline.multidataset_pipeline.real_subtrain_provider --phase inventory`；CPU loader contract smoke：同命令加 `--phase cpu-loader-smoke`。二者不是模型评测。
- 训练 dry preflight：`python -m baseline.multidataset_pipeline.train_shared_window --pipeline P1 --phase preflight`；不会创建 optimizer 或 update。
- L40 入口：`python -m baseline.multidataset_pipeline.l40_preflight --pipeline P1`，默认绑定上述 real subtrain provider。它强制 L40、1–2 个 `subtrain` batches、四 lane 覆盖、frozen encoder、0 optimizer updates、参数不变与 `outer_test_accessed=false`；运行仍需单独服务器批准。

上述 READY 只指 provider/runner/adapters 的本地工程合同与 CPU smoke，不是模型性能结果，也不授权服务器或完整训练。P1/P2 下一门是 L40 CUDA zero-update preflight；P3/P4 仍分别受本地可信 asset/dependency 或 gated asset/license 阻塞。P5 OPERA provenance/overlap 继续 Scientific HOLD；P6 继续 deferred。canonical split authority 目前会为了复现冻结 assignment 审计各 partition 的 source metadata，但 provider 在 waveform decode 前过滤，仅暴露 subtrain/validation，且不调用 terminal target loader；若未来要求完全 label-blind 的 terminal manifest，需要管理侧另行冻结 immutable assignment artifact，不能由本模块重写 split。
