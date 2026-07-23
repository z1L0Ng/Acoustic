# PAFA / SG-SCL -> SPRSound checkpoint-only package handoff

## 结论

状态：`READY_FOR_IMMUTABLE_SNAPSHOT_CONDITIONAL_ON_SERVER_CHECKPOINT_SMOKE`。

已完成 B1 PAFA 与 B2 SG-SCL 的 fresh-checkout runnable package。两个 package
锁定 Patch-Mix B0 的同一 SPRSound BioCAS2022 official inter event target，且本地只做
target-contract、无标签 preprocessing、静态与 verifier 回归测试。没有启动 B1/B2
checkpoint inference，也没有训练、target adaptation、threshold/calibration、intra、
HF_Lung 或 KAUH 工作。

`B0` verified artifacts 与 `C0` package 本轮均未修改。

## Frozen target contract

- Unit / split：SPRSound BioCAS2022 official inter-subject event。
- Events：1,429 unique IDs。
- Ordered event-ID SHA256：
  `81a6b15783a01eb86abe218928884b41e7f975f64eedaefd546e2dbf3deba44b`。
- Raw support：Normal 1,040；Fine Crackle 80；Coarse Crackle 3；Wheeze 305；
  Wheeze+Crackle 1。
- Narrow four：normal 1,040；crackle 83；wheeze 305；both 1。
- Binary broad：four-class argmax 后，normal 保持 normal，其余 collapse abnormal；
  不做 binary threshold search。
- All-normal floor：使用同一 1,429 rows 和最终 scoring labels 独立生成。
- Labels：smoke 和 model inference 阶段不读取；full mode 在全部 logits 已写出后才
  join labels 做 scoring。
- `both` support=1，只报告结果，不作 minority/generalization 结论。

## B1 PAFA

- Code：`baseline/pafa/checkpoint_eval/`。
- Environment：冻结的 `acoustic-pafa-r4`；本轮未修改 environment/runtime-hotfix。
- Source：作者 repo `https://github.com/wa976/PAFA`，commit
  `e49e294d0db0d6af10ac46290512b9c85d3f71e1`；repo 无 LICENSE。
- Task checkpoint gate：必须显式提供 server epoch-27 checkpoint path 和 SHA256；
  要求 `epoch/model/classifier`，strict-load model 与 classifier。
- Backbone gate：BEATs_iter3+ AS2M SHA256
  `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34`。
- Preprocessing：16 kHz mono、recording fade、event crop、5 s repeat/truncate、
  raw waveform、`--nospec`、augmentation off。
- Source status：epoch 27 Sp 76.884 / Se 51.402 / Score 64.143。Score 对 paper
  five-seed mean 64.84+/-0.60 的 gap 为约 -0.70 pp（约 -1.16 SD），但 Sp 更低、
  Se 更高；不是 componentwise exact replication，也不是 five-seed aggregate。
- Canonical output：`result/pafa_sprsound_transfer_<Chicago timestamp>/`。
- Package manifest SHA256：
  `938a023d37e8b26d13b8a808f0e0dabd66db0c577af91b4cd856b96f3e4c7d58`。

## B2 SG-SCL

- Code：`baseline/sg_scl/checkpoint_eval/`。
- Environment：冻结的 `acoustic-sgscl-r4`；本轮未修改 environment/runtime-hotfix。
- Source：作者 repo
  `https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning`，
  commit `66564609595090b61540595d3d27764c00553086`；repo 无 LICENSE。
- Task checkpoint gate：必须显式提供 server epoch-27 checkpoint path 和 SHA256；
  要求 `epoch/model/classifier`，classifier `1.weight` 必须为 `4 x 768`，随后 strict
  load 完整 predictive state。
- Preprocessing：16 kHz mono、recording fade、event crop、8 s repeat/truncate、
  128-bin fbank、resize `798 x 128`、augmentation off。
- Metadata boundary：SG-SCL source training 使用 device/domain metadata；作者
  `validate()` 使用 `model(..., training=False)` 的 audio embedding 和 class
  classifier，不消费 domain label。本 package 不创建或猜测 SPRSound device label。
- Source status：epoch 27 Sp 74.984 / Se 46.984 / Score 60.984；三个分量均在 paper
  five-seed distribution 约 0.6 SD 内，是 successful single-seed numerical alignment，
  不是 five-seed aggregate reproduction。
- Canonical output：`result/sg_scl_sprsound_transfer_<Chicago timestamp>/`。
- Package manifest SHA256：
  `7ac0aa9495bddb966b2fcfd70ae3c0dd5199e4387f63b63bed4ed203e2ce03a9`。

## Verification completed locally

1. Exact target contract：1,429 events；ID SHA matched；mapped support
   `{normal:1040, crackle:83, wheeze:305, both:1}`。
2. Python compile：shared、PAFA、SG-SCL bootstrap/run/verify modules passed。
3. PAFA label-free preprocessing smoke：8 events，shape `80000`，labels not accessed。
4. SG-SCL label-free preprocessing smoke：8 events，shape `1 x 798 x 128`，labels
   not accessed，device metadata not used。
5. Shared smoke verifier regression：8 fixed IDs、no labels/metrics、finite logits/probs、
   softmax、four-class argmax、binary collapse、binary probability aggregation passed。
6. Shared full verifier regression：使用已 verified B0 predictions 作为纯机械 fixture；
   model/floor metrics 独立复算通过，binary/narrow4/model/floor 四份 confusion totals
   均为 1,429。没有重跑 B0 inference。
7. Notebook：两份 nbformat 4，所有 code cells `execution_count=null`、`outputs=[]`。
8. Static package verifier：PAFA 12 hashed files、SG-SCL 12 hashed files；无 `/Users/`、
   `/files1/`、legacy `results/` 路径。
9. `git diff --check` passed；没有 stage/commit/push。

本机不存在 `-r4` CUDA env 和两份 server task checkpoint，因此没有、也不应声称
checkpoint load/forward smoke 已在本机通过。Package readiness 的最后 runtime gate 是
服务器按 README 顺序执行 bootstrap -> fixed 8-event label-free smoke -> independent
smoke verifier；该 gate 通过后才允许单独启动 full 1,429-event inference。

## Server handoff

Exact commands 已写入：

- `baseline/pafa/checkpoint_eval/README.md`
- `baseline/sg_scl/checkpoint_eval/README.md`

服务器必须从已验收 source-run receipt 注入 checkpoint path 与 SHA256；不能用文件名、
epoch 描述或任意 compatible checkpoint 替代 hash gate。PAFA 还必须注入上述固定 SHA 的
BEATs backbone。Bootstrap 会 clone/pin 作者 source 到新的 canonical result root，smoke
不读取 target labels；full 与 verifier 是后续独立命令。

## Claim boundary

未来 B1/B2 数值只能称为：published-model / verified-server-checkpoint /
ICBHI-test-selected / zero-target-tuning exploratory transfer。它们不能称为 clean source
anchor、formal transfer generalization 或 degradation proof；在没有各自 target-trained
reference 时，跨模型 OOD 行为也不能代替 within-model degradation 结论。
