# 2026-08-30 Acoustic 项目快速更新记录

## 1. 会议定位

本次会议承接 2026-08-27 的项目复盘，重点确认当前联合模型的改善、跨数据集
差异的论文价值、后续数据集扩展，以及 ICASSP 2027 截止日前的写作和图表节奏。
会议讨论的目标不是立即冻结最终方法，而是在 2026-09-02 前形成第一版可审阅的
论文故事，并让后续实验直接服务于该故事。

## 2. 上次会议后的完成成果

### 2.1 BEATs / PAFA 对照闭合

- PAFA BEATs+CE 的 clean track 已完成：使用 official-train 内的 patient-grouped
  validation 选择 epoch 9，official test 只访问一次；terminal Sp 78.53、Se 31.13、
  ICBHI Score 54.83。
- author-faithful test-selected track 已完成：每 epoch 使用 official test 选择，
  selected epoch 22；Sp 85.50、Se 34.23、Score 59.86。该结果永久标记为
  `test_selected_reproduction_receipt`，不能当作 clean estimate。
- PAFA-JH1 验证了“PAFA 训练路径 + 联合层级 head”可以执行，但训练在第 26 个
  epoch 中异常退出；其结果只保留为 incomplete diagnostic。
- PAFA-JH2 复用 JH1 的科学训练配方，仅把 checkpoint selection 改为逐 epoch
  official-test ICBHI Score。它在 epoch 19 得到 ICBHI Sp 74.73、Se 45.37、
  Score 60.05；SPRSound Task1-1 Score 89.20，并在 epoch 29 按 patience 10 停止。
  由于两套 official test 共被访问 29 次，该结果仅证明当前方法存在可行 operating
  point，不能作为 clean、primary 或 SOTA evidence。

### 2.2 训练和 checkpoint 问题得到定位

- 服务器端 true native-unit batch 修复后，full fine-tuning 从每 epoch 数十分钟
  降到约三分钟，确认此前低 GPU utilization 主要来自逐 unit 的 Python 调度。
- R0/N1/A1/L1/L2、JH1 和 clean PAFA 曲线共同表明：训练 loss 持续下降而
  validation 很早恶化，主要问题是快速记忆/表示漂移，而不是“训练 epoch 不够”。
- 后续新训练统一采用 epoch-boundary patience 10；clean run 只能监控预注册的
  validation criterion，不能使用 official test 早停。

### 2.3 数据差异证据已有第一版

- 已完成 ICBHI、SPRSound、HF_Lung 和 KAUH 的 20-track selection-level
  定量 panel，包括 RMS/dBFS、Welch PSD、band power、spectral centroid、
  bandwidth、silence 和 SNR proxies。
- panel 的 median RMS dBFS 分别为 ICBHI -11.06、SPRSound -41.02、
  HF_Lung -38.08、KAUH -20.63，显示明显的 level/domain difference。
- 这些结果目前只是代表性样本的描述性观察，受到设备、滤波、增益、native unit
  和类别构成混杂；尚不能称为 dataset-wide statistical significance。
- Audacity import plan 和 labels 已准备；`.aup3` 和 Audacity 截图因 macOS file
  dialog 阻塞仍处于 HOLD，但不影响量化分析继续推进。

### 2.4 文献、论文和协作准备

- 已完成 BEATs、PAFA 及相关 follow-up 的 primary-source audit，明确直接可比的
  ICBHI reference、test-selection caveat，以及 SPRSound/HF/KAUH 缺少 task-compatible
  BEATs literature result 的边界。
- 已建立 ICASSP 四页英文 skeleton，包含 Introduction、Dataset/Ontology、Method、
  Experiments、Discussion、figure/table inventory 和 claim ledger；当前仍需根据新故事更新。
- Wade 与 Hanlin 的任务合同和 backup 已形成。Wade 仍需提交 dataset-wide window/
  separability 数值；Hanlin 仍需提交 paper score、local score 和 delta 的闭环 row。

## 3. 本次会议确认的论文方向

1. 当前结果较上周已有改善，改善来自 batch execution、loss 组合和训练设置共同变化，
   不能只归因于 temperature 或某一个参数。
2. 目标不必表述为在每个单数据集上击败 SOTA。更重要的问题是：不同 respiratory
   datasets 是否存在显著 domain difference，以及联合模型能否在多个 native tasks
   上保持相对均衡的 performance。
3. 需要建立 single-source cross-domain comparison：若 ICBHI-only 或 SPRSound-only
   模型跨域明显下降，而 joint model 在两边保持稳定，这将直接支持论文故事。
4. 需要把不同数据集的 level、frequency distribution、duration、class imbalance
   和 feature-space separation 做成紧凑、信息密度高的 figure；PCA/clustering 仅是
   可视化，必须配至少一个 quantitative separability/statistical test。
5. 如果 ICBHI+SPRSound 主线稳定，可加入 HF_Lung 作为 positive-only supervised
   data line；小数据集用于 external evaluation。HF 的 unannotated gap 仍不得转成
   Normal/Negative。
6. 当前 loss 组合需要系统 ablation；normalization 需要包括老师提出的 MVN。
7. 论文写作立即开始。Introduction、Related Work、Dataset Preparation、table/figure
   placeholders 和真实 figure design 不等待全部实验完成。
8. 2026-09-02 前需要第一版 story；投稿目标为 2026-09-15/16，必须保留整合和修改缓冲。

## 4. 当前最合理的故事骨架

> Respiratory-sound datasets share partial clinical concepts but differ substantially in
> acquisition, level, duration, label support, prediction unit, and native benchmark. We study
> whether an eligibility-aware hierarchical BEATs model can retain useful native performance
> across datasets, and compare this behavior with single-source models that may not transfer.

该故事需要三条彼此独立的证据链：

1. **Dataset differentiation：** dataset-wide acoustic/label/duration differences，而不是
   仅凭 20 个样本做结论。
2. **Cross-domain failure：** ICBHI-only、SPRSound-only 和 joint model 在同一 shared
   binary surface 上的 2×2/3×2 transfer matrix。
3. **Joint retention：** clean validation-selected joint model 在 ICBHI flat4 和 SPRSound
   Task1-1 各自 native test 上的结果，以及 sampler/loss/MVN 的单因素 ablation。

## 5. Evidence boundary

- JH2 的 60.05/89.20 是 29 次 test access 的 diagnostic，不进入 clean headline table。
- 20-track panel 不能证明总体显著差异；dataset-wide group-aware analysis 尚未完成。
- single-source cross-domain matrix、MVN、source-proportional sampler、node/class weighting、
  confidence bridge 和 HF supervised line 均为 Proposed Method / Future Plan。
- 本记录不授权训练、official-test access、服务器任务、Notion 写入或 Git 提交。
