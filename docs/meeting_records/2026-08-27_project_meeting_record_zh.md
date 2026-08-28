# 2026-08-27 Acoustic 项目会议记录

## 1. 会议目标

复盘 Wade 的 window/acoustic-feature 分析、Hanlin 的 baseline reproduction，
以及 BEATs Core-2 normalization 与联合训练结果；在 ICASSP 截止日前重新定义
任务、汇报标准和项目节奏。

## 2. 已确认的会议决策

1. Wade 的任务不再是“提取 RMS/PSD 等特征”，而是使用声学统计与 feature
   separability 为 window/preprocessing 选择提供量化依据。
2. Window 必须同时由 feature-level 分析和 lead-owned end-to-end performance
   判断，尤其检查 4 s 输入对短 ICBHI cycle 的 padding 风险。
3. Hanlin 必须先使用官方 checkpoint/evaluation 在原始 benchmark 上复现论文
   结果，再进入新数据集 adaptation；二者不得混称 baseline reproduction。
4. Baseline 更新必须同时给出 paper result、local reproduction result 和 delta；
   “finished”或“pipeline completed”不构成结果。
5. 目前重点不是继续无解释地增加模型，而是解释 normalization 为什么对
   ICBHI 与 SPRSound、Sp 与 Se 产生不同影响，并系统考虑 waveform 与
   spectrogram/frequency-domain normalization。
6. 项目由每周同步改为每两天一个 checkpoint；小结果完成后立即同步。
7. Wade 和 Hanlin 都不是论文关键路径。若两天内没有形成可审阅的数值产物，
   Zilong 应接管关键分析或缩小论文范围。
8. 2026-08-28 大组组会准备约五分钟的结果与问题更新；若无法参会，提前发送
   材料，不因行程推迟反馈。

## 3. 会议陈述与本地证据核对

| 会议陈述 | 本地证据 | 结论 |
|---|---|---|
| ICBHI 当前最好约 55，仍低于约 63–64 的强参考 | R0 hierarchical posthoc Score 55.40；直接 BEATs + CE Paper Claim 63.49 | 数值方向一致；本地 55.40 是 posthoc/test-informed，不能作为 primary result |
| SPRSound 已达到 90 以上 | 五组 Task1-1 terminal Score 90.68–93.71 | Verified Local Result |
| normalization 改善部分指标，但跨数据集不一致 | R0/N1 与全部五组矩阵已完成，N1 不在 hierarchical ICBHI 上取胜，但 SPR 最好 | 一致；不能总结为 normalization 全面提升 |
| augmentation 还没有做太多 | A1 gain+noise、L1/L2 已完成 50 epochs 和 terminal evaluation | 口头状态已过时，应在明天组会纠正 |
| 已测试 1/2/3/4 s | 本地 matched full-fine-tuning 仅闭合 4 s / 2 s | 只能标为讨论/零散尝试，不能称完整 end-to-end comparison |
| Hanlin 的 AST 已完成、BEATs 正在运行 | 会议未展示 metric、prediction 或 paper delta | Student Report / Unverified，不能进入结果表 |
| 当前 checkpoint 由不同 dataset loss 决定 | 实现使用 ICBHI 与 SPRSound validation eligible-node loss 的等均值 | 口头描述不精确；正式材料使用实现合同 |

## 4. Wade 更新任务

- 候选 window：1/2/3/4 s，16 kHz；先覆盖 ICBHI 与 SPRSound paper core。
- 每个 window 统计 native duration、被截断比例、padding 比例、有效信号覆盖、
  silence/低能量比例。
- 特征至少包括 RMS、average power、SNR proxy、PSD/band energy、MFCC、
  spectral centroid 和 bandwidth。
- 按 Normal/Crackle/Wheeze/可用 Both 标签展示标准化 PCA，并给出至少一个
  quantitative separability 指标；UMAP/t-SNE 仅作辅助。
- 2026-08-30 前交第一份数值表和结论；每个小结果完成后立即更新，不等周会。

## 5. Hanlin 更新任务

- AST 为第一条闭环：Paper/Model、Original Dataset、Paper Score、Official
  Checkpoint、Local Reproduced Score、Delta、protocol/split 说明。
- 只有 AST 原始 benchmark reproduction 经核对后，才记录 cross-dataset
  adaptation；BEATs 按同一格式继续。
- 有官方 checkpoint 时优先直接官方 evaluation，不从零重训来替代 reproduction。
- 每两天提交一个可核查 row；没有具体 metric 的“完成”不计入 progress。

## 6. Lead-owned 下一步

- 2026-08-28：完成五分钟大组 update，明确已做/未做、完整结果、dataset
  conflict、window 风险和下一组 normalization 候选。
- 2026-08-29 至 2026-08-30：基于现有 R0/N1/A1/L1/L2 完成按数据集、Sp/Se
  的解释矩阵；预注册下一步最小 waveform/spectrogram normalization 对照。
- 不等待 Wade 决定 lead-owned end-to-end window 实验，不等待 Hanlin 开始论文
  baseline table 与 paper skeleton。

## 7. Evidence boundary

本记录保存会议决策和对本地证据的核对，不授权任何新训练、terminal evaluation、
服务器任务或外部消息发送。Verified Local Result、posthoc diagnostic、Student
Report、Paper Claim 和 Future Plan 必须继续分开标注。
