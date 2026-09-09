# Hanlin｜Paper baseline registry 任务

Hanlin，你好，我已经看过你发来的 baseline复现 PPT。现在论文主线已经收敛为
ICBHI + SPRSound 的 joint learning，因此你接下来的任务不是继续扩展模型数量，
也不需要重新跑 AST、BEATs、PANNs、OPERA-CT 或 HeAR。我们现在最需要的是把
已有结果整理成可以直接进入论文 Table I 的可信 comparison source。

请在 48 小时内完成下面四项交付：

1. baseline_registry.csv
2. table1_sources.md
3. comparison_notes.md
4. 一份可以直接复制到 LaTeX 的 Table I row draft

每个 baseline row 请包含：

- paper 和 official repository；
- model 与 pretraining checkpoint；
- encoder 是否 frozen；
- dataset、native task 和 prediction unit；
- train/test split 与 patient/recording grouping；
- input duration、preprocessing 和 classifier head；
- trainable scope；
- checkpoint-selection set 与 threshold-selection set；
- seed 数；
- reported metric、paper result、local result 和 delta；
- direct comparison 或 contextual reference；
- known mismatch；
- 项目 artifact 路径。

## Part 1 修订

- 回到原论文核对 4-layer Transformer paper 的 AST 和 BEATs 数字、均值/标准差
  以及具体 table row。PPT 中 BEATs sensitivity 与项目已有 primary-source audit
  不一致，请明确最终采用哪个原文数字和出处。
- PANNs/OPERA-CT 那篇如果原论文 task、unit 或 metric 与 ICBHI flat4 不一致，
  只标为 contextual reference；paper result 写 NR，不计算 paper/local delta。
- 颜色只能表示同一 task/protocol 下的 paper/local 差异；不兼容的 row 不使用
  好坏颜色判断。

## Part 2 修订

- 将 AST、BEATs、PANNs、OPERA-CT 和 HeAR 明确标为 16 kHz、2 s/1 s、
  frozen encoder、projector/head、batch 8、50 epochs、validation-selected 的
  single-dataset foundation-model baselines。
- 它们可以服务 motivation，但不是当前主方法的 matched ablation。主方法使用
  5 s、full fine-tuning、batch 32 和 PAFA joint training。
- 补齐每行 task、split、unit、seed、selection source 和 trainable scope。
  缺字段的 row 不进入主表。

## Strong ICBHI context

- 核对 PAFA、BEATs+CE、Patch-Mix 和 SG-SCL 的 paper row 与项目已有 local
  artifact。
- 主 direct context 优先 PAFA 和 BEATs+CE；Patch-Mix/SG-SCL 只在篇幅允许时
  加入，并写清 metadata 或 selection caveat。
- MVST 因 test support/split 不同，ADD-RSC 因 paper/repository protocol conflict，
  不进入正文 direct comparison。

## Existing zero-target transfer motivation

项目已有 PAFA、SG-SCL 和 Patch-Mix 的 ICBHI checkpoint 到 SPRSound official
inter fixed-checkpoint transfer artifacts。不要重跑，只整理：

- ICBHI source Score；
- SPRSound binary transfer Score；
- all-Normal floor；
- target adaptation 为 none；
- decision mapping；
- support；
- evidence boundary。

这些 row 说明 ICBHI-specialized checkpoint 不能直接保持第二个 benchmark，但不能
写成 joint training 的净收益。

最终请把 rows 分成三类：

1. protocol-compatible ICBHI literature context；
2. verified zero-target transfer motivation；
3. frozen foundation-model contextual baselines。

不要把不同 task、split、unit 或 metric 的数字排成一个 leaderboard，也不要构造
跨数据集 pooled score。完成后请用几句话说明哪些 row 可以进入 Table I，哪些只能放
Related Work，哪些应删除。先回复预计交付时间；有原文或 artifact 找不到时直接列出
缺口，不需要补跑实验。

本任务不包含 independent-head、single-source、clean companion、HF/KAUH 或其他
训练/测试。

