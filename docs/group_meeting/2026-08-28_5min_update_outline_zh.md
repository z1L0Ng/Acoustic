# 2026-08-28 大组组会五分钟更新提纲

## Slide 1｜问题与当前协议（30 秒）

- 目标：一个统一的 respiratory acoustic system 同时服务 ICBHI 与 SPRSound。
- 当前：BEATs full fine-tuning，shared Level1/Crackle/Wheeze hierarchy，16 kHz，
  4 s / 2 s，seed 42。
- 核心问题：SPRSound 已稳定超过 90，但 ICBHI 仍低于直接 BEATs 文献参考。

## Slide 2｜本周实际完成的矩阵（50 秒）

| ID | Normalization | Augmentation | Loss |
|---|---|---|---|
| R0 | None | None | CE + BCE |
| N1 | Peak | None | CE + BCE |
| A1 | Peak | Gain + noise | CE + BCE |
| L1 | Peak | Gain + noise | Focal |
| L2 | Peak | Gain + noise | Class-balanced |

尚未完成：RMS normalization、waveform z-score、spectrogram mean-variance
normalization，以及 matched 1/2/3/4 s end-to-end matrix。

## Slide 3｜完整结果（60 秒）

| ID | ICBHI Sp | ICBHI Se | ICBHI Score | SPRSound Score |
|---|---:|---:|---:|---:|
| R0 | 67.89 | 42.91 | **55.40** | 92.12 |
| N1 | 63.52 | 44.44 | 53.98 | **93.71** |
| A1 | 65.17 | 42.14 | 53.65 | 92.31 |
| L1 | 49.97 | **58.45** | 54.21 | 90.68 |
| L2 | 58.83 | 42.31 | 50.57 | 93.05 |

口头说明：ICBHI 是 hierarchical saved-prediction posthoc diagnostic；
SPRSound 是 local single-seed terminal Task1-1 result。

## Slide 4｜我们学到了什么（50 秒）

- Peak normalization 没有形成跨数据集、跨 metric 的一致胜出。
- Focal loss 提高 ICBHI abnormal sensitivity，但牺牲 specificity 和 SPRSound。
- 旧 bits-only ICBHI readout 忽略 Level1，放大 Normal false positive；hierarchical
  readout 修复部分 specificity，但仍低于直接 BEATs + CE Paper Claim 63.49。
- 当前现象更像 dataset-specific operating-point/conflict，而不是单一 preprocessing
  已经解决问题。

## Slide 5｜Window 与下一步（70 秒）

- 当前 matched run 只有 4 s / 2 s；短 ICBHI cycles 可能被大量 padding。
- Wade：用 padding/truncation、RMS/PSD/SNR proxy、PCA/separability 为 1/2/3/4 s
  提供量化依据。
- Lead：不等待学生，预注册 RMS waveform normalization 与 spectrogram
  mean-variance normalization；并行开始 paper skeleton。
- Hanlin：先完成 official checkpoint 的原始 benchmark reproduction，再做
  cross-dataset adaptation；每条报告 paper/local/delta。

## 希望组内反馈的问题（40 秒）

1. Hierarchical Level1-gated ICBHI readout 是否符合统一系统的预期？
2. 对 ICBHI/SPR 冲突，下一步更应优先 RMS waveform normalization、
   spectrogram normalization，还是 loss/sampling conflict analysis？
3. 在两周 deadline 下，是否应把论文主张收窄为 unified cross-dataset modeling
   与 trade-off analysis，而不是 native-task SOTA？
