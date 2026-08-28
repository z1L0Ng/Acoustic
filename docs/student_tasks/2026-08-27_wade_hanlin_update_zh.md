# Wade / Hanlin 任务更新｜2026-08-27

## Wade｜Window-level acoustic analysis

### 目标

用 quantitative class separability 判断 window/preprocessing 是否有助于
Normal、Crackle、Wheeze 等类别区分，而不是只输出 feature value。

### 第一份交付（2026-08-30）

1. ICBHI + SPRSound，16 kHz，1/2/3/4 s 的 native duration、截断、padding、
   silence/低能量覆盖表。
2. RMS、power、SNR proxy、PSD/band energy、MFCC、spectral centroid、bandwidth
   的 class-wise distribution。
3. 标准化 PCA 作为主图，并给出至少一个 quantitative separability 指标。
4. 用一段结论回答：当前哪个 window 更有依据、为什么、还缺什么证据。

每完成一个小结果立即同步；不要只列 methodology。若 8/30 没有数值产物，
关键部分由 Zilong 接管。

## Hanlin｜Official baseline reproduction

### 目标

先证明能够在原论文 benchmark 上复现原论文，再做 cross-dataset adaptation。

### 第一份交付（AST，2026-08-30）

| Paper / model | Original dataset/task | Paper score | Official checkpoint | Local score | Delta | Protocol note |
|---|---|---:|---|---:|---:|---|
| AST | 待填写 | 待填写 | Yes/No + source | 待填写 | 待填写 | split/metric/evaluation command |

随后以同一格式提交 BEATs。存在官方 checkpoint 时先直接运行官方 evaluation；
修改 head 并在新 dataset 上训练属于 adaptation，必须放在 reproduction 之后单列。
“AST finished”但没有 metric 和 delta，不计为完成。

## 共同汇报规则

- 每两天一个数值型 checkpoint；小结果准备好后可以提前同步。
- 必须说明做了什么、得到了什么、与参考差多少、下一步如何决定。
- 任务延迟应提前报告；本科生结果不作为论文关键路径依赖。
