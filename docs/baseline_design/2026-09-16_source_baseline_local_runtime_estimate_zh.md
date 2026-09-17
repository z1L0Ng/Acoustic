# PC-MCL / DCASE Source Baselines本地时间预算｜2026-09-16

范围更新：DCASE使用ICBHI+SPRSound九输出native-union联合训练；本文已按新增SPR train frame cache、326 source updates/epoch与双source internal validation重算。未启动实验。

状态：**READ-ONLY ESTIMATE / NO NEW FORWARD OR PROFILE**。当前Mac型号未刷新；估计锚定本仓库同一Mac的MPS历史日志。区间不是运行承诺。

## 1. 历史实测依据

### 1.1 BEATs 5-s full-finetuning

`PAFA_JH2_main_multiseed`三seed历史日志：

| Seed | 完成epochs | updates | wall minutes | 普通train step中位数 | epoch wall中位数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 25 | 8150 | 312.55 | 1.864 s/update | 700.89 s |
| 1 | 21 | 6846 | 264.42 | 1.904 s/update | 741.06 s |
| 42 | 27 | 8802 | 273.57 | 1.588 s/update | 607.20 s |

普通step统计排除了epoch末validation summary行。它是本地MPS、5-s、batch32的full BEATs更新，最适合作为PC-MCL近似；PC-MCL的pair读取与SpecAugment仍会增加不确定性。

### 1.2 Frozen BEATs extraction

Generic AudioSet BEATs历史提取：

| Dataset | 实际5-s encoder inputs | seconds | inputs/s |
|---|---:|---:|---:|
| ICBHI | 6898 | 441.27 | 15.63 |
| SPRSound | 8085 | 491.65 | 16.45 |
| HF | 29295 | 1697.81 | 17.25 |
| KAUH | 1306 | 76.83 | 17.00 |

PAFA-trained encoder的对应时间为518.01、612.84、2058.53、102.61秒，提供较保守参照。历史是pooled feature extraction；DCASE新frame保存量更大，不能直接等同。

## 2. PC-MCL当前max50+early-stopping recipe

静态规模：4142 source cycles；dataset size约`1.8N`；batch32/drop-last约232 updates/epoch；50 epochs满跑上界约11600 updates/seed。

按历史1.588–1.904 s/update，训练step本身约6.1–7.4 min/epoch。历史epoch validation/I/O增加约1.5–2.0 min；再为pair waveform读取、2.5-s规范和SpecAugment留余量，采用：

- **每epoch：8–12 min**；
- **每seed满50 epochs：6.7–10 h**；
- **三seed满跑串行：20–30 h**。

当前配置使用strict source-Score improvement、patience10、min_delta0的early stopping。tie计入无提升；PC仍需满足既有Se eligibility。早停可能缩短实际时间，但没有证据保证具体停止epoch，因此预算按三个seed都跑满50轮。

历史来源与当前预算：

| Recipe | 每seed | 三seed串行 |
|---:|---:|---:|
| **当前max50上界** | **6.7–10 h** | **20–30 h** |
| 历史100轮情景 | 13–20 h | 40–60 h |
| 历史400轮原候选 | 53–80 h | 160–240 h |

Selected model的固定target输入数约为：ICBHI test 2756 + SPR inter 1429 + HF 1956×3 windows + KAUH 336 = **10389个5-s inputs**。按历史encoder吞吐约10–12分钟纯前向；计入重复音频读取、写逐样本结果和scoring，估**15–30 min/seed**，三seed约0.75–1.5 h。

## 3. DCASE-inspired CRNN

### 3.1 一次性frame extraction

缓存角色总计：ICBHI 6898 + SPR train/inter 8085 + HF 5868固定窗口 + KAUH 336 = **21187个5-s inputs**。

- 历史pooled extraction线性外推：约20–23 min；
- 新任务需保存约31个temporal frames而非一个pooled vector；预计float32主体约2.0 GB；
- 加上frame写盘、metadata和较保守encoder参照：**35–90 min一次性**。

31 frames和约2.0 GB来自已审计BEATs 25/10-ms frontend及16-frame patch geometry的静态计算，不是新forward；正式提取时记录真实shape。

### 3.2 CRNN source training（当前max50+early stopping）

每epoch326个homogeneous train updates（163+163），再按seed读取ICBHI internal validation和固定SPR internal validation。BEATs已缓存，训练只更新约1M参数的log-Mel CNN/fusion/BiGRU/attention/union head，但当前没有该路径的历史吞吐。

因此采用宽区间：

- **4–17 min/epoch**；
- **50 epochs满跑：3.3–14.2 h/seed**；
- **三seed满跑串行：10–43 h**。

DCASE使用patience10、min_delta0的strict双源internal-validation monitor：`0.5*ICBHI native4 macro multilabel-F1@0.5 + 0.5*SPRSound native7 macro multilabel-F1@0.5`，cosine `T_max=50`。预算不假定早停一定触发。

每个selected DCASE model仍需从缓存读取terminal BEATs frames并计算log-Mel CNN/CRNN；三seedexternal统一估**0.5–2 h**。该段无实测支撑，是静态宽区间。

DCASE合计：frame extraction 0.6–1.5 h + 三seed满跑training 10–43 h + external evaluation 0.5–2 h，即约**11–47 h**。

## 4. 两方法本地串行总预算

| Scenario | PC-MCL | DCASE（含cache） | Target评测/I/O余量 | 总计 |
|---|---:|---:|---:|---:|
| Best plausible | 20 h | 11 h | 1 h | **约32 h** |
| Base planning | 25 h | 25 h | 2 h | **约52 h** |
| Conservative | 30 h | 47 h | 2 h | **约79 h** |

再留约10%共享Mac负载、重启和文件I/O缓冲后，管理排程按**约35–87小时本地串行（1.5–3.6天）**理解。早停若触发只会缩短该上界，不能提前计入承诺。

## 5. 可信度边界

有历史实测：BEATs 5-s MPS full-finetune update、epoch wall、pooled extraction吞吐。
静态推算：PC 232 updates/epoch、max50满跑总量、固定target inputs。
宽区间：DCASE frame写盘放大、CRNN训练、两个方法terminal scoring。
未知：当前共享Mac即时负载、精确硬件型号、DCASE真实frame shape、PC pair loader瓶颈。

没有为估时启动forward、cache、训练、validation/test、profile或服务器任务。
