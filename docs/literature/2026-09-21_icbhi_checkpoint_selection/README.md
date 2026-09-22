# ICBHI选模：原文与位置速查

核对日期：2026-09-21。PDF页码从1开始；标有“印刷页”的另列会议页码。

[完整中文审计报告](ICBHI_CHECKPOINT_SELECTION_AUDIT_2026-09-21_zh.md)另列逐篇判定、原文解释及官方代码的完整调用证据。

**结论：Patch-Mix、SG-SCL、PAFA、PC-MCL的原文均未明确声明用official test Score选择checkpoint。其test-based选模证据来自官方代码，不能表述为论文原句。** 论文写了official train/test split，并不等于交代了模型选择依据。

## 1. Patch-Mix CL（Interspeech 2023正式版）

原题：Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification。

[本地PDF](pdfs/bae23b_patch_mix_interspeech2023.pdf) · [会议原文](https://www.isca-archive.org/interspeech_2023/bae23b_interspeech.pdf)

- 位置：PDF第2页，印刷页5437，§3.1 Dataset Description及§3.3 Training Details；比较表为PDF第4页Table 3。
- 关键原句：“officially split into a train set (60%) and a test set (40%).”
- 中文：官方划分为60%训练集和40%测试集。
- 能证明：使用official split；相关训练段落说明训练设置。
- 不能证明：论文没有明确说明按test Score选择best epoch，也没有说明另拆validation的构建规则。

## 2. SG-SCL（对应ICASSP 2024的作者arXiv版）

原题：Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain Adaptation on Respiratory Sound Classification。

[本地PDF](pdfs/kim_sgscl_arxiv_2312.09603.pdf) · [作者原文](https://arxiv.org/abs/2312.09603)

- 位置：PDF第2页，§2.3 Training Details；PDF第4页§4.2及Table 4。
- 关键原句：“We used the official split of the ICBHI dataset (train-test split as 60-40%)”
- 中文：采用ICBHI的官方60/40训练/测试划分。
- 不能据此认定：论文明确以test选checkpoint。原文没有说明checkpoint选择集及独立validation构建方式。

## 3. PAFA（Interspeech 2025正式版）

原题：Patient-Aware Feature Alignment for Robust Lung Sound Classification: Cohesion-Separation and Global Alignment Losses。

[本地PDF](pdfs/jeong25_pafa_interspeech2025.pdf) · [会议原文](https://www.isca-archive.org/interspeech_2025/jeong25_interspeech.pdf)

- 位置：PDF第3页，印刷页1020，§§3.1、3.3和4.1；损失权重grid search在PDF第2页、印刷页1019、§2.2 Loss Functions。
- 关键原句：“official 60/40 (train/test) split.”
- 中文：采用官方60/40训练/测试划分。
- 未交代：checkpoint选择规则，以及损失权重grid search使用哪个数据子集。不能用grid search或best performance等词反推它使用独立validation。

## 4. PC-MCL（对应ICASSP 2026的作者arXiv版）

原题：PC-MCL: Patient-Consistent Multi-Cycle Learning with Multi-Label Bias Correction for Respiratory Sound Classification。

[本地PDF](pdfs/jeong_kim_pcmcl_arxiv_2601.17080.pdf) · [作者原文](https://arxiv.org/abs/2601.17080)

- 位置：PDF第2页，§2.2 Regularization via a Patient-Matching Auxiliary Task中的Combined Loss条目；official split见PDF第3页§3.1 Experimental Setup。
- 关键原句：“we set α = 0.1 based on a grid search on the validation set.”
- 中文：通过validation set上的grid search，把α设为0.1。
- 这句话说的是损失权重α的选择，不是checkpoint选择。论文没有定义这个validation set如何划分，因此不能认定它独立于official test；也不能仅凭代码中的val_loader身份就反推论文alpha搜索一定用了同一集合。

## 5. 独立validation的反例：Yang et al.（Interspeech 2020正式版）

原题：Adventitious Respiratory Classification using Attentive Residual Neural Networks。

[本地PDF](pdfs/yang20e_attentive_residual_interspeech2020.pdf) · [会议原文](https://www.isca-archive.org/interspeech_2020/yang20e_interspeech.pdf)

- 位置：PDF第3页，印刷页2914，§4.2 Preprocessing；第4页Table 2分列validation/test结果。
- 关键原句：“original test set was only used for evaluation.”
- 中文：原测试集只用于评测。
- 同段明确从original training set中按患者独立再划70%训练、30%验证。因此，不能把“所有ICBHI工作都不拆validation”作为结论。
- 原文仍未给出具体按哪种指标/哪一epoch选择checkpoint的规则；独立划分与具体选模规则是两件事。

## 官方代码证据是另一层依据

四份released implementation都能找到：默认official split；`train_flag=False`读取官方test；这个对象被命名为`val_dataset/val_loader`；每epoch调用评测，并在Score提高且满足额外Se条件时更新best。

| 方法 | best更新位置 | official test来源 |
|---|---|---|
| Patch-Mix | [main.py，Score改善且Se>5](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/main/main.py#L459) | [dataset loader](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/main/util/icbhi_dataset.py#L115) |
| SG-SCL | [main.py，Score改善且Se>5](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/main/main.py#L517) | [dataset loader](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/main/util/icbhi_dataset.py#L100) |
| PAFA | [main.py，Score改善且Se>0.1](https://github.com/wa976/PAFA/blob/main/main.py#L522) | [dataset loader](https://github.com/wa976/PAFA/blob/main/util/icbhi_dataset.py#L118) |
| PC-MCL | [main.py，Score改善且Se>0.1](https://github.com/wa976/PC-MCL/blob/main/main.py#L960) | [dataset loader](https://github.com/wa976/PC-MCL/blob/main/util/dataset_pcmcl.py#L80) |

上述阈值保留代码中的数值写法。main分支行号可能随作者更新变化；这是所查实现的行为，不能据此证明所有published table数字都来自同一份代码。

对外准确表述：我们参照相关released implementations中的test-based checkpoint selection方式。不要写“这些论文明确说用test选模”，也不要写“整个ICBHI领域都采用这个协议”。
