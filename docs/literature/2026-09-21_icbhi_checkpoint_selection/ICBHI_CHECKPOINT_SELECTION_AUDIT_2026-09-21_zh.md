# ICBHI checkpoint selection primary-source audit

**日期：** 2026-09-21  
**范围：** 仅核查 ICBHI official 60/40 test subset 是否被用于选择 checkpoint。未运行作者代码或实验。论文陈述与 released implementation 行为严格分开。

## 1. 判定类别

1. **① 论文明确 test-selection：** 论文正文明确说 test performance 用于选择 checkpoint/model。
2. **② 论文只报告 official split/test：** 论文说明 official train/test evaluation 或 best results，但未交代 checkpoint selection 使用哪个数据子集。
3. **③ 论文明确分离 validation：** 论文从训练数据中另建 validation，并将原始 test 保留为最终评测。
4. **④ 仅 released code 证明 test-selection：** 官方实现能闭合 `official test loader → 每轮评估 → 更新 best checkpoint`，但论文没有这样写。
5. **⑤ 无法确认：** 来源使用 “validation set” 或 “best performance”等措辞，却没有给出足够 provenance 来确认数据子集或选择规则。

## 2. 核心结论

| Work | 论文证据 | Released-code 证据 | 判定 | 可安全使用的结论 |
|---|---|---|---|---|
| Patch-Mix CL, Interspeech 2023 | official 60/40、五次随机运行；没有 validation 构造或 checkpoint rule | official test 每个 epoch 评估；满足 `Se > 5` 时按最高 Score 更新 `best.pth` | **② + ④** | released implementation 是 test-selected；论文未明确写出 |
| SG-SCL, ICASSP 2024 | official 60/40、50 epochs、五个固定 seeds；没有 validation 构造或 checkpoint rule | 与 Patch-Mix 相同的 test-loader/best-Score 逻辑；额外要求 `Se > 5` | **② + ④** | released implementation 是 test-selected；论文未明确写出 |
| PAFA, Interspeech 2025 | official 60/40、五 seeds 平均；提到 grid search，但未说明使用哪个子集 | official test 每个 epoch 评估；满足 `Se > 0.1` 时按最高 Score 保存 | **② + ④** | released implementation 是 test-selected；论文未明确写出 |
| PC-MCL, ICASSP 2026 author arXiv | 说 `alpha` 由 “validation set” grid search 决定，但没有定义该集合或 checkpoint rule | 默认 official split；`train_flag=False` 即 official test；每轮按 Score 保存 best | **⑤ + ④** | 代码证明 released implementation test-select；不能认定论文中的 validation 独立于 test |
| Yang et al., Interspeech 2020 | 从原训练集按 70/30 划 train/validation，并声明原 test 仅评测 | 无需依赖代码即可确定 paper-level 边界 | **③** | 明确的独立 validation 反例；但论文没有说明 exact checkpoint rule |

**这五篇中，没有一篇属于类别①。** Patch-Mix、SG-SCL、PAFA 和 PC-MCL 的 test-selection 证据来自 released code，而不是论文的明确自述。

## 3. 逐篇论文证据

### 3.1 Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification

- **版本/来源：** Interspeech 2023 正式版，[ISCA PDF](https://www.isca-archive.org/interspeech_2023/bae23b_interspeech.pdf)。
- **本地 PDF：** [bae23b_patch_mix_interspeech2023.pdf](./pdfs/bae23b_patch_mix_interspeech2023.pdf)
- **位置：** PDF 第2页（1-indexed），印刷页5437，§3.1 Dataset Description 与 §3.3 Training Details；PDF 第4页，印刷页5439，Table 3。
- **短引文：** “officially split into a train set (60%) and a test set (40%).”
- **已确认：** official split、AST fine-tune 50 epochs、五次随机运行；Table 3 只比较 official-split 工作。
- **未确认：** 没有 validation-set 构造、early stopping/checkpoint selection 数据子集，也没有说报告 best test epoch。
- **论文判定：** **②**。

### 3.2 Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain Adaptation on Respiratory Sound Classification

- **版本/来源：** author arXiv v1（2023-12-15；对应 ICASSP 2024），[arXiv PDF](https://arxiv.org/pdf/2312.09603)。
- **本地 PDF：** [kim_sgscl_arxiv_2312.09603.pdf](./pdfs/kim_sgscl_arxiv_2312.09603.pdf)
- **位置：** PDF 第2页，§2.3 Training Details；PDF 第4页，§4.2 与 Table 4。
- **短引文：** “We used the official split of the ICBHI dataset (train-test split as 60-40%).”
- **已确认：** official 60/40、50 epochs、固定五个 seeds，以及 test-set analysis。
- **未确认：** 没有 validation-set 构造或 checkpoint-selection rule。
- **论文判定：** **②**。

### 3.3 Patient-Aware Feature Alignment for Robust Lung Sound Classification: Cohesion-Separation and Global Alignment Losses

- **版本/来源：** Interspeech 2025 正式版，[ISCA PDF](https://www.isca-archive.org/interspeech_2025/jeong25_interspeech.pdf)。
- **本地 PDF：** [jeong25_pafa_interspeech2025.pdf](./pdfs/jeong25_pafa_interspeech2025.pdf)
- **位置：** PDF 第2页、印刷页1019，§2.2 与 Table 1；PDF 第3页、印刷页1020，§§3.1、3.3、4.1。
- **短引文：** “official 60/40 (train/test) split.”
- **已确认：** 4,142 个训练 cycles、2,756 个 test cycles、100 epochs、五个随机 seeds 平均；§2.2 说 grid search 得到最佳 loss weights。
- **未确认：** 未说明 grid search 使用哪个数据子集，也没有 checkpoint-selection criterion。
- **论文判定：** official split/result 为 **②**；grid-search 子集另记 **⑤**。

### 3.4 PC-MCL: Patient-Consistent Multi-Cycle Learning with Multi-Label Bias Correction for Respiratory Sound Classification

- **版本/来源：** author arXiv v1（2026-01-23；对应 ICASSP 2026），[arXiv PDF](https://arxiv.org/pdf/2601.17080)。
- **本地 PDF：** [jeong_kim_pcmcl_arxiv_2601.17080.pdf](./pdfs/jeong_kim_pcmcl_arxiv_2601.17080.pdf)
- **位置：** PDF 第2页，§2.2 “Regularization via a Patient-Matching Auxiliary Task”中的 **Combined Loss 条目**；PDF 第3页，§3.1 Experimental Setup 与 Table 1。
- **短引文：** “we set α = 0.1 based on a grid search on the validation set.”
- **已确认：** official 60/40、五 seeds 报告，并声称使用 validation-set grid search 选择 `alpha`。
- **未确认：** 论文没有定义这个 “validation set” 如何构建，也没有说它独立于 official test；论文同样没有 checkpoint-selection rule。
- **论文判定：** validation 的身份/独立性为 **⑤**，不能升级为 **③**。

### 3.5 Adventitious Respiratory Classification Using Attentive Residual Neural Networks

- **版本/来源：** Interspeech 2020 正式版，[ISCA PDF](https://www.isca-archive.org/interspeech_2020/yang20e_interspeech.pdf)。
- **本地 PDF：** [yang20e_attentive_residual_interspeech2020.pdf](./pdfs/yang20e_attentive_residual_interspeech2020.pdf)
- **位置：** PDF 第3页、印刷页2914，§4.2 Preprocessing；PDF 第4页、印刷页2915，Table 2 与 §5 Results。
- **短引文1（原句节选）：** “70 % of the samples for the training set and 30 % for the validation set”
- **短引文2（同一句后部）：** “original test set was only used for evaluation.”
- **已确认：** validation 从原训练部分按 subject-independent 方式划出，original official test 只用于评价；Table 2 分列 validation/test。
- **未确认：** 没有描述具体 checkpoint rule。
- **论文判定：** **③**。

## 4. Released-code 证据链

以下结论只描述所列 commit 的实现行为；不能据此证明论文中每个表格数字都来自该 commit。

### 4.1 Patch-Mix CL

- **官方仓库/commit：** [raymin0223/patch-mix_contrastive_learning](https://github.com/raymin0223/patch-mix_contrastive_learning)，`836b09fea1b70eb29fe0b25afa481286b56f5104`。
- 默认 official split：[`main.py` L77–78](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/main.py#L77-L78)。
- `train_flag=False` 构成 `val_dataset`：[`main.py` L214–215](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/main.py#L214-L215)。
- official 路径中它选择标记为 `test` 的 rows：[`util/icbhi_dataset.py` L103–117](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/util/icbhi_dataset.py#L103-L117)。
- 每个 epoch 评估该 loader 并更新 best：[`main.py` L510–527](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/main.py#L510-L527)。
- 条件为 `sc > best_acc[-1] and se > 5`：[`main.py` L459–462](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/main.py#L459-L462)，最终写入 `best.pth`：[`main.py` L533–537](https://github.com/raymin0223/patch-mix_contrastive_learning/blob/836b09fea1b70eb29fe0b25afa481286b56f5104/main.py#L533-L537)。

### 4.2 SG-SCL

- **官方仓库/commit：** [kaen2891/stethoscope-guided_supervised_contrastive_learning](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning)，`2ed1bacc7121653805702f3a1afb5557a56f5a05`。
- 默认 official split：[`main.py` L74–75](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/2ed1bacc7121653805702f3a1afb5557a56f5a05/main.py#L74-L75)。
- `train_flag=False` 构成 `val_dataset`：[`main.py` L212–224](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/2ed1bacc7121653805702f3a1afb5557a56f5a05/main.py#L212-L224)，并映射到 official `test` rows：[`util/icbhi_dataset.py` L76–101](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/2ed1bacc7121653805702f3a1afb5557a56f5a05/util/icbhi_dataset.py#L76-L101)。
- 每轮评估/保存：[`main.py` L570–587](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/2ed1bacc7121653805702f3a1afb5557a56f5a05/main.py#L570-L587)；条件 `sc > best_acc[-1] and se > 5`：[`main.py` L517–520](https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning/blob/2ed1bacc7121653805702f3a1afb5557a56f5a05/main.py#L517-L520)。

### 4.3 PAFA

- **官方仓库/commit：** [wa976/PAFA](https://github.com/wa976/PAFA)，`e49e294d0db0d6af10ac46290512b9c85d3f71e1`。
- 默认 official split：[`main.py` L71–72](https://github.com/wa976/PAFA/blob/e49e294d0db0d6af10ac46290512b9c85d3f71e1/main.py#L71-L72)。
- `train_flag=False` 构成 `val_dataset`：[`main.py` L205–218](https://github.com/wa976/PAFA/blob/e49e294d0db0d6af10ac46290512b9c85d3f71e1/main.py#L205-L218)，并映射到 official `test` rows：[`util/icbhi_dataset.py` L106–120](https://github.com/wa976/PAFA/blob/e49e294d0db0d6af10ac46290512b9c85d3f71e1/util/icbhi_dataset.py#L106-L120)。
- 每轮评估/保存：[`main.py` L647–667](https://github.com/wa976/PAFA/blob/e49e294d0db0d6af10ac46290512b9c85d3f71e1/main.py#L647-L667)；条件 `sc > best_acc[-2] and se > 0.1`：[`main.py` L522–525](https://github.com/wa976/PAFA/blob/e49e294d0db0d6af10ac46290512b9c85d3f71e1/main.py#L522-L525)。

### 4.4 PC-MCL

- **官方仓库/commit：** [wa976/PC-MCL](https://github.com/wa976/PC-MCL)，`85a4e22fc4ea945cf1145c6da61991eb2a2fdd79`。
- 默认 official split：[`main.py` L93–94](https://github.com/wa976/PC-MCL/blob/85a4e22fc4ea945cf1145c6da61991eb2a2fdd79/main.py#L93-L94)。
- 默认 PC-MCL 路径令 `train_flag=False` 产生 `val_dataset`，且 `test_loader=None`：[`main.py` L288–307](https://github.com/wa976/PC-MCL/blob/85a4e22fc4ea945cf1145c6da61991eb2a2fdd79/main.py#L288-L307)。
- 该 dataset class 将其映射到 official `test` rows：[`util/dataset_pcmcl.py` L70–81](https://github.com/wa976/PC-MCL/blob/85a4e22fc4ea945cf1145c6da61991eb2a2fdd79/util/dataset_pcmcl.py#L70-L81)。
- 每轮评估/保存：[`main.py` L1362–1383](https://github.com/wa976/PC-MCL/blob/85a4e22fc4ea945cf1145c6da61991eb2a2fdd79/main.py#L1362-L1383)；条件 `icbhi_score > best_acc[2] and se > 0.1`：[`main.py` L949–963](https://github.com/wa976/PC-MCL/blob/85a4e22fc4ea945cf1145c6da61991eb2a2fdd79/main.py#L949-L963)。

## 5. 合作者安全表述

推荐写法：

> Following the released implementations, Patch-Mix CL, SG-SCL, PAFA, and PC-MCL evaluate the official ICBHI test subset after every epoch and retain the checkpoint with the best test ICBHI Score, subject to an additional Sensitivity condition.

必须紧跟限定：

> Their papers specify official-split test evaluation but do not explicitly document test-based checkpoint selection. Therefore, “test-selected” is an implementation-level finding, not a paper-level claim.

不要写：

- “Patch-Mix/SG-SCL/PAFA/PC-MCL 论文明确说在 test 上选模。”
- “所有 paper table 数字都已证明来自本报告审计的 commit。”
- “PC-MCL 使用独立 validation set。”论文只说 `alpha` 在 “validation set” 上 grid search，但没有定义这个集合；released implementation 的 validation loader 是 official test。

## 6. 证据边界

- 本报告是静态 primary-source audit；未运行作者代码、模型或数据。
- Code behavior 按所列 pinned public commits 核查。
- 审计证明实现逻辑，不证明论文每个表格数字的 exact provenance。
- Yang et al. 证明 paper-level 的 validation/test 分离，但没有给出 per-epoch checkpoint rule。
