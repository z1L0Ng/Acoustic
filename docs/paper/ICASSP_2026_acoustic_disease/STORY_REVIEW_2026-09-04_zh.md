# Paper 故事线复盘与组织方案｜2026-09-04

状态：`ANALYSIS_ONLY / 不授权实验、Notion 写入或 Git 提交`
输入：2026-09-03 会议记录、2026-09-04~06 work plan、当前 LaTeX 骨架、
`result/` 下已完成的实验产物。

---

## 0. 一句话结论

当前故事线的**问题陈述缺一块最强的证据，而这块证据已经躺在 repo 里**：
ICBHI 上的 task-specific SOTA checkpoint 迁移到 SPRSound Task1-1 时接近
trivial floor（PAFA 55.82、SG-SCL 59.98、Patch-Mix 59.38，all-normal floor 50.0），
而同一个 joint model 拿到 90.70。这不是 ablation，这是 **motivation**。

把它从 supporting evidence 提到 Section 1--2，整篇论文的逻辑立刻从
"我们做了个联合模型，ICBHI 差 3 个点" 变成
"专家模型在第二个 benchmark 上失效，我们问一个共享模型能否同时守住两个 native task、
代价是多少"。

---

## 1. Gap：我们到底发现了什么问题

论文的 gap 不应该写成抽象的 "datasets are heterogeneous"。repo 里支持的是三个
可量化的发现，按证据强度排序：

### G-A. 单数据集最优方法不具备跨数据集保持能力（最强，已有证据）

来源：`result/icbhi_strong_method_reproduction/metrics.json`、
`result/pafa_sprsound_transfer_*`、`result/sg_scl_sprsound_transfer_*`、
`result/sprsound_patchmix_frozen_transfer/`

| Method | ICBHI Score（local author ckpt） | → SPRSound Task1-1 binary | 相对 all-normal floor |
|---|---:|---:|---:|
| PAFA | 64.14 | 55.82 | +5.82 |
| SG-SCL | 60.98 | 59.98 | +9.98 |
| Patch-Mix CL | 62.17 | 59.38 | +9.38 |
| all-normal floor | -- | 50.00 | 0 |
| **本文 joint model（3 seeds）** | **61.17 ± 0.31** | **90.70 ± 0.34** | **+40.70** |

边界（必须在正文写明）：迁移是 zero-target-tuning 的 frozen checkpoint 推理，
不是 fine-tune。它证明的是"专家模型自身不具备跨 benchmark 能力"，
**不是**"联合训练的净收益"。后者需要 §4 的 G1 对照。

### G-B. 通用 audio foundation model 的 respiratory 适配不是自动的（已有证据）

- `result/four_dataset_representation_attribution/comparison/encoder_selection.json`：
  ICBHI 上 fine-tune 过的 PAFA task encoder，在 6 个下游 frozen-feature 任务中
  只有 2 个 material win、3 个 material loss，**AudioSet-only BEATs 反而被选为
  neutral default**。→ 在单一数据集上做 respiratory 适配，并不会产出更好的通用
  respiratory representation。
- 外部佐证：frozen BEATs_iter3 + task Transformer 在 ICBHI 只有 57.31 ± 1.40
  （EMBC 2025）。
- Hanlin 的 four-dataset single-dataset foundation-model baselines 是同一结论的
  第四个数据点（待他交表）。

### G-C. Naive pooling 在这四个数据集上是"不可执行"而非"效果差"（结构性论点）

- 标签空间不共享：HF Lung 只有 observed-positive 的 Crackle/Wheeze 时间标注，
  **没有 Normal 标注**；gap/empty ≠ negative。KAUH 是 raw-9 的 disease/sound 混合。
  直接 pool 只能二选一：制造不存在的 negative supervision，或塌缩成
  lowest-common-denominator binary、抹掉 native task 语义。
- Prediction unit 不共享：cycle / event / interval / recording / patient。
- Domain 信息在特征层面 trivially accessible：dataset-ID probe balanced accuracy
  0.988（PAFA encoder）/ 0.995（AudioSet-only），
  `result/four_dataset_shortcut_diagnostic/`。
  **注意**：该 diagnostic 的正式结论是 `not_supported_or_inconclusive`，
  所以只能写成"domain identity 可被线性读出，因此 pooled 训练存在 domain
  shortcut 风险"，**不能**写成"存在 shortcut"。
- 采集条件差异（Figure 1 / 20-track panel）：median RMS dBFS
  ICBHI −11.06、KAUH −20.63、HF −38.08、SPRSound −41.02，跨度约 30 dB。

---

## 2. Method：我们怎么解决的（一句话 + 四个机制）

> 不构造共享的扁平标签空间，而是构造一个 **partial-order attribute hierarchy**，
> 并用 **eligibility mask** 让"标签不可用"成为模型的一等公民，
> 从而在不制造 negative supervision、不抹掉 native prediction unit 的前提下联合训练。

四个必须在 Method 里显式出现的机制：

1. **Hierarchy**：Level-1 Normal/Abnormal + Level-2 Crackle / Wheeze 两个属性；
   Both = 两个属性同时为正。ICBHI flat-4 由此 decode 而来，而不是直接监督。
2. **Eligibility-aware loss**：每个 node 的 loss 只在 *该数据集实际支持该 node* 的
   样本上计算。这是全篇的技术核心——它把 label-support heterogeneity 变成一个
   显式约束，而不是预处理里的一个 hack。
3. **Unit-preserving batching**：native-unit homogeneous batch 32，各数据集
   batch 数相等。保留原生预测单元，避免大数据集主导。
4. **Dataset-native readout，不构造 pooled score**：
   hierarchy → ICBHI flat-4；Level-1 → SPRSound Task1-1；
   attribute prob → HF interval；recording-mean → KAUH patient。

外加一个 **extension**（JH4）：HF positive-only auxiliary loss（λ=0.25），
用来证明同一个 eligibility 机制可以吸收"只有正样本标注"的数据源，
而不需要把 gap 当 negative。

**Novelty 的安全表述**（依据 `docs/surveys/..._prior_art_2026-07-29.md`）：
masked loss / universal taxonomy / label marginalization 都有方法先例
（Schutera 2022、Bevandić 2022、Shi 2021）；ICBHI+SPRSound fusion 也有先例
（SPRSound data fusion、LungMix）。
**可以主张的是**：针对四个数据集 source semantics 审计后定义的
0/1/unknown eligibility contract + unit-preserving native head +
受控的跨数据集 retention 验证。
**不能主张**任何 "first"。

---

## 3. 主结果与 trade-off：一个被低估的发现

`result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/`（3 seeds: 0/1/42）

| System | ICBHI Sp | ICBHI Se | ICBHI Score | SPRSound Score |
|---|---:|---:|---:|---:|
| PAFA（paper） | 82.05 ± 1.95 | 47.63 ± 2.23 | 64.84 ± 0.60 | 55.82（zero-shot transfer） |
| BEATs+CE（paper） | 78.77 ± 3.07 | 48.21 ± 2.32 | 63.49 ± 1.08 | -- |
| PAFA（local author ckpt） | 76.88 | 51.40 | 64.14 | 55.82 |
| **本文 core model** | **74.98 ± 5.69** | **47.35 ± 5.35** | **61.17 ± 0.31** | **90.70 ± 0.34** |

### 关键观察（零成本，立刻可写，直接回答老师"到底改善或损害了什么"）

**ICBHI 的 3.67 分差距几乎完全来自 specificity，不来自 sensitivity。**

- Se：47.35 vs PAFA 47.63（−0.28）、vs BEATs+CE 48.21（−0.86）→ 实质持平。
- Sp：74.98 vs 82.05（−7.07）、vs 78.77（−3.79）→ 差距全在这里。

即：**联合训练没有牺牲异常检出能力，牺牲的是 Normal 的判定精度。**
最可能的机制是 SPRSound 的 abnormal-heavy supervision（inter test 中
abnormal 389 / normal 1040，训练侧属性分布更偏）把 operating point 推向 abnormal。
这条观察把"−3.67 分的亏损"改写成"一个有机制解释的 operating-point 位移"，
而且和 §5 的 HF 实验方向一致（HF supervision 反过来把 ICBHI 推向高 Sp 低 Se）。

注意边界：对 local PAFA author checkpoint（Se 51.40）我们的 Se 是 −4.05，
所以句子要写成"与两条 PAFA published rows 的 Se 在 0.9 分以内"，
不要写成"与所有 PAFA 参考持平"。

其余需要写进 error analysis 的：per-class recall
Normal 0.750 / Crackle 0.602 / Wheeze 0.296 / Both 0.368。
Wheeze 是最弱的一类，且方差最小（±0.021），是系统性而非随机的短板。

### HF / KAUH（建议压缩为 supporting）

- JH4（single-seed）：ICBHI Score 60.05→59.71（−0.34），但 Sp +5.45 / Se −6.12 /
  Both recall 0.357→0.189；SPRSound 89.20→91.81（+2.61）；
  HF D/Crackle AUROC 0.508→0.698、positive-interval recall 0.526→0.931；
  Wheeze AUROC 0.855→0.898 但 interval recall 0.938→0.769。
- KAUH fixed-checkpoint 3-seed：Level-1 recording 0.7099 ± 0.0353 /
  patient 0.7217 ± 0.0164；flat-4 recording 0.4778 ± 0.0914 / patient 0.4700 ± 0.0830。
  → **粗粒度 Normal/Abnormal 迁移得动，细粒度属性迁移不动。** 一句话即可。

---

## 4. 还缺什么证据（按对论文的威胁排序）

### 🔴 G1（Blocker）：JH2 配方下的 single-source 对照

**为什么是 blocker**：论文要说"joint 能同时守住两个 native task"。
现在的对照是*别人的* ICBHI 方法做 zero-shot 迁移。审稿人第一个问题必然是：
**"你自己的架构只用 ICBHI 训练，在 SPRSound 上是多少？反过来呢？"**
没有这个，"joint" 这个因子和"我们的 hierarchy / 我们的 BEATs 配方"完全混淆，
C2 和 C3 都失去因果落点。

**需要**：
- (a) ICBHI-only JH2 → 评 ICBHI + SPRSound
- (b) SPRSound-only JH2 → 评 SPRSound + ICBHI

构成一个 2×2 retention matrix。最少各 1 seed（3 seeds 更好）。
按 multiseed 记录，每 run 约 21--27 epochs、服务器修复后约 3 min/epoch，
成本可控。**这是剩余实验里投入产出比最高的一项。**

现有最接近的替代品是 `result/reproduce/icbhi_attribution_beats/comparison.json`
（B0 joint-hier 49.03 / B1 ICBHI-only-hier 46.13 / B2 ICBHI-only-flat4 50.42），
但那是 2 s window、single-seed、绝对值低，**不能替代**，最多作为脚注。

### 🔴 G2（Blocker）：matched joint baseline，去掉 hierarchy

shared encoder + 两个 independent native heads，其余完全一致。
work plan 已列为未完成。没有它，我们无法把任何改善归因到
*hierarchy + eligibility*，只能说"联合训练有用"。这是 C2 的唯一直接支撑。
1 run。

### 🟠 G3（零成本，最高优先）：Sp/Se + per-class 分解的正式写作

§3 的观察不需要任何新实验，只需要写出来 + 一个 3 行的 per-class 表。
**周五就该做完。** 它同时关闭老师的两个问题
（"拆分 Sp/Se"、"解释改善或损害了什么"）。

### 🟠 G4：Wade 的 quantitative separability + 与模型行为的连接

最小可用版本：
1. 一个 group-balanced 的 domain-separability 数值（domain-classifier AUC /
   MMD / Fréchet），替代"只有 PCA"；
2. 在 class-matched Normal/Abnormal subset 上复核，排除"分离只是类别构成差异"；
3. 与模型行为的**关联式**表述（禁止因果）：
   KAUH 在 level 上最接近 ICBHI，其 Level-1 迁移也最好；
   HF 在 level/标注结构上最远，细粒度迁移最差。

**Fallback**：若周六前不可用，直接用现有 Figure 1 panel C 的 recording-level
descriptors + 20-track panel 的 median dBFS（−11.06 / −41.02 / −38.08 / −20.63），
写成"采集电平跨数据集相差约 30 dB"，够用。论文不等 open-ended outlier exploration。

### 🟠 G5：Hanlin 的 baseline inventory 归一化

用于 Table II 的 "generic foundation models 不能自然适配四个数据集" 行块。
**Fallback**：用已有的 representation-attribution 结论（AudioSet-only ≥ PAFA-tuned
on 3/6 tasks）+ published EMBC frozen-BEATs ICBHI 57.31 行。两者都已在手。

### 🟡 G6：test-selection 的乐观量级（强烈建议，已有数据）

我们已经有一个现成的量级参考：PAFA BEATs+CE 复现里
clean validation-selected 54.83 vs author-faithful test-selected 59.86
→ **test-selection 约值 5 个 ICBHI 点**。
把这一句写进 protocol 小节，等于主动、量化地披露了 Table I 所有行共享的偏差，
比只写一句 caveat 强得多。
如果时间允许，再补 1 个 seed 的 validation-selected JH2 companion 更完整；
但即使不补，上面这句话已经能撑住。

### 🟢 G7（可砍）：JH4 补到 3 seeds。补不了就明确标 single-seed。

---

## 5. 建议砍掉的内容

| 内容 | 处理 |
|---|---|
| Soft Bridge JH3 / JH3.1 / JH3.3（57.21 / 58.69 / 58.66） | 全砍或压成一句"alternative output-coupling variants 未超过 core model"。它们同时改动 method 和 selection，不是 clean single-factor ablation |
| Hard Hierarchy + MVN（58.43） | 一句话 |
| four_dataset_* 探索套件（CRT、event-sensitive pooling、shortcut diagnostic） | 只留 dataset-ID probe 一个从句，用于说明"为什么不 pool"，且必须写成 accessibility 而非 shortcut |
| KAUH B/D/E per-filter 分解 | 不进正文 |
| 任何 pooled cross-dataset score | 已在禁止清单，保持 |

---

## 6. 四页版面的组织方案

| Section | 篇幅 | 承担的论证 | 素材状态 |
|---|---|---|---|
| 1 Introduction | ~0.75 col | G-A 的一句话 + G-C 的结构性论点 + 3 条 contribution | ✅ 证据齐 |
| 2 Data & Task Alignment + **Fig 1** | ~1 col | 异质性四个轴 + hierarchy + eligibility contract | 🟠 待 G4 |
| 3 Method + **Fig 2** | ~1 col | 四个机制 + native readout + HF extension | ✅ 待写 |
| 4 Evaluation + **Tab I** + **Tab II** | ~1.5 col | 主结果 + retention matrix + ablation + error analysis | 🔴 待 G1/G2 |
| 5 Conclusion | ~0.2 col | retention/trade-off + limitation | ✅ |

### 最重要的组织改动

**把 §1 的 G-A 表格化，作为 Table I 的第一个行块**，而不是放进 ablation。
Table I 结构建议：

```
Block A  单数据集专家的跨 benchmark 失效（motivation）
         PAFA / SG-SCL / Patch-Mix：ICBHI Score | → SPRSound Score | floor 50.0
Block B  ICBHI task-compatible published references
         PAFA 64.84±0.60 / BEATs+CE 63.49±1.08 / frozen-BEATs 57.31±1.40
Block C  本文 core model（3 seeds）
         ICBHI 61.17±0.31 | SPRSound 90.70±0.34
```

Table II 结构建议：

```
Row 1  ICBHI-only JH2      → ICBHI | SPRSound     [G1 待补]
Row 2  SPRSound-only JH2   → ICBHI | SPRSound     [G1 待补]
Row 3  joint, independent heads（无 hierarchy）    [G2 待补]
Row 4  joint, hierarchical（本文 core）            ✅
Row 5  + HF positive-only auxiliary（JH4，标 single-seed） ✅
```

Table II 的第 1--4 行就是 **retention matrix**，它同时回答
"joint 有没有用" 和 "hierarchy 有没有用" 两个因子。这是全篇最需要的一张表。

Figure 分工（老师明确要求不重复承担同一功能）：
- **Fig 1** = 数据为什么难合（composition + acoustic separability），
- **Fig 2** = 我们怎么合（input → harmonization → shared BEATs → hierarchy → native readouts），
  **不放** optimizer、不放全部 loss branch、不放实验编号。

---

## 7. 三条 contribution（每条都指向具体证据）

1. **Problem evidence**：在受控设置下量化了 ICBHI-optimized SOTA 方法无法保持
   第二个 native benchmark（64.1→55.8、61.0→60.0、62.2→59.4，trivial floor 50.0），
   并说明 scoped primary-source audit 中不存在 task-compatible 的
   SPRSound official-inter BEATs 结果。
   → 证据：`icbhi_strong_method_reproduction` + 三个 transfer 产物 + 文献 audit。**已有。**

2. **Method**：eligibility-aware hierarchical alignment，保留 label availability 与
   native prediction unit，联合训练不制造 negative supervision，
   并可扩展到只有正样本标注的数据源（HF）。
   → 证据：core model + JH4 + **G1/G2 对照（待补）**。

3. **Analysis**：单模型单 checkpoint 同时达到 ICBHI 61.17±0.31 与 SPRSound
   90.70±0.34；ICBHI 的代价集中在 specificity 而非 sensitivity；
   粗粒度 Normal/Abnormal 可迁移到未见数据集（KAUH 0.72），细粒度属性不可（0.47）。
   → 证据：3-seed 主结果 + fixed-checkpoint external。**已有。**

---

## 8. 与其他 work 的定位（Related Work 的四个 camp）

| Camp | 代表 | 他们做什么 | 我们的区别 |
|---|---|---|---|
| ICBHI task-specific SOTA | PAFA, SG-SCL, Patch-Mix, AddRSC, MVST, SPA | 单数据集单任务最优 | 我们与 PAFA 匹配 backbone/input/selection，在其主场落后约 3.7 分，但证明它守不住第二个 benchmark。**我们不在 ICBHI 上竞争，我们展示专家模型的盲区** |
| Frozen foundation-model evaluation | BEATs 原文, EMBC eval-audio-repr (57.31), OPERA linear probe | 每个数据集当独立 probe task，frozen encoder + per-dataset head | 无共享监督、无 label alignment、不问 retention。且我们自己的 attribution 显示 ICBHI 微调不会改进共享表示 |
| Multi-source pre-training / multimodal | Resp-Agent (ICBHI 72.70) | 把 HF+SPRSound 当 pre-training pool，再回到 ICBHI 微调 | 他们 pool 数据然后专门化到一个数据集；我们对齐任务并用**同一个模型同一个 checkpoint**同时评所有 native task。且 72.70 含 text/Longformer/生成式平衡，非 acoustic-only 可比 |
| Label harmonization / 跨源 DG | LungMix, SPRSound data fusion, BTS-CARD, PC-MCL | 语义 OR 混合标签、single-source DG、metadata debiasing | **最需要小心的一档**。区别点必须落在：0/1/unknown eligibility contract（unknown≠negative）、native unit/head 保留、以及受控的跨数据集 retention 验证 |

**必须预防的审稿人反击**：Resp-Agent 的 72.70 会被指着问。
正文需要一句显式的不可比声明（多模态 + 额外预训练 + 生成式数据平衡 + 目标域微调）。

---

## 9. 三天执行建议（对齐 09-04~09-06 work plan）

**周五**
- 冻结 RQ / gap / 3 条 contribution（用 §1、§7 的措辞）
- 写完 G3 的 Sp/Se + per-class 分析段（零成本、最高优先）
- 决定 G1/G2 是否开跑 —— **建议：跑**，这是唯一能把 C2 从"未支撑"变成"已支撑"的动作
- 按 §4 的 fallback 条款给 Wade / Hanlin 下合同

**周六**
- Fig 2 v1（严格按 §6 的 input→harmonization→model→output）
- Table I / II 按 §6 的 block 结构建骨架，缺的行留 `[G1]` / `[G2]` 占位
- Data + Method 正文 v1
- 写入 G6 的 test-selection 量级披露句

**周日**
- 填 G1/G2 结果（若已回）
- Introduction v1（此时 gap 与 contribution 已冻结，反向写最快）
- page budget 实编译核对
- 输出仅含 submission blocker 的 unresolved list

---

## 10. 措辞红线（沿用 work plan §6，补充两条）

沿用：不写 SOTA / universal / robust / clean test；不把 PCA 写成因果；
HF gap 不当 Normal；KAUH B/D/E 不当独立患者；不构造 pooled score。

**补充：**
1. dataset-ID probe 只能写"domain identity 可被线性读出，存在 shortcut 风险"，
   不能写"存在 shortcut"（该 diagnostic 结论为 `not_supported_or_inconclusive`）。
2. 跨数据集迁移行必须标注 `zero-target-tuning frozen-checkpoint transfer`，
   不能表述为"联合训练的净收益"——净收益只能由 G1 的 single-source 对照给出。
