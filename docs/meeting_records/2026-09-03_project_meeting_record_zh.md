# 2026-09-03 UNC Acoustic 项目会议记录

## 1. 会议定位

本次会议在 ICASSP 截止日期只剩约 13 天的背景下进行。会议首先复盘当前主结果，
随后重点讨论 acoustic feature analysis、single-dataset baselines、主方法的跨数据集
表现，以及论文的 research question、contribution 和 figure 组织。

本记录保留会议中的技术讨论、结果判断和老师反馈。会议末尾与项目无关的闲聊不纳入
主体。会议中口头提到的 61% 和 64% 均为近似值，正式论文数字仍以完成核对的结果
文件为准。

## 2. 当前主结果及老师的初步判断

- 当前 multitask/joint setup 在 ICBHI 四分类任务上的结果约为 61%。
- 当前较强的 ICBHI task-specific work 约为 64%，两者相差约 3 个百分点。
- 老师确认这里比较的是 ICBHI official 四分类任务，而不是其他二分类或跨数据集指标。
- 老师认为 64% 的绝对数值本身并不高，但当前 baseline 与强结果的距离不算过大，
  因此 61% 左右的结果并非不可接受。
- 论文不能只展示一个 aggregate score。后续必须拆分 specificity、sensitivity、
  per-class behavior 和不同数据集上的变化，解释模型到底改善或损害了什么。

老师认为论文重点不应是宣称超过单数据集 SOTA，而应回答 heterogeneous respiratory
acoustic datasets 能否被一个统一框架适配，以及这种适配在不同数据集上带来怎样的
收益和代价。

## 3. Wade：acoustic feature analysis

### 3.1 已展示的工作

Wade 已完成 ICBHI 和 SPRSound 的部分 window-level analysis：

- 将音频统一 resample 到 16 kHz；
- 使用 peak normalization，目标 peak 为 0.95；
- 比较三种 window/stride：
  - 1 s / 0.5 s；
  - 2 s / 1 s；
  - 4 s / 2 s；
- 从每个 window 提取多项 acoustic features；
- 对组合后的完整 feature vector 做 standardization；
- 将所有特征共同输入一个 PCA model，而不是为每种 feature 单独做 PCA；
- 使用两种 outlier detection 方法，并保存包含 outlier window ID 的 CSV；
- 开始按 native label、recording device 和 official split 检查 feature 与 metadata
  的关系。

会议中展示的是 ICBHI 2 s window 的 PCA。SPRSound 已采用类似流程，但四个数据集
尚未形成统一的 cross-dataset comparison。

### 3.2 老师的反馈

老师强调，这项工作的目的不是证明“做过 PCA”或统计出多少 outlier，而是解释主实验
行为。后续分析需要直接回答：

1. 不同数据集的 acoustic characteristics 是否存在可量化的差异；
2. 这些差异是否与 joint training 后不同数据集的提升、保持或下降相对应；
3. 当某个结果不符合预期时，问题更可能来自数据本身，还是模型和 pipeline；
4. SNR、level、spectrum、duration、device 或其他 feature shift 是否能解释
   ICBHI、SPRSound、HF Lung 和 KAUH 的差异。

老师同意将四个数据集组合到一张或少数几张图中，但要求在进入论文前确认：

- 分析方法合理；
- 结果可信；
- 观察与论文中的 experimental observation 有明确关系；
- PCA 不能被单独当作因果或统计显著性证据。

子龙需要向 Wade 提供最新实验结果和明确问题，使后续分析从开放式探索转为
paper-directed analysis。

## 4. Hanlin：reproduction 和 single-dataset baselines

### 4.1 已展示的工作

Hanlin 汇报了两条工作线：

1. 复现与当前项目接近的 foundation-model pipeline：
   - frozen foundation encoder 加四层 Transformer classifier；
   - frozen foundation encoder 加单层 linear classifier；
   - 先按原论文设置复现，再按项目统一设置应用于当前数据。
2. 在四个数据集上建立 single-dataset foundation-model baselines：
   - ICBHI、SPRSound、KAUH 使用 projection 到 256-dimensional representation
     后接分类器；
   - HF Lung 使用其 native temporal/task-specific head；
   - 尽量遵守各数据集的 official split 和 native task。

老师确认这些结果表示 foundation models 在每个数据集上分别训练和测试，而不是
cross-dataset combination。虽然基础 classifier 的绝对分数不高，但它们可以作为
non-combination baseline，说明未经 respiratory-specific adaptation 的 acoustic-event
foundation models 并不能自然适配所有数据集。

### 4.2 尚需补齐的内容

- 增加一到两个 ICBHI 上真正较强的 task-specific baselines；
- 对每个 baseline 明确 paper setting、local setting、task、split、unit、metric 和
  checkpoint-selection protocol；
- 区分“成功复现论文结果”和“把论文方法应用到我们的统一设置”；
- 形成约三组可直接进入论文 comparison table 的强对照；
- 评估一个在 ICBHI 上优化良好的方法是否能够自然迁移到 SPRSound、HF Lung 或
  KAUH，而不是只比较通用 acoustic foundation models。

老师要求子龙与 Hanlin 单独对齐具体方法、配置和交付形式，避免只得到无法支撑论文
argument 的零散实验。

## 5. 主方法、HF supervision 和指标变化

- 当前表现最好的主版本不使用 augmentation，也不包含 HF-specific supervised loss。
- HF-specific loss 的原始目标是改善 HF Lung，同时不改变 core checkpoint-selection
  逻辑。
- 已观察到加入该 loss 后 HF 自身得到改善，SPRSound 也可能受益，但 ICBHI 下降约
  2--3 个百分点。
- HF Lung 缺乏与其他数据集一致的 Normal annotation，Crackle/Wheeze 分布也不均衡。
  这可能影响 Normal/Abnormal 判断，但老师提醒不能未经验证就把下降完全归因于
  “HF 缺 Normal”。
- 另一种可能是其他数据集中的 Normal 比例原本较高，加入 HF 后整体 supervision
  distribution 改变，从而导致 specificity、sensitivity 或具体 abnormal-class recall
  发生变化。

后续需要拆分：

- ICBHI specificity 和 sensitivity；
- Normal、Crackle、Wheeze、Both 的 per-class behavior；
- HF supervision 对 Normal/Abnormal 与细粒度属性判断的不同影响；
- aggregate score 保持或下降时，是否存在某个 component metric 的稳定改善。

这部分适合写成 adaptability 和 trade-off，而不是“所有数据集全面提升”。

## 6. 论文 story 和 figure 设计

### 6.1 Story

老师建议论文围绕 adaptability 展开：

- heterogeneous respiratory datasets 在 acquisition、prediction unit、label support、
  class distribution 和 native task 上存在差异；
- 现有 acoustic-event foundation models 或 single-dataset methods 无法直接证明
  跨数据集适配能力；
- 本项目尝试通过统一但不抹去 native-task meaning 的框架处理这些差异；
- 实验需要说明哪些数据集获益、哪些数据集退化，以及这些差异能否由数据属性或
  supervision structure 解释。

论文不能写成“依次尝试 A、B、C、D 后选择最好组合”。需要形成：

1. 明确 limitation/gap；
2. 可回答的 research question；
3. 针对该问题设计的方法；
4. 能直接支撑结论的实验；
5. 两到四个具体且有证据支持的 contribution points。

### 6.2 Figures and tables

- 需要一张 high-level overview figure，简化为 input、dataset harmonization/processing、
  model 和 output；
- overall figure 不应塞入过多 encoder、branch 或 implementation details；
- 数据集结构、统计和 acoustic differences 放在独立 dataset figure/table；
- 方法内部细节只有在确实帮助读者理解时才单独展示；
- 短篇论文中 figure 和 table 应帮助读者快速理解“做了什么”和“结果说明什么”。

## 7. 写作顺序

- 先完成 Data 和 Method，因为这两部分对最终结果的依赖较小；
- 在完整结果和 method logic 更稳定后，再写 Introduction；
- 在继续大量正文前，先冻结一句 research question、一个 key phrase 和约三条
  contribution；
- 通过 contribution 反向筛选必须保留的实验、图表和分析；
- 截止日前不再引入大规模 architecture modification 或开放式 optimization。

## 8. 会议形成的近期行动

### 子龙

- 将最新主结果和需要解释的现象明确交给 Wade；
- 将 strong ICBHI baseline 候选、复现边界和交付表格明确交给 Hanlin；
- 冻结 research question、gap 和三条 contribution 候选；
- 优先组织 Data、Method、high-level pipeline figure 和核心结果表；
- 拆分 ICBHI specificity/sensitivity 和 per-class behavior；
- 控制新实验范围，只补论文必需的数字和对照。

### Wade

- 完成四数据集统一的 acoustic-feature comparison；
- 将 PCA/outlier 结果转化为与模型行为相关的可量化观察；
- 检查 acoustic shift 是否能解释跨数据集 improvement/degradation；
- 交付能够进入 dataset figure/table 的数字、图和简短结论。

### Hanlin

- 汇总已完成的 four-dataset single-dataset baselines；
- 补充一到两个 strong ICBHI task-specific baselines；
- 明确 reproduction 与 adaptation 的区别；
- 按可直接进入论文 comparison table 的格式交付。

## 9. 当前决策边界

- 当前 61% 左右 ICBHI 结果可以作为可继续推进的主结果，但不构成 SOTA claim。
- Feature analysis 必须服务于解释实验，不再做无边界探索。
- HF supervision 的影响保持为 benefit--trade-off observation，因果解释仍需证据。
- 不启动新的大规模 architecture 或 hyperparameter sweep。
- 新的 baseline 训练、cross-dataset transfer 或主方法实验仍需单独确认具体合同后执行。
