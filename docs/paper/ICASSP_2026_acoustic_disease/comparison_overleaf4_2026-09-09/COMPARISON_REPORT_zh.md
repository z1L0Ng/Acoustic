# 当前本地稿与 Overleaf (4) 导出对比

比较日期：2026-09-09。范围为 Introduction、Data、Method、Evaluation、Figure 1/2、Table 1/2。Abstract 与 Conclusion 按用户要求不作内容对齐判断，不列为本轮补实验或正文对齐的阻塞。

## 结论

**尚未完全一致。Introduction、Data、Method 主体已经一致；Overleaf 尚未同步 Evaluation 最近一次引用/术语修订，Figure 1 图注也落后于本地。现有实验数字、主要方法、四数据集分工和表格结构没有发现版本冲突。**

最小对齐只需：

1. 将 Overleaf 的 Section/4evaluation.tex 更新为主目录当前版本。实质更新范围是 §4.2 的 frozen 背景段、Table 2 的 caption/最后一行名称、§4.3 的 Classification design 段。表内实验条件和数字不变。
2. 仅替换 Overleaf main.tex 中 Figure 1 的 caption 为当前文字：

```latex
\caption{Acoustic cues and annotation granularity across datasets. Figure and analysis pending.}
```

不需要重新改写或复制 Introduction、Data、Method。不要用整份 main.tex 覆盖 Overleaf 来处理这条图注，因为本轮不要求同步 Abstract/Conclusion。完成上述同步后，需要新的 Overleaf 导出才能确认云端已经更新；本次未直接编辑 Overleaf。

## 按章节与图表核对

| 对象 | 对比结果 | 证据与最小动作 |
| --- | --- | --- |
| Introduction | 一致 | 六段结构、无实验数字/xxx、末段工作概述均相同；按双栏提取并规范化断行、连字符、字体连字后文本一致。无需恢复旧 P2 数字槽位。 |
| Data §2.1–2.3 | 一致 | 四数据集分工、4,142/2,756 与 1,429 单位、R/S 只监督 A、HF 的 C/W 条件、KAUH 86 compatible patients、B/D/E 聚合、5 s 输入与日期代理说明一致。§2.2 均仍是占位。 |
| Method §3.1–3.4 | 一致 | BEATs + 三个并行头、可用节点损失、PAFA 权重、两层读出与 fallback、阈值规则、HF 0.25 辅助项及派生负例说明一致。原 §3.5 均未显示，必要评测定义已在 §4.1。公式 (1)–(4) 视觉一致；公式 (2) 的文本提取存在求和符号编码/排列差异，不是公式改动。 |
| Figure 1 | 占位内容相同，图注不同 | 两版均在第 2 页。Overleaf 图注仍为 Cross-dataset acoustic characteristics；本地为 Acoustic cues and annotation granularity across datasets。采用本地图注。真正的图和样本分析两版都没有。 |
| Figure 2 | 内容一致 | 两版均在第 3 页；音频/标签路径、并行 A/C/W、任务读出、HF/KAUH 分工、HF 辅助损失和 B/D/E 聚合相同。放大核对后连线结构一致；细线粗细等渲染外观不同，不据此判为方法版本变化。 |
| Table 1 | 一致 | 两版均在第 4 页跨栏；七列、数据集及模型引用、四条 published 强参考、Ours 三 seed 指标、五条 frozen-context 行和破折号含义一致。表中文字与数字规范化后完全相同。Macro-F1 列均已删除。 |
| Evaluation §4.1、§4.2 第一段 | 一致 | benchmark 的 test-based selection 与独立 validation-selected 对照区分一致；Se 差 0.28 pp、Sp 差 7.07 pp、分差集中于 specificity 的表述均已存在。HF/KAUH 评测定义已内联，无失效的 sec:external_readouts 引用。 |
| Evaluation §4.2 frozen 背景段 | 本地更新 | Overleaf 仍逐个列 AST/BEATs/PANNs/OPERA/HeAR 作为背景，未在此引用 Niizumi。本地段落以 Niizumi et al. [10] 起头，区分文献中的 frozen encoder + task head 评测与本表参考数值的独立来源。 |
| Table 2 | 条件/数字槽位一致，caption 与行名未同步 | 七行均存在；HF on/off 已合并；最后一行 SPR 和 C/W AUROC 均为破折号。Overleaf 为 C/W-only readout (PC-MCL-style)；本地为 Direct C/W thresholding，保留 [16]。采用本地 caption 与行名。 |
| Evaluation §4.3 Classification design | 本地更新 | Overleaf 使用旧的 PC-MCL-style 表述。本地明确区分 PC-MCL 的 Normal 训练标签与 C/W 推理转换，保留本模型表示、属性监督、分数和阈值不变。对照定义没有新增训练条件。 |
| §4.3 source/annotation 与 HF/KAUH 结果段 | 一致 | 原生保留、coarse-label、HF on/off 和外部评测都仍使用 xxx；没有发现一版已有结果而另一版遗漏。 |

## 八项“零训练成本写作工作”的当前状态

| 项目 | 当前判断 | 真正剩余动作 |
| --- | --- | --- |
| gap 数字槽位 | 不再是当前 Intro 待办 | 用户已明确 Intro 不放实验数字；两版都已执行。数值槽位在 Evaluation，待真实结果填入，不恢复旧建议。 |
| contribution 表述 | 已按最新结构处理 | 两版均以问题定义和工作概述收束，原贡献内容融入正文；不预写未经结果支持的 finding。实验完成后再精修结果性结论。 |
| mask 覆盖说明 | 规则已写，定量覆盖未写 | §2.3、§3.2、§3.4 与 Figure 2 已解释哪些目标可监督。若要求 eligible 数量/比例，仍需真实标注统计与确定的划分；不能凭空补数。 |
| frozen 背景块限定 | 已有，最新文献支撑待云端同步 | 两版均注明 frozen、单 seed、单数据集和背景用途。只需同步本地 §4.2 的 Niizumi 段，不是从零重做。 |
| Table 1 破折号语义 | 已完成 | 两版 caption 均写明 unreported endpoints。Table 2 的破折号另表示未改变的输出，不能混成同一解释。 |
| Resp-Agent 从句 | 两版正文均未出现 | citation.bib 有 zhang2026respagent，但正文未引用。若仍保留该近邻覆盖项，需要补一句经过来源核对的定位；本轮只记录缺失，不自动增加论点。 |
| ICBHI-only 作为两套协议的桥梁 | 尚未形成明确规格/表述 | Table 1 的 published 参考与 Table 2 的 validation-selected ICBHI-only 行不构成同一受控桥梁。需管理与模型设计确认对应模型、训练数据、选择方式及报告位置。 |
| Native + attributes 精确定义 | 尚未完整 | 当前只有“同等属性监督”及总体配方，源码注释仍要求固定具体 loss 与 head capacity。需要已冻结的头结构、损失项/权重、mask 与归一化约定、哪些输出参与评价，再写入正文。 |

这里没有替管理/模型设计选择新的实验规格。指定的三项规格应由两方同步冻结后再补写；本轮未训练、重评分、提取特征、建缓存或运行服务器任务。

## 仍缺什么

1. **版本同步**：上述 Evaluation 三处与 Figure 1 caption。属于可直接复制的写作更新，不需要新实验。
2. **Figure 1 与 §2.2 的实际内容**：两版均为占位，尚无可以支撑共享属性讨论的图和分析。
3. **控制实验的正式结果**：Table 2 七行及对应原生/属性/HF/KAUH 数值仍待填。Table 1 的 test-selected benchmark 不能替代这些 validation-selected 结果。
4. **实验定义的冻结与落文**：包括当前注释中仍待确定的划分/选模准则、ICBHI-only 桥梁和 Native + attributes 精确定义等，以管理/模型设计最终规格为准。
5. **相关工作的一处候选缺口**：Resp-Agent 从句尚未落实。是否保留应按当前研究定位决定，不能将这项写作覆盖与模型执行准备混为一谈。

正文对齐是启动条件之一；本次比较不构成实验启动指令，也不判定 runner 或服务器已准备完毕。Abstract/Conclusion 未列入以上缺口。

启动前需处理的是版本同步，以及管理/模型设计确认的实验规格和执行授权。Table 2 与 HF/KAUH 的空值是待启动实验的产出目标，不是要求先有结果才能开工。Figure 1 和 Resp-Agent 等写作项可另行推进，不自动作为独立实验的启动阻塞。

## 排版差异（与内容差异分开）

- Overleaf 导出为 6 页，本地当前完整源码编译为 7 页。两版的首尾区域和编译引擎未统一，因此页数不能直接用来推断正文增删。
- Overleaf 使用 pdfTeX 1.40.27；本地比较件使用 Tectonic/xdvipdfmx。字体连字、求和符号文本编码、断行、段间留白及细线外观存在差异。
- 两版 Figure 1/2 和 Table 1 的页面分别为 2/3/4；Table 2 在 Overleaf 第 4 页、本地第 5 页。它都位于 References 开始之前，之前“表格挂到参考文献中”的问题不再是当前状态。
- **两版 Evaluation 都有内容落到第 5 页。** 四页技术正文的篇幅目标尚需处理；这是版面问题，不是补实验定义或结果的替代动作。
- 本地编译成功，仍有已有的 underfull/overfull 警告（Intro 约 5.23 pt overfull）。按本次范围保留当前状态，没有改字号、页边距或重新排版。

## 产物与来源

- 用户原始导出（未修改）：/Users/zilongzeng/Downloads/ICASSP_2026_acoustic_disease (4).pdf
- 当前本地比较 PDF：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/comparison_local_current_2026-09-09.pdf
- 本报告：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/COMPARISON_REPORT_zh.md
- 渲染页及图件放大核对：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/tmp/pdfs/
- 本次编译采用的当前源码副本：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/tmp/source_snapshot/
- 编译日志与中间文件：/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/comparison_overleaf4_2026-09-09/tmp/build/

对照方法：渲染两版并逐页查看正文与图表；按左右栏分别提取正文，移除跨栏浮动区域后规范化断行、连字符和字体连字；公式另作视觉核对。比较 PDF 由本次捕获的当前主目录源码与 Figure/figure2_method.pdf 生成，没有使用旧 review PDF。编译后核对文本输入未被其他修改改变。未修改主稿或用户 PDF；新增文件均在本次 comparison 目录内。初次编译的默认缓存目录权限问题已通过已有临时缓存解决，没有剩余编译阻塞。
