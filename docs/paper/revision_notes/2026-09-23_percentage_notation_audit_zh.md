# 2026-09-23 百分比表述核对

范围：用户刚从Overleaf同步至main工作区的 `docs/paper/Overleaf_Sync_final/`。检查了main.tex及实际纳入的Sections 1–5和4tables，忽略被LaTeX注释掉的内容；没有编译或改变稿件。
基准：`result/analysis/2026-09-21_tables_anova_hanlin_seed_results/tidy_seed_metrics.csv`中的原始逐seed数值。

## 数值核对：通过
正文11组不同的差值，加上摘要重复的4处，共15处，均对应未舍入三seed均值的绝对差，按两位小数报告。没有将它们计算成相对增幅。

| 文中比较方向 | 指标 | 已核对的绝对差（百分点） |
|---|---|---:|
| LSAA − BEATs | SPRSound Score | −0.69 |
| LSAA − w/o C+W | SPRSound Score | +1.93 |
| LSAA − w/o C+W | HF AUROC | +7.16 |
| LSAA − w/o C+W | KAUH BA | +2.32 |
| LSAA − w/o C+W | ICBHI Score | −0.20 |
| LSAA − w/o PAFA | ICBHI Score | +5.07 |
| LSAA − w/o PAFA | KAUH BA | −0.39 |
| w/HF − LSAA | SPRSound Score | +0.76 |
| w/HF − LSAA | ICBHI Score | −4.78 |
| w/HF − LSAA | HF AUROC | −8.47 |
| w/HF − LSAA | KAUH BA | −2.40 |

部分先把表内均值舍入再相减的结果会与上表差0.01，原因是文中差值从未舍入均值计算；这不属于数值错误。90.86%、49.53%、47.75%作为指标本身的写法也与当前表格一致。

## 替换与语义：尚未通过
1. `Section/4evaluation.tex:37`还保留 `1.93 pp`、`7.16 pp`两处，与摘要同一段数据的%形式不一致。
2. `Section/4evaluation.tex:39`的`0.39 \\%`与其他数值紧邻%的格式不一致。
3. `main.tex:52`及Evaluation中的“improves by / higher / lower X%”沿用了绝对差值数字，容易被理解为相对百分比变化。仅说“Results are percentages”还没有定义变化量的计算方式。

具体例子：
- w/o C+W的SPRSound Score为88.9287978%，LSAA为90.8574392%。
- 绝对差为90.8574392 − 88.9287978 = 1.9286413个百分点。
- 相对增幅为(90.8574392 − 88.9287978) / 88.9287978 × 100% = 2.1687478%。
因此直接写“improves by 1.93%”可能被读为相对增幅，与真实计算不一致。HF和KAUH同理，相对增幅分别约为8.77%与3.06%，不是7.16%与2.32%。

## 与用户格式偏好兼容的改法（供确认）
保留%符号，不强制恢复pp。最明确的局部写法是“SPRSound Score increases from 88.93% to 90.86%”，把两个指标值均用%呈现。如果仍保留差值数字，应在摘要及正文相应句中明确是absolute change，并在首次说明计算定义；不能把原数字直接当作relative percentage improvement。
本次仅提交核对结果，未选择或写入最终措辞。F-01a已勾选数值核对通过；F-01b保留未勾选，整个F-01和S4-05未因此标成完成。

## 协作通知
论文写作、论文写作二及论文绘图已收到通知：后续使用canonical main当前工作区的Overleaf最新内容，尊重未提交的用户修改，等待用户后续具体指示。没有指示任务自动改稿、画图或编译。

