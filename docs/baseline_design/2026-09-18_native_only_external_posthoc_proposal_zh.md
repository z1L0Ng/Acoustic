# Native-only HF/KAUH独立post-hoc固定方案

状态（2026-09-19）：**用户已批准固定方案，代码入口准备中；由管理review/push后服务器执行。** 不修改Native-only训练代码、配置、选模、阈值、ICBHI/SPRSound原生结果或原三seed主汇总。

## 当前模型可用信息

Native-only保留并训练ICBHI四分类头与SPRSound二分类头；C/W辅助头保留但没有监督，不能用于本次外部评测。读取选定checkpoint后使用eval模式，不重新训练或选择epoch。

## 冻结定义：两项统一使用ICBHI四分类头

对每个输入单元，对训练过的ICBHI四分类logits做原生softmax，类别顺序沿用代码的Normal、Crackle、Wheeze、Both。

### HF CAS

- 沿用现有HF source-test及每条录音的三个固定5秒窗口，不修改输入前处理。
- 每窗分数：`P(Wheeze) + P(Both)`，表示原生四分类分布中的Wheeze边缘概率。
- 录音分数：三个窗口分数的最大值。
- 沿用论文CAS评测集合：任何D/Wheeze/Rhonchi/Stridor标注的957条eligible录音，其中CAS阳性661条、D-only阴性296条。
- CAS阳性仍为Wheeze/Rhonchi/Stridor；这个分数是Wheeze边缘概率对CAS的ranking proxy，并非训练过Rhonchi/Stridor的CAS分类头。
- 主报CAS AUROC。AUROC无需阈值；不为它新增目标数据阈值拟合，也不额外声明0.5下的CAS分类能力。

### KAUH

- 使用现有86位compatible患者，保持B/D/E视图、兼容标签和原5秒前处理。
- 每个view的异常分数：`1 - P(Normal)`，等于四分类中三个异常类概率之和。
- 患者分数：B/D/E异常分数的算术均值。
- 固定0.5阈值，分数大于0.5判异常，否则判正常；恰好0.5沿用两类argmax的Normal优先规则。该定义在读取外部结果前固定。
- 主报patient balanced accuracy，同时保存患者ID、target、score、prediction、support及confusion。
- 这是新的独立外部二分类读出，不修改ICBHI原四分类argmax结果。

## 共同边界与交付

- 全部seed固定使用`icbhi_native`四分类头；不再保留SPR head候选或结果后选择空间。
- 不使用未监督C/W头，不访问HF/KAUH来选模型、选择head、调阈值或改变训练。
- 同一确定规则用于seeds 0/1/42；任何seed缺失或失败时不生成`n=3`汇总。
- 入口：`baseline/pafa/native_only_external_posthoc.py`。
- 独立输出：`result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only_external_posthoc/seed_*`；保留逐窗/逐录音、逐view/逐patient的label-free与scored预测和单独三seed汇总，不覆盖Native-only原生run_summary、predictions、selection、thresholds或原三seed主汇总。
- 与LSAA/without-PAFA的外部比较需说明分数来源不同：Native-only用native softmax的边缘概率；LSAA用受监督的属性概率。它不能单独隔离表示学习与读出贡献。
- 不增加训练、cache或checkpoint，不恢复旧队列。HF/KAUH标签只在label-free预测落盘后用于终点计分。

服务器命令：

```bash
conda run --no-capture-output -n acoustic-addrsc python -m unittest \
  tests.test_native_only_external_posthoc.NativeOnlyExternalPosthocTest.test_hf_cas_metrics_uses_standard_auc_and_rejects_nonfinite -v

conda run --no-capture-output -n acoustic-addrsc python \
  -m baseline.pafa.native_only_external_posthoc \
  --repo-root /files1/Zilong/Acoustic --device cuda --queue --run
```

证据边界：三个源checkpoint仍为ICBHI official-test-selected Native-only模型；本项是固定读出的zero-target external post-hoc，不能改写成clean source-selection或target adaptation。
