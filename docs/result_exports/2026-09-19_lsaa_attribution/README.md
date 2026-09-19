# LSAA归因实验与Native-only独立外评：完整本地结果

状态：2026-09-19已完成并回收到本地。两组主实验各三个seed，Native-only另完成三个固定checkpoint的HF/KAUH独立post-hoc。没有重训、重新选模或目标数据调参。

## 汇总展示

数值为百分比，mean ± sample SD，n=3，seeds 0/1/42。Native-only的ICBHI/SPR来自原生主汇总，HF/KAUH来自独立post-hoc；此表仅汇总展示，没有改写原主结果文件。

| 方法 | ICBHI Score | SPR official Score | HF CAS AUROC | KAUH patient BA |
|---|---:|---:|---:|---:|
| Native-only：主结果＋独立外评 | 63.87 ± 1.43 | 88.93 ± 3.36 | 81.62 ± 3.69 | 75.59 ± 5.62 |
| LSAA without PAFA | 58.60 ± 1.60 | 91.04 ± 0.51 | 86.66 ± 1.83 | 78.30 ± 1.95 |

without-PAFA的SPRSound C/W mean AUROC：96.82 ± 0.28%。Native-only的未监督C/W头不报告此项。

## Native-only独立外评逐seed

| Seed | 原selected epoch | HF CAS AUROC (%) | KAUH patient BA (%) |
|---|---:|---:|---:|
| 0 | 23 | 83.1602 | 72.0728 |
| 1 | 15 | 77.4109 | 82.0728 |
| 42 | 16 | 84.2898 | 72.6331 |

## 固定读出与范围

- 全部使用训练过的ICBHI原生四分类softmax头，没有读取未监督C/W辅助头或改用SPR头。
- HF：每窗P(Wheeze)+P(Both)，三个固定5秒窗口取最大值；957条eligible录音，CAS阳性661、阴性296；标准AUROC，无分类阈值。此分数是Wheeze边缘概率对W/R/S union的ranking proxy。
- KAUH：每view的1-P(Normal)，B/D/E患者均值；大于0.5判异常，等于0.5判正常；86位compatible患者。
- HF/KAUH标签只用于终点计分，不训练、不选模型、不挑head、不拟合阈值。源checkpoint仍保留ICBHI official-test selection的原有边界。
- 本post-hoc与原生主结果独立保存；没有修改Native-only ICBHI/SPR指标、选模、阈值或主三seed汇总。
- 与LSAA属性头的外部结果比较时需说明分数来源不同，不能从该比较单独归因于表示学习或某一组件。

## 完整资产位置

- 原始完整目录：[LSAA_ATTRIBUTION_20260918](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918)
- Native-only原生三seed：[multiseed_summary.json](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only/multiseed_summary.json)
- without-PAFA完整三seed：[multiseed_summary.json](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/lsaa_without_pafa/multiseed_summary.json)
- Native-only独立外评：[multiseed_summary.json](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/native_only_external_posthoc/multiseed_summary.json)
- 原始执行日志：[server_logs](/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/LSAA_ATTRIBUTION_20260918/server_logs)

回收694个原始文件，共4,460,601,154 bytes（约4.46 GB），含12个best/last checkpoint、配置、逐epoch/终点预测、逐窗/录音/视图/患者预测、metrics、三seed汇总和3个服务器日志。完整checkpoint保留在results目录，不纳入Git。没有复制数据集、通用预训练资产或旧PC-MCL失败文件。

## Git轻量原始结果快照

本次按用户要求提交本目录的汇总和[raw](raw/)原始结果副本：682个文件，58,025,578 bytes，包含原始JSON、JSONL、NPZ预测和三个服务器日志。目录层次与完整本地结果一致；只排除12个模型权重文件，没有改写原始结果内容。完整best/last checkpoint继续保留在上方完整资产目录，不加入Git。

当前进入paper work：除等待和核验Hanlin已安排的PAFA/五组frozen参考更新外，暂不新增实验。后续重点是Figure1/2、Arian C1–C4、结果表及解释、全文与四页稿；新结果已经可用，不再等待服务器。

## 核对记录

- 两个主variant和独立post-hoc的文件数/字节总数与服务器一致；六个主run、三个post-hoc run均为complete。
- 本地核对Native-only逐窗Wheeze边缘分数及三窗max、KAUH B/D/E聚合/固定0.5预测与BA，均与保存结果一致；三seedmean/sample SD一致。
- 原始JSON保留其服务器绝对路径，本地目录保持相同相对布局；本页给出本地入口。
- 没有重新推理来验收，没有hash/checksum，也没有smoke/probe；回收和核对没有修改论文。
