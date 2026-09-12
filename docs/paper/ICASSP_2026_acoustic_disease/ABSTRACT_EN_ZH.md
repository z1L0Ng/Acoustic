# Abstract｜English and Chinese

与 main.tex 同步：Table 1 使用本文模型（ours）的 ICBHI-test-based benchmark；验证集选模的消融与 HF 对照合并列于 Table 2。具体对照结果补齐后再精修摘要末句。

## English

Respiratory-sound datasets describe related events with different annotation units and label granularity. We study how sharing their supervision affects native-task performance and fine-attribute discrimination. A shared BEATs encoder learns abnormality, crackle, and wheeze from ICBHI and SPRSound, with readouts for their four-class and binary tasks. Under ICBHI-test-based checkpoint selection, the joint model obtains an ICBHI Score of 61.17 ± 0.31% and a SPRSound official Score of 90.70 ± 0.34%; the published ICBHI-only PAFA reference is 64.84 ± 0.60%. HF supports an additional-supervision study and attribute evaluation; KAUH tests external patient-level transfer. Separate validation-selected controls distinguish source sharing and annotation information from classification design, measuring native-task retention and the effect of the final readout.

## 中文翻译

不同呼吸音数据集描述了相关的声学事件，但其标注单元和标签粒度并不一致。本文研究共享这些监督信息如何影响原生任务表现和细粒度属性区分能力。共享的 BEATs 编码器从 ICBHI 和 SPRSound 中学习异常性、爆裂音（crackle）和哮鸣音（wheeze），并通过任务读出完成各自的四分类和二分类任务。在依据 ICBHI 测试集表现选择检查点的设置下，联合模型取得 61.17 ± 0.31% 的 ICBHI Score 和 90.70 ± 0.34% 的 SPRSound 官方 Score；已发表的 ICBHI 单数据集 PAFA 参考结果为 64.84 ± 0.60%。HF 用于额外监督研究和属性评测，KAUH 用于外部患者级迁移评测。另设基于验证集选模的对照，区分来源共享、标注信息和分类设计的作用，衡量原生任务性能保持及最终读出规则的影响。
