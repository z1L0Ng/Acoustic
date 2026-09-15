# Stephen Xia 反馈收集

用户要求：完整接收反馈后，再逐一讨论。当前已获准讨论收到的建议；稿件修改仍待讨论确认。

状态：用户确认目前收到的反馈已全部转交，开始逐条讨论；Stephen 后续如有补充，可继续追加。

## 第一部分：Introduction 的结构与问题衔接

### 用户转交原文

> I think you want to structure the introduction something like 1) existing methods that leverage multiple and labels to enhance performance are labor intensive because you need to map to the same classes, perform data augmentation, etc., 2) we reduce this burden by aiming to leverage native labels from each dataset to enhance training, 3) we propose X to accomplish this.
>
> 1. you kind of have, except instead of just listing each work and describing them, I would categorize them into types of approaches (eg data augmentation, target transfer learning, etc)For 2) I would change the end of the second paragraph and say a statement that summarizes the drawback of all existing approaches (eg burdensome to augment or map to the same set of classes). Then in the third paragraph, I would refactor it. Instead of saying “our method is different because of x”, I would say, “we aim to solve problem x”, which was just described at the end of the second paragraph

### 要点记录（仅转述）

1. 建议 Introduction 按“已有多数据/标签利用方式及其工作负担 → 利用各数据集原生标签、降低负担的目标 → 为此提出的方法”展开。
2. Related work 按方法类型组织，例如 data augmentation、target transfer learning，而不是逐篇罗列。
3. 第二段末尾明确概括已有方法的局限，例如数据增强或统一类别映射所需的工作。
4. 第三段从“我们要解决前述问题”出发，引出设计，而不是先说“我们与已有方法不同”。

以上为 Stephen 的建议原意，不代表已经确认文献普遍具备这些局限，也不代表已经决定以降低工作负担作为本文主张。

## 第二部分：共同声学信号与监督互补（补充原文）

> Then the rest of the second paragraph can be similar to what you have where you’re describing how each dataset, while having different labels, observe the same type of signal and can be used to augment each other without heavy processing like existing work

要点：保留第二段中不同标签指向相关声学现象的观察，用它连接文献讨论与跨数据集监督互补。`augment each other` 的拟议理解是补充监督信息，并非已决定采用音频合成增强；`without heavy processing` 的具体含义仍需与实际标签映射工作相符。

## 完整重构思路（供用户讨论，未改稿）

1. 第一段：自动呼吸音识别的需求、专家标注成本、复用已有数据的机会，止于标注粒度与任务目标不同的困难。
2. 第二段：以如何组织共享监督为比较轴，对现有工作按共同目标映射、声音组成监督、异构/缺失标注学习分类。随后用不同标注指向相关声学现象及 Wheeze 例子，提出复用细标注并保留不同原生输出的问题。
3. 第三段：从上述问题出发提出 LSAA，说明共享 A/C/W 属性、可用标签监督和任务读出如何分别回应问题，再交代四数据集评估范围。
4. 贡献维持用户此前确认的三项职责：框架、读出设计、评估与迁移；不列具体性能数字。
5. 不把减少人工工作量或完全无需映射作为已验证贡献。LSAA 仍需标签到属性的语义映射，能明确主张的是复用既有监督、保留原生任务，而非把所有输出统一成同一分类目标。
6. 共同声学现象不等于同分布或域不变；共同目标映射也不必然丢失所有细标签。后续措辞保留这些边界。

当前仅汇总并提出结构建议，等待用户讨论确认后再修改 Introduction。
