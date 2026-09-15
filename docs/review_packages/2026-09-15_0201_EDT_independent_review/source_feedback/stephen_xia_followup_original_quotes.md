# Stephen Xia 后续反馈：用户转交的英文原文

来源：2026-09-14_stephen_xia_feedback.md中的引述段。本文件只抽取用户转交的原文；未包含该文件中写作任务自行拟定的“完整重构思路”。

以下按原顺序保留；文中示例不自动构成对所有已有方法的事实判断。

> I think you want to structure the introduction something like 1) existing methods that leverage multiple and labels to enhance performance are labor intensive because you need to map to the same classes, perform data augmentation, etc., 2) we reduce this burden by aiming to leverage native labels from each dataset to enhance training, 3) we propose X to accomplish this.


> 1. you kind of have, except instead of just listing each work and describing them, I would categorize them into types of approaches (eg data augmentation, target transfer learning, etc)For 2) I would change the end of the second paragraph and say a statement that summarizes the drawback of all existing approaches (eg burdensome to augment or map to the same set of classes). Then in the third paragraph, I would refactor it. Instead of saying “our method is different because of x”, I would say, “we aim to solve problem x”, which was just described at the end of the second paragraph

> Then the rest of the second paragraph can be similar to what you have where you’re describing how each dataset, while having different labels, observe the same type of signal and can be used to augment each other without heavy processing like existing work
