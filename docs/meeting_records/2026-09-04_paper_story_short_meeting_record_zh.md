# 2026-09-04 Paper Story / Contribution 短会记录

来源：用户在 2026-09-05 提供的录音讨论中文整理。本文按其提供的对话顺序记录，不声称重新听取或独立核验原始录音。日期按用户所说“昨天”记为 2026-09-04，具体会议时间未确认。

## 1. 两页 slides 与最初的 story

讨论开始时，老师在看当时展示的两页快速 update：第一页讲 paper story，第二页讲希望展示的 conclusion。用户解释的逻辑是：先呈现各 dataset-specific models 在其他 datasets 上表现不佳、transfer/adaptation 受限的观察，再进入 gap、提出的方法，最后用 experimental evidence 支撑。

用户说明，列出的三条 contributions 是前一晚整理的初步 ideas，尚不一定是最终贡献，wording 也可以继续调整。

## 2. 老师对原 contribution 描述的反馈

用户解释，不同 datasets 有自己的 native tasks / test formulations，同时又共享部分 labels，因此希望把它们纳入同一个 task/framework。随后介绍两个层级的 classifier：Normal/Abnormal，以及 abnormal 一侧更细的类别，并提到各数据集的 ablation、独立 test 和 analysis。

老师指出：“I don’t feel like this is a cool contribution.” 他希望 contribution 表达工作的价值，而不只是实验步骤或组件：

- “Contributions are the vision of what you have done.”
- “Contribution should be something you figured out or something merit-worthy.”

用户回应，这只是快速整理的 update，会继续深入思考。

## 3. 从结果中提炼发现：gap identification

老师提出，能否从当前 results 本身提炼 contribution。他希望核对目前观察到的问题是否已经在 literature 中出现：其他人是否已经清楚观察过，还是本项目尝试不同 baselines 后才识别和总结出来。

老师强调：“The identification of the gap is also a contribution.” 这是一条有条件的方向：如果系统实验识别了 literature 尚未清楚揭示的问题，那么识别该 gap 本身可以构成贡献。讨论没有据此确认本项目已经具有该新颖性。

## 4. 方法带来的能力：awareness architecture

老师提出，awareness architecture 也可能构成另一条贡献：它使异质数据集能够被 harmonize 到统一体系中，并得到能够适应不同 datasets 的模型。

老师举例，一个统一模型若能在四个 datasets 上取得与各自 single-dataset models 可比的 performance，而且 literature 中尚未实现这样的能力，就可能形成有价值的 contribution。

这段表述是用于讨论 merit 的例子和条件，不是会议中确认的四数据集实验结论、最终主文范围或新的实验启动指令。

## 5. 从图表描述上升到有边界的 claim

老师提出核心问题：

“What is something you have done that could not be done without your methodology?”

他认为原 slides 的 contribution 更像 figure/table captions，即准备往图表中放什么。需要进一步抽象成一般性的 claim，回答工作的 innovation 和 merit：

- “What you want to show me looks more like figure and table captions.”
- “What you want to put in these tables.”
- “You want to abstract the contributions in a general way to make the claim.”
- “What is innovative?”
- “What should be the merit for this paper?”

同时老师明确要求：“But do not overclaim it.” 不能因为有限场景的结果而宣称“one of the best”或解决广泛问题的 adaptive framework。应当做到：“Make sure it’s scoped, but you can clearly state the merits.”

## 6. 下一步写作与讨论安排

用户表示，将核对已有 numbers 和 figures，继续 draft paper 和 figures。

老师表示大概下周二还可以继续讨论，并希望在此之前先准备内容发给他或 Erin。按本次记录日期，“下周二”对应暂定 2026-09-08；具体安排尚未确认。

老师强调当前重点仍是 story 和 paper skeleton：“This is mostly how we want to tell the story, like the current skeleton of the paper.”

记录的是老师提出的准备与沟通安排，不代表本任务已经发送材料或获得代发授权。

## 7. Collaboration 与后续资源

老师提到可以继续让相关协作者了解进展，保持 collaboration，以便未来从 School of Medicine 获得更多资源，包括 actual recordings 和后续 implementation。这属于后续合作方向，讨论未给出本周期新的数据采集或实现任务。

## 8. 结束前的强调

老师再次提醒，claim contribution 的方式与用户原先偏向“做了什么组件、分析和表格”的理解不同，应该回答通过研究发现了什么、解决了什么，以及 methodology 使什么成为可能。

老师强调截止日期临近，需要加快论文推进：“We’re almost at the deadline. We need to move faster to get it pushed out.” 随后讨论转向 Sora 的项目，本文不记录其内容。
