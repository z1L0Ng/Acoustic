# Hanlin核心对照任务｜待确认草案

状态：规划材料，尚未发送；不是立即执行指令。GPU与环境信息待确认，CUDA适配和完整运行包尚未交付。

Hanlin，你好。这一轮拟请你执行一组新的完整核心对照，和此前2s frozen-encoder背景实验分开。请先确认可用GPU型号、显存和是否方便接收这批任务，收到完整运行材料前不要开跑。

计划中的三项是：

1. Full：ICBHI＋SPRSound联合训练。
2. ICBHI-only。
3. SPRSound-only。

每项使用seed0、1、42，共9个run，保持同一CUDA环境、同一初始化、固定分组、训练设置及validation selection。Full必须与两个single-source一起完成，不能用别的机器或旧实验的Full数字替代参照。

这次是5s输入、BEATs full fine-tuning的新控制实验，不是旧frozen表的追加seed。单源模型只用本源训练/校准/选模；selected checkpoint确定后，仍按提供的脚本报告两个数据集的固定模型结果。

我们会单独提供：

- 可运行代码和依赖/环境说明；
- BEATs初始化checkpoint及两源数据入口约定；
- 固定split manifest与完整validation protocol；
- 准确命令、物理batch32/FP32要求；
- 输出示例：config、分组、日志、checkpoint、逐样本probabilities、native指标及C/W AUROC。

请不要自行改变split、标签、窗口、head、loss、batch、selection或训练预算。若显存不足，不自行改成小batch梯度累积或AMP；先说明具体问题，由我们统一调整设计。不要运行smoke、模型probe或哈希验证，也不要影响其他人的GPU任务。

每项请保留完整原始输出与失败说明，不能只交最终表格。原frozen结果可继续作为背景，但不能替代这组matched controls。

这份草案不设个人固定截止时间；项目方会在确认资源与分工后再给出正式启动安排。
