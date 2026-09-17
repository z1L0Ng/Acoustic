# 【已失效】给Hanlin的PC-MCL与DCASE任务草案

9/16用户最新策略已替代本草案：Hanlin不再负责新增PC-MCL/DCASE任务，改为PAFA加论文已有五组baseline复跑。见[PAFA交接清单](2026-09-16_hanlin_pafa_handoff_zh.md)和[已有五组说明](2026-09-16_hanlin_existing_baseline_rerun_zh.md)。以下内容仅保留历史，不是当前执行指令。

状态：供用户审阅和后续转发，尚未发送；不是直接启动命令。DCASE适配方案与正式运行代码包尚未完成确认，不能仅凭本说明开跑。

Hanlin，这一轮拟请你完成两组baseline，每组三个独立seed，共六个正式run。目标是在你自己的电脑可连续运行的条件下，一周内完成训练、评测和结果交回。PAFA由我们本地另行完成。

## 先确认资源

请先告诉我们电脑的GPU型号、显存及可用加速后端。可以只提供GPU信息，不需要任何私人账号、机器登录信息或远程访问。如果只有CPU、显存不足或预计无法在一周内完成，请说明，我们统一调整资源或范围。

## 第一组：PC-MCL，三个独立seed

官方代码：https://github.com/wa976/PC-MCL

优先按原法在ICBHI上训练，之后固定模型，按提供脚本评价ICBHI、SPRSound、HF与KAUH。其余三个数据集不参与这组模型的训练、模型选择或阈值调整。

我们会给最终配置：包括多周期输入、Normal/Crackle/Wheeze三标签、patient-matching任务、损失权重、训练预算与选模规则。论文设置和CLI默认并不完全一致，所以不要直接用仓库默认参数启动。种子预定0、1、42；不能重复运行同一个checkpoint来代替三次训练。

## 第二组：DCASE风格的呼吸音异构监督适配版，三个独立seed（待确认）

参考代码：
https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline

DCASE原始checkpoint输出环境声类别，不能直接作为呼吸音baseline。项目方会先给你确定好的ICBHI+SPRSound适配代码、标签映射、missing-label规则、模型/训练设置及四项评测读出。你不需要自行设计这些科学规格，也不要把已有frozen-BEATs结果改名为DCASE。

这一组的最终名称、范围和代码包确认后再启动。若仍未就绪，可先完成PC-MCL。

## 我们需要提供给你的完整运行包

- 固定版本的代码和简洁环境说明、准确命令；
- 官方BEATs初始化权重来源及本地路径约定；
- 四数据集的获取/目录说明、训练与评测划分及样本支持；
- 标签映射、单位/窗口处理和固定读出实现；
- 每个seed独立输出目录、日志、checkpoint与预测保存；
- 统一计算ICBHI Score、SPRSound official Score、HF CAS AUROC、KAUH patient BA的脚本；
- 三seed汇总和结果交回格式。

## 执行与交付

按照正式包执行，不自行更换seed、窗口、batch、loss、选模规则或目标测试集。若显存/时间不满足要求，报告具体问题再由我们调整；保留已经落盘的数据和失败日志，不删除不利结果。

保留每个seed的配置、完整日志、选择的epoch/checkpoint、逐样本分数与预测、指标/混淆矩阵和实际用时。最后交回完整结果文件，不只交Table1的一行数字。

两组共用一周的总预算，不是每组各一周。初始计划按同机串行；从正式训练的正常日志估算剩余时长，不另外运行smoke/probe/profile或hash/checksum检查。先完成清楚可复现的PC-MCL，不为填表仓促改变DCASE定义。

正式包未交付前，请先回复硬件信息及是否能承担这两组任务。
