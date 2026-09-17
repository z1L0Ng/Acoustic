# PAFA与PC-MCL Frozen-Encoder设计任务

本brief已被9/16最新分工替代：PAFA交Hanlin运行，项目侧集中PC-MCL与DCASE。当前指令见[新设计brief](2026-09-16_frozen_pcmcl_dcase_design_brief_zh.md)。以下保留为此前设计范围的历史记录。

用户2026-09-16指令：本轮新增baseline全部由项目侧负责；Hanlin仅复跑论文已有baseline。调整Work Plan后，由代码设计任务完成PC-MCL与PAFA两个baseline的设计，放入frozen-encoder框架。

接收任务为Acoustic项目的“模型设计”（019fb42d-11d9-7b53-a6ae-d0ab010609c5），工作目录 /Users/zilongzeng/Research/Acoustic。请先读根AGENTS.md和最新Work Plan：
docs/work_plans/2026-09-15_to_2026-09-24_baseline_extension_plan_zh.md
Notion：https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8

## 本轮交付

完成两方法的frozen设计及必要配置/接口/代码准备，不停留在泛泛的可行性描述。先给出每个方案的encoder来源、哪些模块冻结/训练、loss作用路径、任务监督与读出、真实checkpoint和cache缺口、三seed含义、最少实现改动及后续执行入口；再给出明确推荐和仍需用户讨论的科学选择。

所有训练、模型前向、推理、feature extraction、cache构建、服务器执行保持未启动。本次只解除设计/实现准备的暂停，不解除Model Design实验运行的READY_FOR_USER_START边界。

## 先核查的现有资产

- baseline/four_dataset_frozen_encoder/
- baseline/common/frozen_encoder_target_heads.py
- baseline/pafa/frozen_encoder_target_heads/
- baseline/pafa/checkpoint_eval/
- result/four_dataset_pafa_frozen_encoder/run_manifest.json
- 当前稿件 docs/paper/Overleaf_Sync_final/Section/4tables.tex、3method.tex、4evaluation.tex，仅用于理解任务和比较范围。

旧代码有hash、profile-before-full、严格收据等历史要求，已被当前用户AGENTS覆盖。只读参考有用实现，不能调用这些旧入口或恢复这些门槛。

## 科学上必须明确的区别

现有PAFA frozen target-head代码冻结的是已经过PAFA训练的BEATs，并丢弃源classifier/projector后训练新下游头。另一种做法是冻结通用BEATs，再训练PAFA/PC-MCL风格的投影和头；这属于适配版本。请明确用户目标在现有管线下怎样落地，不把两者混称原法复现。

如果冻结通用BEATs，必须说明PAFA PCSL/GPAL和PC-MCL辅助任务究竟更新哪些可训练参数，是否与最终分类路径相连。作用于固定feature或完全独立辅助分支的loss，不能被描述为改善分类表示。不要只给普通BEATs frozen-head结果换名字。

PC-MCL多周期拼接必须区分原始音频拼接后编码与单段embedding拼接；两者不等价。给出输入长度、增强/拼接策略、缓存粒度与何时需要重新编码的设计，当前不实际提取特征。核对论文/代码的10 s对8 s、辅助权重0.1对默认0.5、enable_ssl默认关闭等差异，但不要把原full fine-tuning的400 epochs机械搬到新冻结设置。

PAFA有一个历史源任务checkpoint可查，seed/源监督/选模历史要如实记录；它不是三个原法独立seed。PC-MCL此前未找到训练完成的权重，如果方案需要它，必须显式列出依赖，不能偷偷插入完整源域fine-tuning，也不能用BEATs初始化冒充PC-MCL成品。

如果多个方案各有含义，提出最合适的一套并说明取舍，保留尚未决定的部分为未启动设计接口；不要一口气新增多组训练或替用户做无依据的科学选择。

## 任务与结果语义

目标是可用于当前frozen-encoder比较的设计，按每个数据集明确训练头、监督标签、选模和评测。不沿用旧计划“原法I-only三seed+zero-target外评”作为已经确定的当前方案。

固定encoder下三seed通常是下游头/适配模块的训练随机性；不能称为三个完整PAFA/PC-MCL端到端模型。目标标签用于训练下游头时，不能写zero-target-tuning。比较结果保持ICBHI Score、SPRSound official Score、HF CAS AUROC、KAUH patient BA的明确单位和标签语义；HF真实CAS训练与p_W proxy要区分，KAUH按患者聚合并保留兼容子集范围。

## 执行和协作

- 此次只负责PAFA/PC-MCL frozen方案，DCASE仍是项目侧后续事项，不下放Hanlin，也不自动启动DCASE设计/运行。
- 允许必要文档、配置、接口和代码准备；仅做直接相关静态检查，不做模型前向、smoke/probe/profile或大测试套件。
- 不改active论文、图、Notion或Git，不影响写作任务与既有结果；不重新跑Coarse/JH队列。
- 交付要给明确文件路径，区分设计完成、实现准备、资产缺口和未执行实验。完成、卡点或科学上重要的偏离，向管理任务01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae回报。若工具交接失败，在最终回复提供完整可复制说明，不绕过审批。
