# PC-MCL源模型迁移与DCASE冻结方案｜2026-09-16

本轮已进入完整脚本实现，最新范围见[实现与本地预算brief](2026-09-16_source_baseline_implementation_brief_zh.md)：DCASE也改为ICBHI-only原生四分类后固定迁移；下文早期DCASE联合标签候选保留作历史参考，不是当前指令。两组均未获实验启动授权。

PC-MCL最新用户决定：ICBHI-only源训练，2×2.5 s波形拼成5 s，训练后整个模型固定，直接测试SPR/HF/KAUH。本决定替代下文此前的PC-MCL joint/frozen候选讨论；DCASE段落继续作为待讨论设计。当前仍只允许设计与必要代码准备，不启动实验。

用户最新调整：PAFA交给Hanlin运行；项目侧只集中重新规划PC-MCL和DCASE。Hanlin已有AST、BEATs、PANNs、OPERA-CT、HeAR五组复跑继续。本文件替代本日较早的PAFA/PC-MCL双方法设计brief。

接收任务：Acoustic“模型设计”（019fb42d-11d9-7b53-a6ae-d0ab010609c5），main目录 /Users/zilongzeng/Research/Acoustic。
先读根AGENTS.md与 docs/work_plans/2026-09-15_to_2026-09-24_baseline_extension_plan_zh.md。
Notion：https://app.notion.com/p/3dc309efda29810b84c1c7fd3fd21ad8

## 本轮交付

按已确定的PC-MCL单源固定迁移方向修订设计，DCASE继续提供待讨论的冻结方案。PC-MCL的三seed指三个独立ICBHI源训练模型，每个模型评测全部四列；不为三个目标集训练新模型。允许必要配置/接口准备，未确定的DCASE路线不全面实现多个变体。

不继续扩展PAFA代码，不处理本机PAFA环境，不启动训练、forward、推理、feature extraction、缓存、profile、smoke、服务器或全量数据下载。既有明确授权任务不受影响。不得改主稿、Notion或Git。

## PC-MCL：ICBHI-only 5-s源模型与零目标适配评测

依据用户最新确认和Work Plan第2节，修订PC-MCL设计：保留原方法raw concat、N/C/W additive targets、patient-matching和源模型学习，源训练只使用ICBHI。两个cycles各repeat-pad/center-crop到2.5 s后拼成5 s，单unit验证/测试也为5 s。明确这是相对原文10 s的输入适配。

源训练按原机制准备encoder与预测头优化，训练完成后固定全模型，直接评测SPRSound、HF、KAUH。目标数据不训练新头、不选checkpoint、不拟合阈值；禁止旧ICBHI+SPR validation composite。ICBHI源分区、选模依据和优化设置要与原法及项目已确定的数据/选模口径核对并列明，不擅自创建新的validation协议。

准备固定SPR四类到二类映射、HF maximum-window p_W proxy、KAUH按患者B/D/E概率聚合后的固定正常/异常读出；阈值、聚合顺序与目标支持量处理须预先明确，不读目标表现后挑规则。保留逐样本N/C/W分数和预测，便于区分属性判别与任务读出问题。

三seed指三个独立ICBHI源训练模型，每个模型覆盖四列。先静态核对是否有真实可复用checkpoint；没有则准备源训练方案和预算，不能把generic BEATs权重当成PC-MCL成品。旧frozen cache+0.20M adapter预算不适用于此路线，不要把现有缓存直接用作会更新encoder的源训练输入特征。

主比较是PC-MCL与LSAA主方法：ICBHI上的多标签重构与输入拼接，能否使PC-MCL单源模型直接迁移到SPR/HF/KAUH；与采用多数据集联合学习的LSAA相比表现如何。训练源差异是本组比较的设定，不要求消除，不把LSAA的ICBHI-only消融作为主比较或必要参照，也不增加新控制实验。已有I-only结果仍留在原有消融中。

科学解释不预设迁移失败。先报告改为5 s后的本域效果；跨任务表现用于评价完整方法，不单独归因于N/C/W或某个loss。输入2×2.5 s→5 s、ICBHI-only源训练和固定全模型外评的方案不变。新规格使用单独的source-transfer文件，旧共同规格中的PC部分标为已被用户决定替代。

本轮只做必要规格、配置、接口/代码准备与直接相关静态检查。不运行source训练、模型forward、特征/缓存、验证测试或服务器，不改主稿、Notion或Git。

## DCASE：以官方frozen路线做呼吸音适配

官方入口：https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline

官方README已核对：frame BEATs embeddings冻结，与CRNN输出进行时序融合后接RNN/MLP；loss与attention对缺失标签mask，基于Mean Teacher。进一步核对代码中实际可训练模块和呼吸音需要改动的接口。

提出一套呼吸音方案：训练源、统一或native标签空间、未知标签监督、cycle/event与时间标注的关系、attention/pooling及四列readout。明确是否保留CNN/RNN/Mean Teacher，任何省略都说明理由与方法命名。不能只做普通frozen-BEATs head并叫DCASE，也不能把环境声预训练分类输出直接填呼吸音表。

检查需要frame features还是可复用现有资产；5-s pooled embeddings默认不等价于原frame序列。只列出提取规格和资源预算，不执行提取。

## 共同口径与交付格式

当前目标列：ICBHI Score、SPRSound official Score、HF CAS AUROC、KAUH patient BA。
上一轮新validation、HF dedicated CAS head、KAUH目标监督五折OOF仍是候选，不是默认Table1协议。不要自行切换。给出训练信息和指标对齐方案后由用户讨论。

交付一份紧凑方案，包含：每个baseline回答的问题、冻结权重/可训练模块、输入/监督/选模/readout、与原法保留和变更清单、最少文件改动、已有/缺失资产、三seed真实含义与资源估计、推荐推进顺序和必要选择点。预算应把特征准备与训练拆开，不用未运行的profile推测精确工时。

原 docs/baseline_design/2026-09-16_frozen_pafa_pcmcl_spec_zh.md 及相关代码保留作为前版资产。请把新方案写在单独的PC-MCL/DCASE设计文件中。完成后向管理任务01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae交付，区分设计、实现准备与实际结果。
