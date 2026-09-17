# PC-MCL Source Transfer与DCASE Joint Native-Union实现及本地预算｜2026-09-16

用户最新指令：让代码写作任务完成对应脚本，并估算本地需要的时间。
接收任务为Acoustic“模型设计”019fb42d-11d9-7b53-a6ae-d0ab010609c5，main目录 /Users/zilongzeng/Research/Acoustic。

当前规范说明：DCASE ICBHI-only flat4条款已被九输出ICBHI+SPRSound native-union方案替代；当前代码见`dcase_joint_union_runner.py`。本任务的Git提交授权已由用户明确给出并已完成，下面授权段中的“不改Git”只记录前版实现阶段边界，不再是当前交付状态。

最新预算修订：PC-MCL与DCASE均为max50 epochs、patience10、min_delta0的strict-improvement早停；tie保留较早best并计入无提升。下文原400/100轮要求是历史，已被此修订替代。

## 授权与交付

现在明确授权完整脚本实现及必要直接检查，不再停留在设计/最小接口层。当前不授权source training、模型forward、feature/cache构建、validation/test/target inference、smoke/probe/profile或服务器运行，不安装/改动共享运行环境，不改主稿、Notion、Git。不存在运行授权时不能用短跑估时。

完成两个方法从数据读取、模型/损失、训练、checkpoint、固定评测到三seed串行队列与汇总的脚本；提供精确运行命令。没有预先确认的常规epoch/lr/SpecAugment选项做成显式配置，注明默认依据，不能再次因为这些可配置选项而只交接口。不得私自缩短已定义recipe来让时间看起来可行。

代码保持简单；不新增artifact identity verification或大量receipt/gate文档。训练输出包括config、日志、best/last（含optimizer/epoch以便恢复）、逐样本或患者分数、预测、支持量/混淆矩阵和三seed mean/sample SD。未完成的seed不得汇入n=3。读取既有训练日志和元数据用于预算允许；不运行实验来估计。

## PC-MCL：已确认设计

研究问题与主比较：PC-MCL只在ICBHI学习N/C/W、多周期拼接与patient matching后，直接迁移到SPR/HF/KAUH；主比较是ICBHI+SPR联合LSAA，不增加LSAA I-only匹配消融。

参考 docs/baseline_design/2026-09-16_pcmcl_icbhi5s_source_transfer_spec_zh.md，但实现不能继续停在其“recipe批准前不接runner”的旧状态——用户已要求完整脚本准备，正式运行仍另行授权。

- 三source seeds 0/1/42，ICBHI official source pool；目标集不进入训练或选模。
- 两个cycles各repeat-pad/center-crop到2.5s再在waveform域concat到5s；single validation/test/transfer为5s。
- BEATs encoder与N/C/W、patient heads在源训练中更新；随后固定全模型和readout。
- 保留实际raw concat、additive N/C/W及patient task；loss BCE(N/C/W)+0.1 CE(patient)，不能误用源码convex mixture或只改desired_length而未落实各2.5s。
- 沿用已核查的ICBHI source benchmark选模定义，明确test-selected；不用任何SPR composite或新target calibration。
- PC四类按固定C/W阈值转换；SPR flat4→Normal/Adventitious；HF三个5s窗口max p_W；KAUH每view max(p_C,p_W)，B/D/E均值后0.5作一次patient判断。目标readout严格按已给规格，不根据目标结果试多种择优。
- 本地无真实PC-MCL checkpoint，因此脚本必须覆盖source训练，不冒充只需target推理。generic BEATs是初始化。
- 当前正式预算为max50 epochs+early stopping；PC milestones按相对位置改为15/20。历史400轮只作来源说明，不是当前执行配置。

## DCASE-inspired Native-Class-Union CRNN：当前批准方向

DCASE使用ICBHI+SPRSound联合多标签训练。固定九输出为`N_ICBHI,N_SPR,Crackle,Wheeze,Both,FineCrackle,CoarseCrackle,Rhonchi,Stridor`；两个Normal独立，Wheeze/Both跨源共享，Fine/Coarse对Crackle-only提供单向positive alias，R/S独立。名称使用 DCASE-inspired Native-Class-Union CRNN (ICBHI+SPRSound, 5 s)，不声称复现DCASE 2024。

- Frozen AudioSet BEATs temporal frames + trainable log-Mel CNN、fusion、BiGRU、官方class-wise attention pooling。
- Sigmoid frame outputs；attention在class轴softmax；clip probability按时间加权并除以每类attention和。
- Provider输出target[9]/mask[9]；loss按所有eligible sample-class元素统一mean BCE，不做LSAA式节点重加权。
- 复用现有seed-specific ICBHI grouped subtrain/validation与固定SPR grouped split；official tests仅selected terminal。
- Selection=`0.5*ICBHI native4 validation macro multilabel-F1@0.5 + 0.5*SPR native7 validation macro multilabel-F1@0.5`；alias Crackle不重复进入SPR native7；zero_division=0。
- batch32；每epoch163个ICBHI与163个SPR homogeneous batches；max50、patience10。
- ICBHI native4 scores argmax；SPR native7 scores argmax后binary regroup；HF max(Wheeze,Both,Rhonchi,Stridor)再跨三窗口max；KAUH先mean B/D/E四类scores再argmax regroup。
- Mean Teacher/unlabeled、strong frame loss、mixup与PSDS不进入当前实现。

当前入口为`dcase_joint_union_runner.py`与`dcase_joint_union_run.json`；前版ICBHI-only runner/config标历史并拒绝当前执行。

## 共享边界

两方法都完整输出Table1四列：ICBHI Score、SPRSound official Score=(AS+HS)/2、HF CAS AUROC、KAUH patient BA。HF CAS阳性W/R/S、D-only阴性；wheeze-derived输出只是CAS ranking proxy，不是训练了CAS新头。患者/录音/窗口单位与目标支持量要沿用已接受定义，不使用target真值决定模型输入class mask。四类模型的Wheeze分数包含Both，不能只取Wheeze互斥类概率。

使用已知canonical数据/providers/scorers的必要部分，避免继承旧identity/profile gate。若已知SciPy导入问题会阻止脚本，优先作最小依赖隔离或lazy import，保留数据语义；不可擅自安装、升级整个环境。无法解决的运行依赖要如实列明，但继续完成所有能完成的代码。

## 本地时间估算

基于当前Mac硬件信息及已经完成的本地BEATs/JH2/HF训练、既有feature extraction日志，不做新profile或forward。分别给：
1. PC-MCL每epoch、max50每seed及三seed满跑上界；早停只作为可能缩短，不预设停止epoch；
2. DCASE frame extraction一次性成本、每seed head/CRNN训练、三seed、external evaluation；
3. 两方法本地串行的总预算区间，best/base/conservative情景与关键假设；
4. 哪些量有历史实测支持，哪些只是静态估算；无可信吞吐依据就给宽区间或明确不能精确估计。

不假设两张L40或任何服务器资源可用，不把MPS可用等同已实测当前recipe速度。若超出投稿时间窗口，直说，不隐瞒或改recipe。

## 完成标准

交付可定位的完整脚本/config/README、精确未执行命令、直接相关静态/纯函数检查结果、本地成本估算及剩余真实运行限制。完成后向管理任务01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae报告。不得把脚本完成写成实验完成，也不触发写作任务自动同步。
