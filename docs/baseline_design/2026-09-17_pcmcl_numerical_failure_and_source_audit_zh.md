# PC-MCL 5-s适配：数值失效与源码差异核查

日期：2026-09-17。状态：第二轮静态核查及代码整改已完成；没有新训练、推理、音频解码或模型forward。新运行继续暂停。

## 结论与当前证据

当前5-s适配没有成功，尚不能由此判断作者方法不可复现。三个seed都发生数值失效，但现有epoch日志没有保存首次异常batch的输入、梯度与优化器变化，因此无法唯一定位起因。仅延长到400轮没有解决失效。

| Seed | 首次NaN epoch | 最后完整epoch | 保存状态 |
|---|---:|---:|---|
| 0 | 35 | 400 | 早期有限best、非有限last、脚本生成的terminal/summary均保留 |
| 1 | 45 | 400 | 早期有限best、非有限last、脚本生成的terminal/summary均保留 |
| 42 | 30 | 40 | 用户批准中断；last保留，无best和完整summary |

服务器原始目录：`/files1/Zilong/Acoustic/result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_400epoch/`。原始文件不改写、不删除。seed0/1的complete标记仅说明旧脚本执行到了终点，不能证明训练数值有效。本轮不作为有效PC-MCL三seed基线。

两个已检查的失效last中，encoder约90.34M个元素及全部head参数出现非有限值。NaN与0.5比较得到false，旧C/W读出因此返回Normal；这种情况下的50% Score不是有效模型性能。两张GPU已释放，原15分钟heartbeat已暂停。

## 论文、公开脚本与本项目适配

核查采用[论文v1](https://arxiv.org/html/2601.17080v1)、[公开仓库](https://github.com/wa976/PC-MCL)及以下具体源码；没有执行作者脚本。

| 项目 | 已核对的作者描述/源码 | 本项目实现与判断 |
|---|---|---|
| 输入 | 论文§2.1先规范两半，§3.1使用10 s；CLI默认8 s；dataset脚本先拼接原长cycle，再对整体裁/补 | 用户指定2.5+2.5=5 s及单unit 5 s，属于明确适配，不能叫原配方完整复现 |
| 拼接采样 | 脚本使用短于T/2的候选池，并含三段混合分支；C/W候选池也包含Both | 当前为两段、原生互斥类池、真实label OR，使用全部可配对cycle；采样组成不同 |
| patient身份 | official split分支用录音文件名作patient key | 当前用manifest的实际patient_id；语义更贴近论文，但不同于该源码分支 |
| patient配对 | 正例偏向录音内相邻/非相邻cycle；负例依次搜索病理组 | 当前同patient内均匀配对，在可用病理profile间随机选负组；构成不同 |
| loss | 论文为主BCE加0.1倍辅助CE；脚本实际采用(1−w)主loss+w辅助loss，默认w=0.5，辅助任务默认关闭 | 当前遵循论文加法公式并启用辅助任务；不能把公开默认值视为完整论文命令 |
| SpecAugment | 脚本在p≥标准正态随机数时增强；p=1时约84.1%，宽度不含上界，time mask重新取均值 | 整改前为每batch必增强、宽度包含上界、使用一次均值；当前已对齐这些执行语义 |
| 优化日程 | 公开默认Adam 1e−3、wd1e−4、400轮、120/160下降；warmup和EMA需显式启用 | 当前与这些优化默认值对齐，未启用warmup/EMA；这些默认值是否产生论文结果仍未确认 |
| 本域选模/读出 | official-test Score严格改善且Se>0.1%保存best；C/W阈值0.5还原四类 | 当前数值单位和读出一致；保留test-selected标识，未改阈值或资格 |
| 骨干/环境 | 可见wrapper与优化器路径未冻结BEATs；入口追加仓库外models路径并导入仓库未包含的CNN6 | 当前复用P2的BEATs包装及核心。逐行对照core的差异仅为作者多出的可选FreqMixStyle分支，其默认关闭；未发现默认路径的骨干数学差异 |

具体来源：[训练/参数入口](https://github.com/wa976/PC-MCL/blob/main/main.py)、[dataset与采样](https://github.com/wa976/PC-MCL/blob/main/util/dataset_pcmcl.py)、[音频裁补](https://github.com/wa976/PC-MCL/blob/main/util/icbhi_util.py)、[增强](https://github.com/wa976/PC-MCL/blob/main/util/augmentation.py)、[骨干入口](https://github.com/wa976/PC-MCL/blob/main/models/__init__.py)、[BEATs核心](https://github.com/wa976/PC-MCL/blob/main/models/BEATs/BEATs.py)、[BEATs包装](https://github.com/wa976/PC-MCL/blob/main/models/BEATs/beats.py)。

另一个源码语义问题：C/W池包含Both，但部分混合分支按组合模板赋标签。按可见代码路径推导，抽中Both时可能丢失另一属性，与论文逐元素OR公式不一致。这是静态推导，不是新实验结果；不建议为追求逐行相同而自动复制这种标签行为。

核查时已区分大小写文件名：作者同时提供`BEATs.py`和`beats.py`，在macOS默认文件系统上会冲突；改用独立本地名称后确认核心类存在，不能把本机读取覆盖误当作者缺少核心实现。

上述差异能解释为什么本次不是严格原配方复现，不能证明任一项就是NaN的唯一原因。学习率、增强与优化设置仍是待验证因素；本轮没有偷偷降低LR、加warmup、改loss、改采样或改变选模。

## 已完成的最小代码修复

- 每次更新前分别检查main/patient/total loss和每个参数的gradient；遇非有限值抛出异常，以非零退出码终止。有限梯度不裁剪、不改变大小。
- 检查推理logits及ICBHI/HF/KAUH读入概率，阻止NaN被转成Normal，或Inf经sigmoid成为貌似有效的0/1。
- 在现有`run_summary.json`中记录`failed_nonfinite`，并保留stage、seed、epoch、batch、stable sample IDs、main/patient loss和learning rate；若gradient异常，另记录首个异常参数名。不继续终点评测，也不覆盖上一完整epoch的checkpoint。
- 拒绝从已知非有限的训练历史恢复；队列不会跳过失败后继续补跑。
- 汇总检查旧train log与summary中的所有数值字段，排除带NaN/Inf的伪complete运行，并列出原因；保留原始summary/log。DCASE路径保持原有行为。
- 将本项目SpecAugment改为公开`icbhi_ast_sup`执行语义：`p=1`与标准正态随机数比较（并非每batch必增强）、mask宽度上界不包含、frequency mask后重新计算time-mask mean。没有引入新增强策略。
- patient hard negative恢复已批准的默认：选择总体病理profile相同的两个不同真实patient，再分别抽取cycle；positive仍是同一真实patient内的两条cycle。实际cycle native class强制相同只保留为待用户决定的采样候选，没有进入正式默认。现有证据不足以判断两种条件是否存在或消除病理捷径。

文件：`baseline/frozen_method_baselines/pcmcl_numerics.py`、`pcmcl_source_runner.py`、`pcmcl_source_transfer.py`、`source_transfer_queue.py`、`source_transfer_summary.py`。

## 验证与下一步判断

9项直接单元/错误处理检查通过：有限读出不变；NaN/Inf预测拒绝；NaN loss更新前停止；有限loss产生无限梯度时停止并定位参数；有限标量更新不变；SpecAugment随机gate与time/frequency轴；hard-negative patient-profile匹配；旧伪complete排除及恢复/队列拒绝；失败状态、结构化诊断与既有checkpoint保留。只使用标量/小tensor和临时metadata，未实例化BEATs或运行训练流程。

检查命令：`/opt/anaconda3/envs/Beats/bin/python -m unittest discover -s tests -p test_pcmcl_numerics.py -v`。

这些修复防止数值失效后继续消耗GPU和误收录结果，并使SpecAugment执行语义对齐公开代码；patient-profile默认合同保持不变。它们不证明收敛问题已经解决。

## 仍需用户决定的科学配方

旧`pcmcl_source_run.json`及其`PC_MCL_ICBHI5s_400epoch`输出目录继续作为失败配方历史，不改写成新默认。公开CLI的Adam `1e-3`是源码默认值，不是论文正文给出的完整成功命令；三个seed都在第一次120轮降LR之前失效，因此当前最直接的解释候选是full-BEATs更新过激，但尚无首次异常batch诊断可以证明因果。

若用户批准新的正式单seed，建议使用新输出目录，并从以下未批准候选中另行冻结一个；它们用于稳定训练，不构成对旧失败原因的严格因果检验：

1. **首选前瞻稳定训练候选：全模型Adam lr从`1e-3`降为`1e-4`。** 不是已批准默认值。由于当前代码同时已修正SpecAugment执行语义，未来“新代码+`1e-4`”相对旧失败运行不构成严格单因素对照；即使训练成功，也不能据此证明旧NaN由LR单独导致。
2. **次选候选：encoder `5e-5`、两个随机初始化head `1e-3`。** 它更符合预训练骨干/新head的不同更新尺度，但新增参数组，归因不如候选1简单。
3. **低优先：保留`1e-3`并加10轮linear warmup。** 失效发生在30–45轮而非启动阶段，单独warmup的解释力较弱；EMA只能平滑评测权重，不能阻止训练参数本身变成NaN，均不建议作为第一次修复。

gradient clipping、跳过坏batch、`nan_to_num`与自动重试不进入候选。当前代码已具备“新配方获批后执行一个正式seed，并在首次非有限值处立即终止和定位”的工程条件；**收敛恢复、模型有效性和三seed放行仍为HOLD**。

建议保持暂停。先明确采用论文语义适配还是作者某一完整实验命令，重点确认实际输入长度、辅助loss权重、学习率/EMA/warmup、patient key和拼接策略。可向作者索取完整命令、实际BEATs来源、一个成功seed的日志/训练后权重；当前未联系作者。

若继续本项目5-s方案，应把它明确标为paper-guided adaptation。方案确认后，先完成一个正式seed以验证收敛及本域表现，再决定补其余seed；这不是smoke，也没有在本轮获准启动。SPR/HF/KAUH结果不用于挑参数或readout。尚不能承诺修复所需训练时间。
