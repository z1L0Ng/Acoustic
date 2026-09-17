# Hanlin PAFA交接清单｜2026-09-16

状态：用户最新将PAFA运行安排给Hanlin；既有AST、BEATs、PANNs、OPERA-CT、HeAR五组复跑保留。本文件供用户与本科生对接任务准备材料，尚未直接发送给Hanlin，也不代表实验已启动。

## 分工

Hanlin负责PAFA三seed及已有五组复跑。PC-MCL与DCASE由项目侧设计和运行。项目侧负责固定科学口径、准备必要材料、核验输出和回填paper，不让Hanlin自行猜测未明确的任务定义。

## PAFA交接前需要明确

- 采用原方法三个独立源训练seed，还是同一PAFA-trained encoder上三个下游训练seed；二者结果名称和随机性不同。沿用此前frozen讨论作为候选，负责人变化不等于这条路线已确认。
- 冻结的权重、可训练模块、数据/患者划分、输入、实际训练任务、选模规则及四列readout。
- HF/KAUH是固定源模型外评还是目标监督适配；不能默认采用上一版的新validation、dedicated CAS或KAUH五折OOF。
- Hanlin设备上的环境和资产路径需实际核对。本机SciPy失败不能用于判断他的设备。

## 已有可交接资产

- 设计与来源说明：docs/baseline_design/2026-09-16_frozen_pafa_pcmcl_spec_zh.md
- 准备中的代码：baseline/frozen_method_baselines/；PAFA已有实现参考 baseline/pafa/frozen_encoder_target_heads/
- 历史checkpoint：.cache/checkpoints/pafa/server_epoch27/best.pth
- 相关BEATs文件：.cache/checkpoints/pafa/server_epoch27/BEATs_iter3_plus_AS2M.pt
- 现有5-s特征：.cache/four_dataset_pafa_frozen_encoder/embeddings.npz

以上是项目本地路径，不表示Hanlin已经收到这些文件。一个历史encoder状态不能作为三个源训练seed；其ICBHI official-test selection与源seed未闭合的事实要随使用场景说明。代码是候选协议下的准备，不能在未核对口径前直接作为最终运行命令转发。

## 必须交回的产物

真实seed和配置、权重来源、训练及选模数据、完整日志、selected checkpoint/epoch、逐样本或患者ID与标签/概率/预测、支持量/混淆矩阵、四项指标及mean/sample SD、实际时间和失败情况。已有五组具体要求继续见 2026-09-16_hanlin_existing_baseline_rerun_zh.md。

不设个人硬截止；下一个工作周期前未反馈时项目侧接手尚缺内容。当前PAFA说明仍是交接清单，不能标为已发给Hanlin或已完成运行。
