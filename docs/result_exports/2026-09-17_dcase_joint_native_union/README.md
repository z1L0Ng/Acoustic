# DCASE联合原生类别基线：三seed成功结果

2026-09-17从imec拉回，仅归档成功的DCASE；不包含PC-MCL失败文件。本轮只复制文件并读取既有日志/指标，没有新训练、推理或模型forward。

## 原始资产与Git范围

- 服务器原始目录：`/files1/Zilong/Acoustic/result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s/`。
- 完整本地副本：`result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s/`，47个文件、119,131,074 bytes，含6个best/last checkpoint。按相对路径和文件大小核对一致，没有使用hash/checksum。
- 成功恢复日志：`result/reproduce/source_transfer_baselines/server_logs/dcase_server_evalfix_20260917_012553.log`。
- 本Git目录的`raw/`保留原JSON、JSONL、逐样本NPZ及该日志，不改写原始记录。`results_summary.json`仅从三份已存在的run_summary读取并汇总四项指标。
- checkpoint共117,022,668 bytes，按仓库规则保留在上述完整本地副本及服务器，不加入普通Git；未移动或删除服务器数据。

## 最终结果

三seed为0、1、42，均完成训练与四列固定终点评测。数值为百分比，汇总为均值±样本标准差（ddof=1），从原始小数计算后取两位。

| Seed | ICBHI Score | SPR official Score | HF CAS AUROC | KAUH BA | 完成/选中epoch |
|---|---:|---:|---:|---:|---:|
| 0 | 55.45 | 91.50 | 78.73 | 81.62 | 25/15 |
| 1 | 53.02 | 91.77 | 75.98 | 72.89 | 26/16 |
| 42 | 54.78 | 91.01 | 88.21 | 77.70 | 30/20 |
| 均值±SD | 54.42±1.25 | 91.42±0.39 | 80.97±6.42 | 77.40±4.38 | — |

ICBHI共2756 cycles；SPR official inter共1429 events；HF CAS共957 eligible recordings（661 positive/296 negative）；KAUH共86 compatible patients。

## 实验定义

ICBHI+SPRSound联合训练，冻结BEATs frames，训练CRNN及九个原生类别并集输出，使用标签mask和class-masked attention。两源内部validation的native macro-multilabel F1等权选模；HF/KAUH不参与训练或选模。

ICBHI使用native4 argmax；SPR使用native7 argmax后合并Normal/Adventitious；HF跨窗口取max(Wheeze,Both,Rhonchi,Stridor)；KAUH先对B/D/E四类分数取患者均值，再argmax并归正常/异常。没有额外subset softmax。

这是DCASE-inspired respiratory adaptation，不是未改动的官方SED复现。与LSAA的差异包含编码器训练方式、架构、监督与读出及选模，不能把整体差值单独归因于某一模块。当前仅归档，未改论文。
