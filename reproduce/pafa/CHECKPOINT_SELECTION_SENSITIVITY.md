# BEATs Core-2 checkpoint-selection sensitivity

Status: **validation-log-only design sensitivity**. 本报告不读取 test，也不对现有 R0/N1/A1/L1/L2 事后重选 checkpoint。

## 当前实现

每个 epoch 的 selection loss 只来自 ICBHI 与 SPRSound validation：先在每个数据集内对 eligible Level1、Crackle、Wheeze 的普通 CE/BCE 等权平均，再对两个数据集等权平均。L1/L2 训练时虽使用 focal/class-balanced loss，selection 仍统一重算普通 CE/BCE。严格小于当前 best 才更新，因此完全相同取更早 epoch。

checkpoint 固定后，才在该 epoch 的 validation predictions 上为 Crackle/Wheeze 分别选择一个 core-shared threshold；阈值目标是 ICBHI F1 与 SPRSound F1 的等权平均，tie 取较高 threshold。Level1 使用 softmax argmax。

## 三种 loss-only 视图

| Run | current equal raw loss | e1-normalized mean | normalized worst | Alternative checkpoint |
|---|---:|---:|---:|---|
| R0 | e2 / 0.331620 | e2 / 0.894690 | e1 / 1.000000 | HOLD：e1 checkpoint 未保留 |
| N1 | e2 / 0.330844 | e2 / 0.822529 | e1 / 1.000000 | HOLD：e1 checkpoint 未保留 |
| A1 | e2 / 0.314991 | e2 / 0.912351 | e1 / 1.000000 | HOLD：e1 checkpoint 未保留 |
| L1 | e6 / 0.310946 | e6 / 0.774575 | e6 / 0.987326 | 同一 epoch，无 alternative |
| L2 | e1 / 0.312014 | e3 / 0.948378 | e1 / 1.000000 | HOLD：e3 checkpoint 未保留 |

定义：

- `current equal raw loss`：当前正式 selection。
- `e1-normalized mean`：每个数据集的 epoch loss 除以本数据集 epoch-1 loss，再等权平均。
- `normalized worst`：取两个数据集相对 epoch-1 loss 中较差者，再最小化。

后两者只是 scale sensitivity。看过曲线后再采用 epoch-1 normalization 会形成 validation-method overfit，因此不能替换现有五条的正式 checkpoint。

## Prospective native composite（预注册数学定义）

如果论文 headline 是 ICBHI Sp/Se/Score 与 SPRSound Task1-1，下一条正式实验可采用以下 validation-only 设计：

1. 在 validation 内按 patient/group 划分互斥的 calibration 与 selection 子集；outer/test 不参与。
2. calibration 子集分别为 Crackle、Wheeze 选择一个 core-shared threshold：最大化 ICBHI/SPR F1 的等权平均；tie 取更高 threshold。Level1 始终 argmax。
3. selection 子集计算：
   - ICBHI：由已冻结 threshold 解码 flat4，计算 official `Score=(Sp+Se)/2`；
   - SPRSound：inter Task1-1 Normal/Adventitious official Score；
   - `native_composite = 0.5 * ICBHI_validation_Score + 0.5 * SPRSound_validation_Task1_1_Score`。
4. 最大 composite 选 checkpoint；tie 取更早 epoch。
5. epoch 固定后，按同一预注册规则在 full validation 上重拟合并冻结 shared thresholds；test 只访问一次。

现有五组只有完整 train log 与 selected-epoch predictions；本地没有所有 alternative epoch checkpoint。因此可以重算 loss-only sensitivity，但不能把 alternative epoch 变成可部署或 terminal-scored 模型。native composite 也只能作为未来预注册合同，不能基于现有 test 反选。

## Decision

- 现有 R0/N1/A1/L1/L2：保留 current equal raw loss selection。
- 只读 sensitivity：允许报告，但 evidence label 必须是 design diagnostic。
- 未来 native-composite run：在用户批准新的 validation calibration/selection 协议和训练后才可执行。
