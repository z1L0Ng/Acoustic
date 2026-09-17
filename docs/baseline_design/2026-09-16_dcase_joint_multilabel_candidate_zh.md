# DCASE-inspired Nine-Output Native Union规格｜2026-09-16

状态：**APPROVED DESIGN / CODE READY / NOT RUN**
训练源：ICBHI internal subtrain + SPRSound internal subtrain。
选模：两个source的internal validation；official tests、HF、KAUH均为selected terminal。

## 1. 方法身份

名称：**DCASE-inspired Native-Class-Union CRNN (ICBHI+SPRSound, 5 s)**。

保留DCASE的frozen BEATs temporal frames、trainable log-Mel CNN、temporal fusion、BiGRU、class-aware attention、sigmoid multilabel outputs，以及loss/attention valid-class mask。省略Mean Teacher/unlabeled consistency、strong frame loss、mixup、median/overlap-add与PSDS。这是呼吸音native-union适配，不是DCASE 2024 reproduction，也不是LSAA attribute head。

## 2. 九输出空间

固定顺序：

```text
[N_ICBHI, N_SPR, Crackle, Wheeze, Both,
 FineCrackle, CoarseCrackle, Rhonchi, Stridor]
```

- `N_ICBHI`与`N_SPR`独立；没有证据声称两源Normal生理含义不同，只是保守避免强制合并annotation scope。
- `Wheeze`跨ICBHI Wheeze与SPR Wheeze共享。
- `Both`跨ICBHI Both与SPR Wheeze+Crackle共享；Both不拆成C/W双阳性。
- `FineCrackle/CoarseCrackle`保留SPR原生输出，并单向positive-alias到`Crackle`。
- `Crackle`表示ICBHI原生Crackle-only（C有、W无），不是所有含Crackle样本的通用属性。
- `Rhonchi/Stridor`为SPR原生独立输出。

## 3. Positive / negative / unknown映射

### 3.1 ICBHI

所有SPR-only outputs对ICBHI均unknown/masked。

| Raw | Positive | Observed negative | Unknown |
|---|---|---|---|
| Normal | N_ICBHI | Crackle, Wheeze, Both | N_SPR, Fine, Coarse, Rhonchi, Stridor |
| Crackle | Crackle | N_ICBHI, Wheeze, Both | N_SPR, Fine, Coarse, Rhonchi, Stridor |
| Wheeze | Wheeze | N_ICBHI, Crackle, Both | N_SPR, Fine, Coarse, Rhonchi, Stridor |
| Both | Both | N_ICBHI, Crackle, Wheeze | N_SPR, Fine, Coarse, Rhonchi, Stridor |

### 3.2 SPRSound

`N_ICBHI`对全部SPR rows保持unknown。Fine/Coarse对Crackle提供单向positive alias；该alias不反向展开ICBHI Crackle。

| Raw | Positive | Observed negative | Unknown |
|---|---|---|---|
| Normal | N_SPR | Crackle, Wheeze, Both, Fine, Coarse, Rhonchi, Stridor | N_ICBHI |
| Fine Crackle | Fine, Crackle | N_SPR, Wheeze, Both, Coarse, Rhonchi, Stridor | N_ICBHI |
| Coarse Crackle | Coarse, Crackle | N_SPR, Wheeze, Both, Fine, Rhonchi, Stridor | N_ICBHI |
| Wheeze | Wheeze | N_SPR, Crackle, Both, Fine, Coarse, Rhonchi, Stridor | N_ICBHI |
| Wheeze+Crackle | Both | N_SPR, Crackle, Wheeze, Fine, Coarse, Rhonchi, Stridor | N_ICBHI |
| Rhonchi | Rhonchi | N_SPR, Wheeze, Both, Fine, Coarse, Stridor | N_ICBHI, Crackle |
| Stridor | Stridor | N_SPR, Wheeze, Both, Fine, Coarse, Rhonchi | N_ICBHI, Crackle |

相对官方DCASE alias的差别：官方对一个source开放整组native classes，并把确认的具体类copy到宽类。这里也做单向positive copy。Rhonchi/Stridor与Wheeze/Both属于SPR原生互斥类别集，所以Wheeze/Both为observed negative；额外的Crackle-only alias对R/S保持unknown。N_ICBHI始终unknown，R/S绝不归Normal。

## 4. Loss与attention

每条row带`target[9]`与`mask[9]`：

```text
L_batch = sum(mask * BCE(probability,target)) / sum(mask)
```

所有eligible sample-class元素统一进入分母，不额外做node reweighting。Unknown不进入分子或分母。

每帧class logits先sigmoid。Attention logits在类别轴softmax，invalid classes先mask；clip probability为frame probability按attention沿时间加权求和，再除以每类attention时间和。训练BCE直接接clip probability，不再二次sigmoid。当前输入固定5 s，代码没有batch padding路径，也不声称已实现padding mask；未来若改成变长输入需另补。推理不使用target truth决定mask，且九个score不要求和为1。

## 5. Source split、batch与selection

复用项目现有patient-grouped internal source split：

| Seed/source | Subtrain | Validation | Groups |
|---|---:|---:|---:|
| seed0 ICBHI | 3636 cycles | 506 cycles | 63 / 16 patients |
| seed1 ICBHI | 2880 cycles | 1262 cycles | 63 / 16 patients |
| seed42 ICBHI | 3174 cycles | 968 cycles | 63 / 16 patients |
| all seeds SPRSound | 5219 events | 1437 events | 194 / 49 patients |

两者均来自各自official train；ICBHI official test与SPR inter不冒充validation。

- batch32、dataset-homogeneous；
- 每epoch 163 ICBHI + 163 SPR batches，共326 updates；
- 每源固定5216 sampled units/epoch；ICBHI有放回、SPR接近一遍；
- max50 epochs、patience10、min_delta0；
- validation threshold固定0.5、zero_division=0；
- ICBHI native4 macro multilabel-F1只统计`[N_ICBHI,Crackle,Wheeze,Both]`；
- SPR native7 macro multilabel-F1只统计`[N_SPR,Rhonchi,Wheeze,Stridor,Coarse,Fine,Both]`，Crackle alias不重复计入；
- monitor `M=0.5*F1_ICBHI+0.5*F1_SPR`；strict improvement，tie保留早checkpoint并计无提升；
- HF/KAUH和两个official source tests不参与selection、early stopping或threshold fitting。

## 6. 四列固定读出

`p`表示sigmoid+attention后的native scores；不额外subset softmax。

### ICBHI

对`[N_ICBHI,Crackle,Wheeze,Both]`直接argmax，得到native flat4并报告ICBHI Score。

### SPRSound

对`[N_SPR,Rhonchi,Wheeze,Stridor,Coarse,Fine,Both]`直接argmax；N_SPR归Normal，其余归Adventitious，报告official Task1-1 Score。

### HF CAS

```text
window_score = max(p_Wheeze, p_Both, p_Rhonchi, p_Stridor)
recording_score = max(three fixed 5-s windows)
```

必须包含Both；Crackle/Fine/Coarse不进入CAS。该值是ranking score，不称独立sigmoid之和或union probability。CAS positives为Wheeze/Rhonchi/Stridor，D-only为negative。

### KAUH

每个B/D/E view读取`[N_ICBHI,Crackle,Wheeze,Both]`四个native scores；同一patient先mean三个score vectors，再argmax；N_ICBHI归Normal，其余归Abnormal。只评86位compatible patients，并报告filter-view consistency。

## 7. 实现与产物

当前入口：

- `dcase_joint_union_run.json`
- `dcase_joint_source_groups.json`
- `dcase_joint_union_runner.py`
- `source_transfer_queue.py`
- `source_transfer_summary.py`

旧`dcase_source_runner.py`与`dcase_source_run.json`标记为历史并拒绝当前执行。

每seed输出config、source split summary、train log、selected validation raw scores/targets、best/last resumable checkpoints、四列raw scores与聚合结果、metrics和run summary。三seedsummary只收完整run。

## 8. 预算

- frame inputs：ICBHI 6898 + SPR train/inter 8085 + HF 5868 + KAUH 336 = 21187；
- frame主体静态约2.0 GB；提取35–90 min宽估计；
- 每epoch326 train updates，加两源validation；
- max50三seed训练约10–43 h；
- joint DCASE含cache/external约11–47 h；
- 加PC-MCL三seed后两方法串行约32–79 h，含10%约35–87 h（1.5–3.6天）。

以上没有新profile；早停只可能缩短，预算不预设触发epoch。

## 9. 证据边界

- 这是DCASE-inspired native-union baseline，不是DCASE官方复现；
- 与LSAA差异同时包含native ontology、mask、alias、CRNN、attention与frozen encoder，不能单因子归因；
- SPR参与训练，因此SPR列不是zero-target transfer；
- HF/KAUH没有专属训练head，只是固定source-output readout；
- 当前状态是CODE READY / NOT RUN，没有实验结果。
