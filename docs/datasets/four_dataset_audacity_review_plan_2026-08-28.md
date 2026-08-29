# 四数据集 Audacity 样本与声学观察准备

日期：2026-08-28
状态：`quantitative_panel_complete / audacity_gui_import_hold`

候选表：`docs/datasets/four_dataset_audacity_candidate_manifest_2026-08-28.csv`

## 1. Verified Result

候选阶段只读检查了本地 raw package、现有 ICBHI cycle manifest、SPRSound BioCAS2022 JSON、HF interval TXT 与 KAUH 文件名/工作簿关系。用户随后批准默认 20-selection panel、P20 `C` KAUH abnormal triplet、HF Stridor/empty annotation、跨数据集比较图和 selection-level quantitative analysis。量化阶段只解码 manifest 指定的 20 个 selection，保持各文件 native sample rate，不重采样、不输出或修改音频。

| Dataset | 本地 prediction/annotation unit | 可直接用于候选的标签 | 必须保留的边界 |
|---|---|---|---|
| ICBHI 2017 | annotated respiratory cycle | `normal/crackle/wheeze/both`，由 cycle 的两个 flags 直接确定 | cycle label 不定位 cycle 内具体声响；只选 official train 候选 |
| SPRSound BioCAS2022 | official respiratory event | `Normal/Fine Crackle/Coarse Crackle/Wheeze/Wheeze+Crackle/Rhonchi/Stridor` | Rhonchi/Stridor 只作为 native Other，不并入 Wheeze；只选 official train |
| HF_Lung_V1 | 15 s recording 上的 overlapping positive intervals | `D/Wheeze/Rhonchi/Stridor` positive；`D+Wheeze` 可由两个 observed intervals 的交集得到 | 没有 raw Normal/Negative；phase-only、empty file 和 gap 都不是 Normal |
| KAUH/Fraiwan v3 | recording-level raw sound string | `N/C/I C/E W/I E W/I C E W` 可保留保守解释；`Crep/Bronchial/I C B` 保持 raw/unresolved | 同一 P-number 的 B/D/E 是 Bell/Diaphragm/Extended 三滤波 siblings，必须一起观察 |

候选池共 61 行：ICBHI 8、SPRSound 14、HF 12、KAUH 27。所有路径均为 project-relative；没有复制音频。

## 2. Proposed Minimal Panel

CSV 中 `default_panel=yes` 的 20 行组成默认最小面板。建议分成四个独立 Audacity project，避免不同 native sample rate、prediction unit 和 label semantics 在一个时间轴中被误读。

| Project | 默认候选 | Tracks | 观察目的 |
|---|---|---:|---|
| `01_icbhi_cycle_classes.aup3` | Normal、Crackle、Wheeze、Both 各 1 个 annotated cycle | 4 | cycle-level flat4；同时保留 Meditron/AKGC417L/LittC2SE 设备差异 |
| `02_spr_event_classes.aup3` | Normal、Fine Crackle、Wheeze、Wheeze+Crackle、Rhonchi 各 1 个 event | 5 | event-level形态；Rhonchi 作为 native Other 对照 |
| `03_hf_positive_only.aup3` | D、Wheeze、D+Wheeze overlap、Stridor、empty annotation | 5 | positive-only interval、真实共现、HF-Type-1/Littmann 差异与 not-annotated 对照 |
| `04_kauh_filter_triplets.aup3` | P87 `N` 与 P20 `C` 的完整 B/D/E triplets | 6 | 同患者/同标签条件下比较 Bell、Diaphragm、Extended filter |

候选表中的其余 41 行是替换池：每个 ICBHI/SPRSound 类别另有较长样本；HF 另有各 positive 类、phase-only 和 overlap 候选；KAUH 覆盖完整 raw9 的一个代表患者 triplet。

## 3. Audacity 可视观察合同

每个导入 track 使用 `candidate_id` 作为显示名，并增加一个 label track：

- ICBHI/SPRSound：标出 `selection_start_s` 到 `selection_end_s`，保留完整 recording 作为前后文。
- HF：逐条抄入原始 positive intervals；未覆盖区域显示为 `not_annotated`，不得标记 `Normal`。`D+Wheeze overlap` 另加交集 label。
- KAUH：B/D/E 三条按同一患者连续排列，track name 保留 filter mode；不裁切、不对齐后覆盖原文件。

建议截图统一显示 waveform 与 spectrogram；频率可视范围固定为 `0-2000 Hz`。Audacity `Plot Spectrum` 可作为 qualitative spectrum/PSD-like view，但不能替代统一参数的 Welch PSD。

| 老师要求 | Audacity 可直接观察 | 后续仓库脚本量化 |
|---|---|---|
| Waveform | 波形包络、瞬态、削波、静音段 | peak、DC offset、clipping fraction |
| Spectrogram/PSD | 时频纹理；Plot Spectrum 的定性频谱 | fixed 25 ms/10 ms framing、Welch PSD、band power、spectral centroid/rolloff |
| Duration | selection toolbar/label boundary | annotation duration 与 recording duration 表 |
| RMS/average power | meter 仅作现场参考 | 对精确 selection 计算 RMS、mean-square power、dBFS |
| Silence/padding | 可见平坦/低能量区域 | digital-zero fraction、silence threshold fraction；model padding 与 raw silence 分列 |
| Noise | 定性 noise floor、宽带/窄带干扰 | quiet-frame floor、energy percentile、SNR proxy |
| Device/filter | 并排/solo 比较 | 相同频带定义下的 level/PSD 差异；不作病理因果解释 |

## 4. Non-destructive Import Plan

量化与 label/import-plan 产物已按以下非破坏性规则生成；Audacity GUI project 导入仍为 `HOLD`：

1. 创建 `result/audacity_four_dataset_panel_2026-08-28/`；该目录已由 `result/` 规则排除，不放入 Git。
2. 从 manifest 的原始路径直接导入，不移动、重命名或修改 `dataset/raw/`。
3. Audacity project 另存为上表四个 `.aup3`；Audacity project 内嵌副本只存在于 `result/`。
4. Track 命名为 `<candidate_id>__<native_label>__<device_or_filter>`；KAUH 保持同一 `relationship_group` 相邻。
5. 先建立 label tracks，再设置 waveform/spectrogram view；不执行 normalize、amplify、filter、resample、trim 或 silence removal。
6. 截图与观察笔记分别放入 `screenshots/` 和 `observations.md`。所有判断标为 `direct_visual_observation` 或 `script_measurement`。

## 5. Interpretation

默认 20-track panel 足以展示四类问题：原生类别形态、跨数据集同名声学事件、HF positive-only/gap 语义、KAUH filter-induced variation。它是老师讨论用的代表性 panel，不是随机样本、总体分布或统计显著性证据。

设备、滤波与振幅/频谱差异可能混杂。Audacity 中看到的差异不能直接归因于疾病、标签或模型可学性；跨数据集绝对振幅也不能在未经 calibration 的情况下直接比较。

## 6. 2026-08-29 Execution Receipt

已完成：

- 20 行 `audacity_import_plan.csv` 与 20 个 label-track TXT；
- selection-level RMS、mean-square power、dBFS、exact/near-zero fraction、relative silence proxy、frame-RMS SNR proxy、0–2 kHz Welch PSD、band power、spectral centroid 与 bandwidth；
- 一张跨数据集 quantitative comparison PNG；
- KAUH P87 `N` 与 P20 `C` 的完整 B/D/E triplets；
- HF `D`、Wheeze、observed-positive overlap、Stridor 与 empty source-label example，其中 empty 始终标为 `not_annotated | NOT normal/negative`。

产物位于 `result/audacity_four_dataset_panel_2026-08-28/`，按项目 `result/` 规则不进入 Git。脚本量化结果是 `script_measurement`；20-selection panel 是代表性观察，不是总体分布、显著性或病理因果证据。不同设备、滤波和未校准 gain 会混杂跨数据集绝对幅度。

仍为 `HOLD`：四个 `.aup3` project 与 Audacity waveform/spectrogram GUI 截图。当前 Computer Use 交互未完成，因此不把 quantitative PNG 冒充 Audacity 截图，也不声称 GUI import 已执行。原始音频未复制、重采样、裁切或修改。
