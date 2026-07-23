# 呼吸声学、咳嗽、呼吸与语音健康数据集版图报告（扩展版）

**生成日期：2026-06-17**  
**项目语境：acoustic-based respiratory disease detection**  
**扩展目标：扩大 dataset 范围；新增公开数据集论文、自采集但可复现/模型定义充分论文；方法地图至少 40 篇论文且至少 10 篇为近两年工作。**

> 重要限定：本报告只讨论数据集、标签、协议与机器学习研究路线，不构成临床诊断、筛查或疗效判断。凡无法通过官方数据页、挑战页、Zenodo/PhysioNet/Mendeley/GitHub/Hugging Face、原始论文或机构页面核验的字段均标为 **unknown / not found / requires package inspection**。本报告显式区分疾病标签、症状标签、声学事件标签、自报告标签、PCR-linked 标签和专家/医师听诊标注。

---

## 1. Executive Summary

本扩展版共覆盖 **52 个数据集/数据资源**，其中 **Tier 1 = 6**，**Tier 2 = 20**，**Tier 3 = 16**，**Tier 4 = 10**。相比上一版，本版新增或显著扩展了 **VocalSound、Audio-IMU Multimodal Cough、UCL Speech Breath Monitoring / ComParE 2020 Breathing、ESC-50、cough-speech-sneeze、OpenSLR Nonverbal Vocalization、Resp-229K、FABS、DeepBreath/Pneumoscope、Bridge2AI-Voice Adult/Pediatric、RALE、MEEI、AudioSet Cough class** 等资源。

方法地图共列出 **56 篇论文/数据论文/可复现方法资源**，其中 **23 篇为 2024-2026 年近两年工作**。这些条目被分为三类：第一类是使用公开或半公开呼吸/咳嗽/语音健康数据集的论文；第二类是自采集或私有数据论文，但满足至少一个可复现条件，即有公开 repo、公开数据访问路径、或论文中模型/协议定义足够明确；第三类是泛音频/健康音频 foundation model 或辅助数据集论文。不同论文的结果只有在 **相同数据集、相同标签、相同 split、相同指标和相同 subject-independence 约束** 下才被视为可比。

短期最强路线仍然是 **respiratory acoustic event detection + cross-dataset robustness**。建议主线使用 **ICBHI 2017、HF_Lung_V1、SPRSound** 做 lung/breath event modeling，使用 **COUGHVID、Audio-IMU Cough、Corp Dataset、Clinic Waiting Room Cough** 做 cough/event detection 与质量控制，使用 **UK COVID-19 Vocal Audio** 做更可靠的 PCR-linked disease-label 分析。COVID 音频路线应把 **UK COVID Vocal** 和 **Coppock et al. 2024 的负结果** 作为核心基准，避免只报告音频模型 AUC 而不比较症状基线。

当前直接可行动的 SOTA 方向集中在三类：第一，ICBHI 上的 respiratory sound classification，例如 **MVST 2024、BTS 2024、RepAugment 2024、Stethoscope-guided SupCon 2024、ADD-RSC 2025、Architecture-Agnostic KD 2025、Geometry-aware AST+SAM 2025、Resp-Agent 2026**；第二，OPERA/HeAR/respiratory pretraining，例如 **OPERA 2024、HeAR 2024、Niizumi et al. 2025 M2D+Resp**；第三，COVID/cough validity route，例如 **UK COVID Vocal 2024、Coppock et al. 2024、SympCoughNet 2025、COUGHVID U-Net/DL 2025**。

---

## 2. Search Strategy And Inclusion Criteria

### 2.1 搜索策略

本版检索以一手来源为优先：官方数据页、挑战官网、Zenodo、PhysioNet、Mendeley Data、Dryad、GitHub/GitLab、Hugging Face、论文主页、期刊/会议论文页面和机构页面。关键词包括：respiratory sound dataset, lung sound dataset, wheeze dataset, crackle dataset, adventitious lung sound, pediatric respiratory sound, cough dataset, cough event detection, COVID cough, PCR referenced vocal audio, breathing sound dataset, speech-breath monitoring, voice pathology dataset, voice biomarker dataset, health acoustic foundation model, respiratory acoustic foundation model。

### 2.2 纳入标准

数据集或论文至少满足一项：

1. 直接包含肺音、呼吸音、咳嗽、呼吸、wheeze、crackle、rhonchi、stridor、sniff、sneeze 等呼吸相关声学内容；
2. 包含 COVID、TB、COPD、asthma、pneumonia、bronchiolitis、voice pathology 等呼吸/声学健康相关标签，且音频模态与咳嗽、呼吸、肺音、语音或嗓音健康相关；
3. 虽非疾病数据，但对预训练、域适配、负样本构造、语音健康或远程健康监测有明确方法学价值；
4. 对自采集/私有数据论文，只有在 **repo 公开、数据访问路径公开、或模型/协议定义足够明确** 时纳入 paper map；
5. 对无法直接行动但领域常见的资源，仅列为 Tier 4，不作为核心实验依据。

### 2.3 排除标准

排除仅含非音频生理信号且与声学无直接连接的数据集、无法核验来源的网页列表、普通 ASR/语音识别数据集、只含教学示例且缺少研究标注的音频库、以及无法确认标签语义/访问路径的镜像数据。Kaggle/YouTube/Freesound 镜像只在可以追溯到原始数据源时作为辅助或 Tier 4 处理。

### 2.4 Relevance tiers

| 层级 | 定义 | 本项目用法 |
|---|---|---|
| Tier 1 | Strong direct fit | 可作为核心训练、评测或初步结果主数据集。直接支持 respiratory event detection、cough/breath analysis、lung sound classification 或 PCR-linked disease audio modeling。 |
| Tier 2 | Useful with caveats | 相关性强但存在自报告标签、受限访问、小规模、协议不一致、临床元数据不足、设备/人群 mismatch 等问题。适合补充实验、外部验证或域适配。 |
| Tier 3 | Auxiliary / background evidence | 适合预训练、负样本、语音健康背景、foundation model baseline 或方法启发；不能单独支撑核心呼吸疾病检测结论。 |
| Tier 4 | Mention only / low actionability | 访问不可控、标签/协议不明、教育库/镜像/非音频或弱连接资源。只用于说明潜在线索或不可行动性。 |

---

## 3. Dataset Coverage Table

| # | 数据集/资源 | 层级 | 访问状态 | 主要任务路线 | 音频模态 | 关键规模 | 标签粒度 | 源 |
|---|---|---|---|---|---|---|---|---|
| 1 | ICBHI 2017 Respiratory Sound Database | T1 | open / mirror | event detection; lung sound classification | chest respiratory/lung sounds | 920 recordings; 6,898 annotated respiratory cycles; 126 subjects; about 5.5 h | cycle-level wheeze/crackle; subject/recording metadata | [src](https://ai4eu.dei.uc.pt/respiratory-sounds-dataset/) |
| 2 | HF_Lung_V1 | T1 | open GitLab | event detection; breath phase; adventitious sounds | lung sounds from electronic stethoscope | 9,765 15-s files; 34,095 inhalation, 18,349 exhalation, 13,883 CAS, 15,606 DAS labels | onset-offset event labels for I/E/CAS/DAS | [src](https://gitlab.com/techsupportHF/HF_Lung_V1) |
| 3 | SPRSound | T1 | open GitHub / challenge releases | pediatric respiratory event detection | pediatric lung sounds | 2022 train 1,949 and test 734 records; original paper reports 2,683 records, 292 participants, 9,089 events | record-level classes and event onset-offset annotations | [src](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) |
| 4 | COUGHVID | T1 | open Zenodo / EPFL page | cough quality; cough segmentation; weak disease prediction | crowdsourced cough recordings | over 30k recordings; 25k+ in Scientific Data paper; 2,800+ expert-labeled coughs | file-level cough quality/type; self-report COVID/symptom metadata | [src](https://zenodo.org/records/7024894) |
| 5 | UK COVID-19 Vocal Audio / COVID-19 Vocal Audio | T1 | open Zenodo for cough/exhalation; speech excluded from open release | PCR-linked disease prediction; cough/breath analysis | cough, exhalation, speech collected; open release contains cough and exhalation | 72,999 participants; 70,794 PCR-linked; 53.7 GB open release | PCR-linked subject/session labels; symptom metadata | [src](https://zenodo.org/records/10043978) |
| 6 | Corp Dataset | T1 | paper/page listed; access path unclear | cough event detection in long recordings | long-duration cough audio from respiratory-disease patients | 168 h; 9,969 cough events; 42 respiratory-disease patients | cough onset-offset; disease at subject level | [src](https://pmc.ncbi.nlm.nih.gov/articles/PMC9760237/) |
| 7 | KAUH / Fraiwan lung-sound dataset | T2 | open Mendeley | lung-sound classification; disease-category classification | chest wall lung sounds | patient files P1-P112; exact complete clip count should be inspected after download | sound type and diagnosis; chest location metadata | [src](https://data.mendeley.com/datasets/jwyy9np4gv/3) |
| 8 | RespiratoryDatabase@TR | T2 | open Mendeley | multi-channel lung sounds; COPD severity; domain adaptation | multi-channel lung and heart/lung recordings | 75 subjects reported in associated article; 12-channel lung recordings reported | subject/file-level disease/severity; expert validation reported | [src](https://data.mendeley.com/datasets/p9z4h98s6j/1) |
| 9 | HF_Lung_V2 / HF_Lung_V1_IP | T2 | partially open; expanded labels partly controlled/requested | expanded lung event detection | lung sounds | V2 paper reports train/test 10,554/3,403 files | onset-offset respiratory/adventitious labels | [src](https://www.mdpi.com/2076-3417/12/15/7623) |
| 10 | HF_Tracheal_V1 | T2 | audio open; labels by request/form in some releases | breath phase and CAS detection | tracheal sounds | about 10k 15-s tracheal recordings in paper/repo descriptions; exact count should be verified after download | onset-offset I/E/CAS labels | [src](https://gitlab.com/techsupportHF/HF_Tracheal_V1) |
| 11 | Coswara | T2 | open GitHub | COVID-related cough/breath/speech analysis; symptom modeling | cough, breathing, speech, counting, vowels | Scientific Data paper reports 2,635 individuals; nine sound categories per participant | self-report/test-status subject labels; modality-level files | [src](https://github.com/iiscleap/Coswara-Data) |
| 12 | Cambridge COVID-19 Sounds | T2 | DTA / academic access | COVID/symptom prediction; cough/breath/voice | cough, breathing, voice | 53,449 samples; 552 h; 36,116 participants; 2,106 positive reported | self-reported testing status; participant-level metadata | [src](https://openreview.net/forum?id=9KArJb4r5ZQ) |
| 13 | DiCOVA Challenge datasets | T2 | challenge-only / restricted challenge data | COVID detection from cough/breath/speech | cough, breathing, vowel, counting | Track 1: 1,040 subjects; Track 2: 990 subjects reported | binary subject-level COVID labels; challenge folds | [src](https://dicovachallenge.github.io/) |
| 14 | ComParE 2021 COVID-19 Cough/Speech | T2 | challenge subset | COVID detection from cough/speech | cough and speech | cough split 286/231/208; speech split 315/295/283 train/dev/test | binary subject-level COVID labels in challenge tracks | [src](https://pmc.ncbi.nlm.nih.gov/articles/PMC10031089/) |
| 15 | Sound-Dr | T2 | open GitHub / paper-linked | COVID/symptom/anomaly prediction | cough, mouth breathing, nose breathing | 3,930 recordings; 1,310 subjects | self-report COVID/symptom labels; modality-level files | [src](https://github.com/ReML-AI/Sound-Dr) |
| 16 | smarty4covid | T2 | public release with privacy exclusions | cough/breath/voice COVID-related modeling | cough, regular breath, deep breath, voice | 18,265 recordings; 4,673 users | self-report health/COVID labels; expert-labeled subset | [src](https://www.nature.com/articles/s41597-023-02646-6) |
| 17 | Virufy COVID-19 Open Cough Dataset | T2 | open GitHub | PCR-linked cough; COVID cough analysis | cough | clinical subset described as 121 segmented cough samples from 16 patients | PCR-linked clinical labels; cough segments | [src](https://github.com/virufy/virufy-data) |
| 18 | Solicited TB Cough / CODA TB cough resources | T2 | controlled/research access; exact public package varies | TB cough analysis; disease prediction | cough and possibly respiratory metadata | scale varies by release; exact public count not consistently verifiable from one official page | microbiology/clinical TB status where available | [src](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10031089/) |
| 19 | Audio-IMU Multimodal Cough Dataset | T2 | open Dryad | cough event detection; wearable/bioacoustic multimodal sensing | wearable audio plus IMU | 902 MB zip; 13 student volunteers; two chest-mounted microphones and MetaMotionS IMU | frame/clip labels: cough, speech, sneeze, deep breath, groan, laugh, speech_far, other | [src](https://datadryad.org/dataset/doi:10.5061/dryad.0p2ngf26t) |
| 20 | VocalSound | T2 | open official/MIT/CSAIL + GitHub | cough/sneeze/sniff/throat-clearing pretraining; vocal-event detection | non-speech human vocalizations | 21,024 recordings; 3,365 subjects | clip-level six-class vocalization labels; demographics/health-condition metadata | [src](https://groups.csail.mit.edu/sls/downloads/vocalsound/) |
| 21 | Pulmonary Sound Dataset / ALSD-Net data | T2 | open Mendeley | normal/abnormal lung sound classification | electronic stethoscope lung sounds | Mendeley page reports normal 11.28%, abnormal 88.72%; full count requires download inspection | binary normal/abnormal; no event onset-offset found | [src](https://data.mendeley.com/datasets/6mnp9v3n73/1) |
| 22 | Clinic Waiting Room Cough Dataset | T2 | project/paper-linked; raw access unclear | clinic cough event detection | far-field clinic audio | 3,930 manually annotated 3-s recordings from 54 waiting-room sessions; 348 h deployment validation reported | cough/non-cough clip labels; session/location metadata | [src](https://www.nature.com/articles/s41598-021-03913-5) |
| 23 | DeepBreath / Pneumoscope pediatric auscultation dataset | T2 | raw audio by reasonable request; code public | pediatric pathology/pneumonia/wheeze/bronchiolitis modeling | pediatric lung auscultation | 4,552 lung auscultation recordings; 572 pediatric outpatients; 35.9 h; six outpatient departments across five countries | diagnosis-level labels and attention/localization outputs; by-site validation | [src](https://github.com/pneumonia-diagnosis/DeepBreath) |
| 24 | Formosa Archive of Breath Sound (FABS) | T2 | paper-described; public access not confirmed | ED lung-sound abnormality classification; label-noise analysis | lung sounds in emergency department context | 5,238 annotated recordings; 1,985 patients; 14.6 h; 715 coarse crackles, 234 wheezes, 4,289 normal | physician-annotated normal/coarse crackle/wheeze labels | [src](https://ai.jmir.org/2025/1/e69844/) |
| 25 | PKU Open Auscultation / Open Lung Sound resources | T2 | open page/GitHub; details need inspection | lung-sound classification; external validation | lung auscultation audio | exact public count/version unknown from high-level source | unknown / likely file-level lung sound labels | [src](https://github.com/search?q=open+lung+sound+auscultation+dataset&type=repositories) |
| 26 | Resp-229K | T2 | GitHub + Hugging Face announced; large download | dataset fusion; multimodal respiratory modeling; synthetic/LLM-narrative route | respiratory recordings with LLM-distilled clinical narratives | paper/repo report 229k recordings, 407+ h, about 70 GB; curated from ICBHI, SPRSound, UK COVID, COUGHVID, KAUH and HF Lung V1 | source-derived labels plus LLM-distilled clinical narratives; semantics require audit | [src](https://github.com/zpforlove/Resp-Agent) |
| 27 | HLS-CMDS Simulated Heart/Lung Sound Dataset | T3 | open Mendeley | pretraining; denoising; source separation; simulation sanity checks | clinical manikin heart/lung sounds | 535 recordings; 50 heart, 50 lung, 145 mixed plus source signals; Littmann CORE, WAV/CSV | source/separation labels; simulated context | [src](https://data.mendeley.com/datasets/pxg5f7j5kd/1) |
| 28 | UCL Speech Breath Monitoring / ComParE 2020 Breathing Sub-Challenge | T3 | challenge / research dataset | speech-breath monitoring; breathing-from-speech auxiliary route | speech + respiratory belt signal | 49 speakers; about 4 min per speaker; speech downsampled to 16 kHz; belt 25 Hz | continuous respiratory-belt target over speech | [src](https://www.isca-archive.org/interspeech_2020/schuller20_interspeech.html) |
| 29 | AudioSet respiratory-related ontology labels | T3 | open metadata; YouTube availability varies | generic pretraining; weak auxiliary labels | web audio clips from YouTube | AudioSet full dataset is large; respiratory-related labels include cough, sneeze, sniff, breathing, snoring, wheeze-like labels depending ontology branch | weak clip-level labels, not clinical | [src](https://research.google.com/audioset/ontology/index.html) |
| 30 | AudioSet Cough class | T3 | open metadata; YouTube availability varies | cough detector pretraining; weak positives | YouTube cough audio/video clips | ontology page reports 871 cough clips and about 2.4 h; 60 eval, 60 balanced train, 751 unbalanced train | weak clip-level Cough label | [src](https://research.google.com/audioset/ontology/cough_1.html) |
| 31 | FSD50K / respiratory-sound slices | T3 | open under source clip licenses | generic pretraining; weak respiratory event auxiliary | Freesound clips | 51,197 clips; 200 classes; >100 h; respiratory-related subsets exist by labels/tags | weak labels from AudioSet ontology/Freesound tags | [src](https://zenodo.org/records/4060432) |
| 32 | ESC-50 | T3 | open GitHub; CC BY-NC for ESC-50 | small generic cough/breath/sneeze/snoring auxiliary | environmental sound clips | 2,000 clips; 50 classes; 5 s; 40 examples/class; 44.1 kHz mono WAV | clip-level class labels; includes coughing, breathing, sneezing, snoring | [src](https://github.com/karolpiczak/ESC-50) |
| 33 | cough-speech-sneeze / audEERING corpus | T3 | available via conversion/audb tooling | cough/sneeze/speech auxiliary event modeling | YouTube-derived speech, cough, sneeze, silence | exact count should be inspected from audformat/audb release | clip-level cough/sneeze/speech/silence labels | [src](https://audeering.github.io/datasets/datasets/cough-speech-sneeze.html) |
| 34 | OpenSLR Nonverbal Vocalization Dataset | T3 | open OpenSLR | nonverbal vocal event pretraining; cough/sneeze/panting/yawn auxiliaries | crowdsourced nonverbal vocalizations | scale not transcribed here; official page lists 16 nonverbal classes | clip-level labels include coughing, yawning, throat clearing, sighing, panting, sneezing etc. | [src](https://www.openslr.org/99/) |
| 35 | FluSense hospital audio dataset | T3 | paper/project; public raw access unclear | population-level cough/symptom trend monitoring | waiting-room audio/video/sensor streams | paper reports hospital waiting-area deployment; exact raw release not found | cough/sneeze/audio-event and aggregate illness trend labels | [src](https://dl.acm.org/doi/10.1145/3381014) |
| 36 | NoCoCoDa | T3 | dataset page / paper-linked | non-COVID cough controls; cough negative/positive balance | cough audio | exact scale/access terms need inspection from dataset page | cough and non-COVID status labels | [src](https://www.nature.com/articles/s41597-023-02646-6) |
| 37 | VOICED Database | T3 | open PhysioNet | voice pathology; speech-health background | sustained vowel voice samples | 208 clinically verified voice samples; 150 pathological, 58 healthy | speaker-level pathology labels; demographics and VHI/RSI questionnaire metadata | [src](https://physionet.org/content/voiced/) |
| 38 | Saarbruecken Voice Database (SVD) | T3 | open web interface/download terms vary | voice pathology; speech-health auxiliary | sustained vowels and speech tasks | large voice pathology database; exact current count should be checked from official site | speaker/pathology labels; speech task metadata | [src](https://stimmdb.coli.uni-saarland.de/) |
| 39 | Bridge2AI-Voice Adult Dataset | T3 | PhysioNet derived features open/controlled; raw audio via Synapse/DACO | speech-health and multimodal health monitoring; respiratory/voice-disorder context | voice-derived features; raw audio controlled | PhysioNet v3.1.0 published May 2026; raw audio not in PhysioNet release | derived features, questionnaires, clinical/demographic metadata | [src](https://physionet.org/content/b2ai-voice/) |
| 40 | Bridge2AI-Voice Pediatric Dataset | T3 | PhysioNet derived features; raw access controlled | pediatric speech-health monitoring; respiratory/voice biomarker context | voice-derived features/spectrograms and clinical metadata | PhysioNet v1.1.0 published May 2026 | derived features, spectrograms, questionnaire and clinical metadata | [src](https://physionet.org/content/b2ai-voice-pediatric/) |
| 41 | OPERA benchmark/resource | T3 | open code; underlying datasets have mixed access/licenses | respiratory acoustic foundation-model benchmarking | fused respiratory acoustic tasks | GitHub/OpenReview report about 136k samples, 440 h, 19 tasks | task labels inherited from constituent datasets | [src](https://github.com/evelyn0414/OPERA) |
| 42 | Google HeAR / Health Acoustic Representations | T3 | model gated on Hugging Face; paper/model docs public | frozen embedding baseline; health-audio representation | generic health acoustic embeddings | paper/model docs report 313M two-second clips; evaluation on 33 tasks across six datasets | not a dataset label source; representation model | [src](https://huggingface.co/google/hear) |
| 43 | RALE Lung Sounds Repository | T4 | web-listened education/repository; research download/metadata limited | reference examples only | respiratory sounds | official page groups recordings into Normal, Wheezes, Crackles and Other; exact research-ready metadata unknown | coarse category labels | [src](https://www.rale.ca/repository.htm) |
| 44 | SNUCH Pediatric Lung Sound Dataset | T4 | private / not public | pediatric lung-sound method evidence | pediatric lung sounds | paper reports 680 eligible training/internal-validation clips and 90 prospective-validation clips | normal/crackle/wheeze labels by pediatric pulmonologists | [src](https://pmc.ncbi.nlm.nih.gov/articles/PMC9871007/) |
| 45 | MIT Open Voice / cough AI preliminary resources | T4 | project/repository status unclear | speech/cough health monitoring background | voice/cough depending release | unknown / not found as stable raw-audio dataset | unknown | [src](https://www.media.mit.edu/projects/open-voice/overview/) |
| 46 | MEEI Voice Disorders Database | T4 | commercial/restricted legacy dataset | voice pathology background | voice/speech | secondary papers report >1,400 recordings or 657 pathological + 53 normal subsets; exact accessible package varies | voice pathology labels | [src](https://masseyeandear.org/research/otolaryngology/voice-and-speech) |
| 47 | PEEP / respiratory-support physiological datasets without raw audio | T4 | open/non-audio; not central | clinical context only | not raw respiratory audio | varies | not acoustic labels | [src](https://physionet.org/) |
| 48 | Open-source AI Stethoscope / COPD HeAR demo resources | T4 | GitHub/demo; paper/dataset status not fully verified | method lead only | stethoscope lung sounds | repo claims COPD AUROC but source dataset/access need verification | unknown | [src](https://github.com/search?q=open-source+AI+stethoscope+COPD+HeAR&type=repositories) |
| 49 | Kaggle ICBHI respiratory-sound mirrors | T4 | open mirrors; often unofficial | convenient download only, not primary source | mirrored ICBHI lung sounds | 920 recordings when faithful; mirror metadata may drift | inherits ICBHI labels | [src](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) |
| 50 | YouTube/Freesound cough compilation datasets | T4 | varies; often weakly documented | negative/auxiliary only | web cough/sneeze/breath clips | unknown; many mirrors duplicate AudioSet/FSD/ESC | weak clip tags | [src](https://freesound.org/) |
| 51 | Simulated/body-sound teaching libraries | T4 | varies; educational license | qualitative listening/reference only | lung/heart/respiratory examples | unknown / not research-ready unless metadata available | coarse teaching labels | [src](https://www.easyauscultation.com/) |
| 52 | General speech corpora sometimes used for health-audio pretraining | T4 | open but weak connection | pretraining only; not respiratory-health evidence | ordinary speech | large but not respiratory-specific | transcript/speaker labels, not disease labels | [src](https://www.openslr.org/) |

---

## 4. Relevance Tier Summary

### 4.1 Tier 1: 强直接适配

Tier 1 数据集是短期实验最应优先投入的资源：**ICBHI 2017、HF_Lung_V1、SPRSound、COUGHVID、UK COVID-19 Vocal Audio、Corp Dataset**。其中 ICBHI/HF/SPRSound 覆盖肺音与呼吸事件，COUGHVID/Corp 覆盖咳嗽事件与咳嗽质量，UK Vocal 提供更可靠的 PCR-linked disease label。Tier 1 内部也不能无条件合并：ICBHI 的 cycle-level crackle/wheeze、HF 的 onset-offset CAS/DAS、SPRSound 的 pediatric event ontology、COUGHVID 的 self-report metadata、UK Vocal 的 PCR-linked labels 属于不同语义层。

### 4.2 Tier 2: 有用但有 caveats

Tier 2 包括 KAUH、RespiratoryDatabase@TR、HF_Lung_V2、HF_Tracheal、Coswara、Cambridge、DiCOVA、ComParE COVID、Sound-Dr、smarty4covid、Virufy、TB cough resources、Audio-IMU Cough、VocalSound、Pulmonary Sound、Clinic Waiting Room Cough、DeepBreath、FABS、Resp-229K 等。它们适合外部验证、扩充模态、补充咳嗽/呼吸/语音健康背景或方法复现，但每个都有明确限制：访问受限、自报告标签、私有数据、儿童/成人 mismatch、非临床志愿者、或 derived/LLM labels。

### 4.3 Tier 3: 辅助/背景证据

Tier 3 是表示学习和方法学背景：AudioSet、FSD50K、ESC-50、OpenSLR Nonverbal、cough-speech-sneeze、UCL-SBM/ComParE Breathing、VOICED、SVD、Bridge2AI-Voice、OPERA、HeAR 等。它们适合用作 frozen embedding、pretraining、negative set、domain-adaptation background 或 speech-health route；不应用来直接声明 respiratory disease detection 能力。

### 4.4 Tier 4: 仅提及/低可行动性

Tier 4 包括 RALE、SNUCH private pediatric dataset、MEEI、MIT Open Voice、Kaggle mirrors、teaching libraries 等。它们可以提供背景或联系线索，但不应作为短期可复现实验主数据。

---

## 5. Task-Route Map

### 5.1 Event detection

优先数据集：ICBHI、HF_Lung_V1、SPRSound、Corp Dataset、Audio-IMU Cough、Clinic Waiting Room Cough。任务应拆为：lung cycle classification、adventitious event onset-offset、breath phase segmentation、cough event detection、cough boundary localization。短期最合理的论文目标是 **event-level robust modeling**，而不是直接 clinical diagnosis。

### 5.2 Disease prediction

优先数据集：UK COVID Vocal、DiCOVA、ComParE COVID、Cambridge、Coswara、Sound-Dr、Virufy、TB cough resources、DeepBreath/FABS self-collected papers。需要严格区分 PCR-linked、self-report、challenge-derived、diagnosis-level 和 symptom-level labels。疾病预测路线必须报告 symptom-only 或 metadata-only baseline，尤其在 COVID 音频上。

### 5.3 Cough/breath analysis

优先数据集：COUGHVID、UK Vocal cough/exhalation、Coswara、Audio-IMU Cough、VocalSound、cough-speech-sneeze、OpenSLR NVD、AudioSet Cough、Clinic Waiting Room Cough、Corp Dataset。推荐先做 cough/non-cough、quality filtering、segmentation、artifact rejection，再进入 disease-label modeling。

### 5.4 Speech-health assessment

优先资源：Bridge2AI-Voice Adult/Pediatric、VOICED、SVD、Cambridge/Coswara/UK Vocal speech components（注意 UK open edition不包含 speech）、MEEI 作为受限背景。语音健康路线应和 lung/cough 路线分开写，关注 voice biomarker、vocal fold pathology、respiratory/speech-breath coupling 和 telehealth monitoring，不把 voice pathology label 当作 lung disease label。

### 5.5 Dataset fusion / domain adaptation

优先资源：OPERA、Resp-229K、HeAR、AudioSet/FSD50K/ESC-50、ICBHI/HF/SPRSound/KAUH cross-dataset。Fusion 的核心风险是 label semantics 与 source leakage。若 Resp-229K 使用 ICBHI/HF/SPRSound/UK/COUGHVID/KAUH 源数据，则下游评测必须排除源重叠或至少报告 contamination audit。

### 5.6 Weak auxiliary evidence

AudioSet、FSD50K、ESC-50、OpenSLR、VocalSound、cough-speech-sneeze 可以提供弱标签和预训练信号，但其 web-audio 标签是非临床标签。它们适合降低 false positives、学习 human non-speech events 或作为 frozen embedding baseline 的背景，不应作为临床结论证据。

---

## 6. Dataset Deep-Dive Profiles

### 6.1 ICBHI 2017 Respiratory Sound Database

- **Tier**: Tier 1.
- **Official URL / citation**: ICBHI challenge/mirror and the database paper: `https://ai4eu.dei.uc.pt/respiratory-sounds-dataset/`, `https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge`, and Rocha et al., 2019.
- **Access / license**: open through official/mirror pages; some mirrors state CC0/public-domain style reuse, but the exact license should be preserved from the accessed package.
- **Task route**: core event detection and lung-sound classification.
- **Audio modality/context**: heterogeneous chest respiratory recordings collected with multiple devices and chest positions.
- **Technical format**: WAV files in the common release; sampling rate/device varies by original acquisition; exact per-file specs should be read from WAV headers.
- **Scale**: 920 recordings, 6,898 respiratory cycles, 126 subjects, about 5.5 hours.
- **Age/demographics**: adult and pediatric ages appear in metadata; exact age distribution should be computed from metadata.
- **Labels**: respiratory-cycle annotations for normal, crackle, wheeze, and both crackle+wheeze; diagnosis metadata include COPD/asthma/bronchiectasis/pneumonia/URTI/LRTI/healthy in common metadata.
- **Granularity**: cycle-level acoustic-event labels; subject/recording-level metadata; expert/clinician annotation in dataset paper.
- **Splits/protocol**: official challenge split exists. Many papers use custom or random cycle splits; those are not directly comparable.
- **Leakage risks**: cycle-level random split can place cycles from the same patient or recording in train/test. Use patient-independent split.
- **Recommended use**: first public lung-sound benchmark; use official split and ICBHI score, sensitivity, specificity, macro-F1; include external validation on HF_Lung or SPRSound if claiming robustness.

### 6.2 HF_Lung_V1

- **Tier**: Tier 1.
- **Official URL / citation**: `https://gitlab.com/techsupportHF/HF_Lung_V1`; Hsu et al., PLOS ONE 2021.
- **Access / license**: open GitLab release; paper/repo state CC BY-NC 4.0 in many derived resources.
- **Task route**: event detection, breath-phase segmentation, continuous adventitious sound (CAS) and discontinuous adventitious sound (DAS) detection.
- **Audio modality/context**: lung sound recordings from electronic stethoscope settings; 15-second clips.
- **Technical format**: paper reports 4 kHz sampling, 16-bit WAV, mono.
- **Scale**: 9,765 15-second recordings; 34,095 inhalation labels; 18,349 exhalation labels; 13,883 CAS labels; 15,606 DAS labels.
- **Subjects/metadata**: exact subject count and demographics should be read from metadata; disease labels are not the primary strength.
- **Labels**: onset-offset annotations for inhalation, exhalation, CAS, DAS.
- **Granularity**: event-level onset-offset; expert annotation.
- **Splits/protocol**: user-defined; subject independence must be enforced if subject IDs are available.
- **Leakage risks**: clip-level split across repeated recordings can inflate event metrics.
- **Recommended use**: strongest open event-level lung-sound corpus for model development; useful for boundary detection and temporal models before disease prediction.

### 6.3 SPRSound

- **Tier**: Tier 1.
- **Official URL / citation**: `https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound`; IEEE TBioCAS 2022 database paper and challenge pages.
- **Access / license**: open GitHub/challenge releases; license terms should be checked in the release.
- **Task route**: pediatric respiratory sound event detection and classification.
- **Audio modality/context**: pediatric lung sounds, collected from children aged roughly 1 month to 18 years using Yunting model II stethoscope according to the database paper/challenge material.
- **Technical format**: WAV files; exact sample rate/bit depth should be verified from release.
- **Scale**: 2022 challenge release lists 1,949 training and 734 test records; original paper reports 2,683 records, 292 participants, and 9,089 events.
- **Labels**: normal, poor-quality, CAS types and DAS types in challenge/task definitions; includes wheeze, stridor, rhonchi, coarse/fine crackle in event labels.
- **Granularity**: record-level labels and event onset-offset annotations.
- **Splits/protocol**: challenge train/test splits; later evaluation sets exist.
- **Leakage risks**: pediatric subject independence and site independence need explicit checking.
- **Recommended use**: essential pediatric external domain; do not mix with adult stethoscope data without age-domain analysis.

### 6.4 COUGHVID

- **Tier**: Tier 1.
- **Official URL / citation**: EPFL dataset page, Zenodo release, Orlandic et al., Scientific Data 2021.
- **Access / license**: open Zenodo; license is generally CC BY 4.0 in official releases, but downstream derivatives should preserve metadata and consent constraints.
- **Task route**: cough quality control, cough detection/segmentation, weak COVID/symptom analysis, representation learning.
- **Audio modality/context**: crowdsourced cough through web/mobile; uncontrolled devices and environments.
- **Technical format**: web audio; common release uses Opus/webm/JSON metadata; exact specs vary by uploaded file.
- **Scale**: over 30k cough recordings in later releases; Scientific Data paper describes 25k+ crowdsourced recordings and 2,800+ expert-labeled coughs.
- **Labels**: expert labels for cough/no cough/quality/diagnostic attributes; self-reported COVID and symptoms.
- **Granularity**: file-level labels, metadata-level symptoms, some segmentation from derived tools.
- **Splits/protocol**: no single universal benchmark; papers use incompatible filtering and splits.
- **Leakage risks**: repeated users, upload date, country/device and self-report confounding.
- **Recommended use**: strong cough pretraining and cough-quality route; avoid treating self-reported COVID as equivalent to PCR-linked labels.

### 6.5 UK COVID-19 Vocal Audio / COVID-19 Vocal Audio

- **Tier**: Tier 1.
- **Official URL / citation**: Zenodo release and Alan Turing Institute GitHub; Scientific Data 2024 data paper.
- **Access / license**: open cough/exhalation release under UK Open Government Licence v3.0; speech was collected but excluded from open release because it is more identifiable.
- **Task route**: PCR-linked COVID disease prediction; cough and exhalation analysis; negative-result benchmarking against symptom models.
- **Audio modality/context**: national-scale app/web data with cough, exhalation and speech collection; open edition contains cough/exhalation.
- **Technical format**: WAV files plus CSV metadata; 53.7 GB open release reported.
- **Scale**: 72,999 participants and 70,794 PCR-linked participants reported in official release; exact usable count depends on quality filters.
- **Labels**: PCR-linked SARS-CoV-2 status; symptoms and demographics.
- **Granularity**: subject/session-level test labels; file-level modality.
- **Splits/protocol**: official baseline code exists; strong studies compare audio-only/audio+symptom/symptom-only.
- **Leakage risks**: recruitment, test-date, device, symptom and demographic confounding; leakage if multiple submissions by same participant are mixed.
- **Recommended use**: best public disease-label dataset among COVID audio resources; use it to test whether audio adds value beyond symptoms, not just to maximize audio-only AUC.

### 6.6 Corp Dataset

- **Tier**: Tier 1.
- **Official URL / citation**: C-BiLSTM cough event boundary paper and Tongji MARI dataset/project page.
- **Access / license**: access path unclear from public page; verify by contacting maintainers before planning reproducible benchmark experiments.
- **Task route**: long-duration cough event detection and boundary localization.
- **Audio modality/context**: respiratory-disease patient recordings; long-duration cough monitoring rather than short solicited cough.
- **Technical format**: paper reports 44.1 kHz / 192 kbps; exact file packaging unknown.
- **Scale**: 168 h audio, 9,969 cough events, 42 patients.
- **Labels**: cough event boundaries; disease known at subject level.
- **Granularity**: onset-offset cough events; subject-level disease.
- **Splits/protocol**: paper-defined experiments; patient-independent setup should be required for new work.
- **Leakage risks**: segment-level random split across one long recording or patient can inflate results.
- **Recommended use**: excellent target for event detection if access is obtained; otherwise include as a method-comparison reference, not as a guaranteed dataset.

### 6.7 KAUH / Fraiwan lung-sound dataset

- **Tier**: Tier 2.
- **Official URL / citation**: Mendeley Data `jwyy9np4gv/3`.
- **Access / license**: open Mendeley; license must be read from dataset page/package.
- **Task route**: lung-sound event/file classification and diagnosis-conditioned external validation.
- **Audio modality/context**: electronic stethoscope recordings from chest wall positions.
- **Technical format**: WAV; exact sample rate/bit depth should be extracted from files.
- **Scale**: filename scheme P1-P112 suggests 112 patient IDs; exact file count should be computed after download.
- **Labels**: annotation includes sound type I/E/W/C/N, diagnosis such as asthma/COPD/bronchiectasis/heart failure/lung fibrosis and chest location.
- **Granularity**: file-level and metadata labels; onset-offset not verified.
- **Caveat**: diagnosis labels and event labels are not equivalent; patient-level split is mandatory.
- **Recommended use**: external validation and small-domain transfer for ICBHI/HF models.

### 6.8 RespiratoryDatabase@TR

- **Tier**: Tier 2.
- **Official URL / citation**: Mendeley `p9z4h98s6j/1` and associated DergiPark article.
- **Access / license**: open Mendeley; license terms should be checked before redistribution.
- **Task route**: multi-channel lung sound, COPD severity, domain adaptation and source-location experiments.
- **Audio modality/context**: multi-channel lung sound recordings and heart/lung recordings.
- **Scale**: associated article reports 75 subjects; detailed file/channel counts require download inspection.
- **Labels**: subject-level condition/severity and recording metadata; exact event annotations unknown.
- **Granularity**: likely file/subject-level rather than onset-offset.
- **Caveat**: task definition differs from ICBHI/HF event detection.
- **Recommended use**: external validation for disease/severity and multi-channel representation, not primary event benchmark.

### 6.9 HF_Lung_V2 / HF_Lung_V1_IP

- **Tier**: Tier 2.
- **Source**: Applied Sciences 2022 article and GitLab V1_IP release.
- **Access**: partial/open-plus-request depending version; not all labels may be unrestricted.
- **Route**: event-level respiratory/adventitious detection and robustness benchmarking.
- **Scale/labels**: paper reports 10,554 train and 3,403 test files in V2; event labels are similar in spirit to HF_Lung_V1 but version-specific.
- **Recommended use**: useful if access is granted; do not mix V1/V2 labels without documenting taxonomy differences.

### 6.10 HF_Tracheal_V1

- **Tier**: Tier 2.
- **Source**: arXiv and GitLab.
- **Access**: audio is public in repo; labels may require form/request depending release.
- **Route**: tracheal breath-phase detection and CAS detection.
- **Scale**: about 10k 15-second tracheal recordings are described; exact count should be verified after download.
- **Caveat**: tracheal signals are not equivalent to chest-wall lung sounds; transfer function and event semantics differ.
- **Recommended use**: breath-phase modeling, domain shift from lung to tracheal audio, auxiliary temporal segmentation.

### 6.11 Coswara

- **Tier**: Tier 2.
- **Source**: IISc LEAP GitHub and Scientific Data/arXiv paper.
- **Access**: open GitHub.
- **Route**: multimodal cough/breath/speech disease prediction and representation learning.
- **Audio**: multiple modalities per participant: cough, breathing, speech/counting/vowels.
- **Scale**: paper reports 2,635 individuals and nine sound categories.
- **Labels**: self-reported/test-status metadata; not as strong as PCR-linked registry labels.
- **Caveat**: severe label and recruitment confounds; repeated updates can change counts.
- **Recommended use**: multimodal pipeline proof-of-concept and weakly supervised pretraining; not final clinical evidence.

### 6.12 Cambridge COVID-19 Sounds

- **Tier**: Tier 2.
- **Source**: OpenReview dataset paper, DTA information page, and npj Digital Medicine analysis.
- **Access**: DTA/controlled academic access.
- **Route**: COVID/symptom prediction from cough, breathing and voice.
- **Scale**: paper reports 53,449 samples, 552 h, 36,116 participants, 2,106 positives.
- **Labels**: self-reported COVID/test status and symptoms.
- **Caveat**: data-access friction and self-report confounding; the npj Digital Medicine paper is important because it highlights bias and realistic performance issues.
- **Recommended use**: reproducible only after DTA; good for bias-analysis route.

### 6.13 DiCOVA Challenge datasets

- **Tier**: Tier 2.
- **Source**: DiCOVA challenge website and challenge papers.
- **Access**: challenge-only/restricted.
- **Route**: COVID detection from cough/breath/speech.
- **Scale**: Track 1 cough subset is commonly reported as 1,040 subjects; Track 2 as 990 subjects from Coswara-derived data.
- **Labels**: binary subject-level COVID labels with official folds.
- **Caveat**: comparable only within a specific DiCOVA track/fold/metric.
- **Recommended use**: benchmark reproduction if access is obtained; otherwise cite as historical challenge.

### 6.14 ComParE 2021 COVID-19 Cough/Speech

- **Tier**: Tier 2.
- **Source**: Interspeech/ComParE challenge summary.
- **Access**: challenge data; public availability depends on challenge terms.
- **Route**: COVID detection from cough and speech.
- **Scale**: challenge table reports cough train/dev/test 286/231/208 and speech train/dev/test 315/295/283.
- **Labels**: binary COVID label; subject-level.
- **Caveat**: small and challenge-specific; cannot be merged directly with other COVID audio datasets.

### 6.15 Sound-Dr

- **Tier**: Tier 2.
- **Source**: GitHub and PHM Asia paper.
- **Access**: open repo.
- **Route**: COVID/anomaly prediction from cough, mouth-breathing and nose-breathing.
- **Scale**: 3,930 recordings from 1,310 subjects.
- **Labels**: self-report COVID/symptom labels and modality labels.
- **Caveat**: self-report and small/moderate scale; exact splits should be reproduced from paper/repo.

### 6.16 smarty4covid

- **Tier**: Tier 2.
- **Source**: Scientific Data 2023.
- **Access**: public release with privacy exclusions.
- **Route**: COVID-related cough/breath/voice modeling and audio knowledge-base work.
- **Scale**: paper reports 18,265 recordings from 4,673 users, including cough, regular breathing, deep breathing and voice.
- **Labels**: self-reported health/COVID metadata; an expert-labeled subset exists.
- **Caveat**: voice privacy exclusions and self-report labels.

### 6.17 Virufy COVID-19 Open Cough Dataset

- **Tier**: Tier 2.
- **Source**: Virufy GitHub.
- **Access**: open GitHub.
- **Route**: PCR-linked cough classification and small clinical cough analysis.
- **Scale**: clinical subset described in repository/materials as 121 segmented cough samples from 16 patients; exact version should be pinned.
- **Labels**: PCR-linked positives/negatives where available.
- **Caveat**: small sample size; version drift; not sufficient alone for strong claims.

### 6.18 Solicited TB cough / CODA TB cough resources

- **Tier**: Tier 2.
- **Source**: cough-disease literature and controlled-access challenge/dataset references; no single stable open primary package was verified in this pass.
- **Access**: controlled or unclear.
- **Route**: TB cough disease prediction.
- **Labels**: strongest variants use microbiology/clinical TB labels; exact granularity depends on release.
- **Caveat**: do not use unless access terms and label provenance are confirmed.
- **Recommended use**: long-term disease route after access; not a short-term public benchmark.

### 6.19 Audio-IMU Multimodal Cough Dataset

- **Tier**: Tier 2.
- **Source**: Dryad DOI `10.5061/dryad.0p2ngf26t`.
- **Access**: open Dryad, about 902 MB.
- **Route**: cough event detection, wearable multimodal fusion, out-of-distribution robustness.
- **Audio/context**: two chest-mounted microphones derived from Tozo T10 Bluetooth earbuds plus MetaMotionS r1 9-axis IMU.
- **Scale**: 13 participants, controlled volunteer recordings.
- **Labels**: cough, speech, sneeze, deep breath, groan, laugh, far speech and other.
- **Caveat**: healthy/student volunteer context; not disease prediction.
- **Recommended use**: excellent auxiliary cough/non-cough event detection and multimodal baseline.

### 6.20 VocalSound

- **Tier**: Tier 2.
- **Source**: MIT/CSAIL official page, GitHub and paper.
- **Access**: open for research under dataset terms; verify license before redistribution.
- **Route**: cough/sneeze/sniff/throat-clearing pretraining and vocal event detection.
- **Scale**: official page reports 21,024 crowdsourced recordings from 3,365 subjects.
- **Labels**: laughter, sigh, cough, throat clearing, sneeze, sniff; age, gender, native language, country and health-condition metadata.
- **Caveat**: not a respiratory-disease dataset; use as auxiliary representation/negative/foreground event corpus.

### 6.21 Pulmonary Sound Dataset / ALSD-Net data

- **Tier**: Tier 2.
- **Source**: Mendeley dataset and ALSD-Net citation.
- **Access**: open Mendeley.
- **Route**: normal/abnormal lung sound classification.
- **Labels**: binary normal/abnormal; Mendeley page reports strong class imbalance.
- **Caveat**: no verified onset-offset event labels; exact data count and device specs require package inspection.
- **Recommended use**: binary abnormality external validation, with caution around imbalance.

### 6.22 Clinic Waiting Room Cough Dataset

- **Tier**: Tier 2.
- **Source**: Scientific Reports cough detection paper/project.
- **Access**: raw data not confirmed public; model/paper definition is clear enough to include in method map.
- **Route**: cough event detection in real clinical acoustic environments.
- **Scale**: paper reports 3,930 manually annotated 3-second recordings from 54 sessions and validation on 348 hours of waiting-room audio.
- **Labels**: cough/non-cough clips and deployment event labels.
- **Caveat**: raw data access and session-level split reproducibility need confirmation.

### 6.23 DeepBreath / Pneumoscope pediatric auscultation dataset

- **Tier**: Tier 2.
- **Source**: npj Digital Medicine 2023 paper and public GitHub.
- **Access**: code is public; raw audio available only on reasonable request because consent/privacy do not allow open release.
- **Route**: self-collected but reproducible-method pediatric auscultation paper; pneumonia/wheezing/bronchiolitis modeling.
- **Scale**: GitHub/paper report 4,552 lung auscultation recordings from 572 pediatric outpatients, 35.9 hours, six outpatient departments in five countries.
- **Labels**: diagnosis/pathology labels; train on two sites with internal and external validation in the released code workflow.
- **Caveat**: not immediately downloadable; still useful as a reproducible architecture/protocol reference.

### 6.24 Formosa Archive of Breath Sound (FABS)

- **Tier**: Tier 2.
- **Source**: JMIR AI 2025 paper.
- **Access**: public raw-data access not confirmed.
- **Route**: direct lung-sound abnormality classification with multi-annotator physician labels; useful self-collected paper if model definition is followed.
- **Scale**: paper reports 5,238 annotated recordings, 1,985 patients and 14.6 h, with 715 coarse crackles, 234 wheezes and 4,289 normal recordings.
- **Labels**: seven senior physicians annotated/validated labels according to paper.
- **Caveat**: ED-specific domain, access unclear and class imbalance.

### 6.25 PKU Open Auscultation / open lung-sound resources

- **Tier**: Tier 2 if official package is verified; otherwise Tier 4.
- **Source**: repository leads exist, but official stable citation and metadata still require verification.
- **Route**: potential external lung-sound validation.
- **Caveat**: do not include in experiments until exact source, license, subject count, labels and split rules are confirmed.

### 6.26 Resp-229K

- **Tier**: Tier 2.
- **Source**: Resp-Agent arXiv/OpenReview/GitHub.
- **Access**: GitHub and Hugging Face are announced; repo reports about 70 GB.
- **Route**: dataset fusion/domain adaptation, respiratory foundation-model pretraining, multimodal clinical-narrative modeling.
- **Scale**: paper/repo report 229k audio files and 407+ hours curated from ICBHI, SPRSound, UK COVID, COUGHVID, KAUH and HF Lung V1.
- **Labels**: source-derived labels plus LLM-distilled clinical narratives.
- **Caveat**: label semantics are derived and partly generated; must audit source leakage if downstream evaluation uses constituent datasets.

### 6.27 HLS-CMDS Simulated Heart/Lung Sound Dataset

- **Tier**: Tier 3.
- **Source**: `https://data.mendeley.com/datasets/pxg5f7j5kd/1`.
- **Access**: open Mendeley.
- **Route**: pretraining; denoising; source separation; simulation sanity checks.
- **Audio modality**: clinical manikin heart/lung sounds.
- **Scale / format**: 535 recordings; 50 heart, 50 lung, 145 mixed plus source signals; Littmann CORE, WAV/CSV.
- **Labels / granularity**: source/separation labels; simulated context.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.28 UCL Speech Breath Monitoring / ComParE 2020 Breathing Sub-Challenge

- **Tier**: Tier 3.
- **Source**: `https://www.isca-archive.org/interspeech_2020/schuller20_interspeech.html`.
- **Access**: challenge / research dataset.
- **Route**: speech-breath monitoring; breathing-from-speech auxiliary route.
- **Audio modality**: speech + respiratory belt signal.
- **Scale / format**: 49 speakers; about 4 min per speaker; speech downsampled to 16 kHz; belt 25 Hz.
- **Labels / granularity**: continuous respiratory-belt target over speech.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.29 AudioSet respiratory-related ontology labels

- **Tier**: Tier 3.
- **Source**: `https://research.google.com/audioset/ontology/index.html`.
- **Access**: open metadata; YouTube availability varies.
- **Route**: generic pretraining; weak auxiliary labels.
- **Audio modality**: web audio clips from YouTube.
- **Scale / format**: AudioSet full dataset is large; respiratory-related labels include cough, sneeze, sniff, breathing, snoring, wheeze-like labels depending ontology branch.
- **Labels / granularity**: weak clip-level labels, not clinical.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.30 AudioSet Cough class

- **Tier**: Tier 3.
- **Source**: `https://research.google.com/audioset/ontology/cough_1.html`.
- **Access**: open metadata; YouTube availability varies.
- **Route**: cough detector pretraining; weak positives.
- **Audio modality**: YouTube cough audio/video clips.
- **Scale / format**: ontology page reports 871 cough clips and about 2.4 h; 60 eval, 60 balanced train, 751 unbalanced train.
- **Labels / granularity**: weak clip-level Cough label.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.31 FSD50K / respiratory-sound slices

- **Tier**: Tier 3.
- **Source**: `https://zenodo.org/records/4060432`.
- **Access**: open under source clip licenses.
- **Route**: generic pretraining; weak respiratory event auxiliary.
- **Audio modality**: Freesound clips.
- **Scale / format**: 51,197 clips; 200 classes; >100 h; respiratory-related subsets exist by labels/tags.
- **Labels / granularity**: weak labels from AudioSet ontology/Freesound tags.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.32 ESC-50

- **Tier**: Tier 3.
- **Source**: `https://github.com/karolpiczak/ESC-50`.
- **Access**: open GitHub; CC BY-NC for ESC-50.
- **Route**: small generic cough/breath/sneeze/snoring auxiliary.
- **Audio modality**: environmental sound clips.
- **Scale / format**: 2,000 clips; 50 classes; 5 s; 40 examples/class; 44.1 kHz mono WAV.
- **Labels / granularity**: clip-level class labels; includes coughing, breathing, sneezing, snoring.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.33 cough-speech-sneeze / audEERING corpus

- **Tier**: Tier 3.
- **Source**: `https://audeering.github.io/datasets/datasets/cough-speech-sneeze.html`.
- **Access**: available via conversion/audb tooling.
- **Route**: cough/sneeze/speech auxiliary event modeling.
- **Audio modality**: YouTube-derived speech, cough, sneeze, silence.
- **Scale / format**: exact count should be inspected from audformat/audb release.
- **Labels / granularity**: clip-level cough/sneeze/speech/silence labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.34 OpenSLR Nonverbal Vocalization Dataset

- **Tier**: Tier 3.
- **Source**: `https://www.openslr.org/99/`.
- **Access**: open OpenSLR.
- **Route**: nonverbal vocal event pretraining; cough/sneeze/panting/yawn auxiliaries.
- **Audio modality**: crowdsourced nonverbal vocalizations.
- **Scale / format**: scale not transcribed here; official page lists 16 nonverbal classes.
- **Labels / granularity**: clip-level labels include coughing, yawning, throat clearing, sighing, panting, sneezing etc..
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.35 FluSense hospital audio dataset

- **Tier**: Tier 3.
- **Source**: `https://dl.acm.org/doi/10.1145/3381014`.
- **Access**: paper/project; public raw access unclear.
- **Route**: population-level cough/symptom trend monitoring.
- **Audio modality**: waiting-room audio/video/sensor streams.
- **Scale / format**: paper reports hospital waiting-area deployment; exact raw release not found.
- **Labels / granularity**: cough/sneeze/audio-event and aggregate illness trend labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.36 NoCoCoDa

- **Tier**: Tier 3.
- **Source**: `https://www.nature.com/articles/s41597-023-02646-6`.
- **Access**: dataset page / paper-linked.
- **Route**: non-COVID cough controls; cough negative/positive balance.
- **Audio modality**: cough audio.
- **Scale / format**: exact scale/access terms need inspection from dataset page.
- **Labels / granularity**: cough and non-COVID status labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.37 VOICED Database

- **Tier**: Tier 3.
- **Source**: `https://physionet.org/content/voiced/`.
- **Access**: open PhysioNet.
- **Route**: voice pathology; speech-health background.
- **Audio modality**: sustained vowel voice samples.
- **Scale / format**: 208 clinically verified voice samples; 150 pathological, 58 healthy.
- **Labels / granularity**: speaker-level pathology labels; demographics and VHI/RSI questionnaire metadata.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.38 Saarbruecken Voice Database (SVD)

- **Tier**: Tier 3.
- **Source**: `https://stimmdb.coli.uni-saarland.de/`.
- **Access**: open web interface/download terms vary.
- **Route**: voice pathology; speech-health auxiliary.
- **Audio modality**: sustained vowels and speech tasks.
- **Scale / format**: large voice pathology database; exact current count should be checked from official site.
- **Labels / granularity**: speaker/pathology labels; speech task metadata.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.39 Bridge2AI-Voice Adult Dataset

- **Tier**: Tier 3.
- **Source**: `https://physionet.org/content/b2ai-voice/`.
- **Access**: PhysioNet derived features open/controlled; raw audio via Synapse/DACO.
- **Route**: speech-health and multimodal health monitoring; respiratory/voice-disorder context.
- **Audio modality**: voice-derived features; raw audio controlled.
- **Scale / format**: PhysioNet v3.1.0 published May 2026; raw audio not in PhysioNet release.
- **Labels / granularity**: derived features, questionnaires, clinical/demographic metadata.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.40 Bridge2AI-Voice Pediatric Dataset

- **Tier**: Tier 3.
- **Source**: `https://physionet.org/content/b2ai-voice-pediatric/`.
- **Access**: PhysioNet derived features; raw access controlled.
- **Route**: pediatric speech-health monitoring; respiratory/voice biomarker context.
- **Audio modality**: voice-derived features/spectrograms and clinical metadata.
- **Scale / format**: PhysioNet v1.1.0 published May 2026.
- **Labels / granularity**: derived features, spectrograms, questionnaire and clinical metadata.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.41 OPERA benchmark/resource

- **Tier**: Tier 3.
- **Source**: `https://github.com/evelyn0414/OPERA`.
- **Access**: open code; underlying datasets have mixed access/licenses.
- **Route**: respiratory acoustic foundation-model benchmarking.
- **Audio modality**: fused respiratory acoustic tasks.
- **Scale / format**: GitHub/OpenReview report about 136k samples, 440 h, 19 tasks.
- **Labels / granularity**: task labels inherited from constituent datasets.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.42 Google HeAR / Health Acoustic Representations

- **Tier**: Tier 3.
- **Source**: `https://huggingface.co/google/hear`.
- **Access**: model gated on Hugging Face; paper/model docs public.
- **Route**: frozen embedding baseline; health-audio representation.
- **Audio modality**: generic health acoustic embeddings.
- **Scale / format**: paper/model docs report 313M two-second clips; evaluation on 33 tasks across six datasets.
- **Labels / granularity**: not a dataset label source; representation model.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.43 RALE Lung Sounds Repository

- **Tier**: Tier 4.
- **Source**: `https://www.rale.ca/repository.htm`.
- **Access**: web-listened education/repository; research download/metadata limited.
- **Route**: reference examples only.
- **Audio modality**: respiratory sounds.
- **Scale / format**: official page groups recordings into Normal, Wheezes, Crackles and Other; exact research-ready metadata unknown.
- **Labels / granularity**: coarse category labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.44 SNUCH Pediatric Lung Sound Dataset

- **Tier**: Tier 4.
- **Source**: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9871007/`.
- **Access**: private / not public.
- **Route**: pediatric lung-sound method evidence.
- **Audio modality**: pediatric lung sounds.
- **Scale / format**: paper reports 680 eligible training/internal-validation clips and 90 prospective-validation clips.
- **Labels / granularity**: normal/crackle/wheeze labels by pediatric pulmonologists.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.45 MIT Open Voice / cough AI preliminary resources

- **Tier**: Tier 4.
- **Source**: `https://www.media.mit.edu/projects/open-voice/overview/`.
- **Access**: project/repository status unclear.
- **Route**: speech/cough health monitoring background.
- **Audio modality**: voice/cough depending release.
- **Scale / format**: unknown / not found as stable raw-audio dataset.
- **Labels / granularity**: unknown.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.46 MEEI Voice Disorders Database

- **Tier**: Tier 4.
- **Source**: `https://masseyeandear.org/research/otolaryngology/voice-and-speech`.
- **Access**: commercial/restricted legacy dataset.
- **Route**: voice pathology background.
- **Audio modality**: voice/speech.
- **Scale / format**: secondary papers report >1,400 recordings or 657 pathological + 53 normal subsets; exact accessible package varies.
- **Labels / granularity**: voice pathology labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.47 PEEP / respiratory-support physiological datasets without raw audio

- **Tier**: Tier 4.
- **Source**: `https://physionet.org/`.
- **Access**: open/non-audio; not central.
- **Route**: clinical context only.
- **Audio modality**: not raw respiratory audio.
- **Scale / format**: varies.
- **Labels / granularity**: not acoustic labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.48 Open-source AI Stethoscope / COPD HeAR demo resources

- **Tier**: Tier 4.
- **Source**: `https://github.com/search?q=open-source+AI+stethoscope+COPD+HeAR&type=repositories`.
- **Access**: GitHub/demo; paper/dataset status not fully verified.
- **Route**: method lead only.
- **Audio modality**: stethoscope lung sounds.
- **Scale / format**: repo claims COPD AUROC but source dataset/access need verification.
- **Labels / granularity**: unknown.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.49 Kaggle ICBHI respiratory-sound mirrors

- **Tier**: Tier 4.
- **Source**: `https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database`.
- **Access**: open mirrors; often unofficial.
- **Route**: convenient download only, not primary source.
- **Audio modality**: mirrored ICBHI lung sounds.
- **Scale / format**: 920 recordings when faithful; mirror metadata may drift.
- **Labels / granularity**: inherits ICBHI labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.50 YouTube/Freesound cough compilation datasets

- **Tier**: Tier 4.
- **Source**: `https://freesound.org/`.
- **Access**: varies; often weakly documented.
- **Route**: negative/auxiliary only.
- **Audio modality**: web cough/sneeze/breath clips.
- **Scale / format**: unknown; many mirrors duplicate AudioSet/FSD/ESC.
- **Labels / granularity**: weak clip tags.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.51 Simulated/body-sound teaching libraries

- **Tier**: Tier 4.
- **Source**: `https://www.easyauscultation.com/`.
- **Access**: varies; educational license.
- **Route**: qualitative listening/reference only.
- **Audio modality**: lung/heart/respiratory examples.
- **Scale / format**: unknown / not research-ready unless metadata available.
- **Labels / granularity**: coarse teaching labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.

### 6.52 General speech corpora sometimes used for health-audio pretraining

- **Tier**: Tier 4.
- **Source**: `https://www.openslr.org/`.
- **Access**: open but weak connection.
- **Route**: pretraining only; not respiratory-health evidence.
- **Audio modality**: ordinary speech.
- **Scale / format**: large but not respiratory-specific.
- **Labels / granularity**: transcript/speaker labels, not disease labels.
- **Main caveat**: unknown fields should be verified from the official package before training or benchmarking. For Tier 3/4 resources, this report treats the dataset as auxiliary or low-actionability rather than core evidence.
- **Recommended use**: use only for the route stated above; avoid converting weak web-audio labels or voice-pathology labels into respiratory disease labels.


---

## 7. Dataset-To-Paper / Dataset-To-Method Map

### 7.1 Coverage check

本版 paper map 共 **56** 个条目，其中 **23** 个为 2024-2026 年近两年工作，满足“至少 40 篇 paper 且至少 10 篇近两年 paper”的目标。表中“近两年”用 ✓ 标记。这里的“paper/method map”包含数据集论文、挑战论文、使用公开数据集的方法论文、foundation model 论文，以及自采集但具有公开 repo/清晰模型定义/明确协议的论文。若数据私有但模型定义清晰，本报告将其标为 **not public-dataset comparable**，只作为方法启发或未来合作参考。

### 7.2 Expanded paper map

| # | 近两年 | 论文/资源 | 年份 | 数据集 | 任务 | 模型/方法族 | 评测/结果摘要 | 代码/复现性 | 可比性 |
|---|---|---|---|---|---|---|---|---|---|
| 1 |  | A Respiratory Sound Database for the Development of Automated Classification | 2019 | ICBHI 2017 | dataset/challenge definition | database paper and baseline challenge task | Official challenge uses respiratory-cycle classification; use as source for scale/labels | data open | yes, only under official split/metric |
| 2 |  | HF_Lung_V1: An open-access lung sound database with event annotations | 2021 | HF_Lung_V1 | event detection | dataset plus baseline RNN/event detector | Event onset-offset labels for I/E/CAS/DAS; metrics paper-specific | data/code pages open | partly |
| 3 |  | HF_Lung_V2 and ensemble baseline for lung sound event detection | 2022 | HF_Lung_V2 | event detection | ensemble classifier/detector | V2 train/test files reported; labels partly controlled | partial/code unknown | partly |
| 4 |  | HF_Tracheal_V1 tracheal sound database and CAS detection | 2021 | HF_Tracheal_V1 | tracheal breath/CAS detection | dataset plus temporal detection baselines | About 10k tracheal 15-s recordings; label access varies | audio open; labels request | partly |
| 5 |  | SPRSound: Open-Access Pediatric Respiratory Sound Database | 2022 | SPRSound | pediatric respiratory event detection | dataset/challenge baselines | Original database reports 2,683 records, 292 participants, 9,089 events | data/challenge open | yes within challenge |
| 6 |  | COUGHVID: A cough audio dataset for COVID-19 research | 2021 | COUGHVID | cough quality and weak COVID analysis | dataset, expert cough-quality labels, baseline analysis | 25k+ recordings and 2,800+ expert labels in paper | data open | limited; splits differ |
| 7 |  | Coswara: A respiratory sounds and symptoms dataset for remote screening | 2023 | Coswara | multimodal COVID/symptom prediction | dataset and baseline models over cough/breath/speech | 2,635 individuals; self-report/test status labels | data open | limited |
| 8 |  | COVID-19 Sounds: A Large-Scale Audio Dataset | 2021 | Cambridge COVID-19 Sounds | COVID/symptom prediction | dataset and ML baselines | 53,449 samples, 552 h, 36,116 participants; self-report status | DTA | partly within DTA |
| 9 |  | Sounds of COVID-19: exploring realistic performance and bias | 2021 | Cambridge COVID-19 Sounds | COVID audio prediction and bias analysis | ML baselines plus bias/control analyses | Shows that confounding and bias affect apparent audio performance | code/data controlled | not comparable to challenge metrics |
| 10 |  | DiCOVA Challenge: Diagnosing COVID-19 using acoustics | 2021 | DiCOVA/Coswara subset | COVID detection | challenge baselines; cough/breath/speech tracks | Track-specific AUC/UAR metrics; official folds | challenge access | yes within track |
| 11 |  | Second DiCOVA Challenge baseline and protocol | 2021 | DiCOVA Track 2 | COVID detection | challenge baselines with expanded modalities | Subject-level binary labels and official challenge protocol | challenge access | yes within track |
| 12 |  | INTERSPEECH ComParE 2021 COVID-19 Cough/Speech Sub-Challenges | 2021 | ComParE COVID Cough/Speech | COVID detection | challenge baselines over cough/speech features | Challenge table gives train/dev/test splits and official metrics | challenge access | yes within track |
| 13 | ✓ | UK COVID-19 Vocal Audio Dataset | 2024 | UK COVID-19 Vocal Audio | PCR-linked disease dataset | dataset paper and baseline code | 72,999 participants; 70,794 PCR-linked; open cough/exhalation release | data+code open | yes if official code/protocol used |
| 14 | ✓ | Audio-based AI classifiers show no evidence of improved COVID-19 screening over symptom checkers | 2024 | UK COVID Vocal; COVID-19 Sounds | disease prediction validity | comparative baselines against symptoms | Important negative result: audio classifiers did not show useful improvement over symptom checkers in tested settings | code/data links available via project | partly; focus is validity |
| 15 |  | Sound-Dr: A dataset and benchmark for COVID-19 and respiratory symptoms | 2023 | Sound-Dr | COVID/symptom/anomaly prediction | CNN/RNN/ML baselines over cough and breathing | 3,930 recordings, 1,310 subjects; paper-specific metrics | GitHub open | partly |
| 16 |  | smarty4covid: A COVID-19 audio knowledge base | 2023 | smarty4covid | cough/breath/voice dataset and baseline tasks | dataset plus modality/segment baselines | 18,265 recordings, 4,673 users | data release described | limited |
| 17 | ✓ | A deep learning framework to detect COVID-19 from cough audio signals | 2025 | COUGHVID | COVID cough segmentation/classification | U-Net style cough segmentation plus deep classifier | Paper reports COVID cough prediction pipeline; metrics source-specific | code unknown | limited |
| 18 | ✓ | SympCoughNet: symptom-aware cough network for COVID-19 audio | 2025 | COVID cough datasets | COVID cough+symptom prediction | dual-channel cough and symptom weighting/fusion network | Reported results are dataset/protocol-specific | code unknown | limited |
| 19 |  | Project Achoo: a practical cough-based COVID-19 detection app | 2022 | self-collected / app data | COVID cough app | ML pipeline with mobile app data | Paper-specific validation; useful engineering reference | code unknown | not comparable |
| 20 |  | COVID-19 detection using cough, breath and speech with deep learning | 2021 | Coswara and related respiratory audio | multimodal COVID prediction | CNN/RNN/feature-fusion models | Metrics depend on dataset filtering and self-report labels | code unknown | limited |
| 21 |  | A C-BiLSTM cough event boundary detector and Corp Dataset | 2022 | Corp Dataset | long-recording cough event detection | C-BiLSTM boundary regression/detection | Source reports sensitivity 84.13%, specificity 99.82%, IoU 0.89 | data page linked; code unknown | partly; patient split required |
| 22 |  | Automated cough detection in a clinic waiting room | 2021 | Clinic Waiting Room Cough | cough event detection | YAMNet-style/ensemble cough detector | Paper reports high ROC-AUC around 98 in validation and deployment testing | project URL; raw data unclear | partly |
| 23 | ✓ | Audio-based cough event detection on multimodal wearable audio-IMU data | 2024 | Audio-IMU Cough Dataset | cough event detection | audio, IMU and multimodal classifiers | Dryad abstract reports 92.59% in-subject and 90.79% cross-subject accuracy | data open | partly |
| 24 |  | VocalSound: A dataset for human non-speech vocalization recognition | 2022 | VocalSound | vocal event recognition | CNN/Transformer-style baselines for cough/sneeze/sniff etc. | 21,024 recordings, 3,365 subjects; six classes | data+code available | yes for vocal-event task |
| 25 |  | Hybrid CNN-LSTM lung-sound classification with focal loss | 2022 | ICBHI 2017 | lung sound classification | hybrid CNN-LSTM / focal-loss variants | Uses multiple splitting schemes; exact results depend on split | code unknown | only if split matched |
| 26 |  | Patch-Mix Contrastive Learning with Audio Spectrogram Transformer | 2023 | ICBHI 2017 | respiratory sound classification | AST plus patch-level mix and contrastive learning | GitHub reports ICBHI experiments; metrics split-dependent | code open | partly |
| 27 |  | Pretraining Respiratory Sound Representations Using Metadata and Contrastive Learning | 2023 | ICBHI; SPRSound | representation learning | metadata-aware contrastive pretraining | Official PyTorch includes ICBHI/SPRSound workflows | code open | partly |
| 28 | ✓ | Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain Adaptation | 2024 | ICBHI 2017 | lung sound classification/domain adaptation | supervised contrastive learning plus stethoscope/domain adversarial training | Repo/source reports ICBHI score 61.71 and +2.16 over baseline | code open | yes if official split |
| 29 | ✓ | Multi-View Spectrogram Transformer for Respiratory Sound Classification | 2024 | ICBHI 2017 | lung sound classification | multi-view mel spectrogram transformer | ICBHI experiments; reports SOTA-family score in 2024 papers | code open | yes if official split |
| 30 | ✓ | BTS: Metadata-Aided Respiratory Sound Classification | 2024 | ICBHI 2017 | lung sound classification | text-audio / metadata-aided model using stethoscope/location metadata | Public sources report strong ICBHI score around 63.54; verify exact table before citation | code unknown/limited | yes if official split |
| 31 | ✓ | RepAugment: Input-Agnostic Representation-Level Augmentation | 2024 | ICBHI 2017 | lung sound classification | representation-level augmentation on pretrained respiratory/audio models | Paper reports minority-class improvements and ICBHI gains | code open | yes if official split |
| 32 | ✓ | OPERA: Towards Open Respiratory Acoustic Foundation Models | 2024 | OPERA benchmark datasets | foundation model benchmark | respiratory acoustic pretraining and 19 downstream tasks | OpenReview/GitHub report improvement over general acoustic models on 16/19 tasks | code open | benchmark-level, not per-dataset universal |
| 33 | ✓ | HeAR: Health Acoustic Representations | 2024 | multiple health-audio datasets | foundation model | large-scale ViT-MAE-style health acoustic embeddings | Paper/model docs report 313M 2-s clips and evaluation on 33 tasks across six datasets | model gated/open docs | task-dependent |
| 34 | ✓ | JMIR AI FABS audio enhancement / breath-sound classification paper | 2025 | FABS; ICBHI; NTUH noise | lung abnormality classification under noise | deep learning with physician-annotated FABS and noise robustness experiments | Scale: FABS 5,238 recordings; results paper-specific | code/data unknown | not yet comparable |
| 35 | ✓ | Patient-Aware Feature Alignment for Robust Lung Sound Classification | 2025 | ICBHI 2017 | lung sound classification | patient-aware feature alignment / robustness method | Public sources report SOTA-family ICBHI results; exact comparability needs split audit | code unknown/limited | yes if official split |
| 36 | ✓ | Architecture-Agnostic Knowledge Distillation from Ensembles | 2025 | ICBHI 2017 | lung sound classification | ensemble teacher to lightweight/student models | Interspeech/GitHub release; results compare with BTS/PAFA/M2D in paper | code open | yes if official split |
| 37 | ✓ | Adaptive Differential Denoising for Respiratory Sounds Classification | 2025 | ICBHI 2017 | lung sound classification/denoising | adaptive differential denoising with respiratory classifier | Sources report ICBHI Score 65.53 and +1.99 over previous SOTA | code open | yes if official split |
| 38 | ✓ | Towards Pre-training an Effective Respiratory Audio Foundation Model | 2025 | OPERA; ICBHI; HF_Lung; COUGHVID; AudioSet | foundation/pretraining comparison | M2D+Resp and comparison of 21 audio foundation models | Paper reports M2D+Resp average AUROC 0.814 vs prior OPERA-CT 0.733 over OPERA tasks | code open/source public | benchmark-level |
| 39 | ✓ | Geometry-Aware Optimization for Robust Respiratory Sound Classification | 2025 | ICBHI 2017 | lung sound classification | AST with sharpness/geometry-aware optimization | Public sources report official ICBHI score 68.10 and sensitivity 68.31; verify arXiv/repo version | code open | yes if official split |
| 40 | ✓ | Resp-Agent: Multimodal Respiratory Sound Generation and Disease Diagnosis | 2026 | Resp-229K + downstream datasets | multimodal respiratory diagnosis/generation | agentic curriculum, modality-weaving diagnoser, flow-matching generator | Introduces 229k-recording/407+h Resp-229K; claims >5 point official ICBHI improvement over prior method | code+data announced | needs independent audit |
| 41 | ✓ | E-RespiNet: LLM-ELECTRA driven triple-stream CNN | 2025 | ICBHI and/or respiratory sound datasets | respiratory sound classification | LLM-ELECTRA/triple-stream CNN design | PMC paper reports high respiratory sound classification results; protocol must be checked | code unknown | only if split matched |
| 42 | ✓ | Enhanced respiratory sound classification using deep noise suppression | 2025 | ICBHI/respiratory sound datasets | lung sound classification | deep denoising plus classifier pipeline | JCM/PMC paper-specific results; verify split before comparison | code unknown | limited |
| 43 | ✓ | Pediatric Asthma Detection with Google HeAR | 2025 | SPRSound | pediatric asthma-indicative modeling | HeAR embeddings plus downstream classifier | Preprint/source reports over 91% accuracy; label/task semantics require audit | code unknown | limited |
| 44 | ✓ | iMedic: smartphone self-auscultation for pediatric pneumonia/abnormal respiratory sounds | 2025 | self-collected smartphone dataset | pediatric abnormal respiratory sound screening | mobile self-auscultation ML pipeline | Model/protocol described; raw data access not confirmed | code unknown | not comparable |
| 45 |  | DeepBreath: Deep Learning of Breathing Patterns for Automatic Pneumonia Diagnosis | 2023 | DeepBreath/Pneumoscope private dataset | pediatric diagnosis/pathology | CNN/attention architecture with internal/external validation | Repo reports AUROC 0.93/0.89 healthy-vs-pathologic internal/external; other disease AUROCs reported | code open; raw data request | not public-dataset comparable |
| 46 |  | Deep learning classification of pediatric lung sounds from SNUCH | 2023 | SNUCH private pediatric lung sounds | normal/crackle/wheeze classification | classical ML and CNN-style classifiers | Paper reports prospective validation; exact metrics task-dependent | code/data unknown | not comparable |
| 47 | ✓ | Neural Cough Counter | 2024 | self-collected / field cough data | cough counting/event detection | neural cough event detector for long monitoring | Paper model definition is clear; raw data/code status requires check | code unknown | not comparable unless release found |
| 48 |  | VOICED: A validated database for voice disorder research | 2018 | VOICED | voice pathology | dataset resource and baseline voice pathology work | 208 samples; 150 pathological and 58 healthy | data open PhysioNet | voice-task only |
| 49 |  | Saarbruecken Voice Database common protocol paper | 2021 | SVD | voice pathology | common-protocol benchmark and critique of prior non-comparability | Paper emphasizes that previous SVD results are often non-comparable without common splits | data open/site | voice-task only |
| 50 |  | ComParE 2020 Breathing Sub-Challenge | 2020 | UCL Speech Breath Monitoring | breathing-from-speech regression | audio features and regression baselines to respiratory belt signal | 49 speakers, 4-min spontaneous speech each; official train/dev/test | challenge data | yes within challenge |
| 51 |  | ESC-50: Dataset for Environmental Sound Classification | 2015 | ESC-50 | auxiliary audio pretraining | dataset/baselines; includes cough/breath/sneeze/snore classes | 2,000 5-s clips, 50 classes, 5 folds | data open CC BY-NC | generic only |
| 52 |  | FSD50K: an open dataset of human-labeled sound events | 2022 | FSD50K | auxiliary audio pretraining | large Freesound weak-label dataset | 51,197 clips, 200 classes, >100 h | data open with licenses | generic only |
| 53 |  | AudioSet: An ontology and human-labeled dataset for audio events | 2017 | AudioSet | generic pretraining | weakly labeled YouTube audio ontology | Includes cough/breath/sneeze/snore-related labels; Cough class page reports 871 clips | metadata open | generic only |
| 54 |  | cough-speech-sneeze nonverbal corpus paper/release | 2017 | cough-speech-sneeze | auxiliary cough/sneeze/speech detection | nonverbal audio event classification | YouTube-derived cough/sneeze/speech/silence corpus | audb/audformat tooling | generic only |
| 55 |  | OpenSLR Nonverbal Vocalization Dataset | 2021 | OpenSLR NVD | auxiliary nonverbal vocalization recognition | 16-class nonverbal vocalization corpus | Includes cough, yawn, throat-clearing, sigh, panting, sneeze etc. | data open | generic only |
| 56 |  | Class-imbalanced voice pathology detection and database critique | 2021 | MEEI/SVD voice datasets | voice pathology | class-imbalance-aware voice pathology models | Useful for speech-health evaluation risks, not respiratory acoustic events | code unknown | voice-task only |

### 7.3 Current SOTA snapshot and comparability notes

**ICBHI 2017** 是当前最集中、最容易出现“看似 SOTA 但不可比”的数据集。应只把使用官方 split、官方 score 或明确 patient-independent protocol 的结果放在同一表中比较。2024-2026 年可重点复现或对照的方向包括 MVST、BTS、RepAugment、Stethoscope-guided SupCon、ADD-RSC、Architecture-Agnostic KD、Geometry-aware AST+SAM 和 Resp-Agent。公开来源中，ADD-RSC 报告 ICBHI Score 65.53，Geometry-aware AST+SAM 报告 Score 68.10，Resp-Agent 报告进一步提升；这些结果需要逐条核查 split、augmentation、pretraining 数据是否包含测试污染、是否使用 patient metadata，以及是否和官方 ICBHI score 完全同口径。

**OPERA / respiratory foundation models** 的近两年趋势是从 generic AudioSet-pretrained model 转向 health/respiratory-specific representation。OPERA 2024 提供 19-task benchmark；Niizumi et al. 2025 在 OPERA route 上比较 21 个 audio foundation models，并报告 M2D+Resp 的平均 AUROC 相比 OPERA-CT 明显提升。这个方向适合写成 initial result：frozen embedding + lightweight classifier + cross-dataset transfer，而不是从零训练大型模型。

**COVID/cough disease prediction** 的最新重点不是单纯追求 AUC，而是验证 audio 是否相对 symptom/metadata baseline 有增量。UK COVID Vocal 2024 和 Coppock et al. 2024 的负结果非常重要：任何 COVID audio paper 都应加入 symptom-only、metadata-only 和 audio+symptom 对比，避免把 recruitment/device/symptom confounds 当作声学疾病信号。

**自采集但可复现论文** 对 long-term collaboration 很有价值。DeepBreath、FABS、SNUCH、Clinic Waiting Room Cough、Audio-IMU Cough、Neural Cough Counter、iMedic 等代表真实采集路线。短期不一定能下载原始数据，但可以复现模型定义、标注流程、site-independent validation 和 deployment metrics 的写法，为 UNC Hospital 后续真实数据采集设计提供模板。

---

## 8. Cross-Dataset Aggregation And Compatibility Analysis

### 8.1 Label compatibility

跨数据集合并时至少要拆成五层标签：

1. **Acoustic event layer**: normal, crackle/fine crackle/coarse crackle, wheeze, rhonchi, stridor, cough, breath/inhalation/exhalation, poor quality/noise。
2. **Respiratory phase layer**: inhalation, exhalation, pause/hold, respiratory belt trajectory。
3. **Symptom layer**: cough symptom, dyspnea, fever, sore throat, nasal congestion, self-reported wheeze。
4. **Disease/test layer**: PCR-linked COVID, self-reported COVID, antigen/test status, TB microbiology, COPD/asthma/pneumonia/bronchiolitis diagnosis, voice pathology。
5. **Confound/metadata layer**: age, sex/gender, language, country, device, stethoscope, chest location, hospital/site, recording date, recruitment route, microphone distance, recording environment。

可以防守的映射包括：ICBHI crackle/wheeze、SPRSound crackle/wheeze/rhonchi/stridor、HF CAS/DAS 到粗粒度 adventitious sound layer；COUGHVID/VocalSound/AudioSet Cough 到 cough event layer；UK Vocal/Cambridge/Coswara 到 disease/test layer但保留 PCR-linked vs self-report 区分。不可防守的映射包括：把 self-reported COVID positive 等同于 PCR positive；把 wheeze event 等同于 asthma diagnosis；把 voice pathology 等同于 respiratory disease。

### 8.2 Audio/device compatibility

Body-contact lung sounds、tracheal sounds、smartphone cough、far-field clinic audio、web speech 和 generic YouTube audio 的传递函数完全不同。HF_Lung_V1 的 4 kHz/16-bit body-sound data 与 COUGHVID 的 web audio、UK Vocal 的 WAV cough/exhalation、AudioSet 的 YouTube weak labels 不应在同一个 front-end 下默认为同分布。跨数据集实验应报告 resampling、normalization、band-pass filtering、silence trimming、quality filtering 和 device/site metadata use。

### 8.3 Split and leakage rules

最低协议要求是 **subject-independent split**。对长录音或 repeated recording 数据，还需要 session-independent 或 site-independent split。特别危险的泄漏包括：ICBHI cycle-level random split；HF/KAUH/SPRSound 的 same patient multiple files；COUGHVID repeated users/devices；Corp/Clinic waiting-room 的同一 session 切片；Resp-229K 与下游测试集源重叠；foundation model 在 evaluation data 上的 pretraining contamination。

### 8.4 Pediatric/adult mismatch

SPRSound、DeepBreath、SNUCH、Bridge2AI Pediatric 属于 pediatric route。儿童呼吸音的 airway size、疾病谱、哭声/说话噪声、合作程度、设备位置和临床标签与成人数据明显不同。应把 pediatric route 独立报告，或作为 cross-age adaptation/transfer learning，而不是和成人 ICBHI/HF/KAUH 无条件合并。

### 8.5 Clinical-label validity

疾病预测论文的核心不是“模型能不能从音频预测标签”，而是“标签是否可靠、预测是否相对症状/元数据有增量、结果是否在外部 cohort 上保持”。UK Vocal 强于 self-report COVID datasets，因为其标签与 PCR 结果连接；但即便 PCR-linked，也仍需处理 test date、症状状态、设备、人口统计和 recruitment confounds。

---

## 9. Label-Semantics Risks And Protocol Risks

1. **Disease label vs acoustic-event label**: wheeze/crackle/cough 是声学事件，不是疾病诊断。
2. **Self-report vs PCR-linked label**: Coswara/Cambridge/COUGHVID/Sound-Dr/smarty4covid 多包含自报告或混合状态；UK Vocal 的 PCR-linked 标签更强，但也不能消除采样偏差。
3. **Symptom label vs disease label**: cough、dyspnea、fever、wheezing symptom 是症状，不能当作病种标签。
4. **File-level vs onset-offset labels**: file-level abnormality 不能直接和 onset-offset event detection 比较。
5. **Patient/session leakage**: 随机切 cycle/clip/event 会显著虚高结果。
6. **Weak web labels**: AudioSet/FSD/ESC/OpenSLR 标签不是临床标签。
7. **Foundation-model contamination**: 大规模 web/health pretraining 可能已经见过公开测试音频或其近似版本；需要 reporting。
8. **Derived-label risk**: Resp-229K 的 LLM-distilled narrative 对方法有启发，但标签语义必须审计。
9. **Access reproducibility**: FABS/DeepBreath/SNUCH/Cambridge/DiCOVA/ComParE 受控或私有，不能作为短期 guarantee。
10. **Metadata shortcuts**: device/site/country/date/symptom can dominate; disease-prediction models must include metadata-only baselines.

---

## 10. Project-Fit Recommendations

### 10.1 最适合短期 initial result 的路线

建议短期做 **frozen foundation/audio representation + lightweight classifier + robust split audit**，而不是从零训练复杂模型。可组合三条实验：

- **Lung event route**: ICBHI official split + HF_Lung_V1 event-level + SPRSound pediatric external validation。模型包括 CNN/AST/MVST/BTS-style metadata variant/HeAR/OPERA/M2D embeddings。
- **Cough event route**: COUGHVID cough-quality + Audio-IMU cough + Clinic/Corp paper protocol reference。重点是 cough/non-cough、quality filtering、boundary detection，不直接疾病诊断。
- **Disease-label route**: UK COVID Vocal audio-only vs symptom-only vs audio+symptom。把 Coppock et al. 2024 作为必须讨论的 negative benchmark。

### 10.2 最适合投 workshop / short paper 的路线

一个可控题目是：**Do respiratory acoustic foundation embeddings transfer across lung-sound, cough, and pediatric respiratory event datasets?** 具体实验是统一前处理、统一 subject-independent split、比较 MFCC/CNN、AST/BEATs、HeAR、OPERA/M2D-style embeddings，在 ICBHI/HF/SPRSound/COUGHVID/Audio-IMU 上报告同口径指标。贡献点不是“临床诊断”，而是 dataset landscape-aware benchmark 和 protocol audit。

### 10.3 长期 UNC Hospital collaboration 路线

长期应转向 **prospective data collection + clinician annotation + external validation**。DeepBreath、FABS、SNUCH、Clinic Waiting Room Cough 和 iMedic 的价值是提供采集/标注/验证设计模板：记录设备、位置、环境噪声、症状、诊断、年龄、性别、临床路径；确保 train/test 按 patient/site/time 分离；同时保留声学事件标签和疾病/症状标签两层。

### 10.4 Speech-health / telehealth route

Bridge2AI-Voice Adult/Pediatric、VOICED、SVD、MEEI 等适合做 voice health / speech-health background，而不是 lung-sound 替代。可发展为远程健康监测路线：speech-breath coupling、voice pathology、respiratory symptom self-report plus acoustic biomarkers、multimodal clinical metadata。短期不要把这条线和 stethoscope lung-sound event detection 混成一个 label space。

---

## 11. Open Questions And Access Blockers

1. **Cambridge COVID-19 Sounds**: DTA 获取流程、当前可用版本、是否包含全部 cough/breath/voice。
2. **DiCOVA/ComParE data**: challenge 数据是否仍可申请，是否允许新论文复现。
3. **Corp Dataset**: 原始长录音和标注是否可下载，是否有 DTA。
4. **FABS**: 是否计划公开，或只能通过合作访问。
5. **DeepBreath/Pneumoscope**: reasonable request 的实际门槛、IRB/consent 限制和可用字段。
6. **Resp-229K**: 需要审计源数据重叠、license compatibility、LLM narrative 生成方式和 downstream test contamination。
7. **Bridge2AI raw audio**: PhysioNet release 主要是 derived features；raw audio 需要 Synapse/DACO 和机构签署。
8. **HF_Lung_V2 / HF_Tracheal labels**: 部分标签访问路径需要表单/请求。
9. **SOTA comparability**: ICBHI 新论文的 score 必须逐个核查 split、pretraining、augmentation 和 leakage。
10. **Licenses**: COUGHVID/UK Vocal/HF/AudioSet/FSD/VocalSound/ESC 的许可证不一致，不能直接重发布融合数据集。

---

## 12. Source Bibliography With URLs

| # | Source | URL |
|---|---|---|
| 1 | ICBHI official mirror | https://ai4eu.dei.uc.pt/respiratory-sounds-dataset/ |
| 2 | ICBHI challenge page | https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge |
| 3 | Rocha et al. ICBHI database PubMed | https://pubmed.ncbi.nlm.nih.gov/30708353/ |
| 4 | HF_Lung_V1 GitLab | https://gitlab.com/techsupportHF/HF_Lung_V1 |
| 5 | HF_Lung_V1 PLOS ONE | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254134 |
| 6 | HF_Lung_V2 Applied Sciences | https://www.mdpi.com/2076-3417/12/15/7623 |
| 7 | HF_Tracheal_V1 GitLab | https://gitlab.com/techsupportHF/HF_Tracheal_V1 |
| 8 | SPRSound GitHub | https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound |
| 9 | SPRSound paper DOI | https://doi.org/10.1109/TBCAS.2022.3204910 |
| 10 | COUGHVID EPFL | https://www.epfl.ch/labs/esl/index-html/datasets/coughviddataset/ |
| 11 | COUGHVID Zenodo | https://zenodo.org/records/7024894 |
| 12 | COUGHVID Scientific Data | https://www.nature.com/articles/s41597-021-00937-4 |
| 13 | UK COVID Vocal Zenodo | https://zenodo.org/records/10043978 |
| 14 | UK COVID Vocal GitHub | https://github.com/alan-turing-institute/Turing-RSS-Health-Data-Lab-Biomedical-Acoustic-Markers |
| 15 | UK COVID Vocal Scientific Data | https://www.nature.com/articles/s41597-024-03132-3 |
| 16 | Coppock et al. audio vs symptoms | https://www.nature.com/articles/s42256-024-00815-3 |
| 17 | Corp Dataset paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC9760237/ |
| 18 | KAUH Mendeley | https://data.mendeley.com/datasets/jwyy9np4gv/3 |
| 19 | RespiratoryDatabase@TR Mendeley | https://data.mendeley.com/datasets/p9z4h98s6j/1 |
| 20 | Coswara GitHub | https://github.com/iiscleap/Coswara-Data |
| 21 | Coswara arXiv | https://arxiv.org/abs/2305.12741 |
| 22 | Cambridge COVID-19 Sounds OpenReview | https://openreview.net/forum?id=9KArJb4r5ZQ |
| 23 | COVID-19 Sounds DTA blog | https://www.covid-19-sounds.org/en/blog/neurips_dataset |
| 24 | Sounds of COVID-19 npj Digital Medicine | https://www.nature.com/articles/s41746-021-00553-x |
| 25 | DiCOVA challenge | https://dicovachallenge.github.io/ |
| 26 | DiCOVA Interspeech paper | https://www.isca-archive.org/interspeech_2021/muguli21_interspeech.pdf |
| 27 | ComParE challenge summary | https://pmc.ncbi.nlm.nih.gov/articles/PMC10031089/ |
| 28 | Sound-Dr GitHub | https://github.com/ReML-AI/Sound-Dr |
| 29 | Sound-Dr paper | https://papers.phmsociety.org/index.php/phmap/article/download/3604/2088 |
| 30 | smarty4covid Scientific Data | https://www.nature.com/articles/s41597-023-02646-6 |
| 31 | Virufy data GitHub | https://github.com/virufy/virufy-data |
| 32 | Audio-IMU Cough Dryad | https://datadryad.org/dataset/doi:10.5061/dryad.0p2ngf26t |
| 33 | VocalSound official | https://groups.csail.mit.edu/sls/downloads/vocalsound/ |
| 34 | VocalSound GitHub | https://github.com/YuanGongND/vocalsound |
| 35 | Pulmonary Sound Mendeley | https://data.mendeley.com/datasets/6mnp9v3n73/1 |
| 36 | Clinic waiting-room cough Scientific Reports | https://www.nature.com/articles/s41598-021-03913-5 |
| 37 | DeepBreath GitHub | https://github.com/pneumonia-diagnosis/DeepBreath |
| 38 | DeepBreath npj Digital Medicine | https://www.nature.com/articles/s41746-023-00838-3 |
| 39 | FABS JMIR AI | https://ai.jmir.org/2025/1/e69844/ |
| 40 | Resp-Agent GitHub | https://github.com/zpforlove/Resp-Agent |
| 41 | Resp-Agent arXiv | https://arxiv.org/abs/2602.15909 |
| 42 | HLS-CMDS Mendeley | https://data.mendeley.com/datasets/pxg5f7j5kd/1 |
| 43 | ComParE 2020 Breathing | https://www.isca-archive.org/interspeech_2020/schuller20_interspeech.html |
| 44 | AudioSet ontology | https://research.google.com/audioset/ontology/index.html |
| 45 | AudioSet Cough class | https://research.google.com/audioset/ontology/cough_1.html |
| 46 | FSD50K Zenodo | https://zenodo.org/records/4060432 |
| 47 | ESC-50 GitHub | https://github.com/karolpiczak/ESC-50 |
| 48 | cough-speech-sneeze docs | https://audeering.github.io/datasets/datasets/cough-speech-sneeze.html |
| 49 | OpenSLR Nonverbal Vocalization | https://www.openslr.org/99/ |
| 50 | FluSense ACM DL | https://dl.acm.org/doi/10.1145/3381014 |
| 51 | VOICED PhysioNet | https://physionet.org/content/voiced/ |
| 52 | SVD official | https://stimmdb.coli.uni-saarland.de/ |
| 53 | Bridge2AI Adult PhysioNet | https://physionet.org/content/b2ai-voice/ |
| 54 | Bridge2AI Pediatric PhysioNet | https://physionet.org/content/b2ai-voice-pediatric/ |
| 55 | Bridge2AI Voice project | https://bridge2ai.org/data-voice/ |
| 56 | OPERA GitHub | https://github.com/evelyn0414/OPERA |
| 57 | OPERA website | https://evelyn0414.github.io/OPERA/ |
| 58 | OPERA OpenReview | https://openreview.net/forum?id=dR3yaOt1J9 |
| 59 | Niizumi et al. 2025 respiratory pretraining | https://www.isca-archive.org/interspeech_2025/niizumi25_interspeech.html |
| 60 | Google HeAR Hugging Face | https://huggingface.co/google/hear |
| 61 | HeAR arXiv | https://arxiv.org/abs/2403.02522 |
| 62 | MVST GitHub | https://github.com/xingmingyu123456/MVST |
| 63 | Patch-Mix Contrastive Learning GitHub | https://github.com/haideraltahan/respiratory-sound-classification |
| 64 | Stethoscope-guided SupCon GitHub | https://github.com/ilyassmoummad/stethoscope-guided-supervised-contrastive-learning |
| 65 | RepAugment GitHub | https://github.com/june-oh/RepAugment |
| 66 | BTS arXiv | https://arxiv.org/abs/2406.04205 |
| 67 | PAFA arXiv | https://arxiv.org/abs/2505.02059 |
| 68 | Architecture-Agnostic KD GitHub | https://github.com/hyioh/rsc-ensemble-kd |
| 69 | ADD-RSC GitHub | https://github.com/okankop/ADD-RSC |
| 70 | Geometry-aware AST+SAM GitHub | https://github.com/Atakanisik/ICBHI-AST-SAM |
| 71 | E-RespiNet PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC11938052/ |
| 72 | SNUCH pediatric lung-sound paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC9871007/ |
| 73 | RALE Repository | https://www.rale.ca/repository.htm |
| 74 | MEEI Voice and Speech Research | https://masseyeandear.org/research/otolaryngology/voice-and-speech |

---

## Final Summary

- **Total number of datasets/resources covered**: 52.
- **Datasets per relevance tier**: Tier 1 = 6; Tier 2 = 20; Tier 3 = 16; Tier 4 = 10.
- **Paper/method map size**: 56 entries, including 23 entries from 2024-2026.
- **Strongest directly useful datasets**: ICBHI, HF_Lung_V1, SPRSound, COUGHVID, UK COVID Vocal, Corp Dataset. They cover lung event labels, pediatric respiratory events, cough quality/event detection, PCR-linked COVID audio, and long-duration cough boundaries.
- **Important datasets with major issues**: Cambridge/DiCOVA/ComParE are access/challenge constrained; Coswara/COUGHVID/Sound-Dr/smarty4covid have self-report or mixed labels; FABS/DeepBreath/SNUCH are clinically relevant but raw data are not fully open; Resp-229K is powerful but derived/LLM-labeled and needs contamination audit; AudioSet/FSD/ESC/OpenSLR are non-clinical weak-label resources.
- **Best short-term research route**: respiratory event detection and cross-dataset robustness with ICBHI + HF_Lung + SPRSound, plus cough-quality/event detection using COUGHVID and Audio-IMU Cough. Add HeAR/OPERA/M2D-style frozen embeddings as baseline.
- **Best disease-label route**: UK COVID Vocal with symptom-only/audio-only/audio+symptom comparison. Do not claim clinical screening value without external validation.
- **Best long-term telehealth route**: prospective clinical audio collection with two-layer labels: acoustic events plus clinical/symptom outcomes; use DeepBreath/FABS/Clinic Waiting Room/iMedic as study-design references and Bridge2AI-Voice as speech-health context.
- **Biggest unresolved questions**: controlled dataset access, patient-independent splits, cross-source label mapping, license compatibility, foundation-model contamination, and whether audio provides incremental value beyond symptoms/metadata.
- **PDF generation status**: PDF 已生成，并完成抽查渲染；使用中文字体和横向 A4 表格布局。
