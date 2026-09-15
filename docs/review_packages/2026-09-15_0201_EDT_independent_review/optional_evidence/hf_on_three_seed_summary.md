# JH4 HF-on multiseed external-transfer summary

Status: `complete`. Evidence: `fixed_JH4_HF_on_multiseed_external_transfer_diagnostic`.

HF-on is evaluated with the JH2 core recipe plus one HF batch per core update and lambda=0.25. HF/KAUH are selected-checkpoint external diagnostics; neither tunes thresholds or selects checkpoints.

## Per-seed core and C/W results

| Seed | Selected epoch | Completed epochs | ICBHI Score | SPR official Score | SPR C/W mean AUROC |
|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 27 | 0.562442 | 0.922095 | 0.974020 |
| 1 | 13 | 23 | 0.607275 | 0.908346 | 0.970701 |
| 42 | 9 | 19 | 0.597148 | 0.918086 | 0.969505 |

## 3-seed mean ± sample SD

| Metric | Mean ± sample SD |
|---|---:|
| icbhi_score | 0.588955 ± 0.023512 |
| sprsound_official_score | 0.916176 ± 0.007071 |
| spr_cw_mean_auroc | 0.971409 ± 0.002339 |
| icbhi_specificity | 0.749419 ± 0.080484 |
| icbhi_sensitivity | 0.428491 ± 0.037942 |
| icbhi_macro_f1 | 0.475379 ± 0.015674 |
| icbhi_uar | 0.487049 ± 0.033723 |
| sprsound_specificity | 0.856410 ± 0.020986 |
| sprsound_sensitivity | 0.980291 ± 0.009028 |
| sprsound_macro_f1 | 0.874237 ± 0.012837 |
| sprsound_uar | 0.918351 ± 0.006011 |
| hf_D_auroc | 0.641104 ± 0.056091 |
| hf_D_auprc | 0.454205 ± 0.058813 |
| hf_D_interval_recall | 0.954930 ± 0.034944 |
| hf_Wheeze_auroc | 0.829178 ± 0.064738 |
| hf_Wheeze_auprc | 0.776187 ± 0.088573 |
| hf_Wheeze_interval_recall | 0.665035 ± 0.135997 |
| kauh_recording_level1_score | 0.704233 ± 0.014809 |
| kauh_recording_flat4_score | 0.534298 ± 0.121099 |
| kauh_patient_level1_score | 0.755089 ± 0.057786 |
| kauh_patient_flat4_score | 0.562278 ± 0.104755 |
| kauh_patient_flat4_macro_f1 | 0.353519 ± 0.052576 |
| kauh_patient_flat4_uar | 0.459393 ± 0.076680 |

## Paired core references

- seed0: paired Full/core `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_0` (selected epoch 15); HF-off external `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/seed_0`
- seed1: paired Full/core `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_1` (selected epoch 11); HF-off external `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/seed_1`
- seed42: paired Full/core `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_test_selected_seed42_attempt2` (selected epoch 19); HF-off external `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2`

HF uses 957 eligible recordings, D/W interval supports 1812/1430, and excludes gap/phase-only/empty and unsupported native tasks. KAUH uses 336 recordings/112 patients with the compatible overlay; unresolved raw labels remain outside metrics.

This is a fixed protocol, test-exposed diagnostic multiseed summary, not a clean estimate, native HF/KAUH reproduction, or paper primary claim.

JSON: `result/reproduce/pafa_joint_hierarchy/PAFA_JH4_JH2_HFaux_multiseed/hf_on_multiseed_summary.json`.
