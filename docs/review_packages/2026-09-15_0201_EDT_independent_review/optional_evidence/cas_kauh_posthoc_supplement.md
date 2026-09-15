# Table 1/2 post-hoc HF CAS and KAUH supplement

Status: `completed_posthoc_test_informed_diagnostic`. Evidence: `fixed_checkpoint_posthoc_hf_cas_kauh_table_supplement`.

## Fixed HF CAS ranking score

CAS target is positive when a source-test recording has any Wheeze, Rhonchi, or Stridor interval; negatives are the other recordings within the 957 recordings eligible by any D/Wheeze/Rhonchi/Stridor annotation. Score is the maximum Wheeze-head probability across the three fixed 5-s windows.
This is a Wheeze-head ranking diagnostic against the CAS union, not a CAS/Rhonchi/Stridor prediction head or an equivalent CAS readout.

## Absolute results by seed

| Condition | Seed | Selected epoch | HF CAS AUROC (%) | KAUH patient binary BA (%) |
|---|---:|---:|---:|---:|
| LSAA/Full | 0 | 15 | 85.47 | 72.27 |
| LSAA/Full | 1 | 11 | 88.87 | 73.75 |
| LSAA/Full | 42 | 17 | 84.97 | 70.48 |
| ICBHI-only | 0 | 30 | 86.44 | 69.66 |
| ICBHI-only | 1 | 4 | 88.03 | 72.18 |
| ICBHI-only | 42 | 18 | 86.69 | 69.47 |
| SPRSound-only | 0 | 10 | 78.76 | 62.30 |
| SPRSound-only | 1 | 16 | 80.49 | 69.69 |
| SPRSound-only | 42 | 24 | 78.50 | 69.78 |

## Mean ± sample SD

| Condition | HF CAS AUROC (%) | KAUH patient BA (%) |
|---|---:|---:|
| LSAA/Full | $86.44_{\pm2.13}$ | $72.17_{\pm1.64}$ |
| ICBHI-only | $87.05_{\pm0.86}$ | $70.44_{\pm1.52}$ |
| SPRSound-only | $79.25_{\pm1.08}$ | $67.25_{\pm4.29}$ |

## Table 2(a) paired deltas (pp)

| Delta | Metric | Seed 0 | Seed 1 | Seed 42 | Mean ± sample SD |
|---|---|---:|---:|---:|---:|
| Delta_I_LSAA_minus_ICBHI_only | cas_auroc_pp | -0.97 | +0.84 | -1.72 | $-0.61_{\pm1.32}$ |
| Delta_I_LSAA_minus_ICBHI_only | kauh_patient_ba_pp | +2.61 | +1.57 | +1.01 | $+1.73_{\pm0.81}$ |
| Delta_S_LSAA_minus_SPRSound_only | cas_auroc_pp | +6.70 | +8.39 | +6.47 | $+7.19_{\pm1.05}$ |
| Delta_S_LSAA_minus_SPRSound_only | kauh_patient_ba_pp | +9.97 | +4.06 | +0.70 | $+4.91_{\pm4.69}$ |

## Table 2(c) HF-on minus actual paired HF-off (pp)

| Seed | HF-on/off epoch | HF-on CAS | HF-off CAS | CAS Δ (pp) | KAUH BA Δ (pp) | HF-off reference |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 17/15 | 76.31 | 85.47 | -9.16 | +9.36 | `/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/seed_0` |
| 1 | 13/11 | 80.06 | 88.87 | -8.82 | +1.01 | `/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed_external_HF_KAUH/seed_1` |
| 42 | 9/19 | 84.58 | 88.84 | -4.27 | -4.62 | `/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_JH2_external_HFtest_KAUHall_seed42_attempt2` |

- HF CAS AUROC Δ: $-7.41_{\pm2.73}$
- KAUH patient BA Δ: $+1.91_{\pm7.03}$
- Existing native paired deltas reused from the fixed HF-on summary: ICBHI `$-1.90_{\pm2.89}$`, SPRSound `$+1.31_{\pm1.43}$`.

## Supports and source artifacts

- HF: 1956 source-test recordings; 957 CAS-evaluable recordings (661 positive / 296 negative), three 5-s windows each.
- KAUH: 336 recordings / 112 patients overall; 86 compatible patients scored after B/D/E probability averaging. Crep, Bronchial, and I C B remain unresolved.
- No training, checkpoint reselection, target threshold fitting, paper edit, or Git operation was performed.
- JSON: `/Users/zilongzeng/Research/Acoustic/result/reproduce/pafa_joint_hierarchy/PAFA_TABLE2_POSTHOC_HF_CAS_KAUH_20260914/results/hf_cas_kauh_table2_supplement.json`
