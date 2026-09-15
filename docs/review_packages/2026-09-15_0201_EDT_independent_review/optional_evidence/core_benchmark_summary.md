# Multi-seed benchmark control summary

Status: `complete`.

All rows are official-test-selected benchmark controls. Full is the existing same-seed reference; Coarse seed42 uses `seed_42_attempt2`. The interrupted Coarse seed42 directory is excluded.

## Primary metrics

| Condition | n | Selected epochs | ICBHI Score mean ± sample SD | SPR official Score mean ± sample SD | SPR C/W mean AUROC mean ± sample SD |
|---|---:|---|---:|---:|---:|
| Full | 3 | s0=15, s1=11, s42=17 | 0.611681 ± 0.003139 | 0.906986 ± 0.003424 | 0.973181 ± 0.001481 |
| ICBHI-only | 3 | s0=30, s1=4, s42=18 | 0.577800 ± 0.014652 | 0.495276 ± 0.063964 | 0.750069 ± 0.054748 |
| SPRSound-only | 3 | s0=10, s1=16, s42=24 | 0.477465 ± 0.005135 | 0.928248 ± 0.006497 | 0.976523 ± 0.002358 |
| Coarse SPR | 3 | s0=1, s1=19, s42=20 | 0.546618 ± 0.040474 | 0.672343 ± 0.366236 | 0.888429 ± 0.085870 |
| Native+attributes | 3 | s0=15, s1=13, s42=17 | 0.636728 ± 0.016691 | 0.908574 ± 0.012067 | 0.972651 ± 0.001422 |

## Paired deltas versus same-seed Full

The delta aggregates use only the intersection of control and Full seeds. A one-seed sample SD is intentionally blank.

| Condition | Paired n | Δ ICBHI Score | Δ SPR official Score | Δ SPR C/W mean AUROC |
|---|---:|---:|---:|---:|
| ICBHI-only | 3 | -0.033881 ± 0.015884 | -0.411711 ± 0.067047 | -0.223112 ± 0.056132 |
| SPRSound-only | 3 | -0.134216 ± 0.008177 | +0.021262 ± 0.003299 | +0.003342 ± 0.003826 |
| Coarse SPR | 3 | -0.065063 ± 0.043315 | -0.234643 ± 0.366243 | -0.084753 ± 0.085829 |
| Native+attributes | 3 | +0.025047 ± 0.019032 | +0.001588 ± 0.015478 | -0.000530 ± 0.002496 |

## Per-seed sources

- Full seed0: epoch 15, `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_0`
- Full seed1: epoch 11, `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_1`
- Full seed42: epoch 17, `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/seed_42`
- ICBHI-only seed0: epoch 30, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/icbhi_only/seed_0`
- ICBHI-only seed1: epoch 4, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/icbhi_only/seed_1`
- ICBHI-only seed42: epoch 18, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/icbhi_only/seed_42`
- SPRSound-only seed0: epoch 10, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/sprsound_only/seed_0`
- SPRSound-only seed1: epoch 16, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/sprsound_only/seed_1`
- SPRSound-only seed42: epoch 24, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/sprsound_only/seed_42`
- Coarse SPR seed0: epoch 1, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/coarse_spr/seed_0`
- Coarse SPR seed1: epoch 19, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/coarse_spr/seed_1`
- Coarse SPR seed42: epoch 20, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/coarse_spr/seed_42_attempt2`
- Native+attributes seed0: epoch 15, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/native_attributes/seed_0`
- Native+attributes seed1: epoch 13, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/native_attributes/seed_1`
- Native+attributes seed42: epoch 17, `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_seed42/native_attributes/seed_42`

## Boundary

This is a seed-matched official-test-selected benchmark-control summary, not a clean estimate, pure causal ablation, or multi-seed claim beyond the seeds present in each condition. SPR C/W support is Crackle 84 positive/1345 negative and Wheeze 306/1123 for each available row.

Summary JSON: `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/multiseed_summary.json`.
