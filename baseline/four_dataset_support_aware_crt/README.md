# Four-Dataset Support-Aware cRT Control

This local controlled baseline compares the corrected R0+D2 native heads (`T0`)
with classifier-only class-balanced retraining (`T1`). The PAFA representation
is frozen and remains **ICBHI official-test-selected**.

The accepted tail eligibility contract is read mechanically. Only
`primary_evaluable` labels may satisfy the preregistered go rule. HF date
proxies are grouping proxies, never patient IDs, and SPR inter predictions are
written label-free before terminal scoring.

```bash
conda run -n acoustic-pafa python -m \
  baseline.four_dataset_support_aware_crt.run --phase all

conda run -n acoustic-pafa python -m \
  baseline.four_dataset_support_aware_crt.verify
```

Generated outputs are isolated under `result/four_dataset_support_aware_crt/`.
