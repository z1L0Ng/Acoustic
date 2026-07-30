# Four-Dataset Shortcut Diagnostic

This bounded post-hoc diagnostic asks whether recording-level acquisition
covariates add grouped out-of-fold information about corrected T0 errors or
true-class margin after native label and support controls. It also compares
linear dataset-ID accessibility in frozen R0 PAFA and R1 AudioSet embeddings.

The result may support **acquisition-correlated error** or **accessible domain
information**. It cannot establish a causal shortcut. R0 remains an ICBHI
official-test-selected PAFA representation.

The runner only reads existing predictions, embeddings and recording features.
It never reads or processes raw audio and never trains an acoustic model.

```bash
conda run -n acoustic-pafa python -m \
  baseline.four_dataset_shortcut_diagnostic.run --phase smoke

conda run -n acoustic-pafa python -m \
  baseline.four_dataset_shortcut_diagnostic.run --phase full

conda run -n acoustic-pafa python -m \
  baseline.four_dataset_shortcut_diagnostic.verify
```

Generated artifacts are isolated under
`result/four_dataset_shortcut_diagnostic/`.
