# Shared Compatible-Head Harmonization

This cached-feature experiment retains the six four-dataset native heads while
testing compatible supervision between ICBHI cycles and SPRSound events.
HF_Lung and KAUH never supply compatible-head labels.

The frozen encoder is selected only by the independently verified Step 1
receipt and is fixed to AudioSet-only BEATs cache SHA
`3b3798cc9d01dbdfa8168a1cd641d658eb2fd4553799e59b84b7aae7ad0f5a69`.

Conditions:

- `h0_native_plus_independent_compatible`: conventional full 256-dimensional
  separate compatible heads.
- `h1_eligibility_masked_shared`: one binary and one narrow-four shared head.
- `h2_parameter_matched_independent`: separate heads after fixed orthonormal
  128-dimensional projections. H1 and H2 each have exactly 1,548 compatible
  trainable parameters.

SPRSound inter predictions are written without labels. Labels are loaded only
for terminal scoring; Rhonchi, Stridor, and Poor Quality are omitted from
narrow-four loss/scoring and are never converted to negatives.

```bash
conda run -n acoustic-pafa python -m baseline.shared_compatible_head_harmonization.run --phase smoke
conda run -n acoustic-pafa python -m baseline.shared_compatible_head_harmonization.verify --mode gate
conda run -n acoustic-pafa python -m baseline.shared_compatible_head_harmonization.run --phase train
conda run -n acoustic-pafa python -m baseline.shared_compatible_head_harmonization.run --phase analyze
conda run -n acoustic-pafa python -m baseline.shared_compatible_head_harmonization.verify --mode full
```

Outputs are isolated under
`result/four_dataset_shared_compatible_head_harmonization/`; runtime caches are
under `.cache/four_dataset_shared_compatible_head_harmonization/`.
