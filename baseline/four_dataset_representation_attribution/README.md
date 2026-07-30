# Four-Dataset Representation Attribution

This controlled experiment isolates the frozen encoder representation while
holding the existing D2 downstream schedule fixed.

Representations:

- `r0_pafa_icbhi_task_encoder`: immutable PAFA task-trained cache, with the
  ICBHI official-test-selection caveat.
- `r1_beats_as2m_audioset_only`: BEATs_iter3+ AS2M AudioSet initialization
  only. No PAFA model, classifier, or projector state is loaded.
- `r2_beats_random_init_sanity`: the identical BEATs architecture initialized
  from seed `20260728`, with no pretrained tensors. This is a random-feature
  sanity floor and cannot be selected.

All conditions use the same 768-to-256 adapter, six native heads,
dataset-balanced update schedule, five KAUH-fold-conditioned models, and
validation-only checkpoint selection. Results remain dataset/task specific;
cross-dataset raw scores are never pooled.

Run in the existing `acoustic-pafa` environment:

```bash
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.run --phase smoke
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.run --phase profile
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.verify --mode gate
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.run --phase extract
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.run --phase train
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.run --phase analyze
conda run -n acoustic-pafa python -m baseline.four_dataset_representation_attribution.verify --mode full
```

Generated artifacts are restricted to
`result/four_dataset_representation_attribution/` and
`.cache/four_dataset_representation_attribution/`. Extraction is atomic per
representation and dataset and can resume only from complete shard/receipt
pairs.
