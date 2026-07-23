# Patch-Mix C0 on SPRSound BioCAS2022

This package defines a target-native Patch-Mix reference. It is not zero-shot,
pooled training, domain adaptation, or a reproduction of a published SPRSound
result. Binary-broad and narrow-four are trained separately. The official
inter-subject events are evaluated once after checkpoint selection on a fixed
patient-grouped validation fold drawn only from the official train events.

## Fixed protocol

- Unit: official SPRSound event segment.
- Train: 6,656 official train events from 1,772 event-bearing recordings and
  243 event-bearing patients. The archive has 1,949 recording annotations from
  251 patients; 177 recordings contain no event annotation and therefore
  produce no event-level training row.
- Validation: `StratifiedGroupKFold(5, shuffle=True, random_state=20260722)`,
  fold 0, grouped by patient ID; expected subtrain/validation = 5,219/1,437.
- Final test: 1,429 official inter-subject events; never used for checkpoint,
  preprocessing, threshold, or hyperparameter selection.
- Model/training: Patch-Mix CL AST base384, source 16 kHz/8 s/128-fbank
  preprocessing, SpecAugment on subtrain only, Adam `5e-5`, weight decay
  `1e-6`, batch 8, 50 epochs, cosine, EMA 0.5, seed 1.
- Selection: maximum inner-validation ICBHI Score after each epoch, retaining
  the author's sensitivity-greater-than-five eligibility gate; all 50 epochs
  run. The selected checkpoint is evaluated on inter exactly once.
- Narrow-four excludes Rhonchi and Stridor. Binary-broad includes them as
  abnormal. Poor Quality is fail-closed if it appears as an event label.

The AST initialization is the current artifact referenced by the pinned author
runtime loader. Its SHA256 is verified, but this does not establish historical
2023 serialized-byte identity. See `protocol.json` for the complete receipt.

## Fresh-checkout server commands

Run from the repository root at the immutable snapshot selected by management.
All generated data stays under `result/` or the parameterized cache root.

```bash
RUN_ROOT="result/c0_patch_mix_sprsound_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
CACHE_ROOT="$RUN_ROOT/cache"
DATASET_ROOT="dataset/raw/sprsound"

conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.bootstrap \
  --result-root "$RUN_ROOT" --cache-root "$CACHE_ROOT"
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.prepare_data \
  --dataset-root "$DATASET_ROOT" --result-root "$RUN_ROOT"
```

Build and verify the bounded smoke cache. Inter rows are selected by sorted ID;
no inter labels are opened by cache construction or smoke inference.

```bash
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.build_cache \
  --result-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" --cache-name smoke \
  --max-train-events 32 --max-inter-events 8
for TASK in binary_broad narrow_four; do
  conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.train \
    --mode smoke --task "$TASK" --result-root "$RUN_ROOT" \
    --cache-root "$CACHE_ROOT" --cache-name smoke --device cuda \
    --num-workers 0 --max-steps 1
done
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.verify \
  --mode smoke --result-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" --cache-name smoke
```

After management accepts the smoke receipt, build the full cache and run the
two tasks sequentially. Do not run both AST jobs on the same device.

```bash
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.build_cache \
  --result-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" --cache-name full
for TASK in binary_broad narrow_four; do
  conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.train \
    --mode full --task "$TASK" --result-root "$RUN_ROOT" \
    --cache-root "$CACHE_ROOT" --cache-name full --device cuda --num-workers 4
done
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.verify \
  --mode full --result-root "$RUN_ROOT" --cache-root "$CACHE_ROOT" --cache-name full
```

Resume only from the matching task's receipted `last.pth`:

```bash
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.train \
  --mode full --task binary_broad --result-root "$RUN_ROOT" \
  --cache-root "$CACHE_ROOT" --cache-name full --device cuda --num-workers 4 \
  --resume "$RUN_ROOT/full/binary_broad/checkpoints/last.pth"
```

The smoke command exports label-free inter logits only. It must not produce an
inter metric file. Full mode opens inter labels only after the best validation
checkpoint is fixed and all inter logits have been written.

## Evaluation-only uncertainty supplement

The statistical layer is descriptive and cannot select any model or decision.
It uses patient-grouped bootstrap because the official SPRSound README defines
the first filename field as patient number. B0/floor intervals and chance floors
can be generated without rerunning B0 inference:

```bash
conda run -n acoustic-patchmix-r4 python -m baseline.patch_mix_cl.c0_sprsound.statistical_comparison \
  --b0-result-root result/patch_mix_cl_author_checkpoint_transfer_20260722_175414
```

After C0 exists, add `--c0-result-root "$RUN_ROOT"` to compute the preregistered
same-ID paired patient-grouped C0-minus-B0 intervals. A five percentage-point
absolute gap is a project practical-materiality reference, not a community
standard or a significance threshold. Binary and narrow-four are judged
separately; inter Both support is one and cannot support a minority claim.
