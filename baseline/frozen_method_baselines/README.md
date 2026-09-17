# Frozen Method Baselines: PAFA handoff, PC-MCL, and DCASE

This document preserves historical frozen-head candidates and PAFA handoff
assets. The current approved PC-MCL source-training and DCASE joint nine-output
native-union entries are documented in `SOURCE_TRANSFER_README.md`.
The older candidate interfaces below are not the current execution queue.

This package separates a **method-trained frozen representation** from a
**method-inspired adapter on a generic frozen representation**.

## PAFA — recommended runnable design

`pafa_trained_encoder_frozen_target_adaptation` uses the accepted PAFA-trained
BEATs state and the existing four-dataset 768-d cache.  The PAFA source
classifier and source projector are not reused.  A new target-supervised
adapter/head is trained separately for each Table 1 endpoint.

This row must be named **PAFA-trained BEATs (frozen)** or equivalent.  It is not
a PAFA reproduction: PCSL/GPAL acted during the historical ICBHI source run;
they are not silently reapplied during downstream target-head training.  The
fixed source state was selected using the ICBHI official test.

## PC-MCL — current source-transfer authority

`pcmcl_icbhi5s_source_trained_fixed_transfer` performs three independent full
ICBHI source trainings.  Each concatenated example contains two cycles that are
independently repeat-padded or center-cropped to 2.5 s before waveform
concatenation to 5 s.  BEATs, the N/C/W classifier, and the patient classifier
are all trainable during source training.  After ICBHI source selection, the
entire model and the 0.5 threshold are frozen for direct SPRSound, HF, and KAUH
readouts; no target head, selection, or threshold fitting is allowed.

The old frozen-head, joint-core, and pooled-embedding options are superseded.
Generic BEATs is only initialization, not a PC-MCL-trained checkpoint.  The
current specification is
`docs/baseline_design/2026-09-16_pcmcl_icbhi5s_source_transfer_spec_zh.md`.

## DCASE-style respiratory adaptation

`dcase_masked_crnn_respiratory_adaptation` keeps frozen BEATs frame embeddings,
an official-style trainable log-Mel CNN, temporal late fusion, a BiGRU, and
class-masked frame/attention outputs.  It uses explicit eligibility for the
shared respiratory channels Abnormal/Crackle/Wheeze/Other.

The candidate omits Mean Teacher because the proposed core has no approved
unlabeled respiratory pool, and omits mixup so augmentation is not another
comparison axis.  It must be named **DCASE-style respiratory adaptation**, not
a DCASE Task 4 reproduction.  Existing pooled caches are not compatible with
its frame-level input.

The retained DCASE design, endpoint semantics, and pending decisions are in
`docs/baseline_design/2026-09-16_frozen_pcmcl_dcase_spec_zh.md`.

For the existing HF CAS table column, both candidates use the same primary
readout as the current manuscript: weak/clip-level `p_W` for each of the three
fixed 5-s windows, followed by a maximum over windows.  Crackle is excluded.
The DCASE-specific `max(p_W, p_Other)` score is a separate diagnostic only when
Other remains strictly Rhonchi/Stridor; it is not silently substituted into the
primary CAS column.  External inference uses one fixed output-class mask for
all HF recordings and never derives that mask from target annotations.

## Design inspection entry

These commands print the frozen plan and local asset availability.  They do not
load a model or start training.

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.run \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --method pafa_trained_encoder_frozen_target_adaptation --seed 0

/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.run \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --method pcmcl_icbhi5s_source_trained_fixed_transfer --seed 0
```

Legacy target-adapter tasks in `task_protocol.json` do not govern the current
PC-MCL source-transfer design.  That design is fully specified in
`pcmcl_icbhi5s_source_transfer.json`; its official-test-selected source recipe
still requires explicit execution approval.
The prepared PAFA target-head training entry is `train.py`; for example:

```bash
/opt/anaconda3/envs/Beats/bin/python -u -m baseline.frozen_method_baselines.train \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --task icbhi_flat4 --seed 0 --device mps --run
```

KAUH additionally requires `--kauh-fold 0..4`.  These commands are prepared
interfaces only.  No training or feature extraction is authorized by this
package.

Current local environment HOLD: both the `Beats` and `acoustic-pafa` Python
environments fail while importing the installed SciPy PROPACK extension, which
is pulled in by the existing canonical data loader.  No dependency was changed
in this design task.  The describe-only CLIs remain usable; formal execution
needs a separately approved working environment.

The former PC-MCL-inspired route and its target eligibility interface remain
historical assets only.  They are not part of the current source-transfer run.

Complete unexecuted runners and future commands are documented in
`SOURCE_TRANSFER_README.md`.  The read-only local runtime estimate is
`docs/baseline_design/2026-09-16_source_baseline_local_runtime_estimate_zh.md`.
