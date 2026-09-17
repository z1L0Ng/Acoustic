# ICBHI-source PC-MCL and DCASE-inspired runners

These scripts are complete execution entries but have **not** been run.  They
implement PC-MCL ICBHI-only source transfer and DCASE ICBHI+SPRSound joint
native-union training. PC-MCL keeps its audited ICBHI official-test source
selection. DCASE uses only the accepted internal source validation partitions;
official tests and HF/KAUH are terminal.

## Methods

- PC-MCL: full trainable BEATs, N/C/W BCE, patient CE, exact 2.5 s + 2.5 s
  waveform concatenation, at most 50 epochs with patience-10 early stopping.
- DCASE-inspired CRNN: frozen BEATs frames, trainable log-Mel CNN/fusion/BiGRU,
  nine-output native union, sigmoid masked BCE and class-masked attention,
  at most 50 epochs with patience-10 early stopping.

PC-MCL monitors ICBHI official-test Score and retains its sensitivity gate.
DCASE monitors the equal mean of ICBHI native4 and SPR native7 internal-
validation macro multilabel F1 at threshold 0.5. Strict improvement resets
patience; ties count as no improvement and retain the earlier checkpoint.
HF/KAUH and both official source tests never enter DCASE stopping.

PC-MCL milestones 15/20 preserve the original 120/160 relative positions after
the 400-to-50 epoch budget adaptation.  DCASE cosine uses `T_max=50`.

Neither row is a reproduction of the published method: PC-MCL changes the
paper's 10-s input to 5 s, and DCASE removes Mean Teacher/heterogeneous-label
training in favor of ICBHI native flat4 CE.

## Future commands — do not run without a new start instruction

One PC-MCL seed:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.pcmcl_source_runner \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --seed 0 --device mps --run
```

Resume an incomplete seed only from its own last checkpoint:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.pcmcl_source_runner \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --seed 0 --device mps \
  --resume result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s/seed_0/last_checkpoint.pt \
  --run
```

DCASE frame extraction once:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.dcase_joint_union_runner \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --phase extract-frames --device mps --run
```

One DCASE joint-union training seed after the frame cache exists:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.dcase_joint_union_runner \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --phase train --seed 0 --device mps --run
```

Complete local serial queue:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.source_transfer_queue \
  --repo-root /Users/zilongzeng/Research/Acoustic \
  --methods pcmcl,dcase --device mps --run
```

The queue skips seeds with a completed run summary.  For an incomplete seed it
resumes from that seed's `last_checkpoint.pt`; a non-empty seed directory with
no last checkpoint is rejected rather than overwritten.  Fresh runners also
refuse non-empty result directories.

Resume restores model, optimizer, scheduler, epoch, and best-selection state.
It also restores the consecutive no-improvement count, completed source epoch,
and early-stop reason.  A last checkpoint already marked early-stopped or at
max epoch skips source training and proceeds to load the best model for target
evaluation.
It does not restore the full Python/NumPy/PyTorch/DataLoader random stream, so a
resumed run is not claimed to be bitwise identical to an uninterrupted run.

Rebuild a summary without running models:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.source_transfer_summary \
  --repo-root /Users/zilongzeng/Research/Acoustic --method pcmcl
```

## Outputs

Each seed writes config, JSONL train log, best/last resumable checkpoint,
source and target prediction NPZs, metrics, and run summary below
`result/reproduce/source_transfer_baselines/`.  The summary includes only seeds
whose run status is complete.

The DCASE frame cache uses memory-mappable `frames.npy` plus aligned `ids.npy`
under `.cache/frozen_method_baselines/dcase_joint_union_frames/`. Existing
pooled BEATs caches are not accepted as frame features.

`dcase_source_runner.py` and `dcase_source_run.json` are historical ICBHI-only
artifacts and refuse current execution. The default queue invokes only
`dcase_joint_union_runner.py` with `dcase_joint_union_run.json`.

## Runtime dependencies

The configured `Beats` environment must import PyTorch, torchaudio, and the
local BEATs source.  The implementation avoids sklearn and legacy identity
gates.  Full runtime remains unverified because model forward, extraction, and
training were prohibited during implementation.
