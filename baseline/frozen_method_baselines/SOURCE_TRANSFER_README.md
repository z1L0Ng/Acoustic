# PC-MCL source transfer and DCASE joint native-union runners

Current status (2026-09-17): DCASE completed three seeds. PC-MCL is stopped
after non-finite training in all three seeds; no automatic restart is authorized.
See `docs/baseline_design/2026-09-17_pcmcl_numerical_failure_and_source_audit_zh.md`.
The checks added below prevent invalid updates, decoding, resume and aggregation;
they do not establish that the convergence problem has been fixed.

These scripts implement PC-MCL ICBHI-only source transfer and DCASE ICBHI+SPRSound joint
native-union training. PC-MCL keeps its audited ICBHI official-test source
selection. DCASE uses only the accepted internal source validation partitions;
official tests and HF/KAUH are terminal.

## Methods

- PC-MCL: full trainable BEATs, N/C/W BCE, patient CE, exact 2.5 s + 2.5 s
  waveform concatenation, 400 full epochs with early stopping disabled.
- DCASE-inspired CRNN: frozen BEATs frames, trainable log-Mel CNN/fusion/BiGRU,
  nine-output native union, sigmoid masked BCE and class-masked attention,
  at most 50 epochs with patience-10 early stopping.

PC-MCL monitors ICBHI official-test Score and retains its sensitivity gate and
strict best-checkpoint selection, even when patience stopping is disabled.
DCASE monitors the equal mean of ICBHI native4 and SPR native7 internal-
validation macro multilabel F1 at threshold 0.5. Strict improvement resets
patience; ties count as no improvement and retain the earlier checkpoint.
HF/KAUH and both official source tests never enter DCASE stopping.

PC-MCL milestones 120/160 follow the public source's 400-epoch defaults, restored
by the user's 2026-09-17 decision. DCASE cosine uses `T_max=50`.
The failed ten-epoch PC-MCL attempt remains under `PC_MCL_ICBHI5s`; the three
stopped/invalid 400-epoch attempts remain under `PC_MCL_ICBHI5s_400epoch` on
the server. Both are historical recipes and must not be resumed. A new run
needs an agreed optimization recipe, a new output directory, and explicit
start authorization.

Both rows are respiratory-task adaptations. PC-MCL changes the paper's
10-s input to 5 s. DCASE retains native-class-union joint training and missing-
label masks, while omitting Mean Teacher, strong frame supervision and SED
post-processing. Its current head is the approved nine-output sigmoid union.

## Commands and execution scope

The failed historical queue remains paused. The current execution authorization
is limited to three independent runners for seeds 0/1/42 using the new lr1e-4
config below; it does not authorize the old queue or a duplicate local run.

The approved prospective server runs use `pcmcl_source_lr1e4_run.json`. They
keep the 400-epoch 5-s contract and
120/160 milestones, change Adam learning rate to `1e-4`, and write to
`PC_MCL_ICBHI5s_lr1e4_20260917`. The corrected SpecAugment implementation also
differs from the historical failed run, so this is a stability recipe rather
than a single-factor causal experiment. Management/server task owns launch and
monitoring; this model-design task does not start it.

```bash
CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.pcmcl_source_runner \
  --repo-root /files1/Zilong/Acoustic \
  --config baseline/frozen_method_baselines/pcmcl_source_lr1e4_run.json \
  --seed 0 --device cuda:0 --run
CUDA_VISIBLE_DEVICES=1 /opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.pcmcl_source_runner \
  --repo-root /files1/Zilong/Acoustic \
  --config baseline/frozen_method_baselines/pcmcl_source_lr1e4_run.json \
  --seed 1 --device cuda:0 --run
CUDA_VISIBLE_DEVICES=2 /opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.pcmcl_source_runner \
  --repo-root /files1/Zilong/Acoustic \
  --config baseline/frozen_method_baselines/pcmcl_source_lr1e4_run.json \
  --seed 42 --device cuda:0 --run
```

Do not invoke the default `source_transfer_queue` for these runs. After all
three finish, aggregate the new root explicitly:

```bash
/opt/anaconda3/envs/Beats/bin/python -m baseline.frozen_method_baselines.source_transfer_summary \
  --repo-root /files1/Zilong/Acoustic --method pcmcl \
  --output-root result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_lr1e4_20260917
```

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
  --resume result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_400epoch/seed_0/last_checkpoint.pt \
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

PC-MCL also checks its existing training log for non-finite loss. A failed
numerical history cannot be resumed or treated as a completed seed. Future
non-finite component/total loss, parameter gradient or inference logits
terminate the run with `failed_nonfinite` in the existing run summary;
seed/epoch/batch/sample IDs, component losses, learning rate and the first bad
gradient tensor are localized while previous completed-epoch checkpoints are
kept.
The summary excludes such runs even if an older script marked them complete.

The author-code alignment pass also restored the stochastic
`icbhi_ast_sup` SpecAugment gate and upper-exclusive mask widths. Patient hard
negatives retain the approved default: two different real patients with the
same aggregate pathology profile. Requiring the sampled cycles themselves to
share one native class is only an unapproved candidate, not the runtime
default. These code changes are not evidence that convergence recovered.

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

## Server time budget (2026-09-17)

The following launch-time estimate is historical and no longer a completion
promise: the PC-MCL queue failed numerically and was stopped.

The previous real CUDA run measured about 1.243 min/epoch on an L40. A fresh
400-epoch seed therefore takes about 497 min of training and epochwise source
selection. Three seeds take about 24.9 h serially or 16.6 h on two GPUs in two
batches. Allow 17–19 h for the two-GPU queue including fixed terminal evaluation
and runtime variation. This is an extrapolation, not a convergence guarantee.
DCASE completed all three seeds and released its GPU before this scheduling
decision. The server task verifies current availability and owns 15-minute
read-only progress reports.

## Historical local 50-epoch estimate

- PC-MCL three seeds at the 50-epoch upper bound: 20–30 h training, plus
  approximately 0.75–1.5 h fixed evaluation.
- DCASE joint frame extraction: 35–90 min; three-seed training: 10–43 h;
  three-seed external evaluation: 0.5–2 h; DCASE total: approximately 11–47 h.
- Both methods serially: approximately 32–79 h, or 35–87 h with a 10% local
  load/I/O allowance. Early stopping may shorten this but is not assumed.

These old local estimates do not apply to the current 400-epoch server schedule.

The DCASE frame cache uses memory-mappable `frames.npy` plus aligned `ids.npy`
under `.cache/frozen_method_baselines/dcase_joint_union_frames/`. Existing
pooled BEATs caches are not accepted as frame features.

`dcase_source_runner.py` and `dcase_source_run.json` are historical ICBHI-only
artifacts and refuse current execution. The default queue invokes only
`dcase_joint_union_runner.py` with `dcase_joint_union_run.json`.

## Runtime dependencies

The configured `Beats` environment must import PyTorch, torchaudio, and the
local BEATs source.  The implementation avoids sklearn and legacy identity
gates. The server environment is `acoustic-addrsc`; actual PC-MCL epochs and
DCASE training plus terminal evaluation have run there. The revised 400-epoch
PC-MCL completion and convergence remain to be observed.
