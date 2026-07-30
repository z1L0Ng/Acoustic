# Four-Dataset PAFA Frozen-Encoder Matrix

This package tests the project idea before any full shared-encoder training.
It uses the accepted PAFA checkpoint because PAFA has the highest
project-reproduced ICBHI Score among reusable single encoders evaluated on the
official recording split (`64.143`). MVST's larger numeric Score is excluded
from encoder selection because its author-code run uses a different fixed
random file split and the retained checkpoint is a fusion model.

The PAFA BEATs encoder is frozen. Source classifier and PAFA projector states
are discarded. The checkpoint was selected on the ICBHI official test set, so
all results are exploratory target-supervised representation experiments, not
clean generalization.

## Native Tasks

- ICBHI cycle flat4.
- SPRSound BioCAS2022 event binary and seven-class, inter-subject test.
- HF_Lung recording-level phase/adventitious positive-presence diagnostics.
  Each 15-second recording is covered by three non-overlapping author-length
  5-second inputs; frozen embeddings are mean-pooled to one recording vector.
  A recording enters a task only when at least one source label in that pool is
  observed. Peer-label absence is negative only inside that eligible pool.
  Unannotated gaps and empty annotation files are not normal/negative. These
  diagnostics are not a reproduction of the source temporal detector.
- KAUH raw nine-way recording sound strings. `C`, `Crep`, `I C B`, and
  `Bronchial` remain distinct. B/D/E siblings stay patient-grouped. Results are
  five-fold patient-grouped OOF because no official split exists. Several raw
  labels have fewer patients than folds, so no per-fold stratification-success
  claim is made; the primary KAUH result is aggregate fixed-nine-label OOF.

## Conditions

- `D0`: independent native heads on frozen 768-d embeddings.
- `D1`: shared `768 -> 256` adapter and native heads, source-proportional.
- `D2`: same model, equal update counts per dataset.
- `D3`: D2 plus multiclass train-time `+log(class prior)` Logit Adjustment and
  multilabel BCE `pos_weight=negative/positive`. These are explicit
  task-kind-specific policies, not one generic prior-balanced loss.

Raw challenge Scores are never pooled across datasets. Model selection uses
validation macro-F1 only. SPR inter predictions are written without labels
before the raw annotations are joined for terminal scoring. ICBHI, SPRSound,
and HF results are summarized as mean and sample standard deviation across the
five KAUH-fold-conditioned joint models; KAUH uses aggregate OOF.

## Run

```bash
PY=/opt/anaconda3/envs/acoustic-pafa/bin/python

$PY -m baseline.four_dataset_frozen_encoder.run --phase audit
$PY -m baseline.four_dataset_frozen_encoder.run --phase smoke
$PY -m baseline.four_dataset_frozen_encoder.run --phase profile
$PY -m baseline.four_dataset_frozen_encoder.verify --mode gate
$PY -m baseline.four_dataset_frozen_encoder.run --phase extract
$PY -m baseline.four_dataset_frozen_encoder.run --phase train
$PY -m baseline.four_dataset_frozen_encoder.verify --mode full
```

Canonical outputs:

- `result/four_dataset_pafa_frozen_encoder/`
- `.cache/four_dataset_pafa_frozen_encoder/`

The experiment is a method-development demo. It does not establish zero-shot
transfer, full-backbone multi-dataset learning, absence of dataset shortcuts,
or statistical significance.

The one-run local execution gate is 120 active minutes and 24 GiB peak RSS.
This bounded exception was approved because the measured center estimate was
54.14 minutes for extraction plus 0.69 minutes for heads; only the prior 2x
safety projection (109.66 minutes) exceeded the former 90-minute threshold.
It changes runtime authorization only, not the scientific protocol.
