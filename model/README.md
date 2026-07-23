# Project-authored models

This directory is reserved for new methods proposed by this project. Existing
paper implementations belong in `baseline/`; reusable infrastructure belongs in
`acoustic/`; experiment configuration belongs in `experiments/`.

Each future model subdirectory should include a concise README, implementation,
and tests. Do not add a model until its matched comparison and stop rule are
recorded in an experiment definition. Model binaries do not live beside the
implementation; an accepted experiment may publish only its single eligible
best file under `checkpoints/<experiment_id>/best.*`.
