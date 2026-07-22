# SG-SCL server-runnable gate report

## Verdict

SG-SCL is **server-runnable / metadata-aware official-like / test-selected**.
It is not an audio-only baseline. No local full numerical reproduction was
attempted; the author repository has no license and no task checkpoint.

## Source and protocol

- Author repo: `https://github.com/kaen2891/stethoscope-guided_supervised_contrastive_learning`
- Commit: `66564609595090b61540595d3d27764c00553086`
- Contract: `codex/2026-07-21/paper_contracts/sg_scl.json`
- Five-seed paper target: Sp `79.87+/-8.89`, Se `43.55+/-5.93`, Score `61.71+/-1.61`

The executable configuration agrees with the contract: official 60/40 split,
16 kHz, 8 s repeat padding, 128-bin fbank, AST ImageNet+AudioSet initialization,
batch 8, 50 epochs, Adam `5e-5`, weight decay `1e-6`, cosine, EMA `0.5`, and
device-domain SG-SCL. The official test is evaluated each epoch and the best
test Score is retained. One L40 run is therefore one `test-selected` seed, not
a reproduction of the five-seed aggregate.

## Data, checkpoint, and smoke

- 6,898 unique cycles; official train/test `4,142/2,756`.
- Device labels from the author filename rule cover all cycles; unknown count 0.
- Train device counts: AKGC417L 2,510; Litt3200 41; LittC2SE 594; Meditron 997.
- Test device counts: AKGC417L 1,836; Litt3200 461; Meditron 459; LittC2SE 0.
- Checkpoint SHA256: `dfc313e5082dc37ece8bd3bd6e7ea8bfee6598179a14eedd15c1727ad0af788f`.

The checkpoint is the current artifact referenced by the author runtime URL,
not a verified byte-identical 2023 original. The eight-cycle smoke produced
input `8 x 1 x 798 x 128`, embeddings `8 x 768`, and logits `8 x 4`. CE
`1.28424`, MetaCL `5.72137`, total loss `7.00561`, and gradients were finite.
Eight unique prediction IDs and confusion/Sp/Se/Score wiring verified. This is
a wiring result, not a quality estimate.

The compatibility patch only makes MetaCL allocations device-aware, adds
complete rolling resume state, and exports sample IDs/predictions. Its clean
patch digest is
`db6829b0f66e6ba1d52e5f4c67ad1c4e79cc0972bbad0d7e6216cd616f3d8a2f`.

## Fresh-checkout L40 commands

```bash
conda env create -f baseline/sg_scl/environment.linux-cu118.yml
conda activate acoustic-sgscl
RUN_ROOT="result/sg_scl_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
python -m baseline.sg_scl.run_reproduction bootstrap \
  --project-root . --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" --device cuda
python -m baseline.sg_scl.run_reproduction verify-bootstrap --result-root "$RUN_ROOT"
python -m baseline.sg_scl.run_reproduction smoke \
  --result-root "$RUN_ROOT" --device cuda --steps 1
python -m baseline.sg_scl.run_reproduction profile \
  --result-root "$RUN_ROOT" --device cuda --steps 100
python -m baseline.sg_scl.run_reproduction full \
  --project-root . --result-root "$RUN_ROOT" --device cuda:0
```

Acceptance requires 2,756 unique official-test IDs, confusion total 2,756,
finite probabilities, prediction-derived metric identity, and prominent
`test-selected` and metadata-aware labeling. Lack of a repository license
limits redistribution.
