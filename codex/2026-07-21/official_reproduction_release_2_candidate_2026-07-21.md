# Official reproduction Release 2 candidate

## Candidate status

`SAFE_TO_SNAPSHOT`

Release 2 is an additive snapshot on top of Release 1 branch
`codex/l40-training-release`, commit
`51626840f6ec325086f68bd88446ff956f7e0357`. Release 1 Patch-Mix/PAFA files are
not modified. Release 2 adds only SG-SCL, MVST, ADD-RSC and their contracts and
reports.

## Required tracked scope

- `baseline/sg_scl/**`
- `baseline/mvst/**`
- `baseline/add_rsc/**`
- `codex/2026-07-21/paper_contracts/sg_scl.json`
- `codex/2026-07-21/paper_contracts/mvst.json`
- `codex/2026-07-21/paper_contracts/add_rsc.json`
- `codex/2026-07-21/sg_scl_reproduction_gate_report_2026-07-21.md`
- `codex/2026-07-21/mvst_reproduction_gate_report_2026-07-21.md`
- `codex/2026-07-21/add_rsc_reproduction_gate_report_2026-07-21.md`
- `codex/2026-07-21/baseline_five_strong_icbhi_reproduction_report_2026-07-21.md`
- `codex/2026-07-21/baseline_five_strong_icbhi_reproduction_matrix_2026-07-21.csv`
- `codex/2026-07-21/official_reproduction_release_2_clean_checkout_receipt_2026-07-21.json`
- `codex/2026-07-21/official_reproduction_release_2_candidate_2026-07-21.md`
- `codex/2026-07-21/official_reproduction_release_2_candidate_sha256_2026-07-21.txt`

Do not include any `result/`, legacy `results/`, checkpoint, source clone, cache,
log, prediction, `.DS_Store`, or `__pycache__` object.

## Reconstruction verification

For each method, create a new timestamped root and run:

```bash
python -m baseline.<method>.run_reproduction bootstrap \
  --project-root . --dataset-root dataset/raw/icbhi_2017 \
  --result-root "$RUN_ROOT" --device cuda
python -m baseline.<method>.run_reproduction verify-bootstrap \
  --result-root "$RUN_ROOT"
```

Then run the method-specific smoke from its README. Clean-checkout tests based
on the exact Release 1 commit successfully cloned all three pinned repos,
downloaded public checkpoints, verified SHA256, rebuilt 6,898-cycle manifests,
and passed all 8-cycle model/gradient/prediction wiring. Detailed evidence is in
`official_reproduction_release_2_clean_checkout_receipt_2026-07-21.json`.

## Server order and claim boundaries

Run SG-SCL first within Release 2, then MVST, then ADD-RSC. Perform an L40
100-step timing/VRAM profile before each full run. SG-SCL must be labelled
metadata-aware; MVST must be labelled author-code fixed-random-file-split;
ADD-RSC must use one of its two explicit bounded track names. All author flows
that select by test Score must remain labelled test-selected.

## Final verification

- Python compilation and shell syntax: passed.
- Three expected Release 2 notebooks: nbformat valid, outputs empty,
  `execution_count=null`.
- Formal v2 regression verifier: `architecture=42`, `imbalance=36`,
  `predictions=78`, identity true, combined 6.
- Scoped `git diff --check`: passed.
- `result/` roots: ignored; legacy `results/`: absent.
- Candidate baseline directories contain no checkpoint, cache, log, NPZ,
  `.DS_Store`, or `__pycache__` file.
- No full Mac training was started.
