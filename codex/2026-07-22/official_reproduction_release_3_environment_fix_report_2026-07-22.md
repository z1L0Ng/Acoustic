# Official reproduction Release 3 environment fix report

## Verdict

Release 3 is **SAFE TO SNAPSHOT for a fresh Linux/L40 environment gate** on top
of `3f757adcc12fcc5b5e2f1058a593345f750de2a5`. It is not yet accepted for
bootstrap or training: the server must create the five new `-r3` environments
and produce five `status=verified` runtime receipts first. Release 2
environments remain immutable.

This candidate changes environment declarations, environment verification,
command metadata, notebook kernel labels, and bootstrap receipt preconditions
only. It does not change a model, dataset, split, preprocessing path, loss,
hyperparameter, checkpoint, metric, or checkpoint-selection rule.

## Blocker classification

The four CUDA 11.8 environments install the conda artifact
`torchtriton-2.0.0-py310.tar.bz2`. Its conda index lists only `filelock`,
Python, and PyTorch, but the installed Python distribution metadata declares
`cmake`, `filelock`, `torch`, and `lit` unconditionally. `cmake` supports the
source-build extension and `lit` the test suite; neither is used by the
prebuilt model forward path. They are nevertheless real metadata requirements,
so waiving strict `pip check` would be incorrect. Release 3 pins
`cmake==3.26.4` and `lit==16.0.6` in Patch-Mix, PAFA, SG-SCL, and MVST.

All five methods use `librosa==0.9.2`, which imports `pkg_resources`.
`setuptools==83.0.0` on the server did not expose that module. Release 3 pins
`setuptools==80.9.0` in every local and Linux declaration; clean local imports
of both `librosa` and `pkg_resources` pass in all five new environments.

MVST additionally exposed an inherited dependency-closure issue: `cmapy==0.6.6`
requires the `opencv-python` distribution while the server declaration used
the headless distribution. Both are now explicitly pinned to `4.11.0.86`,
along with NumPy `1.26.4`, preventing an unbounded OpenCV 5 install from pulling
NumPy 2. The two wheels share the `cv2` namespace, so the server verifier treats
`cv2`/`cmapy` import failure as a hard stop; this compatibility compromise is
recorded in `baseline/common/official_environment_r3_contract.json`.

## Changes

- New environment names: `acoustic-patchmix-r3`, `acoustic-pafa-r3`,
  `acoustic-sgscl-r3`, `acoustic-mvst-r3`, and `acoustic-addrsc-r3`.
- Added a strict Linux verifier at
  `baseline/common/verify_official_environment_r3.py`. It runs `pip check`,
  exact-version and required-import checks, validates CUDA metadata, and in
  runtime mode executes a finite CUDA kernel before writing a package receipt.
- Added `baseline/common/official_environment_r3_server_gate.sh`. It refuses
  to mutate/reuse an existing Release 3 environment, creates each environment,
  and fails closed before bootstrap or training.
- Bootstrap now requires a valid `environment_r3.json` in the same new
  timestamped result root and records the environment YAML SHA256.
- Updated five READMEs and five clean notebooks; the notebooks contain no
  outputs or execution counts.

## Verification

| Check | Result |
|---|---|
| Clean checkout from base SHA + candidate overlay | pass |
| Five `linux-64` conda solver dry-runs | 5/5 pass |
| Five new local macOS arm64 environment creates | 5/5 pass |
| `librosa` + `pkg_resources` local imports | 5/5 pass |
| Full method dependency imports | 5/5 pass after MVST pins |
| ADD-RSC local strict `pip check` | pass |
| Four Torch 2.0.1 local strict `pip check` | platform-limited: the macOS wheel declares itself unsupported |
| Python compile / shell syntax | pass |
| Notebook JSON / clean output state | 5/5 pass, 0 outputs, 0 executed cells |
| Candidate whitespace audit | pass |
| SG-SCL/MVST/ADD-RSC compatibility diff SHA regression | exact Release 2 identity |
| Linux strict `pip check` and CUDA kernel | pending server gate |

The repository-wide `git diff --check` could not run to completion because the
current post-migration working tree asks Git to diff the removed legacy binary
`results/model_design_2026-07-08/icbhi_sequence_pooling/features/beats_sequence_full.npz`.
This path is outside the Release 3 candidate. All 33 candidate files passed
individual `git diff --no-index --check`, and the clean checkout checks above
used only the base SHA plus the candidate overlay.

The local Mac cannot certify a Linux CUDA runtime. Docker is installed but its
daemon is not running, and emulation would not certify L40 CUDA. Therefore the
candidate is safe to snapshot, while the five server runtime receipts remain a
mandatory acceptance gate.

## Server migration

Management should create one immutable Release 3 commit based directly on
`3f757adcc12fcc5b5e2f1058a593345f750de2a5`, using the candidate SHA manifest.
On the server, check out that exact new commit in a fresh tree and run:

```bash
bash baseline/common/official_environment_r3_server_gate.sh "$PWD"
```

The command creates five new timestamped `result/<method>_YYYYMMDD_HHMMSS/`
roots and writes `receipts/environment_r3.json`. It does not clone method
sources, download checkpoints, run method smoke tests, or train. If any `-r3`
environment already exists, the script stops rather than mutating it.

Only a method whose receipt has `status=verified`, `pip_check.returncode=0`,
all imports `status=ok`, matching CUDA metadata, `cuda.available=true`, and
`cuda.finite_kernel=true` may proceed to the bootstrap command in that method's
README. A failed method must retain its receipt and stop; do not patch the
server environment operationally.

## Artifacts

- Dependency and artifact contract:
  `baseline/common/official_environment_r3_contract.json`
- Clean-checkout receipt:
  `codex/2026-07-22/official_reproduction_release_3_clean_checkout_receipt_2026-07-22.json`
- Candidate SHA manifest:
  `codex/2026-07-22/official_reproduction_release_3_candidate_sha256_2026-07-22.txt`
- Server gate entry:
  `baseline/common/official_environment_r3_server_gate.sh`

No training or dataset migration was started. No files were staged, committed,
or pushed, and Notion was not edited.
