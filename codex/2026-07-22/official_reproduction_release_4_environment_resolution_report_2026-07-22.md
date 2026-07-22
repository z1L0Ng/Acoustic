# Official reproduction Release 4 environment resolution

## Verdict

Release 4 is **SAFE TO SNAPSHOT** directly on parent
`4172524a0e5d7b792de248820439f30874e2ae6d`.

The authoritative server patch and final receipt were retrieved read-only over
the configured imec SSH connection. Their SHA256 values match the handoff:

- Patch: `d2f2077f3d65ca5104e8e2a6d8a202c2bc35c3db31370c76e84d2a45ea3845a5`
- Final receipt: `9cfd849dd52868da444674aabd3f9b29205884fed1e01ae9c18dfc40e0a32be5`

This release changes only environment declarations and environment-command
metadata. No model, data, split, preprocessing, loss, hyperparameter,
checkpoint, metric, or checkpoint-selection behavior changed.

## Root cause and exact resolution

CMake 3.26.4's `WHEEL` metadata contains a blank line before its `Tag` headers.
The server matrix showed that pip 24.2 and later parse an empty wheel-tag set;
pip 25.3 therefore fails with the same unsupported-platform report. Pip 24.1.2
passes strict `pip check` while preserving the exact CMake 3.26.4 artifact.

The server also found that unbounded PAFA `transformers` resolved to 5.14.1,
which disables its PyTorch backend under Torch 2.0.1. The verified PAFA pin is
`transformers==4.38.2`.

Release 4 applies exactly this recipe:

1. Rename the five environments to `acoustic-patchmix-r4`,
   `acoustic-pafa-r4`, `acoustic-sgscl-r4`, `acoustic-mvst-r4`, and
   `acoustic-addrsc-r4`.
2. Pin conda `pip=24.1.2` in all five Linux YAMLs.
3. Pin PAFA `transformers==4.38.2` in Linux and local declarations.
4. Preserve Release 3 CMake, lit, setuptools, NumPy, OpenCV, and cmapy pins.

## Server evidence

The authoritative scratch run created all five full environments. Every method
returned `pip check=0`, passed required imports, reported the expected CUDA
11.8 or 12.1 metadata, saw the NVIDIA L40, and completed a finite CUDA matrix
operation. No checkpoint was downloaded and no training was started.

The five local Release 4 Linux YAML hashes exactly match the five server
candidate hashes:

| Method | SHA256 |
|---|---|
| Patch-Mix CL | `489b4ace63b547a63c2dffc9f0a7c8d074f35acd1982964a429de3e85c4f83e2` |
| PAFA | `986d5ebf92d315858c7c561a7639a214c53f02608226eb5e9a323c3b8dd027b1` |
| SG-SCL | `a8dea5e0b069cbd2369b532b5fd9cecee08188d565b48af3a735096f652a27f5` |
| MVST | `6ce315048b78fba12c8442a777645162e11868cdd7c849d9ea3286d40f6bdee6` |
| ADD-RSC | `200ab7245b81df321ed243df1a3377cf0bef3cd0d920032e102972e2391238ae` |

## Candidate changes

- Replaced the Release 3 contract, verifier, and server gate with versioned
  Release 4 files.
- Release 4 verifier now requires exact `pip==24.1.2` for every method and
  exact PAFA `transformers==4.38.2`.
- Bootstrap accepts only `receipts/environment_r4.json` from the same new
  timestamped run root and records the Release 4 environment YAML hash.
- Updated all environment mappings, READMEs, command examples, notebook entry
  cells, and notebook kernel display names to `-r4`.
- Release 3 historical reports remain unchanged in the parent history.

## Verification

| Check | Result |
|---|---|
| Release 3 baseline files matched parent before editing | pass |
| Authoritative patch and receipt SHA | pass |
| Linux YAML identity against server candidates | 5/5 exact |
| Clean-checkout Linux solver dry-run | 5/5 pass |
| Python compile and shell syntax | pass |
| Fresh-checkout CLI help entry points | 6/6 pass |
| Notebook JSON and clean output state | 5/5 pass; 0 outputs/executed cells |
| SG-SCL/MVST/ADD-RSC compatibility diff SHA | unchanged |
| Model/data/protocol semantic change | none |

## Server command

After management creates a new immutable Release 4 commit directly on
`4172524a0e5d7b792de248820439f30874e2ae6d`, the server must use a fresh
checkout of that exact SHA and run only:

```bash
bash baseline/common/official_environment_r4_server_gate.sh "$PWD"
```

The gate refuses existing `-r4` environments, creates a new timestamped result
root per method, and writes `receipts/environment_r4.json`. It verifies exact
pip and PAFA Transformers versions in addition to strict `pip check`, imports,
CUDA metadata, and a finite CUDA kernel. Bootstrap and training remain blocked
until management accepts all five durable receipts.

## Artifacts

- `baseline/common/official_environment_r4_contract.json`
- `baseline/common/verify_official_environment_r4.py`
- `baseline/common/official_environment_r4_server_gate.sh`
- `codex/2026-07-22/official_reproduction_release_4_clean_checkout_receipt_2026-07-22.json`
- `codex/2026-07-22/official_reproduction_release_4_candidate_sha256_2026-07-22.txt`

No files were staged, committed, or pushed. Notion was not edited. No
bootstrap, training, or cross-dataset migration was started.
