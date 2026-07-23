# Server Operation

The server is an execution target for the same Git repository, not a second
development workspace.

## Standard flow

1. Finish and verify code locally on `main`.
2. Commit and push `main`.
3. On the server, fetch and fast-forward the single canonical checkout.
4. Verify dataset/checkpoint hashes and create or reuse only an environment whose
   exact specification is tracked by the selected baseline.
5. Run bootstrap, smoke, and bounded profile before full training.
6. Start long runs inside durable `tmux`; logs and exit status must survive SSH loss.
7. Write generated files under `result/<experiment_id>/` and caches under `.cache/`.
8. Return compact metrics, predictions, hashes, and any narrow runtime patch to
   local management. Do not commit generated outputs.

## Git rule

The normal server branch is `main`. If a server-only runtime fix is unavoidable,
make the smallest patch on a temporary `codex/` branch, commit it, and return the
commit or patch for local review. Never mix model changes, data changes, and
runtime fixes in one server commit. After integration, return the server to main.

## Safety gate

Before changing commits, moving the checkout, or deleting an old checkout:

- all training/export processes using it are terminal;
- no process has a working directory or open file inside it;
- terminal result hashes have been recorded;
- required raw data and checkpoints have independent parity checks;
- free-space policy is satisfied.

The intended server root is `/files1/Zilong/Acoustic`. Multiple timestamped Git
checkouts are temporary migration state and should be removed only after these
checks pass.
