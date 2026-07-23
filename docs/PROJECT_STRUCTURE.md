# Project Structure

## Ownership

The repository uses ownership by scientific role, not by date or task thread.

- `acoustic/` owns reusable project infrastructure.
- `baseline/` owns prior methods and their faithful or explicitly bounded reproductions.
- `model/` owns only methods proposed by this project.
- `experiments/` describes questions and runs without prescribing a global schema.
- `dataset/` owns source acquisition, immutable raw packages, and lightweight metadata.
- `result/` owns generated evidence. It is local/server state, never Git source.
- `.cache/` owns all rebuildable large artifacts.
- `docs/` owns the current human-readable project state.

Threads and machines must edit the owning surface only. A server executes a
checked-out Git commit; it does not become a second development workspace.

## File lifecycle

| Class | Location | Git | Retention |
|---|---|---:|---|
| Source code and notebooks | `acoustic/`, `baseline/`, `model/`, `dataset/script/` | yes | maintained |
| Experiment definitions | `experiments/` | yes | maintained |
| Small schemas and tables | `dataset/processed/`, `docs/tables/` | yes | maintained |
| Raw source packages | `dataset/raw/` | no | immutable locally/server |
| Checkpoints and feature caches | `.cache/` | no | rebuildable or source-hash pinned |
| Metrics and predictions | `result/` | no | compact canonical runs only |
| Scratch files | `tmp/` | no | disposable |

Generated result directories use one level: `result/<experiment_id>/`. A final
run should normally contain `run.json`, `metrics.json`, compressed predictions,
confusion tables, and at most one best checkpoint or log when they are needed.
Source repositories, environments, preprocessing caches, profiler traces, and
receipt fragments do not belong in a final result directory.

## Git and server

`main` is the local integration branch and the source of server execution. The
server keeps one checkout and uses temporary branches only when a server-only
runtime fix must be returned as a patch or commit. Before switching or deleting
a server checkout, verify that no process has its working directory, executable,
checkpoint, log, or open file inside that tree.

See `server/README.md` for the operating sequence.
