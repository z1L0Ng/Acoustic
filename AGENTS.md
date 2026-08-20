# Acoustic Project Agent Policy

This file applies to every agent and every subdirectory in the Acoustic project. These rules override older project-task prompts when they conflict on validation or implementation style.

## Mandatory execution defaults

1. **Do not use SHA-256 or other hash/checksum verification unless the user explicitly requests it.**
   - Do not calculate, compare, record, or require hashes for code, data, checkpoints, receipts, logs, predictions, exports, or environments.
   - Do not add hash fields, checksum manifests, artifact-identity chains, or hash-based execution gates.
   - Existing hashes are historical records only. Do not extend or retroactively remove them unless requested.

2. **Do not run smoke tests unless the user explicitly requests them.**
   - This includes smoke runs, synthetic end-to-end runs, probe runs, zero-update preflights, minimal-forward checks, profile runs, and tests whose main purpose is only to prove that a path starts.
   - Do not substitute smoke evidence for experimental results.
   - When a code change needs verification, use only the smallest directly relevant unit or functional check. Do not run broad regression, adversarial, exhaustive fail-closed, or full test suites unless requested.

3. **Minimize protective and defensive programming.**
   - Implement the simplest clear happy path that satisfies the approved task.
   - Avoid redundant validators, immutable identity chains, duplicated guards, defensive fallback layers, retry orchestration, excessive exception wrapping, and speculative compatibility code.
   - Do not create extra receipts or gate documents unless they are needed for the requested scientific result or explicitly requested by the user.
   - Preserve only essential safeguards for destructive actions, data leakage, train/validation/test separation, patient/group split correctness, label semantics, and research integrity.

## Project-wide behavior

- Keep Verified Result, Interpretation, Proposed Method, Future Plan, and HOLD separate.
- Do not start experiments, server runs, Notion writes, or Git commits/pushes without the authorization required by the management task.
- The current Working Plan permits contract and implementation preparation, but Model Design training remains paused at `READY_FOR_USER_START`. No experiment, feature extraction, cache build, validation, test, or server run may start until the user gives an explicit start instruction.

## Mandatory management handoff

- Every Acoustic project task must report to the project-management task when its assigned work completes, reaches a decision gate, or becomes blocked. Do not finish silently inside a specialist task.
- The management destination is the task titled `Acoustic项目管理` (thread ID `019fb42d-0edd-7aa1-aecf-9c6669109279`).
- A completion handoff must state: execution status; completed scope; changed files or external pages; Verified Results; Interpretation; Issues/HOLD; actions not performed; and the next decision required from management or the user.
- Report planning, code readiness, partial cache, validation-only evidence, and completed experimental results as different statuses. Never upgrade one into another.
- Send routine progress only when requested. Completion, blockers, approval requests, and scientifically material deviations must always be returned to management.
- If the task cannot directly message the management task, place a compact management-ready handoff in its final response so management can collect it without reconstructing the work from logs.
