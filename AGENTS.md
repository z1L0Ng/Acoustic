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
- Current additional execution authorization: the user has explicitly approved the local-training task to complete HF-on seeds 0 and 1, reusing the completed historical seed42 run. This is a separate finite two-run task with the existing HF-on recipe and required native/HF/KAUH evaluation. It does not authorize other new experiments, and writing must not wait for these two runs.

## Paper-writing workspace and workflow

- Current workflow: the user has split writing into two parallel tasks on the main checkout. The existing `论文写作` task owns Sections 1/4/5 and Abstract, including Evaluation and its result tables; the second writing task owns Sections 2/3 and their associated figures/tables, starting discussion from Section 2. Discuss the relevant chapter before rewriting it, complete agreed edits, and continue within the assigned scope without returning to management for approval. Neither writer waits for the other writer's entire chapter sequence. Align Abstract/Conclusion with settled body claims before finalization. HF-on training remains independent and must not block writing. Both writing tasks follow the user-controlled management-synchronization rule below.

- The paper-writing task (`01a08442-92e6-7110-8399-e42eca520ea8`) works directly in `/Users/zilongzeng/Research/Acoustic` on the `main` branch, as requested by the user on 2026-09-08.
- The second writing task (`01a0a275-3711-73f3-9fec-7dbfd04c1611`) works in the same `/Users/zilongzeng/Research/Acoustic` main checkout at the user's explicit request, not a worktree. It owns `Section/2data.tex` and `Section/3method.tex` under the active manuscript. The existing writer owns `Section/1introduction.tex`, `Section/4evaluation.tex`, and `Section/5conclusion.tex`, plus Abstract in `main.tex`.
- Shared-file ownership: the existing writer is the sole direct editor of `main.tex` (including Abstract and global assembly), `citation.bib`, and shared template/global macro settings. The second writer sends exact, localized caption/bibliography/global-setting changes to that writer after the relevant user discussion; it does not overwrite those shared files or edit Sections 1/4/5. Both writers may read the other sections for consistency and coordinate only the needed terminology, symbols, citations, and interfaces. Preserve concurrent edits; do not restore entire files from an earlier snapshot.
- Figure coordination: the second writer directly coordinates with `论文绘图` (`01a084ea-4764-79a1-815b-e3b79948caca`) on Figures 1/2 and other data/method figures or tables belonging to Sections 2/3, owning the agreed brief and final asset integration. The existing writer owns Section 4 result tables (including Tables 1/2) and any corresponding Evaluation figures, and applies localized caption changes in `main.tex`. The drawing task receives the brief from the writer responsible for that item. Do not issue competing briefs for the same figure. User instructions can explicitly reassign an individual edit.
- The sole manuscript directory synchronized with Overleaf is `docs/paper/Overleaf_Sync_final/` under this main checkout. Use this exact path and capitalization for all subsequent paper edits and figure integration.
- `docs/paper/ICASSP_2026_acoustic_disease/` and previous manuscript directories are historical assets. Read them for reference only; do not resume editing them or automatically overwrite the active Overleaf copy with their content.
- Keep manuscript sources, figures, bibliography, and required template files in the active Overleaf directory. Keep internal discussion, provenance, management notes, and drafting comments outside the synchronized manuscript files. Hanlin Liu and his author email have been removed from the active author block at the user's request.
- Review markers are disabled at the user's request. Keep the active manuscript free of `DONE` completion markers and the `\revisiondone` macro. Do not add or restore these markers in sections, abstracts, captions, figures, or PPTs unless the user explicitly approves adding them again. Completing a revision does not authorize restoration. This supersedes all earlier instructions requiring automatic DONE markers. Other internal notes remain outside the synchronized manuscript.
- Routine manuscript edits do not trigger local LaTeX compilation, review-PDF generation, or PDF rendering/visual inspection. The user compiles in Overleaf; compile locally only when the user explicitly requests it.
- Send writing progress or a handoff to management only when the user explicitly permits synchronization. This also applies to completion, blockers, decision points, and scientific findings: discuss them with the user in the writing task, and wait for permission before syncing to management. Do not generate an automatic management receipt after each edit.
- Management-side collection of writing progress and its Work Plan/Notion synchronization follow the same user-authorized synchronization rule.

## Independent reviewer workflow

- The user requested a separate task titled `论文独立审阅`. It starts with fresh task history, is not forked from the writing task, and remains idle until the user explicitly requests a review in that task.
- Creation authorizes only a brief waiting acknowledgment. Do not inspect a manuscript, run tools, or begin a review during initialization. Management messages, a Work Plan time slot, manuscript edits, and training completion do not trigger reviewer work. Finish each user-requested review and return to waiting; do not create automatic monitoring or recursive review runs.
- Preserve independence from writing-task memory: do not read either writing task's conversations, summaries, handoffs, or memory-derived scientific narrative. The first review uses the user-designated manuscript and that request's explicit review criteria. Meeting critiques and management plans are not review inputs unless the user separately supplies them for that purpose.
- Review output is for the user in the reviewer task. Do not automatically edit the manuscript or synchronize findings to writing/management; synchronization requires the user's instruction. A review request does not authorize training, new inference, local compilation, Notion writes, or Git commits/pushes.
- If the user requests the latest local manuscript, the source is `/Users/zilongzeng/Research/Acoustic/docs/paper/Overleaf_Sync_final/` on main, not an older copy in the review worktree. Read that source only after review is requested; use an explicitly supplied PDF or snapshot when the user chooses one.

## Mandatory management handoff

- The paper-writing and independent-reviewer tasks are exempt from the automatic reporting requirements in this section and follow their user-controlled synchronization rules above.
- Every other Acoustic project task must report to the project-management task when its assigned work completes, reaches a decision gate, or becomes blocked. Do not finish silently inside a specialist task.
- The management destination is the task titled `接管 Acoustic 项目管理` (thread ID `01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae`).
- A completion handoff must state: execution status; completed scope; changed files or external pages; Verified Results; Interpretation; Issues/HOLD; actions not performed; and the next decision required from management or the user.
- Report planning, code readiness, partial cache, validation-only evidence, and completed experimental results as different statuses. Never upgrade one into another.
- Send routine progress only when requested. Completion, blockers, approval requests, and scientifically material deviations must always be returned to management.
- If the task cannot directly message the management task, place a compact management-ready handoff in its final response so management can collect it without reconstructing the work from logs.
