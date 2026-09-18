# Acoustic lab meeting presentation brief — 2026-09-18

## Requested deliverable

Prepare an approximately 10-minute research presentation to seek lab feedback on the paper. English slide text and English speaker notes; a complete Chinese speaking script in Notion, matched to slide numbers and timings. The user explicitly requested a dedicated task using GPT-5.6 Luna at max reasoning effort. This is presentation preparation, not a new Work Plan.

Calendar verified on September 18: **NIcE X F26 Lab Meeting**, Friday **September 18, 2026, 14:30–15:30 America/Chicago / 15:30–16:30 America/New_York**. Organizer: Jingping Nie. Zoom: https://unc.zoom.us/my/jingpingnie. The current occurrence starts at 14:30 Chicago time, although its original recurring start was 14:00. The user said “tomorrow” before the overnight work; prepare for this upcoming Friday afternoon meeting.

Create an editable PPTX, rendered slide previews for visual checking, and the Chinese script page. Work in the new task's worktree; keep the main manuscript unchanged. Read canonical main files by absolute path where needed, because recent feedback and this brief are not committed and will not automatically exist in the worktree. Do not create a Work Plan, launch experiments, access the server, edit the manuscript, contact collaborators, commit/push Git, or activate the independent reviewer. No hashes or ML smoke/probe runs.

## Story and audience

The audience is the lab and collaborators; the goal is useful feedback before submission. Organize the talk around the scientific argument: existing work and observed limitations → precise research question → task formulation → LSAA → evidence and comparisons → open feedback questions. Do not present project-management history or a training/debugging diary.

Current paper title: **Learning Shared Acoustic Attributes Across Respiratory Sound Datasets with Heterogeneous Annotations**. Method: **LSAA (Learning Shared Acoustic Attributes)**. Current authors: Zilong Zeng, Wade Wu, Arian Azarang, Stephen Xia, Jingping Nie.

Frame the question as learning shared respiratory acoustic information from datasets with differing annotation schemes, prediction units, and native tasks, while retaining task-appropriate outputs. Avoid claiming to be the first heterogeneous-label joint-learning method: DCASE already offers relevant multi-source mapping/masking principles. Prior respiratory label merging/remapping and single-dataset multilabel methods also exist. The paper's scope and evidence, rather than an exaggerated universality claim, should lead the discussion.

The advisor asks for contributions that express findings or capabilities, not lists of classifiers, analyses, or planned figures. Make clear what the methodology enables and what the experiments support. Specific observed transfer weaknesses do not establish that all single-source methods or all learned features fail to generalize.

## Suggested 10-minute sequence

This is a starting outline; improve it to avoid repetition while keeping the total near ten minutes.

1. Paper question and feedback objective — 0:30.
2. Four datasets, heterogeneous annotations, and native tasks — 1:00.
3. Existing approaches and the specific observed gap — 1:30.
4. Problem formulation: shared supervision, distinct native predictions — 1:15.
5. LSAA architecture, annotation masks, and task readouts — 1:30.
6. Evaluation design and source/target roles — 0:45.
7. Main results, including completed PC-MCL and DCASE comparisons — 1:30.
8. What ablations support and what remains unresolved — 1:15.
9. Focused feedback questions — 0:45.

Use roughly nine main slides and at most a few useful backup slides. Keep the main deck legible, with editable data charts/tables and a clear editable method diagram where appropriate. Do not paste full paper pages or dense tables as the main presentation. Use meaningful titles and a coherent academic design. The Presentations skill and its render-and-inspect workflow apply; reuse the already settled language preferences rather than asking again. Rendering this PPT is authorized; local LaTeX compilation is not required or authorized by this request.

## Canonical scientific inputs

Canonical main repository: `/Users/zilongzeng/Research/Acoustic`.

The only active manuscript is `/Users/zilongzeng/Research/Acoustic/docs/paper/Overleaf_Sync_final/`. Read `main.tex`, `Section/1introduction.tex`, `Section/2data.tex`, `Section/3method.tex`, `Section/4evaluation.tex`, `Section/4tables.tex`, `Section/5conclusion.tex`, and `citation.bib`. Earlier manuscript directories are historical references.

Method details to check against those sources:

- Joint source training uses ICBHI cycles and SPRSound events with a shared, fine-tuned BEATs encoder. Main inputs are standardized to 5 seconds.
- Shared attributes are abnormality A, crackle C, and wheeze W. A uses Normal/Abnormal logits; C/W use sigmoid outputs. Annotation-aware masks prevent treating unobserved attributes as negatives. SPR Rhonchi/Stridor labels supervise A without asserting unsupported C/W negatives.
- SPRSound native binary output uses A. ICBHI native four-class output gates on A, then decodes C/W. Both positive produces Both; when A is abnormal and neither attribute clears its source-fitted threshold, use the larger probability-minus-threshold margin, with C winning a tie.
- HF transfer ranks broad CAS (Wheeze/Rhonchi/Stridor) with the maximum W-head probability across three 5-second windows. This is a proxy, not a claim that W semantics equal every CAS label.
- KAUH evaluates Normal/Abnormal on the 86 compatible patients, averaging probabilities over B/D/E views.
- Main LSAA uses ICBHI official-test Score for checkpoint selection/early stopping; source-internal validation fits C/W thresholds. Do not call the whole protocol validation-selected. HF and KAUH remain outside fitting and selection.
- PAFA-derived PCSL/GPAL regularization is part of the LSAA recipe. Its effects are not isolated by comparisons that also change encoder training, output parameterization, or source data.

Bibliography keys for grounded references: `zhang2022sprsound`, `ge2025lungmix`, `chua2024multibreath`, `jeong2026pcmcl`, `cornell2024dcase`, `jeong2025pafa`, `rocha2019icbhi`, `hsu2021hflung`, `fraiwan2021kauh`. Verify wording in the original local paper/source materials if making a detailed literature claim. Put full citations in the relevant English speaker notes, with short visible citations where useful.

## Results available for the talk

All values below are percentages, mean ± sample standard deviation over three seeds. These are completed results, not projections. Different methods do not share every experimental factor; present system-level comparisons with the relevant distinctions visible.

| Method | Source training | Encoder | ICBHI Score | SPRSound official Score | HF CAS AUROC | KAUH balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| LSAA | ICBHI + SPRSound | Fine-tuned BEATs | 61.17 ± 0.31 | 90.70 ± 0.34 | 86.44 ± 2.13 | 72.17 ± 1.64 |
| PC-MCL 5-second adaptation | ICBHI | Fine-tuned BEATs | 59.81 ± 0.80 | 48.91 ± 2.59 | 78.64 ± 7.00 | 75.39 ± 4.55 |
| DCASE-style respiratory adaptation | ICBHI + SPRSound | Frozen BEATs + trainable CRNN | 54.42 ± 1.25 | 91.42 ± 0.39 | 80.97 ± 6.42 | 77.40 ± 4.38 |

Within these three systems, LSAA has the highest mean ICBHI and HF values; DCASE has the highest mean SPRSound and KAUH values. LSAA's KAUH mean is below both comparison methods. No significance test has been performed. PC-MCL does not receive SPRSound training labels; its SPRSound difference cannot isolate a readout advantage.

PC-MCL is the successful revised 5-second adaptation with Adam learning rate 1e-4, seeds 0/1/42, 400 full epochs per seed and milestones 120/160. It is not an unchanged original-paper reproduction. All three runs completed without the old numerical-failure issue. Management directly checked the server summaries on September 18. The successful artifacts remain at `/files1/Zilong/Acoustic/result/reproduce/source_transfer_baselines/PC_MCL_ICBHI5s_lr1e4_20260917/`; this presentation task need not access or download them. Selected epochs are 3, 3, and 23 for seeds 0, 1, and 42 respectively. The older failed PC-MCL attempts are not evidence for this comparison.

Exact completed PC-MCL per-seed proportions for reconstruction of plots:

| Seed | ICBHI | SPRSound | HF CAS | KAUH |
| --- | --- | --- | --- | --- |
| 0 | 0.5994195265708645 | 0.5068228741460691 | 0.8123185591037331 | 0.7014005602240896 |
| 1 | 0.5895117146619043 | 0.5010568258272391 | 0.8396675798339943 | 0.7831932773109244 |
| 42 | 0.6053221901949063 | 0.45936785824453086 | 0.7071390603917079 | 0.77703081232493 |

DCASE uses native-class-union masked multi-label supervision, class-masked attention, frozen BEATs frames, and trainable CRNN fusion. Its nine outputs preserve distinct native classes and appropriate mappings. Selection averages the ICBHI native-four and SPR native-seven validation macro multilabel F1 at 0.5; it is a respiratory adaptation of a multi-source validation principle. Its advantage/disadvantage relative to LSAA cannot be attributed to only output decoding.

Relevant local evidence, relative to the canonical repository:

- LSAA main native results: `result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.json` and `.md`.
- Single-source controls: `result/reproduce/pafa_joint_hierarchy/PAFA_BENCHMARK_4COND_multiseed/multiseed_summary.json`; readable summary in `docs/review_packages/2026-09-15_0201_EDT_independent_review/optional_evidence/core_benchmark_summary.md`.
- HF/KAUH results: `result/reproduce/pafa_joint_hierarchy/PAFA_TABLE2_POSTHOC_HF_CAS_KAUH_20260914/results/hf_cas_kauh_table2_supplement.json`.
- Successful DCASE evidence: `docs/result_exports/2026-09-17_dcase_joint_native_union/README.md`, `results_summary.json`, and original per-seed exports. Full local artifacts also exist under `result/reproduce/source_transfer_baselines/DCASE_Joint_NativeUnion_5s/`.

Single-source context, if useful for the gap/result slides:

| LSAA recipe/control | ICBHI Score | SPRSound Score | HF CAS AUROC | KAUH BA |
| --- | --- | --- | --- | --- |
| ICBHI only | 57.78 ± 1.47 | 49.53 ± 6.40 | 87.05 ± 0.86 | 70.44 ± 1.52 |
| SPRSound only | 47.75 ± 0.51 | 92.82 ± 0.65 | 79.25 ± 1.08 | 67.25 ± 4.29 |
| Joint | 61.17 ± 0.31 | 90.70 ± 0.34 | 86.44 ± 2.13 | 72.17 ± 1.64 |

Relative to each corresponding native single-source control, joint LSAA changes ICBHI Score by +3.39 percentage points and SPRSound Score by −2.13 points. Do not describe this as a gain on every task.

Mechanism evidence from current paper tables:

- Native heads + matched C/W supervision relative to LSAA: ICBHI +2.50 ± 1.90 pp; SPRSound +0.16 ± 1.55 pp; SPR C/W AUROC −0.05 ± 0.25 pp. This challenges claims that the specific hierarchical readout is necessary or superior; shared acoustic supervision and native-task retention should be discussed separately from that readout.
- LSAA SPR C/W AUROC: 97.32 ± 0.15%.
- Coarse SPR relative to LSAA: ICBHI −6.51 ± 4.33 pp; SPRSound −23.46 ± 36.62 pp; C/W AUROC −8.48 ± 8.58 pp. One coarse-label seed collapsed, so show its variance and keep the inference qualified.
- HF-on is optional backup material only. Its existing seed42 pairing uses a historical reference and should not silently be mixed into fresh formal-seed comparisons.

## Revision feedback to shape the discussion

Read these canonical main inputs (recent Arian files are uncommitted):

- `/Users/zilongzeng/Research/Acoustic/docs/paper/revision_notes/2026-09-17_arian_feedback_zh.md`
- `/Users/zilongzeng/Research/Acoustic/docs/source_materials/collaborator_feedback/2026-09-17_arian_azarang/ICASSP-Comments.docx`
- `/Users/zilongzeng/Research/Acoustic/docs/review_packages/2026-09-15_0201_EDT_independent_review/02_ADVISOR_REVISION_BRIEF.md`
- `/Users/zilongzeng/Research/Acoustic/docs/meeting_records/2026-09-14_paper_revision_meeting_record_zh.md`

Arian's four main concerns are: selection/validation terminology; confounding between frozen baselines and fine-tuned LSAA plus PAFA regularization; the Native+C/W result versus claims about hierarchy; and missing implementation/split details. Address these honestly in the scientific narrative and backup material. A suggested paired significance test is not an existing result.

Possible closing questions: Is the scoped contribution compelling relative to respiratory multilabel and heterogeneous multi-source learning? How should shared supervision and the particular readout be positioned given Native+C/W? Which matched comparison or statistical analysis is essential before submission?

## Visual sources

Read/reuse without overwriting:

- `docs/paper/Overleaf_Sync_final/Figure/figure1_dataset_characteristics.pdf`
- `docs/paper/Overleaf_Sync_final/Figure/figure2_method.pdf`
- Editable PPT figure sources under `docs/paper/figure_sources/teacher_method_2026-09-14/`, including `ICASSP Figures_CAS_core_final.pptx` and `ICASSP Figures_HF_annotated_user_final.pptx`.
- Historical decks under `slides/`, especially `Acoustic_ICASSP_story_and_next_evidence_2026-09-04_v2.pptx` and `acoustic_multi_dataset_lab_intro_northwestern_v1.pptx`.

Historical decks are content/visual references, not authority for current metrics or an automatically selected template. Follow the Presentations skill's template choice workflow if needed. Required diagrams and data visualizations should remain editable. Render and inspect every final slide for legibility, clipping, and overlaps. Do not add DONE markers or internal management vocabulary.

## Chinese script in Notion

Verified Acoustic project parent page: `380309ef-da29-8071-b6c6-c1ee1031bbbc`, URL `https://app.notion.com/p/380309efda298071b6c6c1ee1031bbbc`.

Create a dedicated page titled **2026-09-18 组会论文汇报 中文讲稿** under that project, or in its Meeting Report database after fetching the schema. Meeting Report database ID: `380309efda298068ab68cbcfaa83d852`; data source: `collection://380309ef-da29-8041-a74f-000b300b8c92`. Do not place this in Working Plan or change the existing plan.

The Chinese script must be a usable spoken presentation, not only bullets: slide-by-slide explanations, transitions, timing totaling approximately ten minutes, and the closing requests for feedback. Match it to the actual deck and English notes after the slide sequence settles. Fetch back the page to verify it and return its URL. If Notion access is unavailable, keep a local script and report the actual blocker; do not claim it was published.

On completion, return the deck path, preview path, Notion script URL, talk duration, and any material unresolved issue. The management task is `01a06de3-f57c-7bf3-a7b2-2a6bd7c33aae` (Acoustic 项目管理). A single completion handoff is appropriate; do not continuously report unchanged progress or launch monitoring.
