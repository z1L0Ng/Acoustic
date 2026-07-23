# Datasets

Source facts are kept separate from proposed benchmark policy. Canonical field
definitions and mappings are under `dataset/processed/schema/`.

| Dataset | Native unit | Size | Labels | Identity and split facts | Current use |
|---|---|---:|---|---|---|
| ICBHI 2017 | recording and annotated cycle | 920 recordings, 126 patients, 6,898 cycles, 5.49 h | normal, crackle, wheeze, both; disease metadata | Official recording split is 4,142/2,756 cycles, but patients 156 and 218 cross it | source benchmark plus future strict patient-grouped robustness |
| HF_Lung_V1 | 15 s recording and overlapping interval | 9,765 recordings, 81,933 intervals, 40.69 h | inspiration, expiration, discontinuous and continuous sounds | No package patient ID; date/session is a proxy; source folders have no date-key overlap | temporal multilabel task; shared binary waits for negative-interval policy |
| SPRSound BioCAS2022 | recording and event | 2,683 recordings, 292 patients, 9,089 events, 8.16 h | record labels plus seven event types | Train/inter have zero patient overlap; intra intentionally repeats 162 train patients | inter is primary cross-subject target; intra is a separate diagnostic |
| KAUH/Fraiwan v3 | recording | 336 recordings, 112 patients, 1.62 h | sound type and diagnosis | Three B/D/E recordings per patient; no official split | recording task after patient-grouped split and label abbreviations are approved |

## Integration principle

The project unifies lifecycle, IDs, lineage, mapping, QC, and reporting. It does
not pretend that recordings, cycles, intervals, and events are interchangeable.
Three compatible evaluation surfaces remain available:

1. Segment/event tasks for ICBHI cycles and SPRSound events.
2. Recording tasks for SPRSound and KAUH.
3. Dataset-specific heads for HF intervals, disease labels, quality labels, and
   any source-native category that cannot be represented honestly in a shared task.

Raw labels are retained. Shared binary or narrow-four targets are derived views
with explicit coverage masks. HF unlabeled gaps are not normal by default;
SPRSound Poor Quality is excluded from shared disease/sound targets; Rhonchi and
Stridor are not silently relabeled as wheeze. Proposed splits never overwrite
source split fields.

## Unresolved decisions

- HF negative intervals and whether the source train/test proxy is acceptable.
- KAUH B/D/E and `I C B` semantics, `Crep`/`C` normalization, and source-family naming.
- Whether binary or a richer multilabel task is the primary shared surface.
- The role of disease heads and Poor Quality.
- When to materialize an ICBHI strict patient-grouped split in addition to the
  official recording split.
