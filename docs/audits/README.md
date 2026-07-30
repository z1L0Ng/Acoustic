# Audits

Independent, read-only research audits. Each audit re-derives headline numbers
from one-hand `result/` artifacts at execution time rather than transcribing
reports, and separates `source_official_facts`, project-measured results, and
proposed policy.

| Audit | Scope | Reproducibility |
|---|---|---|
| [2026-07-23_research_audit.ipynb](2026-07-23_research_audit.ipynb) | ICBHI source alignment, SPRSound zero-target transfer, frozen-encoder target-head diagnostic, four-dataset curation, candidate paper-line novelty vs BTS-CARD / LungMix | `Restart & Run All`; code cells read only `result/*.json` and `event_manifest.jsonl` (no GPU / torch / raw audio) |

Companion analysis: [dataset/script/acoustic_distribution_analysis.ipynb](../../dataset/script/acoustic_distribution_analysis.ipynb)
implements decision gate **E6** (acoustic domain shift + dataset-identity
separability) from the 2026-07-23 audit.
