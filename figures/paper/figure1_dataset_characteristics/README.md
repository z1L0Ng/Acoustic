# Figure 1: annotations, study roles, and acoustic input distributions

The deliverables are `figure1_dataset_characteristics.svg`, `.pdf`, `.png`,
and the plotting script `.py`. The canvas is 178 × 65 mm. SVG text is editable;
the PDF remains vector and the PNG is 400 dpi. The small `source_data/` folder
must accompany the plotting script when it is moved.

Panel (a) gives schematic annotation units and the study roles. N/C/W mean
Normal/Crackle/Wheeze; Both is C/W co-presence. The small HF intervals are an
annotation schematic, not a sampled recording. KAUH B/D/E denotes the Bell,
Diaphragm and Extended filter versions within one patient.

Panel (b) plots every row of the saved
`paper_scope_spectral_only_balanced_scores.csv`, without fitting PCA, changing
signs, rescaling scores, selecting a projection, or dropping points. Its definition
is taken from the matching JSON. PC1 and PC2 explain 63.145% and 33.249%,
displayed as 63.15% and 33.25% on the axes. Stored values are unchanged.
The fitted features are log frequencies of the 80–2000 Hz power-spectrum
centroid, bandwidth and peak, after global waveform centering and common 5-s
preparation. The existing PCA uses common z-scoring and equal source weights.
Its 561 groups include patients and HF recording-date proxies. It describes
inputs and is not a learned model representation or causal transfer analysis.

Panels (c,d) plot `median`, `ci95_low` and `ci95_high` directly from the frozen
`class_matched_summary.csv`, with `dbfs_dc` and `band80_centroid_hz`. These are
95% intervals for each source median, not intervals for a source difference.
The original aggregation was within recording and then patient, and the existing
intervals came from a 1,000-replicate patient bootstrap. This script does not
repeat aggregation or bootstrap estimation. SPRSound Both has 10 patients.

Original read-only sources:

- PCA scores, definition, loadings and source summary:
  `/Users/zilongzeng/.codex/worktrees/5513/Acoustic/result/wade_acoustic_validation/pca/`.
- Frozen class summaries, supports and underlying patient medians:
  `/Users/zilongzeng/Research/Acoustic/docs/paper/ICASSP_2026_acoustic_disease/Figure/figure1_source_data/`.
- Analysis definitions:
  `/Users/zilongzeng/.codex/worktrees/5513/Acoustic/docs/analysis/wade_acoustic_validation/REPORT_zh.md`.

Figure scope: ICBHI 6,898 cycles / 126 patients; SPRSound train+inter 8,085 events /
284 patients; HF source-test 1,956 recordings / 5,868 windows / 39 recording-date
groups; KAUH 336 filter-version files / 112 patients. The compatible KAUH
classification evaluation is separately restricted to 86 patients. The displayed
HF source-test count does not describe the auxiliary-training partition.

Re-render with Python, Matplotlib and Arial fonts. This machine's existing
`/opt/anaconda3/envs/bsm/bin/python` was used because the bundled PDF Python
does not include Matplotlib; no packages or environments were changed:

```sh
/opt/anaconda3/envs/bsm/bin/python figure1_dataset_characteristics.py
```

No audio, model execution, statistical re-estimation or manuscript compilation is
part of this package. Integrate through the writing task into `Overleaf_Sync_final`
and retain `fig:dataset_characteristics`. The writing task adds red DONE at the
caption start only after completing its review and integration; the figure has no
DONE marker.
