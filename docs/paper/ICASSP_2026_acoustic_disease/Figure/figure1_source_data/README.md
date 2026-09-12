# Figure 1: acoustic dataset characteristics

The final assets are `../figure1_dataset_characteristics.pdf`, `.svg`, `.png`,
and `.py`. The PDF is exactly **7.00 x 2.26 inches** (504 x 162.72 pt), suitable
for the existing spconf `figure*` at `width=\textwidth`. Arial, slate text,
muted blue/green/purple/tan and thin rules coordinate with the current Figure 2.

## Caption

Acoustic characteristics of 5-s inputs. (a) Group-level spectral centroid
distributions for ICBHI, SPRSound training/inter-test data, HF source-test data,
and KAUH. Group counts appear below the dataset names; HF groups are recording-date
proxies, whereas the other groups are patients. Boxes show the median and
interquartile range, with 1.5-IQR whiskers; outliers are omitted from display.
(b,c) Class-matched patient-level medians and 95% group-bootstrap intervals for
ICBHI and SPRSound. Summaries aggregate within recording and then within group;
KAUH B/D/E versions are aggregated within patient. Level is measured in dBFS,
and centroids use power within 80-2000 Hz after global mean removal.

## Definitions and scope for Section 2

- **Input:** the current method's mono 16 kHz waveform preparation, with a
  62.5 ms recording fade at each end. ICBHI cycles, SPR events, and KAUH recordings
  are front-truncated or repeated to 5 s; repeated/equal-length inputs receive
  the existing final fade-out. HF uses consecutive [0,5], [5,10], [10,15] s windows.
  The native cycle/event labels are retained; no new time-local labels are inferred.
- **Level:** `10 log10(mean((x - mean(x))^2))`, reference full-scale amplitude 1.
  This is **dBFS, not calibrated sound pressure**. No per-recording amplitude
  normalization is applied.
- **Spectrum:** prepared waveforms are resampled to 4 kHz for a common analysis
  band; the global waveform mean is removed. Welch power spectra use a periodic
  Hann window of 1 s, 50% overlap, FFT length 4000, and mean aggregation. Spectral
  centroid is `sum(f P(f)) / sum(P(f))` restricted to **80-2000 Hz inclusive**.
  The band restriction renormalizes spectral statistics; it is not a new waveform
  filtering step or a claim of an optimal diagnostic band.
- **Aggregation:** median of the relevant 5-s descriptors within each recording,
  then median across recordings within each patient/date group. Class-matched
  summaries apply this sequence separately within each compatible native class.
  Thus KAUH's three filter versions contribute one patient summary.
- **Intervals:** the existing 1,000-replicate, seed-20260911 percentile bootstrap
  over patient summaries, independently within each source/class. Panels (b,c)
  use per-source median intervals, not confidence intervals for the source difference.

| Overview source | 5-s inputs | Recordings | Groups |
|---|---:|---:|---:|
| ICBHI (all) | 6,898 | 920 | 126 patients |
| SPRSound (train + inter) | 8,085 | 2,119 | 284 patients |
| HF Lung (source-test) | 5,868 | 1,956 | 39 date proxies |
| KAUH (all B/D/E) | 336 | 336 | 112 patients |
| Total | 21,187 | 5,331 | 561 groups |

**561 is not a patient count.** SPR intra-test is excluded from this figure.
The overview includes all available inputs in these data roles, rather than
only a model's selected subtraining or attribute-eligible subset.

| Compatible class | ICBHI patients | SPRSound patients |
|---|---:|---:|
| Normal | 124 | 274 |
| Crackle | 74 | 89 |
| Wheeze | 63 | 63 |
| Both | 35 | 10 |

Class counts must **not** be added: a patient may contribute to multiple classes.
SPR Fine/Coarse Crackle map to Crackle, and Wheeze+Crackle maps to Both.
Rhonchi/Stridor are not forced into the compatible four-class comparison.
HF has no class-matched Normal lane; its unannotated regions are not negatives.
KAUH unresolved sound categories remain in the overview without a four-class mapping.

The supported observation is that source differences persist within compatible
native labels under the common waveform preparation. The figure does not
identify disease mechanisms, establish a cause of transfer errors, or compare
hierarchical and native readout performance. It contains no PCA result.

## Small input tables and provenance

| Packaged CSV | Contents | Existing verification source |
|---|---|---|
| `overview_group_centroids.csv` | All 561 group centroids used by (a) | `paper_scope_model5s_group_medians.csv`, selected columns |
| `overview_support.csv` | Input/recording/group counts | `paper_scope_support.csv` |
| `class_matched_summary.csv` | Sixteen source/class/feature medians and their existing intervals | `core_class_group_medians.csv` and the unchanged bootstrap sequence used in `04_class_matched_5s` |
| `class_group_medians.csv` | The 732 source/class patient summaries underlying (b,c) | `core_class_group_medians.csv`, model5s rows and selected columns |
| `class_support.csv` | Class-specific unit/recording/group counts | `class_support_model5s.csv` |

The original verification sources are under
`/Users/zilongzeng/.codex/worktrees/5513/Acoustic/result/wade_acoustic_validation/`.
Methods and results are documented in
`/Users/zilongzeng/.codex/worktrees/5513/Acoustic/docs/analysis/wade_acoustic_validation/REPORT_zh.md`
and `analysis/wade_acoustic_validation/PROTOCOL_zh.md` in that same worktree.
The waveform implementation was read from the current main checkout's
`baseline/pafa/joint_hierarchy.py:_prepare_waveforms` and
`baseline/pafa/jh2_hf_kauh_external.py:_load_hf_windows`.

No waveform analysis was repeated for this figure. To preserve the already
delivered per-source intervals, the existing bootstrap RNG sequence was replayed
using only its saved group counts and group medians. The resulting interval
endpoints match the previously exported `04_class_matched_5s.svg` to its vector
coordinate precision (maximum difference below 0.000001 in either displayed
unit). These frozen values are now explicit in `class_matched_summary.csv`;
the final plotting script does not run a bootstrap or any other analysis branch.

## Re-rendering and manuscript integration

Copy the four `figure1_dataset_characteristics.*` files **together with this
entire `figure1_source_data/` directory** into the main paper's `Figure/` directory.
The plotting script resolves inputs relative to itself and does not reference
the original analysis directory or audio. Python, NumPy, pandas, and Matplotlib
are sufficient; Arial is preferred with a sans-serif fallback.

From any working directory:

```bash
python /path/to/Figure/figure1_dataset_characteristics.py
```

The local creation used `/opt/anaconda3/envs/bsm/bin/python` (Python 3.11.14,
NumPy 2.4.2, pandas 3.0.0, Matplotlib 3.10.8). The final PNG is 400 dpi; the PDF
embeds TrueType text, and the SVG retains editable text. The figure includes
whiskers and intervals at their saved values without new normalization or clipping
of those plotted summaries. Only boxplot outlier markers in (a) are omitted.

After management has copied the package to main, the writing task can replace
the Figure 1 placeholder with:

```latex
\includegraphics[width=\textwidth]{Figure/figure1_dataset_characteristics.pdf}
```

Retain the existing `fig:dataset_characteristics` label. Before the management
copy, that relative asset path exists only in this worktree and must not be
assumed to exist in the main manuscript. No main.tex, Section file, old Figure 1
proposal, or Figure 2 was changed by this task.
