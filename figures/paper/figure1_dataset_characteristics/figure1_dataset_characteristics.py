"""Plot frozen source tables only; no PCA fitting, bootstrap or audio processing.

Run with Python and Matplotlib. All inputs are relative to this script.
Outputs: editable-text SVG, vector PDF and a 400-dpi PNG.
"""
from pathlib import Path
import csv
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

FONT_DIR = Path('/System/Library/Fonts/Supplemental')
for name in ['Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf']:
    font_manager.fontManager.addfont(FONT_DIR / name)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'text.color': '#263D50', 'axes.labelcolor': '#263D50',
    'xtick.color': '#364A5C', 'ytick.color': '#364A5C',
    'axes.linewidth': .55, 'axes.edgecolor': '#8C9AA6',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.unicode_minus': False, 'savefig.facecolor': 'white',
})

DATA = ROOT / 'source_data'
def rows(name):
    with (DATA / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

scores = rows('paper_scope_spectral_only_balanced_scores.csv')
definition = json.loads((DATA / 'paper_scope_spectral_only_balanced_definition.json').read_text())
summary = {
    (r['feature'], r['dataset'], r['compatible_label']): r
    for r in rows('class_matched_summary.csv')
}
W, H = 178 / 25.4 * 72, 65 / 25.4 * 72
fig = plt.figure(figsize=(178 / 25.4, 65 / 25.4), facecolor='white')
INK, RULE = '#263D50', '#D5DDE4'
COLORS = {'ICBHI': '#3B739E', 'SPRSound': '#328F76', 'HF_Lung': '#9365A9', 'KAUH': '#AF864B'}
MARKERS = {'ICBHI': 'o', 'SPRSound': '^', 'HF_Lung': 's', 'KAUH': 'D'}

def txt(x, y, value, size=9, bold=False, color=INK, ha='left', va='baseline', **kwargs):
    return fig.text(x/W, y/H, value, fontsize=size, weight='bold' if bold else 'normal',
                    color=color, ha=ha, va=va, **kwargs)

def line(x0,y0,x1,y1,color=RULE,lw=.5):
    fig.add_artist(Line2D([x0/W,x1/W],[y0/H,y1/H], transform=fig.transFigure,
                         color=color, linewidth=lw, solid_capstyle='butt'))

def box(x,y,w,h,color,label=None):
    fig.add_artist(Rectangle((x/W,y/H),w/W,h/H,transform=fig.transFigure,
                            facecolor=color+'14', edgecolor=color, linewidth=.55))
    if label is not None:
        txt(x+w/2,y+h/2,label,ha='center',va='center')

def source_header(dataset, label, x, y):
    fig.add_artist(Line2D([x/W],[y/H],transform=fig.transFigure, linestyle='none',
                         marker=MARKERS[dataset],markersize=4,markerfacecolor=COLORS[dataset],
                         markeredgecolor=COLORS[dataset],markeredgewidth=.55))
    txt(x+7,y-3,label,size=10,bold=True,color=COLORS[dataset])

# (a) Shared dataset legend and concise annotation/task information.
txt(4,174,'(a)',size=10,bold=True)
source_header('ICBHI','ICBHI',29,177)
source_header('SPRSound','SPRSound',141,177)
source_header('HF_Lung','HF Lung',265,177)
source_header('KAUH','KAUH',391,177)
line(129,138,129,182)
for x in [254,379]:
    line(x,128,x,182)

txt(10,160,'Cycles (four classes)')
for i,label in enumerate(['N','C','W','Both']):
    box(10+i*27,142,23,12,COLORS['ICBHI'],label)

txt(137,160,'7 event categories')
txt(137,145,'Task: Normal/Adventitious')
line(10,138,246,138)
txt(128,127.5,'Core joint training and native evaluation',ha='center')

txt(262,160,'Timed intervals')
# Schematic interval overlap, not a measured recording or fabricated waveform.
line(337,158.5,371,158.5,color='#B4A4BD',lw=.55)
fig.add_artist(Rectangle((339/W,159/H),19/W,2.8/H,transform=fig.transFigure,
                        facecolor=COLORS['HF_Lung'],edgecolor='none'))
fig.add_artist(Rectangle((351/W,154.5/H),18/W,2.8/H,transform=fig.transFigure,
                        facecolor=COLORS['HF_Lung'],edgecolor='none'))
txt(262,145,'Separate C/W auxiliary')
txt(262,127.5,'source-test: attributes')

txt(387,160,'Recording-level classes')
txt(387,145,'Per patient:')
for i,label in enumerate(['B','D','E']):
    box(440+i*19,141.5,15,12,COLORS['KAUH'],label)
txt(387,127.5,'External evaluation')
line(4,122,500,122)

# (b) All saved coordinates; fixed axis bounds include every saved point.
txt(4,110,'(b) Spectral PCA',size=9.5,bold=True)
ax_b = fig.add_axes([30/W,29/H,129/W,74/H])
for dataset in ['KAUH','ICBHI','SPRSound','HF_Lung']:
    selected=[r for r in scores if r['dataset']==dataset]
    artist=ax_b.scatter([float(r['PC1']) for r in selected], [float(r['PC2']) for r in selected],
                        s=7.5,marker=MARKERS[dataset],color=COLORS[dataset],alpha=.64,
                        edgecolors='none',rasterized=False,zorder=3)
    artist.set_gid('pca-'+dataset)
ax_b.set_xlim(-4.5,8)
ax_b.set_ylim(-2.75,6.5)
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xticks([-4,0,4,8])
ax_b.set_yticks([-2,2,6])
ax_b.set_xlabel(f"PC1 ({100*definition['explained_variance_ratio'][0]:.2f}%)",labelpad=2)
ax_b.set_ylabel(f"PC2 ({100*definition['explained_variance_ratio'][1]:.2f}%)",labelpad=2)
ax_b.axhline(0,color=RULE,lw=.45,zorder=0)
ax_b.axvline(0,color=RULE,lw=.45,zorder=0)

# (c,d) The exact saved per-source medians and 95% interval endpoints.
txt(182,110,'(c) Level (dBFS)',size=9.5,bold=True)
txt(366,110,'(d) Centroid (Hz)',size=9.5,bold=True)
ax_c = fig.add_axes([222/W,29/H,120/W,74/H])
ax_d = fig.add_axes([386/W,29/H,110/W,74/H])
classes=['Normal','Crackle','Wheeze','Both']
for ax,feature in [(ax_c,'dbfs_dc'),(ax_d,'band80_centroid_hz')]:
    for dataset,offset in [('ICBHI',.13),('SPRSound',-.13)]:
        for i,label in enumerate(classes):
            row=summary[(feature,dataset,label)]
            median=float(row['median'])
            low=float(row['ci95_low'])
            high=float(row['ci95_high'])
            # Differences below encode bar lengths only; no new intervals are estimated.
            artists=ax.errorbar(median,3-i+offset,xerr=[[median-low],[high-median]],
                                fmt=MARKERS[dataset],markersize=3.5,color=COLORS[dataset],
                                markeredgewidth=.65,elinewidth=.9,capsize=1.7,
                                linestyle='none',zorder=3)
            artists.lines[0].set_gid(f'{feature}-{dataset}-{label}')
    ax.set_ylim(-.5,3.5)
    ax.set_yticks([3,2,1,0])
    ax.grid(axis='y',color='#E6EBEF',linewidth=.5,zorder=0)
    ax.tick_params(axis='y',length=0,pad=4)
ax_c.set_yticklabels(classes)
ax_d.set_yticklabels([])
ax_c.set_xlim(-55,-10)
ax_c.set_xticks([-50,-30,-10])
ax_d.set_xlim(125,235)
ax_d.set_xticks([140,180,220])

for ax in [ax_b,ax_c,ax_d]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x',direction='out',length=2.5,width=.55,pad=2)
ax_b.tick_params(axis='y',direction='out',length=2.5,width=.55,pad=2)
txt(343,5,'Source medians + 95% CIs; SPRSound Both: n=10',ha='center')

for extension in ['svg','pdf','png']:
    filename=ROOT/f'figure1_dataset_characteristics.{extension}'
    fig.savefig(filename,dpi=400,metadata={'Creator':'Matplotlib; frozen-table Figure 1'} if extension!='png' else None)
    print(filename)
print(f'Plotted {len(scores)} saved PCA coordinates and {len(summary)} frozen summary rows.')
