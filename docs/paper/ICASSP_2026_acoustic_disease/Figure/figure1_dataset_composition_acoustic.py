"""Create Figure 1: dataset composition and acoustic heterogeneity."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "dataset" / "raw"
FEATURE_CSV = ROOT / "result" / "acoustic_distribution" / "recording_features.csv"
OUT_DIR = Path(__file__).resolve().parent

CLASS_ORDER = ["Normal", "Crackle", "Wheeze", "Both", "Other / unmapped"]
CLASS_COLORS = {
    "Normal": "#4C78A8",
    "Crackle": "#F28E2B",
    "Wheeze": "#E15759",
    "Both": "#B279A2",
    "Other / unmapped": "#BAB0AC",
}
DATASET_ORDER = ["ICBHI", "SPRSound", "HF Lung", "KAUH"]
FEATURE_DATASET_NAMES = {
    "ICBHI": "ICBHI",
    "SPRSound": "SPRSound",
    "HF_Lung": "HF Lung",
    "KAUH": "KAUH",
}
DATASET_COLORS = {
    "ICBHI": "#3B6FB6",
    "SPRSound": "#4E9F6D",
    "HF Lung": "#D9893D",
    "KAUH": "#8E6BBE",
}
DATASET_MARKERS = {"ICBHI": "o", "SPRSound": "s", "HF Lung": "^", "KAUH": "D"}

PANEL_TITLE_SIZE = 8.0
FEATURE_TITLE_SIZE = 6.8
AXIS_LABEL_SIZE = 6.6
TICK_LABEL_SIZE = 6.0
ANNOTATION_SIZE = 5.8
LEGEND_SIZE = 5.8


def icbhi_counts() -> Counter:
    path = ROOT / "dataset" / "processed" / "manifests" / "icbhi_2017_cycles.csv"
    labels = pd.read_csv(path)["native_four_class_label"].str.lower()
    mapping = {"normal": "Normal", "crackle": "Crackle", "wheeze": "Wheeze", "both": "Both"}
    return Counter(mapping.get(label, "Other / unmapped") for label in labels)


def sprsound_counts() -> Counter:
    base = RAW / "sprsound" / "source_original" / "SPRSound-874eeb8736ddb78937c2fb5332fc7e7293d0f0ca" / "BioCAS2022"
    mapping = {
        "Normal": "Normal",
        "Fine Crackle": "Crackle",
        "Coarse Crackle": "Crackle",
        "Wheeze": "Wheeze",
        "Wheeze+Crackle": "Both",
    }
    counts = Counter()
    for path in sorted(base.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload.get("event_annotation", []):
            counts[mapping.get(event["type"], "Other / unmapped")] += 1
    return counts


def hf_counts() -> Counter:
    counts = Counter()
    for wav in sorted((RAW / "hf_lung_v1" / "source_original").rglob("*.wav")):
        label_path = wav.with_name(f"{wav.stem}_label.txt")
        tokens = set()
        if label_path.is_file():
            for line in label_path.read_text(errors="replace").splitlines():
                parts = line.split()
                if parts:
                    tokens.add(parts[0])
        crackle = "D" in tokens
        wheeze = "Wheeze" in tokens
        if crackle and wheeze:
            label = "Both"
        elif crackle:
            label = "Crackle"
        elif wheeze:
            label = "Wheeze"
        else:
            label = "Other / unmapped"
        counts[label] += 1
    return counts


def kauh_counts() -> Counter:
    counts = Counter()
    base = RAW / "kauh_fraiwan" / "source_original" / "audio_files"
    for wav in sorted(base.glob("*.wav")):
        fields = wav.name.split(",")
        sound = " ".join(fields[1].split()) if len(fields) > 1 else ""
        if sound == "N":
            label = "Normal"
        elif sound in {"C", "I C"}:
            label = "Crackle"
        elif sound in {"E W", "I E W"}:
            label = "Wheeze"
        elif sound == "I C E W":
            label = "Both"
        else:
            label = "Other / unmapped"
        counts[label] += 1
    return counts


def native_unit_counts() -> dict[str, Counter]:
    return {
        "ICBHI": icbhi_counts(),
        "SPRSound": sprsound_counts(),
        "HF Lung": hf_counts(),
        "KAUH": kauh_counts(),
    }


def balanced_group_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(FEATURE_CSV)
    df["dataset"] = df["dataset"].map(FEATURE_DATASET_NAMES)
    feature_cols = [
        "dbfs",
        "crest_db",
        "snr_proxy_db",
        "centroid_hz",
        "bandwidth_hz",
        "rolloff85_hz",
        "flatness",
        "zcr",
        "band_0_100",
        "band_100_250",
        "band_250_500",
        "band_500_1000",
        "band_1000_2000",
    ]
    grouped = df.groupby(["dataset", "patient_id"], as_index=False)[feature_cols].median()
    n_groups = min(grouped.groupby("dataset").size())
    balanced = (
        grouped.groupby("dataset", group_keys=False)
        .sample(n=n_groups, random_state=20260903)
        .reset_index(drop=True)
    )
    x = balanced[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    for column in ["centroid_hz", "bandwidth_hz", "rolloff85_hz"]:
        x[column] = np.log10(x[column].clip(lower=0) + 1)
    x["flatness"] = np.log10(x["flatness"].clip(lower=1e-8))
    x["zcr"] = np.log10(x["zcr"].clip(lower=1e-8))
    for column in x.columns:
        low, high = x[column].quantile([0.01, 0.99])
        x[column] = x[column].clip(low, high)
    x_scaled = StandardScaler().fit_transform(x)
    model = PCA(n_components=2)
    coords = model.fit_transform(x_scaled)
    return balanced, coords, model.explained_variance_ratio_


def add_covariance_ellipse(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    scale = np.sqrt(chi2.ppf(0.80, 2))
    width, height = 2 * scale * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        points.mean(axis=0),
        width,
        height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=0.9,
        alpha=0.08,
        zorder=1,
    )
    ax.add_patch(ellipse)


def plot_panel_a(ax: plt.Axes, counts: dict[str, Counter]) -> None:
    x = np.arange(len(DATASET_ORDER))
    bottoms = np.zeros(len(DATASET_ORDER))
    totals = np.array([sum(counts[d].values()) for d in DATASET_ORDER])
    for label in CLASS_ORDER:
        values = np.array([counts[d].get(label, 0) for d in DATASET_ORDER])
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.68,
            color=CLASS_COLORS[label],
            edgecolor="#FBFAF7",
            linewidth=0.7,
            label=label,
        )
        for i, (value, bottom, total) in enumerate(zip(values, bottoms, totals)):
            if total and value / total >= 0.12 and value >= 500:
                ax.text(
                    i,
                    bottom + value / 2,
                    f"{value / total:.0%}",
                    ha="center",
                    va="center",
                    fontsize=ANNOTATION_SIZE,
                    color="white" if label != "Other / unmapped" else "#3D3D3D",
                    fontweight="bold",
                )
        bottoms += values
    units = ["cycles", "events", "recordings", "recordings"]
    ax.set_xticks(x, [f"{d}\n({u})" for d, u in zip(DATASET_ORDER, units)])
    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, pad=2)
    ax.set_ylabel("Native annotated units", labelpad=4, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, totals.max() * 1.30)
    ax.grid(axis="y", color="#DDD8CE", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    for i, total in enumerate(totals):
        ax.text(i, total + totals.max() * 0.018, f"{total:,}", ha="center", va="bottom", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(
        "(a) Dataset scale and class composition",
        loc="left",
        fontweight="bold",
        fontsize=PANEL_TITLE_SIZE,
        pad=5,
    )
    ax.set_facecolor("#FFFEFB")


def plot_panel_b(ax: plt.Axes, balanced: pd.DataFrame, coords: np.ndarray, variance: np.ndarray) -> None:
    for dataset in DATASET_ORDER:
        mask = balanced["dataset"].to_numpy() == dataset
        points = coords[mask]
        color = DATASET_COLORS[dataset]
        add_covariance_ellipse(ax, points, color)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=11,
            alpha=0.52,
            c=color,
            marker=DATASET_MARKERS[dataset],
            edgecolors="white",
            linewidths=0.20,
            label=dataset,
            zorder=2,
        )
    ax.axhline(0, color="#D8D4CA", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#D8D4CA", linewidth=0.5, zorder=0)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}% variance)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}% variance)", fontsize=AXIS_LABEL_SIZE)
    ax.legend(
        loc="lower left",
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.7,
        fontsize=LEGEND_SIZE,
        markerscale=0.8,
    )
    ax.set_title(
        "(b) Acoustic-feature PCA",
        loc="left",
        fontweight="bold",
        fontsize=PANEL_TITLE_SIZE,
        pad=4,
    )
    ax.text(
        0.02,
        0.97,
        "112 groups per dataset · 80% covariance ellipses",
        transform=ax.transAxes,
        fontsize=ANNOTATION_SIZE,
        color="#4D4D4D",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.set_facecolor("#FFFEFB")


def plot_panel_c(fig: plt.Figure, subgrid, balanced: pd.DataFrame) -> None:
    data = balanced.copy()
    data["Low-frequency ratio"] = data[["band_0_100", "band_100_250", "band_250_500"]].sum(axis=1)
    panels = [
        ("dbfs", "RMS level"),
        ("centroid_hz", "Spectral centroid"),
        ("bandwidth_hz", "Spectral bandwidth"),
        ("Low-frequency ratio", "Energy below 500 Hz"),
        ("flatness", "Spectral flatness"),
    ]
    transformed = {}
    for column, _ in panels:
        values = data[column].astype(float).copy()
        if column in {"centroid_hz", "bandwidth_hz", "flatness"}:
            floor = 1e-8 if column == "flatness" else 0.0
            values = np.log10(values.clip(lower=floor) + (0 if column == "flatness" else 1))
        mean = values.mean()
        std = values.std(ddof=0)
        transformed[column] = (values - mean) / std
        data[f"{column}_z"] = transformed[column]

    limit = 0.0
    for column, _ in panels:
        z_column = f"{column}_z"
        for dataset in DATASET_ORDER:
            values = data.loc[data["dataset"] == dataset, z_column].dropna().to_numpy()
            q05, q95 = np.quantile(values, [0.05, 0.95])
            limit = max(limit, abs(q05), abs(q95))
    limit = np.ceil(limit * 2) / 2 + 0.25

    for i, (column, title) in enumerate(panels):
        ax = fig.add_subplot(subgrid[0, i])
        y_positions = np.arange(len(DATASET_ORDER))[::-1]
        for y, dataset in zip(y_positions, DATASET_ORDER):
            values = data.loc[data["dataset"] == dataset, f"{column}_z"].dropna().to_numpy()
            q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            color = DATASET_COLORS[dataset]
            ax.hlines(y, q05, q95, color=color, linewidth=1.1, alpha=0.60)
            ax.vlines([q05, q95], y - 0.07, y + 0.07, color=color, linewidth=0.9, alpha=0.70)
            ax.hlines(y, q25, q75, color=color, linewidth=4.0, alpha=0.88)
            ax.scatter(median, y, s=25, color=color, edgecolor="white", linewidth=0.55, zorder=3)
            ax.vlines(median, y - 0.12, y + 0.12, color="#263238", linewidth=0.65, zorder=4)
        ax.set_title(title, fontsize=FEATURE_TITLE_SIZE, pad=5, fontweight="semibold")
        ax.set_ylim(-0.55, 3.55)
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks(y_positions)
        if i == 0:
            ax.set_yticklabels(DATASET_ORDER, fontsize=TICK_LABEL_SIZE)
        else:
            ax.set_yticklabels([])
        ax.axvline(0, color="#8FA3B5", linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.grid(axis="x", color="#DCE4EC", linewidth=0.4, alpha=0.75)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor("#FCFEFF")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": AXIS_LABEL_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="ticks", rc=mpl.rcParams)

    counts = native_unit_counts()
    balanced, coords, variance = balanced_group_features()

    fig = plt.figure(figsize=(7.16, 4.60), constrained_layout=False, facecolor="#F8F7F3")
    top_frame = FancyBboxPatch(
        (0.022, 0.515),
        0.956,
        0.402,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor="#FFF8E8",
        edgecolor="#D8B46A",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=-10,
    )
    bottom_frame = FancyBboxPatch(
        (0.022, 0.055),
        0.956,
        0.390,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor="#F1F7FC",
        edgecolor="#7EA5C8",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=-10,
    )
    fig.add_artist(top_frame)
    fig.add_artist(bottom_frame)

    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.64)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.20, 0.80], wspace=0.30)
    bottom = outer[1].subgridspec(1, 5, wspace=0.35)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    plot_panel_a(ax_a, counts)
    plot_panel_b(ax_b, balanced, coords, variance)
    plot_panel_c(fig, bottom, balanced)

    class_handles = [Patch(facecolor=CLASS_COLORS[label], edgecolor="none", label=label) for label in CLASS_ORDER]
    fig.legend(
        handles=class_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.15,
        handlelength=1.5,
        title="Mapped class",
        title_fontsize=AXIS_LABEL_SIZE,
    )
    fig.text(
        0.070,
        0.448,
        "(c) Standardized acoustic feature distributions across balanced recording groups",
        fontweight="bold",
        fontsize=PANEL_TITLE_SIZE,
        va="center",
    )

    fig.text(
        0.590,
        0.068,
        "Standardized feature value (z-score)",
        ha="center",
        fontsize=AXIS_LABEL_SIZE,
        color="#36454F",
    )
    fig.subplots_adjust(left=0.085, right=0.970, top=0.858, bottom=0.120)

    stem = OUT_DIR / "figure1_dataset_composition_acoustic"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Panel (a) counts:")
    for dataset in DATASET_ORDER:
        print(dataset, dict(counts[dataset]), "total", sum(counts[dataset].values()))
    print("PCA explained variance:", variance.tolist())
    print("Balanced groups:", balanced.groupby("dataset").size().to_dict())
    print("Wrote", stem.with_suffix(".png"))
    print("Wrote", stem.with_suffix(".pdf"))
    print("Wrote", stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
