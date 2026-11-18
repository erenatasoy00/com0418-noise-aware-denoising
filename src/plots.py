"""Figures for reports and slides, built from results/<dataset>_raw.csv.

Usage:
    python -m src.plots --dataset set12
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from skimage.restoration import estimate_sigma

from src.benchmark import ALL_METHODS, HYBRID_CHOICE, hybrid_preview
from src.config import FIGURES, RESULTS
from src.dncnn import warm_up as dncnn_warm_up
from src.filters import warm_up
from src.metrics import psnr
from src.noise import NOISE_LEVELS, NOISE_NAMES, level_label, list_images, noisy_version


# Validated reference palette (dataviz skill): 3 categorical slots, chrome and a blue sequential ramp.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# The three hybrid-pipeline branches; BM3D and the rest are in the heatmap (max 3 hues per small multiple).
SERIES = {"adaptive_median": "#2a78d6", "nlm": "#eb6834", "dncnn_sigma": "#1baf7a"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"noisy": "Noisy input", "mean": "Mean", "gaussian": "Gaussian", "median": "Median",
          "adaptive_median": "Adaptive median", "bilateral": "Bilateral", "wiener": "Wiener",
          "wavelet": "Wavelet (BayesShrink)", "nlm": "NLM", "bm3d": "BM3D",
          "dncnn_sigma": "DnCNN (σ-specific)", "dncnn_blind": "DnCNN (blind)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGURES / name}")


def psnr_vs_level(df, dataset):
    """Small multiples: one panel per noise type, three key methods + noisy baseline."""
    mean = df.groupby(["noise", "level", "method"])["psnr"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, layout="constrained")
    for ax, (noise_type, levels) in zip(axes, NOISE_LEVELS.items()):
        x = np.arange(len(levels))
        ax.plot(x, [mean[noise_type, l, "noisy"] for l in levels], color=AXIS, lw=1.5, marker="o", ms=5,
                label=LABELS["noisy"])
        for method, color in SERIES.items():
            ax.plot(x, [mean[noise_type, l, method] for l in levels], color=color, lw=2, marker="o", ms=6,
                    label=LABELS[method], markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(x, [level_label(noise_type, l) for l in levels])
        ax.set_xlim(-0.25, 2.25)
        ax.set_title(NOISE_NAMES[noise_type], fontsize=13, fontweight="bold", loc="left")
        ax.grid(axis="x", visible=False)
    # Direct labels on the Gaussian panel, where the line ends are well separated.
    ax = axes[0]
    ax.set_xlim(-0.25, 2.95)
    for method in ["noisy", *SERIES]:
        ax.annotate(LABELS[method], (2, mean["gaussian", 50, method]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK2)
    axes[0].set_ylabel("Mean PSNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"PSNR vs. noise level on {dataset.upper()} — no single method wins every noise type",
                 fontsize=14, x=0.01, ha="left")
    _save(fig, f"{dataset}_psnr_vs_level.png")


def psnr_heatmap(df, dataset):
    """All methods x all conditions; colour = closeness to the best method in that column."""
    df = df[df["method"] != "noisy"]
    cols = [(t, l) for t, levels in NOISE_LEVELS.items() for l in levels]
    methods = [m for m in ALL_METHODS if m in set(df["method"])]
    table = df.groupby(["method", "noise", "level"])["psnr"].mean().unstack(["noise", "level"]).loc[methods, cols]
    rel = (table - table.min()) / (table.max() - table.min())

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(13, 5.6), layout="constrained")
    ax.pcolormesh(rel.values, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    best = table.values.argmax(axis=0)
    for i in range(len(methods)):
        for j in range(len(cols)):
            ax.text(j + 0.5, i + 0.5, f"{table.values[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if rel.values[i, j] > 0.55 else INK,
                    fontweight="bold" if best[j] == i else "normal")
    ax.set_xticks(np.arange(len(cols)) + 0.5, [f"{NOISE_NAMES[t]}\n{level_label(t, l)}" for t, l in cols], fontsize=9.5)
    ax.set_yticks(np.arange(len(methods)) + 0.5, [LABELS[m] for m in methods])
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Mean PSNR (dB) on {dataset.upper()}.  Darker = closer to the best method for that condition;"
                 " best in bold.", fontsize=12, loc="left", pad=12)
    _save(fig, f"{dataset}_psnr_heatmap.png")


def hybrid_bar(df, dataset):
    """Key message for the hypothesis: oracle hybrid vs. best single method."""
    h = hybrid_preview(df)
    names = ["Noisy input", f"Best single method\n({LABELS[h['best_single']]})", "Oracle hybrid\n(true noise type)"]
    values = [h["noisy"], h["best_single_psnr"], h["oracle_hybrid_psnr"]]
    colors = [AXIS, AXIS, SERIES["adaptive_median"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    bars = ax.bar(names, values, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.2f} dB", ha="center", va="bottom", fontsize=11)
    gain = h["oracle_hybrid_psnr"] - h["best_single_psnr"]
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Mean PSNR over 9 noise conditions (dB)")
    ax.grid(axis="x", visible=False)
    choice = ", ".join(f"{NOISE_NAMES[k]} → {LABELS[v]}" for k, v in HYBRID_CHOICE.items())
    ax.set_title(f"Oracle hybrid beats the best single method by {gain:+.2f} dB (target ≥ +1 dB)",
                 fontsize=12.5, loc="left")
    fig.text(0.01, -0.09, f"{dataset.upper()}. Hybrid choice: {choice}.",
             fontsize=9, color=MUTED, ha="left")
    _save(fig, f"{dataset}_hybrid_preview.png")


def visual_comparison(dataset, image_stem="01"):
    """Same image, middle noise level of each type, key methods side by side."""
    warm_up()
    dncnn_warm_up()
    path = next(p for p in list_images(dataset) if p.stem == image_stem)
    methods = ["median", "adaptive_median", "nlm", "bm3d", "dncnn_sigma"]
    fig, axes = plt.subplots(3, len(methods) + 2, figsize=(18.5, 8.4), layout="constrained")
    for row, (noise_type, levels) in enumerate(NOISE_LEVELS.items()):
        clean, noisy = noisy_version(path, noise_type, levels[1])
        sigma = estimate_sigma(noisy)
        outs = {m: ALL_METHODS[m](noisy, sigma) for m in methods}
        scores = {m: psnr(clean, o) for m, o in outs.items()}
        winner = max(scores, key=scores.get)
        panels = [("Clean", clean, None), (f"Noisy ({level_label(noise_type, levels[1])})", noisy, psnr(clean, noisy))]
        panels += [(LABELS[m], np.clip(outs[m], 0, 1), scores[m]) for m in methods]
        for col, (title, img, score) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            text = title if score is None else f"{title}\n{score:.2f} dB"
            bold = col > 1 and methods[col - 2] == winner
            ax.set_title(text, fontsize=10.5, fontweight="bold" if bold else "normal")
        axes[row, 0].set_ylabel(NOISE_NAMES[noise_type], fontsize=13, fontweight="bold")
    fig.suptitle("Visual comparison (best PSNR per row in bold)", fontsize=14, x=0.01, ha="left")
    _save(fig, f"{dataset}_visual_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="set12")
    parser.add_argument("--image", default="01", help="image stem for the visual comparison")
    args = parser.parse_args()

    raw = pd.read_csv(RESULTS / f"{args.dataset}_raw.csv")
    psnr_vs_level(raw, args.dataset)
    psnr_heatmap(raw, args.dataset)
    hybrid_bar(raw, args.dataset)
    visual_comparison(args.dataset, args.image)
