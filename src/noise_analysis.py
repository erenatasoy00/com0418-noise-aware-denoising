"""Noise-type classifier and noise-level estimator (SO4).

Two hand-crafted features, both cheap to compute:
  * impulse purity: among pixels that differ strongly (> 0.3) from their 3x3 median,
    the fraction that sit exactly at 0 or 1. Close to 1 only for salt-and-pepper.
  * brightness slope: the noise std measured in the finest wavelet HH sub-band,
    per brightness bin, fitted as log(std) = a + slope * log(mean). Additive Gaussian
    noise has slope ~0; multiplicative speckle (y = x + x*n) has slope ~1.
Thresholds are set on Set12 (validation) and evaluated on BSD68 (test).

Usage:
    python -m src.noise_analysis
"""
import cv2
import numpy as np
import pandas as pd
import pywt
from skimage.restoration import estimate_sigma

from src.config import FIGURES, RESULTS
from src.noise import NOISE_LEVELS, conditions, list_images, noisy_version

TYPES = list(NOISE_LEVELS)
UNSEEN_LEVELS = [("gaussian", 35), ("sp", 0.15), ("speckle", 0.07)]  # robustness test (SO4)


def _brightness_bins(noisy, n_bins=8):
    """(mean, noise std) per brightness bin, from the Haar LL and HH sub-bands."""
    ll, (_, _, hh) = pywt.dwt2(noisy, "db1")
    mean = ll / 2
    valid = (mean > 0.08) & (mean < 0.92)  # avoid clipped extremes
    edges = np.quantile(mean[valid], np.linspace(0, 1, n_bins + 1))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (mean >= lo) & (mean <= hi)
        if sel.sum() > 100:
            bins.append((np.median(mean[sel]), np.median(np.abs(hh[sel])) / 0.6745))
    return np.array(bins)


def extract_features(noisy):
    img = noisy.astype(np.float32)
    deviation = np.abs(img - cv2.medianBlur(img, 3))
    extreme = (noisy <= 0) | (noisy >= 1)
    large = deviation > 0.3
    bins = _brightness_bins(noisy)
    slope = np.polyfit(np.log(bins[:, 0]), np.log(bins[:, 1] + 1e-6), 1)[0] if len(bins) > 2 else 0.0
    return dict(purity=(large & extreme).sum() / max(large.sum(), 1), extreme_ratio=extreme.mean(), slope=slope)


def fit_thresholds(features):
    """Midpoint between the classes on the validation set."""
    is_sp = features["noise"] == "sp"
    purity = (features.loc[~is_sp, "purity"].max() + features.loc[is_sp, "purity"].min()) / 2
    gauss, speckle = features[features["noise"] == "gaussian"], features[features["noise"] == "speckle"]
    slope = (gauss["slope"].max() + speckle["slope"].min()) / 2
    return dict(purity=purity, slope=slope)


def classify(feat, thresholds):
    if feat["purity"] > thresholds["purity"]:
        return "sp"
    return "speckle" if feat["slope"] > thresholds["slope"] else "gaussian"


def estimate_level(noisy, noise_type, feat=None):
    """Gaussian: sigma (0-255) | sp: density | speckle: variance of n."""
    if noise_type == "gaussian":
        return estimate_sigma(noisy) * 255
    if noise_type == "sp":
        return (feat or extract_features(noisy))["extreme_ratio"]
    bins = _brightness_bins(noisy)
    return float(np.median((bins[:, 1] / bins[:, 0]) ** 2))


def analyse(dataset, conds, thresholds=None):
    rows = []
    for path in list_images(dataset):
        for noise_type, level in conds:
            _, noisy = noisy_version(path, noise_type, level)
            feat = extract_features(noisy)
            row = dict(image=path.stem, noise=noise_type, level=level, **feat)
            if thresholds:
                row["predicted"] = classify(feat, thresholds)
                row["level_est"] = estimate_level(noisy, noise_type, feat)  # level estimated for the true type
            rows.append(row)
    return pd.DataFrame(rows)


def class_report(df):
    cm = pd.crosstab(df["noise"], df["predicted"]).reindex(index=TYPES, columns=TYPES, fill_value=0)
    lines = ["| true \\ predicted | " + " | ".join(TYPES) + " |", "|---" * (len(TYPES) + 1) + "|"]
    lines += [f"| {t} | " + " | ".join(str(v) for v in cm.loc[t]) + " |" for t in TYPES]
    lines += ["", "| class | precision | recall | F1 |", "|---|---|---|---|"]
    for t in TYPES:
        tp = cm.loc[t, t]
        p, r = tp / max(cm[t].sum(), 1), tp / max(cm.loc[t].sum(), 1)
        lines.append(f"| {t} | {p:.3f} | {r:.3f} | {2 * p * r / max(p + r, 1e-9):.3f} |")
    return (df["noise"] == df["predicted"]).mean(), cm, "\n".join(lines)


def level_report(df):
    lines = ["| noise | true level | estimated (mean ± std) |", "|---|---|---|"]
    for (noise_type, level), g in df.groupby(["noise", "level"], sort=False):
        lines.append(f"| {noise_type} | {level:g} | {g['level_est'].mean():.3f} ± {g['level_est'].std():.3f} |")
    return "\n".join(lines)


def confusion_figure(cm, title, out_file):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from src.plots import BLUE_RAMP, INK, SURFACE  # shared chart styling
    names = {"gaussian": "Gaussian", "sp": "Salt & pepper", "speckle": "Speckle"}
    fig, ax = plt.subplots(figsize=(5.6, 4.6), layout="constrained")
    norm = cm.values / cm.values.sum(axis=1, keepdims=True)
    ax.pcolormesh(norm, cmap=LinearSegmentedColormap.from_list("b", BLUE_RAMP), vmin=0, vmax=1,
                  edgecolors=SURFACE, linewidth=2)
    for i in range(len(TYPES)):
        for j in range(len(TYPES)):
            ax.text(j + 0.5, i + 0.5, str(cm.values[i, j]), ha="center", va="center", fontsize=13,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ticks = np.arange(len(TYPES)) + 0.5
    ax.set_xticks(ticks, [names[t] for t in TYPES]), ax.set_yticks(ticks, [names[t] for t in TYPES])
    ax.invert_yaxis(), ax.grid(False)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11.5, loc="left")
    fig.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_file}")


if __name__ == "__main__":
    val = analyse("set12", conditions())
    thresholds = fit_thresholds(val)
    test = analyse("bsd68", conditions(), thresholds)
    unseen = analyse("bsd68", UNSEEN_LEVELS, thresholds)

    acc, cm, report = class_report(test)
    acc_unseen, _, report_unseen = class_report(unseen)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "# Noise-type classifier and noise-level estimator\n",
        f"Thresholds fitted on Set12 (validation, {len(val)} images): "
        f"impulse purity > {thresholds['purity']:.3f} → salt & pepper; "
        f"brightness slope > {thresholds['slope']:.3f} → speckle; otherwise Gaussian.\n",
        f"## Test: BSD68, 9 seen conditions ({len(test)} images) — accuracy {acc:.1%}\n", report,
        f"\n## Robustness: BSD68, unseen levels σ=35 / 15% / var 0.07 ({len(unseen)} images) — "
        f"accuracy {acc_unseen:.1%}\n", report_unseen,
        "\n## Noise-level estimation on BSD68 (true type given)\n", level_report(pd.concat([test, unseen])),
    ]) + "\n"
    (RESULTS / "noise_classifier.md").write_text(text)
    pd.concat([test.assign(split="test"), unseen.assign(split="unseen")]).to_csv(RESULTS / "noise_classifier_raw.csv",
                                                                                  index=False)
    print(text)
    confusion_figure(cm, f"Noise-type classifier on BSD68: accuracy {acc:.1%}",
                     FIGURES / "noise_classifier_confusion.png")
