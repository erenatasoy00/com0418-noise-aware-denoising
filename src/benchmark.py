"""Run every classical method on every (image, noise type, level) and save the results.

Noisy images are regenerated on the fly with the same fixed seeds as src/noise.py,
so the float noise is used directly (no 8-bit rounding). The noise level given
to the methods is estimated blindly from the noisy image (skimage estimate_sigma,
i.e. median(|HH|) / 0.6745), exactly as the hybrid pipeline will do.

Usage:
    python -m src.benchmark --dataset set12
    python -m src.benchmark --dataset bsd68 --skip bm3d
"""
import argparse
import time

import pandas as pd
from skimage.restoration import estimate_sigma

from src.dncnn import DEEP_METHODS, DEVICE, warm_up as dncnn_warm_up
from src.filters import METHODS, warm_up
from src.metrics import psnr, ssim, timed
from src.config import RESULTS
from src.noise import conditions, list_images, noisy_version

ALL_METHODS = METHODS | DEEP_METHODS


def run(dataset, methods):
    warm_up()
    dncnn_warm_up()
    files = list_images(dataset)
    rows, total, start = [], len(files) * len(conditions()), time.time()
    for n, (path, (noise_type, level)) in enumerate(((p, c) for p in files for c in conditions()), 1):
        clean, noisy = noisy_version(path, noise_type, level)
        sigma = estimate_sigma(noisy)
        base = dict(image=path.stem, noise=noise_type, level=level, sigma_est=sigma * 255)
        rows.append(base | dict(method="noisy", psnr=psnr(clean, noisy), ssim=ssim(clean, noisy), ms=0.0))
        for name in methods:
            out, ms = timed(ALL_METHODS[name], noisy, sigma)
            device = str(DEVICE) if name in DEEP_METHODS else "cpu"
            rows.append(base | dict(method=name, psnr=psnr(clean, out), ssim=ssim(clean, out), ms=ms, device=device))
        print(f"\r[{n}/{total}] {path.stem} {noise_type} {level}   ({time.time() - start:.0f} s)", end="", flush=True)
    print()
    return pd.DataFrame(rows)


# Method the hybrid pipeline picks for each noise type (proposal, Section 3).
HYBRID_CHOICE = {"sp": "adaptive_median", "gaussian": "dncnn_sigma", "speckle": "nlm"}
# Alternative suggested by the Set12 validation results: blind DnCNN beats NLM on speckle.
HYBRID_ALT = HYBRID_CHOICE | {"speckle": "dncnn_blind"}


def hybrid_preview(df, choice=HYBRID_CHOICE):
    """Oracle hybrid (true noise type known) vs. the best single fixed method."""
    methods = df[df["method"] != "noisy"]
    per_method = methods.groupby("method")["psnr"].mean()
    best_single = per_method.idxmax()
    oracle = methods[methods["method"] == methods["noise"].map(choice)]["psnr"].mean()
    return dict(noisy=df[df["method"] == "noisy"]["psnr"].mean(), best_single=best_single,
                best_single_psnr=per_method.max(), oracle_hybrid_psnr=oracle)


def summarize(df, dataset):
    df = df.assign(condition=df["noise"] + " " + df["level"].map("{:g}".format))
    order = list(dict.fromkeys(df["condition"]))
    methods = list(dict.fromkeys(df["method"]))

    stats = df.groupby(["method", "condition"])[["psnr", "ssim", "ms"]].agg(["mean", "std"])
    stats.to_csv(RESULTS / f"{dataset}_summary.csv")

    lines = [f"# Benchmark results on {dataset} ({df['image'].nunique()} images)\n",
             "Noise level estimated blindly from each noisy image. Best value per column in **bold**.\n"]
    for metric, fmt in [("psnr", "{:.2f}"), ("ssim", "{:.3f}")]:
        table = df.pivot_table(index="method", columns="condition", values=metric, aggfunc="mean")[order]
        table["average"] = table.mean(axis=1)
        table = table.loc[methods]
        best = table.drop(index="noisy").max()
        lines.append(f"\n## Mean {metric.upper()}\n")
        lines.append("| method | " + " | ".join(table.columns) + " |")
        lines.append("|---" * (len(table.columns) + 1) + "|")
        for method, row in table.iterrows():
            cells = [f"**{fmt.format(v)}**" if method != "noisy" and v == best[c] else fmt.format(v)
                     for c, v in row.items()]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")

    runtime = df[df["method"] != "noisy"].groupby("method")["ms"].agg(["mean", "std"]).loc[methods[1:]]
    devices = df[df["method"].isin(list(DEEP_METHODS))]["device"].dropna().unique() if "device" in df else []
    lines.append(f"\n## Runtime per image (ms) — classical on CPU, DnCNN on {', '.join(devices) or 'cpu'}\n")
    lines.append("| method | mean | std |\n|---|---|---|")
    lines += [f"| {m} | {r['mean']:.1f} | {r['std']:.1f} |" for m, r in runtime.iterrows()]

    h = hybrid_preview(df)
    lines.append("\n## Hypothesis preview: oracle hybrid vs. best single method (mean PSNR over all 9 conditions)\n")
    lines.append("| | PSNR (dB) |\n|---|---|")
    lines.append(f"| noisy input | {h['noisy']:.2f} |")
    lines.append(f"| best single fixed method ({h['best_single']}) | {h['best_single_psnr']:.2f} |")
    choice = ", ".join(f"{k} → {v}" for k, v in HYBRID_CHOICE.items())
    lines.append(f"| oracle hybrid ({choice}) | **{h['oracle_hybrid_psnr']:.2f}** |")
    alt = hybrid_preview(df, HYBRID_ALT)["oracle_hybrid_psnr"]
    lines.append(f"| oracle hybrid, alternative (speckle → dncnn_blind) | {alt:.2f} |")
    lines.append(f"\nGain of the oracle hybrid over the best single method: "
                 f"**{h['oracle_hybrid_psnr'] - h['best_single_psnr']:+.2f} dB** (hypothesis target: ≥ +1 dB); "
                 f"alternative: {alt - h['best_single_psnr']:+.2f} dB.")

    report = "\n".join(lines) + "\n"
    (RESULTS / f"{dataset}_summary.md").write_text(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="set12")
    parser.add_argument("--methods", default=",".join(ALL_METHODS), help="comma-separated subset of methods")
    parser.add_argument("--skip", default="", help="comma-separated methods to leave out")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild the summary from the saved raw CSV")
    args = parser.parse_args()

    if args.summarize_only:
        print(summarize(pd.read_csv(RESULTS / f"{args.dataset}_raw.csv"), args.dataset))
        raise SystemExit

    skip = set(filter(None, args.skip.split(",")))
    methods = [m for m in args.methods.split(",") if m not in skip]
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = run(args.dataset, methods)
    df.to_csv(RESULTS / f"{args.dataset}_raw.csv", index=False)
    print(summarize(df, args.dataset))
