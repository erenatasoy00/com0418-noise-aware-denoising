"""Noise simulation module (SO1).

Three noise types x three levels. Every (image, noise type, level) triple gets
its own fixed seed, so all experiments see exactly the same noisy images.

Usage:
    python -m src.noise --dataset set12            # save noisy PNGs + grid figure
    python -m src.noise --dataset bsd68 --no-grid
"""
import argparse
import zlib

import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.util import random_noise

from src.config import DATA, FIGURES

# gaussian: sigma on the 0-255 scale | sp: pixel density | speckle: variance of n in y = x + x*n
NOISE_LEVELS = {
    "gaussian": [15, 25, 50],
    "sp": [0.05, 0.10, 0.20],
    "speckle": [0.02, 0.05, 0.10],
}
NOISE_NAMES = {"gaussian": "Gaussian", "sp": "Salt & pepper", "speckle": "Speckle"}


def conditions():
    """All 9 (noise_type, level) pairs."""
    return [(t, lvl) for t, levels in NOISE_LEVELS.items() for lvl in levels]


def level_label(noise_type, level):
    if noise_type == "gaussian":
        return f"σ={level}"
    if noise_type == "sp":
        return f"d={int(level * 100)}%"
    return f"var={level}"


def list_images(dataset):
    folder = DATA / "clean" / dataset
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif"})
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    return files


def load_image(path, gray=True):
    """Read an image as float in [0, 1]."""
    img = img_as_float(io.imread(path))
    if gray and img.ndim == 3:
        img = rgb2gray(img[..., :3])
    return img


def make_seed(name, noise_type, level):
    """Deterministic seed per (image, noise type, level)."""
    return zlib.crc32(f"{name}|{noise_type}|{level}".encode())


def add_noise(img, noise_type, level, seed=0, clip=True):
    """Corrupt a [0, 1] float image. clip=False reproduces the published DnCNN/BM3D setting."""
    rng = np.random.default_rng(seed)
    if noise_type == "gaussian":
        return random_noise(img, mode="gaussian", var=(level / 255) ** 2, rng=rng, clip=clip)
    if noise_type == "sp":
        return random_noise(img, mode="s&p", amount=level, rng=rng)
    if noise_type == "speckle":
        return random_noise(img, mode="speckle", var=level, rng=rng, clip=clip)
    raise ValueError(f"Unknown noise type: {noise_type}")


def noisy_version(path, noise_type, level, clip=True):
    """Clean image and its reproducible noisy version."""
    clean = load_image(path)
    return clean, add_noise(clean, noise_type, level, make_seed(path.stem, noise_type, level), clip)


def save_noisy_dataset(dataset):
    out_dir = DATA / "noisy" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(dataset)
    for path in files:
        for noise_type, level in conditions():
            _, noisy = noisy_version(path, noise_type, level)
            io.imsave(out_dir / f"{path.stem}_{noise_type}_{level}.png", img_as_ubyte(np.clip(noisy, 0, 1)),
                      check_contrast=False)
    print(f"{len(files) * len(conditions())} noisy images -> {out_dir}")


def make_noise_grid(path, out_file):
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio as psnr

    mosaic = [["clean"] + [f"{t}{i}" for i in range(3)] for t in NOISE_LEVELS]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 9.4), layout="constrained")
    clean = load_image(path)
    axes["clean"].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes["clean"].set_title("Clean", fontsize=13, fontweight="bold")
    for noise_type, levels in NOISE_LEVELS.items():
        for i, level in enumerate(levels):
            _, noisy = noisy_version(path, noise_type, level)
            ax = axes[f"{noise_type}{i}"]
            ax.imshow(noisy, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{level_label(noise_type, level)}  |  PSNR {psnr(clean, noisy, data_range=1):.1f} dB",
                         fontsize=10)
            if i == 0:
                ax.set_ylabel(NOISE_NAMES[noise_type], fontsize=13, fontweight="bold")
    for ax in axes.values():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Synthetic noise models: 3 types × 3 levels", fontsize=15)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Grid figure -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="set12")
    parser.add_argument("--grid-image", default="05", help="image stem used for the grid figure")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    save_noisy_dataset(args.dataset)
    if not args.no_grid:
        img = next(p for p in list_images(args.dataset) if p.stem == args.grid_image)
        make_noise_grid(img, FIGURES / "noise_grid.png")
