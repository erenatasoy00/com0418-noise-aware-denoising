"""Pretrained DnCNN [5] with the public KAIR weights [17] (SO3).

Weights: https://github.com/cszn/KAIR/releases/tag/v1.0 -> pretrained/
    dncnn_15.pth, dncnn_25.pth, dncnn_50.pth  (17 layers, one model per sigma)
    dncnn_gray_blind.pth                      (20 layers, blind, sigma in [0, 55])
In the released files BatchNorm is already merged into the convolutions, so the
network is Conv+ReLU, (Conv+ReLU) x (D-2), Conv, with residual learning.

Usage:
    python -m src.dncnn       # sanity check against the published BSD68 numbers
"""
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from src.config import PRETRAINED, RESULTS

# GPU when available (Colab T4), else CPU. Override with DENOISE_DEVICE=cpu for CPU runtimes.
DEVICE = torch.device(os.environ.get("DENOISE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
SIGMA_MODELS = {15: "dncnn_15", 25: "dncnn_25", 50: "dncnn_50"}
PUBLISHED_BSD68 = {15: 31.73, 25: 29.23, 50: 26.23}   # DnCNN-S, Table II of [5]


class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=64, image_channels=1):
        super().__init__()
        layers = [nn.Conv2d(image_channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, image_channels, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.model(x)  # clean estimate = noisy - predicted noise map R(y)


@lru_cache(maxsize=None)
def load_model(name):
    model = DnCNN(depth=20 if "blind" in name else 17)
    state = torch.load(PRETRAINED / f"{name}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(DEVICE)


@torch.inference_mode()
def denoise(noisy, name):
    x = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))[None, None].to(DEVICE)
    return load_model(name)(x)[0, 0].cpu().numpy().astype(np.float64)  # .cpu() also syncs the GPU for timing


def nearest_sigma_model(sigma):
    """sigma is on the [0, 1] scale; returns the name of the closest sigma-specific model."""
    return SIGMA_MODELS[min(SIGMA_MODELS, key=lambda s: abs(s - sigma * 255))]


def dncnn_sigma(noisy, sigma):
    """DnCNN-S: the sigma-specific model nearest to the estimated noise level."""
    return denoise(noisy, nearest_sigma_model(sigma))


def dncnn_blind(noisy, sigma):
    """DnCNN-B: one blind model for every noise level (sigma is ignored)."""
    return denoise(noisy, "dncnn_gray_blind")


DEEP_METHODS = {"dncnn_sigma": dncnn_sigma, "dncnn_blind": dncnn_blind}


def warm_up():
    for name in [*SIGMA_MODELS.values(), "dncnn_gray_blind"]:
        denoise(np.zeros((32, 32)), name)


def sanity_check(dataset="bsd68"):
    """Unclipped Gaussian noise, as in the paper, with the matching sigma-specific model."""
    from src.metrics import psnr
    from src.noise import list_images, noisy_version

    rows = ["| sigma | ours (dB) | published (dB) | difference |", "|---|---|---|---|"]
    for sigma, model in SIGMA_MODELS.items():
        scores = []
        for path in list_images(dataset):
            clean, noisy = noisy_version(path, "gaussian", sigma, clip=False)
            scores.append(psnr(clean, denoise(noisy, model)))
        ours = float(np.mean(scores))
        rows.append(f"| {sigma} | {ours:.2f} | {PUBLISHED_BSD68[sigma]:.2f} | {ours - PUBLISHED_BSD68[sigma]:+.2f} |")
    return f"# DnCNN sanity check on {dataset.upper()} (unclipped Gaussian noise)\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = sanity_check()
    (RESULTS / "dncnn_sanity_check.md").write_text(report)
    print(report)
