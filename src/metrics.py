"""Evaluation helpers: PSNR, SSIM and runtime."""
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(clean, out):
    return peak_signal_noise_ratio(clean, np.clip(out, 0, 1), data_range=1.0)


def ssim(clean, out):
    return structural_similarity(clean, np.clip(out, 0, 1), data_range=1.0)


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed milliseconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
