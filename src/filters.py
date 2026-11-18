"""Classical denoising methods (SO2).

Every method has the same signature: f(noisy, sigma) -> denoised, where noisy is
a float image in [0, 1] and sigma is the (estimated) noise std on the same scale.
Parameters below are reasonable defaults; grid-search tuning on Set12 comes next.
"""
import cv2
import numpy as np
from numba import njit
from scipy.signal import wiener as _wiener
from skimage.restoration import denoise_nl_means, denoise_wavelet
import bm3d as _bm3d


def _f32(img):
    return np.ascontiguousarray(img, dtype=np.float32)


def mean_filter(noisy, sigma, ksize=3):
    return cv2.blur(_f32(noisy), (ksize, ksize)).astype(np.float64)


def gaussian_filter(noisy, sigma, ksize=5):
    return cv2.GaussianBlur(_f32(noisy), (ksize, ksize), 0).astype(np.float64)


def median_filter(noisy, sigma, ksize=3):
    return cv2.medianBlur(_f32(noisy), ksize).astype(np.float64)


@njit(cache=True)
def _adaptive_median(padded, h, w, pad, max_size):
    out = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            ci, cj = i + pad, j + pad
            zxy = padded[ci, cj]
            out[i, j] = zxy
            r = 1
            while 2 * r + 1 <= max_size:
                win = padded[ci - r:ci + r + 1, cj - r:cj + r + 1]
                zmin, zmax, zmed = win.min(), win.max(), np.median(win)
                if zmin < zmed < zmax:          # stage A: median is not an impulse
                    if not (zmin < zxy < zmax):  # stage B: centre pixel is an impulse
                        out[i, j] = zmed
                    break
                r += 1
                if 2 * r + 1 > max_size:         # window limit reached
                    out[i, j] = zmed
    return out


def adaptive_median_filter(noisy, sigma, max_size=7):
    """Hwang & Haddad adaptive median filter [7] (Numba-accelerated)."""
    pad = max_size // 2
    padded = np.pad(np.asarray(noisy, dtype=np.float64), pad, mode="reflect")
    return _adaptive_median(padded, noisy.shape[0], noisy.shape[1], pad, max_size)


def bilateral_filter(noisy, sigma, d=7, sigma_space=2.0):
    sigma_color = max(2.0 * sigma, 0.02)
    return cv2.bilateralFilter(_f32(noisy), d, sigma_color, sigma_space).astype(np.float64)


def wiener_filter(noisy, sigma, size=5):
    return _wiener(noisy, mysize=size, noise=max(sigma, 1e-4) ** 2)


def wavelet_filter(noisy, sigma):
    return denoise_wavelet(noisy, sigma=sigma, wavelet="sym4", wavelet_levels=3,
                           method="BayesShrink", mode="soft", rescale_sigma=True)


def nlm_filter(noisy, sigma):
    sigma = max(sigma, 1e-3)
    return denoise_nl_means(noisy, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6, fast_mode=True)


def bm3d_filter(noisy, sigma):
    return _bm3d.bm3d(noisy, sigma_psd=max(sigma, 1e-3))


METHODS = {
    "mean": mean_filter,
    "gaussian": gaussian_filter,
    "median": median_filter,
    "adaptive_median": adaptive_median_filter,
    "bilateral": bilateral_filter,
    "wiener": wiener_filter,
    "wavelet": wavelet_filter,
    "nlm": nlm_filter,
    "bm3d": bm3d_filter,
}


def warm_up():
    """Trigger Numba compilation so it is not counted in runtime measurements."""
    adaptive_median_filter(np.random.default_rng(0).random((16, 16)), 0.1)
