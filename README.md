# Noise-Type-Aware Image Denoising

COM0418 Image Processing term project (2026–2027 Fall). Classical filters and DnCNN
are compared on Gaussian, salt-and-pepper and speckle noise, and a noise-type
classifier selects the best method automatically.

## Run on Google Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erenatasoy00/com0418-noise-aware-denoising/blob/main/notebooks/colab_run.ipynb)

Open the notebook with the badge above, choose `Runtime → Change runtime type → T4 GPU` and
`Runtime → Run all`. Code comes from this repository, data is downloaded to `/content` and
results are saved to `MyDrive/COM0418/results` on Google Drive.

## Run on a laptop

```bash
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m src.download     # add --all for BSD400 and Kodak24
```

Data and weights are not in git; `src.download` fetches them:

| folder | content | source |
|---|---|---|
| `data/clean/set12` | 12 grayscale images (validation) | `cszn/DnCNN` → `testsets/Set12` |
| `data/clean/bsd68` | 68 grayscale images (test) | `cszn/DnCNN` → `testsets/BSD68` |
| `data/clean/bsd400` | 400 grayscale images (training / CV) | `cszn/DnCNN` → `Train400` |
| `data/clean/kodak24` | 24 color images (color test) | r0k.us Kodak suite |
| `pretrained/` | `dncnn_15/25/50.pth`, `dncnn_gray_blind.pth` | `cszn/KAIR` release v1.0 |

Paths can be changed with `DATA_DIR`, `RESULTS_DIR` and `PRETRAINED_DIR` (see `src/config.py`).
DnCNN uses the GPU when available; set `DENOISE_DEVICE=cpu` to measure CPU runtimes.

## Pipeline

| step | command | output |
|---|---|---|
| 1. noise generation (SO1) | `python -m src.noise --dataset set12` | `data/noisy/set12/`, `results/figures/noise_grid.png` |
| 2. benchmark (SO2, SO3) | `python -m src.benchmark --dataset set12` | `results/set12_raw.csv`, `results/set12_summary.md` |
| 3. figures | `python -m src.plots --dataset set12` | `results/figures/set12_*.png` |
| 4. DnCNN sanity check | `python -m src.dncnn` | `results/dncnn_sanity_check.md` |
| 5. noise classifier (SO4) | `python -m src.noise_analysis` | `results/noise_classifier.md` |

Use `--dataset bsd68` for the test set. `--skip bm3d` makes the benchmark much faster.

## Code layout

| file | content |
|---|---|
| `src/config.py` | data / results / weights paths (overridable with environment variables) |
| `src/download.py` | downloads datasets and pretrained weights |
| `src/noise.py` | 3 noise types × 3 levels, fixed seed per (image, type, level) |
| `src/filters.py` | mean, Gaussian, median, adaptive median (Numba), bilateral, Wiener, wavelet (BayesShrink), NLM, BM3D |
| `src/dncnn.py` | DnCNN network + pretrained KAIR weights (σ-specific and blind) |
| `src/metrics.py` | PSNR, SSIM, runtime |
| `src/benchmark.py` | runs every method on every noisy image, writes CSV + Markdown tables |
| `src/noise_analysis.py` | noise-type classifier (impulse purity + brightness slope) and noise-level estimator |
| `src/plots.py` | figures for the report and slides |

## Conventions

- Images are float in [0, 1]; Gaussian σ is given on the 0–255 scale.
- Noise is clipped to [0, 1] as in the proposal; `add_noise(..., clip=False)` reproduces the
  published (unclipped) DnCNN/BM3D setting.
- Methods receive a **blindly estimated** noise level (`estimate_sigma`), never the true one.
- Set12 is used for tuning and threshold selection; BSD68 is reported as the test set.
