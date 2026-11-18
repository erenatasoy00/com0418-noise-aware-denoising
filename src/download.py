"""Download datasets and pretrained weights (same result on a laptop and on Colab).

Usage:
    python -m src.download                      # Set12 + BSD68 + DnCNN weights (~17 MB)
    python -m src.download --sets bsd400 kodak24
    python -m src.download --all               # everything (~60 MB)
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from src.config import DATA, PRETRAINED

GITHUB_DIRS = {  # dataset -> (repo, folder) on GitHub
    "set12": ("cszn/DnCNN", "testsets/Set12"),
    "bsd68": ("cszn/DnCNN", "testsets/BSD68"),
    "bsd400": ("cszn/DnCNN", "TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400"),
}
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/{}.pth"
WEIGHTS = ["dncnn_15", "dncnn_25", "dncnn_50", "dncnn_gray_blind"]
DEFAULT_SETS = ["set12", "bsd68"]
HEADERS = {"User-Agent": "com0418-denoising-downloader"}


def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        dest.write_bytes(response.read())
    return True


def _github_files(repo, folder):
    api = f"https://api.github.com/repos/{repo}/contents/{folder}"
    with urllib.request.urlopen(urllib.request.Request(api, headers=HEADERS), timeout=60) as response:
        return [(item["name"], item["download_url"]) for item in json.load(response) if item["type"] == "file"]


def download_set(name):
    out = DATA / "clean" / name
    if name == "kodak24":
        files = [(f"kodim{i:02d}.png", KODAK_URL.format(i)) for i in range(1, 25)]
    else:
        files = [(f, u) for f, u in _github_files(*GITHUB_DIRS[name])
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    with ThreadPoolExecutor(8) as pool:
        new = sum(pool.map(lambda item: _fetch(item[1], out / item[0]), files))
    print(f"{name:8s} {len(files):4d} images ({new} new) -> {out}")


def download_weights():
    new = sum(_fetch(WEIGHTS_URL.format(w), PRETRAINED / f"{w}.pth") for w in WEIGHTS)
    print(f"weights  {len(WEIGHTS):4d} files  ({new} new) -> {PRETRAINED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=[*GITHUB_DIRS, "kodak24"], default=DEFAULT_SETS)
    parser.add_argument("--all", action="store_true", help="all datasets and the weights")
    parser.add_argument("--no-weights", action="store_true")
    args = parser.parse_args()

    for name in [*GITHUB_DIRS, "kodak24"] if args.all else args.sets:
        download_set(name)
    if not args.no_weights:
        download_weights()
