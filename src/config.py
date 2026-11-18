"""Project paths. Each one can be overridden with an environment variable, e.g. on Colab:

    DATA_DIR=/content/data                                  (fast local disk, re-downloaded each session)
    RESULTS_DIR=/content/drive/MyDrive/COM0418/results      (Drive, survives the session)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIGURES = RESULTS / "figures"
PRETRAINED = Path(os.environ.get("PRETRAINED_DIR", ROOT / "pretrained"))
