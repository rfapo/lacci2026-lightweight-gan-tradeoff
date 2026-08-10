"""Root paths for every script in this repository.

Nothing here is hardcoded to the machine the experiments were run on. Point the
four environment variables below at your own copies, or accept the defaults,
which assume the layout described in the README:

    BLOODMNIST_ROOT   real BloodMNIST PNGs, one directory per class:
                        $BLOODMNIST_ROOT/bloodmnist_<class>_64/raw/*.png
    RUNS_ROOT         training outputs (checkpoints and generated images)
    OUT_ROOT          where analysis results are written
    FASTGAN_ROOT      a checkout of FastGAN-pytorch with the patch of patch/ applied

Example:

    export BLOODMNIST_ROOT=~/data
    export RUNS_ROOT=~/runs
    python analysis/kid_and_resampling.py --verify
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _root(var: str, default: str) -> Path:
    return Path(os.environ.get(var, str(_HERE / default))).expanduser()


DATA_ROOT = _root("BLOODMNIST_ROOT", "data")
RUNS_ROOT = _root("RUNS_ROOT", "runs")
OUT_ROOT = _root("OUT_ROOT", "outputs")
FASTGAN_ROOT = _root("FASTGAN_ROOT", "FastGAN-pytorch")

CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

#: Fixed across every script so that the three systems are scored on the same
#: protocol. Changing it invalidates comparability with the published numbers.
PROTOCOL_SEED = 42
N_SAMPLE = 200


def real_dir(cls: str) -> Path:
    """Directory holding the real PNGs of one BloodMNIST class."""
    return DATA_ROOT / f"bloodmnist_{cls}_64" / "raw"
