"""Where the code lives and where run outputs go.

All simulation data, plots, queues and Slurm logs are written under OUTPUT_ROOT
(``<repo>/output`` by default, gitignored). Set ``FLEX_OUTPUT_ROOT`` to put
them somewhere else, e.g. a /scratch directory on Della.

The pipeline modules use paths relative to the current directory
("samples/", "results/", "SUSC_RUNS/", "plots/..."), so the ``run/`` scripts
``chdir`` into OUTPUT_ROOT before doing anything. Relative paths given on the
command line are therefore relative to OUTPUT_ROOT too.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
OUTPUT_ROOT = Path(os.environ.get("FLEX_OUTPUT_ROOT", REPO_ROOT / "output")).resolve()

# Slurm/tmux shell scripts that ship with the package.
COEX_DIR = PACKAGE_DIR / "coex"
SUSC_SLURM_DIR = PACKAGE_DIR / "susc" / "slurm"


def enter_output_root() -> Path:
    """Create OUTPUT_ROOT if needed and make it the current directory."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "slurm_reports").mkdir(exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    return OUTPUT_ROOT
