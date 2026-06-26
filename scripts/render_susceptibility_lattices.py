#!/usr/bin/env python3
"""
Render final_lattice_*.npy from susceptibility production runs as PNG images.

Color map:
  - active (BONDING, 2) -> red
  - inert  (INERT, 1)   -> blue
  - empty  (EMPTY, 0)   -> white

Usage:
    python scripts/render_susceptibility_lattices.py
    python scripts/render_susceptibility_lattices.py --results susceptibility_results
    python scripts/render_susceptibility_lattices.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from susceptibility_paths import PROD_RESULTS_BASE

EMPTY, INERT, BONDING = 0, 1, 2

COLOR = {
    EMPTY: (1.0, 1.0, 1.0),
    INERT: (0.0, 0.0, 1.0),
    BONDING: (1.0, 0.0, 0.0),
}


def lattice_to_rgb(state: np.ndarray) -> np.ndarray:
    """Map uint lattice state (Lx, Ly) to an (Lx, Ly, 3) float RGB image."""
    if state.ndim != 2:
        raise ValueError(f"expected 2D lattice, got shape {state.shape}")

    rgb = np.zeros((*state.shape, 3), dtype=float)
    for value, color in COLOR.items():
        rgb[state == value] = color

    unknown = ~np.isin(state, list(COLOR))
    if unknown.any():
        rgb[unknown] = (0.5, 0.5, 0.5)
    return rgb


def render_lattice_png(npy_path: Path, png_path: Path, dpi: int = 150) -> None:
    state = np.load(npy_path)
    rgb = lattice_to_rgb(state)

    height, width = rgb.shape[:2]
    fig_size = max(2.0, width / 40), max(2.0, height / 40)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.imshow(rgb, interpolation="nearest", origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def find_npy_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("**/final_lattice_*.npy"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render susceptibility final_lattice_*.npy files as PNG images",
    )
    parser.add_argument(
        "--results",
        default=PROD_RESULTS_BASE,
        help="Root directory to search (default: susceptibility_results)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when PNG already exists and is newer than the .npy",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    npy_paths = find_npy_files(results_dir)
    if not npy_paths:
        print(f"No final_lattice_*.npy files under {results_dir}")
        return

    n_written = 0
    n_skipped = 0
    for npy_path in npy_paths:
        png_path = npy_path.with_suffix(".png")
        if args.skip_existing and png_path.is_file():
            if png_path.stat().st_mtime >= npy_path.stat().st_mtime:
                n_skipped += 1
                continue

        if args.dry_run:
            print(f"would render: {npy_path} -> {png_path}")
        else:
            render_lattice_png(npy_path, png_path, dpi=args.dpi)
            print(f"Wrote {png_path}")
        n_written += 1

    verb = "Would render" if args.dry_run else "Rendered"
    print(f"\n{verb} {n_written} file(s), skipped {n_skipped}")


if __name__ == "__main__":
    main()
