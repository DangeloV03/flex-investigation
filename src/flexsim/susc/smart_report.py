"""Per-round jump report for a smart sweep, and detection of rounds that added no data."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def latest_round(results_base: str) -> int:
    """Count existing round PNGs to determine the current round number."""
    existing = glob.glob(os.path.join(results_base, "jump_check_round_*.png"))
    return len(existing) + 1


def plot_jump_figure(
    summary: pd.DataFrame,
    png_path: str,
    *,
    title: str,
    threshold: float = 10.0,
) -> None:
    """Per-L block of three stacked panels vs ε: ⟨J⟩, #replicas with J > 10, max J.

    Red/blue markers follow the ⟨J⟩ pass/fail status in every panel.
    """
    L_vals = sorted(summary["L"].unique()) if not summary.empty else []
    ncols = min(4, len(L_vals)) if L_vals else 1
    nrows = max(1, math.ceil(len(L_vals) / ncols))
    n_panels = 3
    fig, axes = plt.subplots(nrows * n_panels, ncols,
                             figsize=(5 * ncols, 2.6 * n_panels * nrows),
                             squeeze=False)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for ax_idx, L in enumerate(L_vals):
        r0, c = (ax_idx // ncols) * n_panels, ax_idx % ncols
        ax_mean, ax_count, ax_max = (axes[r0 + k][c] for k in range(n_panels))
        sub = summary[summary["L"] == L].sort_values("epsilon")
        eps_arr = sub["epsilon"].to_numpy(float)
        colors = ["tab:red" if not bool(p) else "tab:blue" for p in sub["passes"].tolist()]

        # <J>
        j_arr = sub["J_mean"].to_numpy(float)
        if "J_stderr" in sub.columns:
            err_arr = sub["J_stderr"].fillna(0).to_numpy(float)
        elif "J_std" in sub.columns:
            err_arr = sub["J_std"].fillna(0).to_numpy(float)
        else:
            err_arr = np.zeros_like(j_arr)
        ax_mean.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax_mean.errorbar(eps_arr, j_arr, yerr=err_arr, fmt="none", ecolor="gray",
                         capsize=2, linewidth=0.8, alpha=0.7)
        ax_mean.scatter(eps_arr, j_arr, c=colors, s=28, zorder=3)
        ax_mean.set_title(f"L = {L}", fontsize=9)
        ax_mean.set_ylabel(r"$\langle J \rangle$", fontsize=8)

        # number of replicas with J > 10, against the replica count
        ax_count.plot(eps_arr, sub["n_replicas"].to_numpy(float), color="gray",
                      linestyle="--", linewidth=1.0, alpha=0.7, label="n_replicas")
        ax_count.scatter(eps_arr, sub["n_over_10"].to_numpy(float), c=colors, s=28, zorder=3)
        ax_count.set_ylabel("# replicas J > 10", fontsize=8)
        ax_count.set_ylim(0, float(sub["n_replicas"].max()) * 1.1 + 0.5)
        ax_count.legend(fontsize=6, loc="best")

        # max J over replicas
        ax_max.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax_max.scatter(eps_arr, sub["J_max"].to_numpy(float), c=colors, s=28, zorder=3)
        ax_max.set_ylabel(r"max $J$", fontsize=8)
        ax_max.set_xlabel(r"$\varepsilon$", fontsize=8)

        for ax in (ax_mean, ax_count, ax_max):
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

    # Hide unused axes
    for ax_idx in range(len(L_vals), nrows * ncols):
        r0, c = (ax_idx // ncols) * n_panels, ax_idx % ncols
        for k in range(n_panels):
            axes[r0 + k][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    results_base: str,
    *,
    round_num: int,
    threshold: float = 10.0,
) -> str:
    """Write jump_report.md and a round PNG figure; returns the md path."""
    os.makedirs(results_base, exist_ok=True)
    png_path = os.path.join(results_base, f"jump_check_round_{round_num}.png")
    md_path = os.path.join(results_base, "jump_report.md")

    plot_jump_figure(
        summary, png_path,
        title=f"Jump analysis — round {round_num}  (threshold J ≥ {threshold:.0f})",
        threshold=threshold,
    )

    # ---- markdown ----
    n_total = len(summary) if not summary.empty else 0
    n_pass = int(summary["passes"].sum()) if not summary.empty else 0
    n_fail = n_total - n_pass

    lines = [
        f"# Jump Analysis Report — Round {round_num}",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Results base:** `{results_base}`  ",
        f"**Threshold:** J ≥ {threshold:.0f}  ",
        f"**Status:** {n_pass}/{n_total} pairs passing — "
        + (f"**{n_fail} pairs queued for top-up**" if n_fail > 0 else "✓ All pairs pass"),
        "",
        f"![Round {round_num} figure]({os.path.basename(png_path)})",
        "",
        "## Results table",
        "",
        "| L | ε | ⟨J⟩ | stderr | #J>10 | max J | n_replicas | status |",
        "|---|---|-----|--------|-------|-------|------------|--------|",
    ]

    if not summary.empty:
        for _, row in summary.sort_values(["L", "epsilon"]).iterrows():
            status = "✓" if bool(row["passes"]) else "**RERUN**"
            j_str = f"{row['J_mean']:.2f}"
            err_val = row.get("J_stderr", float("nan"))
            err_f = float(err_val) if err_val is not None else float("nan")
            err_str = f"{err_f:.2f}" if math.isfinite(err_f) else "—"
            lines.append(
                f"| {int(row['L'])} | {row['epsilon']:.4f} | {j_str} | {err_str} "
                f"| {int(row['n_over_10'])} | {int(row['J_max'])} "
                f"| {int(row['n_replicas'])} | {status} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — | no data |")

    lines += [
        "",
        "## Timing",
        "",
        f"Wall time per L and Slurm CPU/time/memory efficiency: "
        f"[timing_round_{round_num}.md](timing_round_{round_num}.md).",
        "",
        "---",
        f"*Generated by flexsim.susc.smart_sweep*",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[check] Round {round_num}: {n_pass}/{n_total} passing → {md_path}", flush=True)
    return md_path


def progress_fingerprint(summary: pd.DataFrame) -> str:
    """Hash of the data the check just saw — identical means no new samples."""
    if summary.empty:
        return "empty"
    cols = ["L", "epsilon", "n_replicas", "J_mean"]
    payload = summary.sort_values(["L", "epsilon"])[cols].round(6).to_csv(index=False)
    return hashlib.sha1(payload.encode()).hexdigest()


def check_stalled(results_base: str, fingerprint: str) -> int:
    """Record this round's fingerprint; return how many rounds in a row matched."""
    path = os.path.join(results_base, ".jump_progress.json")
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    if state.get("fingerprint") == fingerprint:
        stalled = int(state.get("stalled_rounds", 0)) + 1
    else:
        stalled = 0
    with open(path, "w") as f:
        json.dump({"fingerprint": fingerprint, "stalled_rounds": stalled}, f)
    return stalled
