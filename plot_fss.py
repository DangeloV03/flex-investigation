"""
plot_fss.py

Finite-size scaling (FSS) collapse plots analogous to Figs 11 & 13 of Kumar &
Dasgupta (Phys. Rev. E 102, 052111, 2020).

Quality function and parameter search follow Melchert (arXiv:0910.5403,
autoScale.py), ported to Python 3 with scipy Nelder-Mead in place of the
custom amoeba implementation.

Scaling ansatz (autoScale convention):
    x → (ε − εc) L^a       a  = 1/ν
    y → y · L^b             b  = −γ/ν  for χ   (Fig 11 analog)
                            b  =  β/ν  for |m|  (Fig 13 analog)

Usage examples:
    python plot_fss.py --results susceptibility_results --xc -1.75
    python plot_fss.py --pooled --xc -1.75 --xr -5 5 --peak_shift
    python plot_fss.py --fix_xc --xc -1.75 --fix_nu --nu 1.0
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_susceptibility import aggregate, aggregate_pooled, L_PLOT_STYLE


# ---------------------------------------------------------------------------
# Core FSS quality function — Python-3 port of Melchert autoScale.py
# ---------------------------------------------------------------------------

def _lls_fit(piv_x: float, subset: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    Weighted linear least-squares fit y = A + B·x through subset points.
    Returns (Y, dY²) — estimated master-curve value and its squared error at piv_x.
    """
    K = Kx = Ky = Kxx = Kxy = 0.0
    for x, y, dy in subset:
        w = 1.0 / (dy * dy)
        K += w
        Kx += x * w
        Ky += y * w
        Kxx += x * x * w
        Kxy += x * y * w
    fac = K * Kxx - Kx * Kx
    if abs(fac) < 1e-15:
        return (Ky / K if K > 0 else 0.0), 0.0
    A = (Ky * Kxx - Kx * Kxy) / fac
    B = (K * Kxy - Kx * Ky) / fac
    Y = A + B * piv_x
    dY2 = abs((Kxx - 2.0 * piv_x * Kx + piv_x ** 2 * K) / fac)
    return Y, dY2


def fss_quality(
    scale_par: list[float],
    dataset: dict[float, np.ndarray],
    x_range: tuple[float, float] = (-np.inf, np.inf),
) -> float:
    """
    Data-collapse quality S for scaling parameters [xc, a, b].

    For each scaled point (L, x_s, y_s, dy_s):
      - find bracketing points from every other L on the rescaled axis
      - linear-interpolate a master-curve estimate Y ± dY at x_s
      - accumulate chi² = (y_s − Y)² / (dy_s² + dY²)

    S = mean(chi²) over all valid points;  smaller S = better collapse.

    dataset : {L: ndarray shape (n, 3)}, columns [ε, y, dy]
    """
    xc, a, b = scale_par
    L_list = list(dataset.keys())

    # Scale all datasets once
    scaled: dict[float, np.ndarray] = {}
    for L, raw in dataset.items():
        xs = (raw[:, 0] - xc) * (L ** a)
        ys = raw[:, 1] * (L ** b)
        dys = np.abs(raw[:, 2] * (L ** b))
        scaled[L] = np.column_stack([xs, ys, dys])

    chi2_list: list[float] = []

    for L_piv in L_list:
        for xs_piv, ys_piv, dys_piv in scaled[L_piv]:
            if not (x_range[0] <= xs_piv <= x_range[1]):
                continue
            if dys_piv <= 0:
                continue

            # Collect one bracketing pair per other L value
            subset: list[tuple[float, float, float]] = []
            for L_other in L_list:
                if L_other == L_piv:
                    continue
                pts = scaled[L_other]
                left_mask = pts[:, 0] <= xs_piv
                right_mask = pts[:, 0] > xs_piv
                if left_mask.any() and right_mask.any():
                    left = pts[left_mask][np.argmax(pts[left_mask, 0])]
                    right = pts[right_mask][np.argmin(pts[right_mask, 0])]
                    if left[2] > 0 and right[2] > 0:
                        subset.extend([tuple(left), tuple(right)])

            if len(subset) < 2:
                continue

            Y, dY2 = _lls_fit(xs_piv, subset)
            chi2_list.append((ys_piv - Y) ** 2 / (dys_piv ** 2 + dY2))

    return float(np.mean(chi2_list)) if chi2_list else 1e9


def optimise_fss(
    dataset: dict[float, np.ndarray],
    x0: list[float],
    x_range: tuple[float, float] = (-np.inf, np.inf),
    fixed: dict[str, float] | None = None,
) -> dict:
    """
    Minimise fss_quality via Nelder-Mead.

    x0     : [xc, a, b] initial guess
    fixed  : e.g. {'xc': -1.75, 'a': 1.0} — parameters held constant
    Returns dict with keys xc, a, b, S, nfev, success.
    """
    names = ['xc', 'a', 'b']
    fixed = fixed or {}

    x0 = list(x0)
    for name, val in fixed.items():
        x0[names.index(name)] = val

    free_idx = [i for i, n in enumerate(names) if n not in fixed]
    free_x0 = [x0[i] for i in free_idx]

    def objective(free_vals: np.ndarray) -> float:
        full = list(x0)
        for k, fi in enumerate(free_idx):
            full[fi] = float(free_vals[k])
        return fss_quality(full, dataset, x_range)

    res = minimize(
        objective, free_x0, method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000, 'adaptive': True},
    )

    full_best = list(x0)
    for k, fi in enumerate(free_idx):
        full_best[fi] = float(res.x[k])

    return {
        'xc': full_best[0], 'a': full_best[1], 'b': full_best[2],
        'S': float(res.fun), 'nfev': res.nfev, 'success': res.success,
    }


# ---------------------------------------------------------------------------
# Collapse plot
# ---------------------------------------------------------------------------

def plot_collapse(
    dataset: dict[float, np.ndarray],
    xc: float,
    a: float,
    b: float,
    outpath: str,
    xlabel: str,
    ylabel: str,
    title: str,
    x_range: tuple[float, float] = (-np.inf, np.inf),
    use_peak_shift: bool = False,
) -> None:
    """
    Draw FSS collapse.  x → (ε − shift) L^a,  y → y L^b.

    use_peak_shift : if True, shift each L by its own peak ε*(L) rather than εc
                     (reproduces the T*(L) convention used in Fig 11 of the paper).
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for L in sorted(dataset.keys()):
        raw = dataset[L]
        L_int = int(L)
        style = L_PLOT_STYLE.get(L_int, {'color': 'gray', 'marker': 'o'})
        color = style['color']

        shift = raw[np.argmax(raw[:, 1]), 0] if use_peak_shift else xc

        xs = (raw[:, 0] - shift) * (L ** a)
        ys = raw[:, 1] * (L ** b)
        dys = np.abs(raw[:, 2] * (L ** b))
        mask = (xs >= x_range[0]) & (xs <= x_range[1]) & np.isfinite(ys)

        ax.errorbar(
            xs[mask], ys[mask], yerr=dys[mask],
            fmt=f"{style['marker']}-",
            color=color,
            markerfacecolor='none',
            markeredgecolor=color,
            markeredgewidth=1.2,
            capsize=3,
            label=f'L = {L_int}',
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_dataset(
    agg: pd.DataFrame,
    y_col: str,
    yerr_col: str,
    min_error_floor: float = 1e-8,
) -> dict[float, np.ndarray]:
    """
    Return {L: array(n, 3)} with columns [epsilon, y, dy].
    Drops NaN/non-finite rows; applies a small error floor so dy > 0 always.
    """
    dataset: dict[float, np.ndarray] = {}
    for L, sub in agg.groupby('L'):
        sub = sub.sort_values('epsilon')
        x = sub['epsilon'].to_numpy(float)
        y = sub[y_col].to_numpy(float)
        dy = sub[yerr_col].to_numpy(float)
        dy = np.maximum(dy, min_error_floor)
        mask = np.isfinite(y) & np.isfinite(dy)
        if mask.sum() < 3:
            continue
        dataset[float(L)] = np.column_stack([x[mask], y[mask], dy[mask]])
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='FSS collapse plots (Figs 11 & 13 analog)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results', default='susceptibility_results',
                        help='Directory containing susceptibility_data.csv files')
    parser.add_argument('--outdir', default='plots/fss')
    parser.add_argument('--pooled', action='store_true',
                        help='Pool replicas before computing χ/|m| (vs per-trajectory average)')

    grp = parser.add_argument_group('initial guesses')
    grp.add_argument('--xc', type=float, default=None,
                     help='εc initial guess (auto-detected from χ peak at largest L if omitted)')
    grp.add_argument('--nu', type=float, default=1.0,
                     help='1/ν initial guess (x-exponent a)')
    grp.add_argument('--gamma_nu', type=float, default=1.75,
                     help='γ/ν initial guess (χ y-exponent magnitude)')
    grp.add_argument('--beta_nu', type=float, default=0.125,
                     help='β/ν initial guess (|m| y-exponent)')

    grp2 = parser.add_argument_group('optimisation control')
    grp2.add_argument('--xr', nargs=2, type=float, metavar=('XMIN', 'XMAX'),
                      default=None,
                      help='Restrict quality function to rescaled x in [XMIN, XMAX]')
    grp2.add_argument('--fix_xc', action='store_true', help='Hold εc fixed during optimisation')
    grp2.add_argument('--fix_nu', action='store_true', help='Hold 1/ν fixed during optimisation')

    grp3 = parser.add_argument_group('visualisation')
    grp3.add_argument('--peak_shift', action='store_true',
                      help='For χ: shift each L by its own ε*(L) instead of εc (Fig-11 style)')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('Loading data...')
    agg = aggregate_pooled(args.results) if args.pooled else aggregate(args.results)
    print(f'  L values: {sorted(agg["L"].unique())}')
    print(f'  ε range:  [{agg["epsilon"].min():.4f}, {agg["epsilon"].max():.4f}]')

    # Auto-detect εc from χ peak at the largest L available
    if args.xc is None:
        L_max = agg['L'].max()
        row = agg.loc[agg['L'] == L_max].sort_values('chi_mean').iloc[-1]
        args.xc = float(row['epsilon'])
        print(f'  Auto εc ≈ {args.xc:.4f}  (χ peak at L={L_max})')

    x_range: tuple[float, float] = tuple(args.xr) if args.xr else (-np.inf, np.inf)
    fixed: dict[str, float] = {}
    if args.fix_xc:
        fixed['xc'] = args.xc
    if args.fix_nu:
        fixed['a'] = args.nu

    # ------------------------------------------------------------------ χ --
    print('\n=== χ FSS collapse (Fig 11 analog) ===')
    ds_chi = _build_dataset(agg, 'chi_mean', 'chi_stderr')
    if not ds_chi:
        print('  No χ data found — skipping.')
    else:
        x0 = [args.xc, args.nu, -args.gamma_nu]
        print(f'  Initial: εc={x0[0]:.4f}  1/ν={x0[1]:.3f}  b=−γ/ν={x0[2]:.3f}')
        res = optimise_fss(ds_chi, x0, x_range, fixed)
        xc, inv_nu, b_chi = res['xc'], res['a'], res['b']
        gamma_nu = -b_chi
        print(f'  Best:    εc={xc:.6f}  1/ν={inv_nu:.4f}  γ/ν={gamma_nu:.4f}'
              f'  S={res["S"]:.4f}  (nfev={res["nfev"]})')

        x_lbl = (r'$(\varepsilon - \varepsilon^*(L))\,L^{1/\nu}$' if args.peak_shift
                 else r'$(\varepsilon - \varepsilon_c)\,L^{1/\nu}$')
        plot_collapse(
            ds_chi, xc, inv_nu, b_chi,
            outpath=os.path.join(args.outdir, 'fss_chi_collapse.png'),
            xlabel=x_lbl,
            ylabel=r'$\chi\,L^{-\gamma/\nu}$',
            title=(rf'$\chi$ FSS — '
                   rf'$\varepsilon_c={xc:.4f}$,  $1/\nu={inv_nu:.3f}$,  $\gamma/\nu={gamma_nu:.3f}$'),
            x_range=x_range,
            use_peak_shift=args.peak_shift,
        )

    # --------------------------------------------------------------- |m| --
    print('\n=== |m| FSS collapse (Fig 13 analog) ===')
    ds_m = _build_dataset(agg, 'abs_m_mean', 'abs_m_mean_stderr')
    if not ds_m:
        print('  No |m| data found — skipping.')
    else:
        x0 = [args.xc, args.nu, args.beta_nu]
        print(f'  Initial: εc={x0[0]:.4f}  1/ν={x0[1]:.3f}  b=β/ν={x0[2]:.3f}')
        res = optimise_fss(ds_m, x0, x_range, fixed)
        xc_m, inv_nu_m, beta_nu = res['xc'], res['a'], res['b']
        print(f'  Best:    εc={xc_m:.6f}  1/ν={inv_nu_m:.4f}  β/ν={beta_nu:.4f}'
              f'  S={res["S"]:.4f}  (nfev={res["nfev"]})')

        plot_collapse(
            ds_m, xc_m, inv_nu_m, beta_nu,
            outpath=os.path.join(args.outdir, 'fss_m_collapse.png'),
            xlabel=r'$(\varepsilon - \varepsilon_c)\,L^{1/\nu}$',
            ylabel=r'$\langle|m|\rangle\,L^{\beta/\nu}$',
            title=(rf'$|m|$ FSS — '
                   rf'$\varepsilon_c={xc_m:.4f}$,  $1/\nu={inv_nu_m:.3f}$,  $\beta/\nu={beta_nu:.3f}$'),
            x_range=x_range,
        )


if __name__ == '__main__':
    main()
