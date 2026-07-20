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

By default LOO jackknife errors (drop one L at a time) are printed and saved to
fss_fit_results.json. Pass --no-errors for point estimates only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_susceptibility import aggregate, aggregate_pooled, L_PLOT_STYLE, resolve_repo_path


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
    verbose: bool = False,
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
    nfev = 0

    def objective(free_vals: np.ndarray) -> float:
        nonlocal nfev
        full = list(x0)
        for k, fi in enumerate(free_idx):
            full[fi] = float(free_vals[k])
        val = fss_quality(full, dataset, x_range)
        nfev += 1
        if verbose and (nfev == 1 or nfev % 25 == 0):
            print(f'  ... optimizer eval {nfev}, S={val:.4f}', flush=True)
        return val

    if verbose:
        print('  Running Nelder-Mead collapse optimizer...', flush=True)

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


def loo_jackknife_fss_errors(
    dataset: dict[float, np.ndarray],
    x0: list[float],
    x_range: tuple[float, float],
    fixed: dict[str, float] | None,
    verbose: bool = False,
) -> dict[str, float]:
    """Leave-one-L-out jackknife stderr on (εc, a, b) after a collapse fit."""
    L_list = sorted(dataset.keys())
    n = len(L_list)
    nan = {'xc_err': float('nan'), 'a_err': float('nan'), 'b_err': float('nan')}
    if n < 3:
        return nan

    partials: list[list[float]] = []
    for i, L_drop in enumerate(L_list):
        sub = {L: dataset[L] for L in L_list if L != L_drop}
        if verbose:
            print(f'  ... LOO jackknife {i + 1}/{n} (drop L={int(L_drop)})', flush=True)
        res = optimise_fss(sub, x0, x_range, fixed, verbose=False)
        partials.append([res['xc'], res['a'], res['b']])

    arr = np.array(partials)
    mean = arr.mean(axis=0)
    err = np.sqrt((n - 1) / n * np.sum((arr - mean) ** 2, axis=0))
    return {'xc_err': float(err[0]), 'a_err': float(err[1]), 'b_err': float(err[2])}


def _fmt_pm(val: float, err: float, prec: int = 4) -> str:
    if not np.isfinite(err):
        return f'{val:.{prec}f}'
    return f'{val:.{prec}f} ± {err:.{prec}f}'


def _fit_record(
    label: str,
    res: dict,
    err: dict[str, float] | None,
    *,
    b_is_gamma_nu: bool = False,
) -> dict:
    """Normalised fit dict for printing and JSON export."""
    gamma_nu = -res['b'] if b_is_gamma_nu else None
    gamma_nu_err = err['b_err'] if (b_is_gamma_nu and err) else None
    beta_nu = res['b'] if not b_is_gamma_nu else None
    beta_nu_err = err['b_err'] if (not b_is_gamma_nu and err) else None
    rec: dict = {
        'observable': label,
        'epsilon_c': res['xc'],
        'epsilon_c_err': err['xc_err'] if err else float('nan'),
        'inv_nu': res['a'],
        'inv_nu_err': err['a_err'] if err else float('nan'),
        'S': res['S'],
        'nfev': res['nfev'],
        'success': res['success'],
    }
    if b_is_gamma_nu:
        rec['gamma_nu'] = gamma_nu
        rec['gamma_nu_err'] = gamma_nu_err
    else:
        rec['beta_nu'] = beta_nu
        rec['beta_nu_err'] = beta_nu_err
    return rec


def _print_fit_record(rec: dict) -> None:
    print(f"  εc     = {_fmt_pm(rec['epsilon_c'], rec['epsilon_c_err'], 6)}")
    print(f"  1/ν    = {_fmt_pm(rec['inv_nu'], rec['inv_nu_err'])}")
    if 'gamma_nu' in rec:
        print(f"  γ/ν    = {_fmt_pm(rec['gamma_nu'], rec['gamma_nu_err'])}")
    else:
        print(f"  β/ν    = {_fmt_pm(rec['beta_nu'], rec['beta_nu_err'])}")
    print(f"  S      = {rec['S']:.4f}  (nfev={rec['nfev']})")


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
    grp3.add_argument('--no-errors', action='store_true',
                      help='Skip LOO jackknife stderr (faster; point estimates only)')

    args = parser.parse_args()
    args.results = resolve_repo_path(args.results)
    args.outdir = resolve_repo_path(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    print('Loading data...', flush=True)
    print(f'  Results: {args.results}', flush=True)
    if args.pooled:
        agg, _ = aggregate_pooled(args.results, verbose=True)
    else:
        agg, _ = aggregate(args.results, verbose=True)
    print(f'  Done loading. L values: {sorted(agg["L"].unique())}', flush=True)
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

    fit_results: dict = {
        'results_dir': args.results,
        'x_range': list(x_range) if np.isfinite(x_range[0]) else None,
        'pooled': args.pooled,
        'peak_shift_chi': args.peak_shift,
    }

    # ------------------------------------------------------------------ χ --
    print('\n=== χ FSS collapse (Fig 11 analog) ===')
    ds_chi = _build_dataset(agg, 'chi_mean', 'chi_stderr')
    if not ds_chi:
        print('  No χ data found — skipping.')
    else:
        x0 = [args.xc, args.nu, -args.gamma_nu]
        print(f'  Initial: εc={x0[0]:.4f}  1/ν={x0[1]:.3f}  b=−γ/ν={x0[2]:.3f}')
        res = optimise_fss(ds_chi, x0, x_range, fixed, verbose=True)
        xc, inv_nu, b_chi = res['xc'], res['a'], res['b']
        gamma_nu = -b_chi
        err = None
        if not args.no_errors:
            print('  LOO jackknife errors (drop one L at a time):')
            err = loo_jackknife_fss_errors(ds_chi, x0, x_range, fixed, verbose=True)
        chi_rec = _fit_record('chi', res, err, b_is_gamma_nu=True)
        fit_results['chi'] = chi_rec
        print('  Best:')
        _print_fit_record(chi_rec)

        x_lbl = (r'$(\varepsilon - \varepsilon^*(L))\,L^{1/\nu}$' if args.peak_shift
                 else r'$(\varepsilon - \varepsilon_c)\,L^{1/\nu}$')
        title = (
            rf'$\chi$ FSS — $\varepsilon_c={_fmt_pm(xc, chi_rec["epsilon_c_err"], 4)}$, '
            rf'$1/\nu={_fmt_pm(inv_nu, chi_rec["inv_nu_err"], 3)}$, '
            rf'$\gamma/\nu={_fmt_pm(gamma_nu, chi_rec["gamma_nu_err"], 3)}$'
        )
        plot_collapse(
            ds_chi, xc, inv_nu, b_chi,
            outpath=os.path.join(args.outdir, 'fss_chi_collapse.png'),
            xlabel=x_lbl,
            ylabel=r'$\chi\,L^{-\gamma/\nu}$',
            title=title,
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
        res = optimise_fss(ds_m, x0, x_range, fixed, verbose=True)
        xc_m, inv_nu_m, beta_nu = res['xc'], res['a'], res['b']
        err = None
        if not args.no_errors:
            print('  LOO jackknife errors (drop one L at a time):')
            err = loo_jackknife_fss_errors(ds_m, x0, x_range, fixed, verbose=True)
        m_rec = _fit_record('abs_m', res, err, b_is_gamma_nu=False)
        fit_results['abs_m'] = m_rec
        print('  Best:')
        _print_fit_record(m_rec)

        title = (
            rf'$|m|$ FSS — $\varepsilon_c={_fmt_pm(xc_m, m_rec["epsilon_c_err"], 4)}$, '
            rf'$1/\nu={_fmt_pm(inv_nu_m, m_rec["inv_nu_err"], 3)}$, '
            rf'$\beta/\nu={_fmt_pm(beta_nu, m_rec["beta_nu_err"], 3)}$'
        )
        plot_collapse(
            ds_m, xc_m, inv_nu_m, beta_nu,
            outpath=os.path.join(args.outdir, 'fss_m_collapse.png'),
            xlabel=r'$(\varepsilon - \varepsilon_c)\,L^{1/\nu}$',
            ylabel=r'$\langle|m|\rangle\,L^{\beta/\nu}$',
            title=title,
            x_range=x_range,
        )

    if fit_results.keys() - {'results_dir', 'x_range', 'pooled', 'peak_shift_chi'}:
        json_path = os.path.join(args.outdir, 'fss_fit_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fit_results, f, indent=2)
            f.write('\n')
        print(f'\nWrote {json_path}')


if __name__ == '__main__':
    main()
