"""
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
    python run/plots.py fss --results susceptibility/results --xc -1.75
    python run/plots.py fss --pooled --xc -1.75 --xr -5 5 --peak_shift
    python run/plots.py fss --fix_xc --xc -1.75 --fix_nu --nu 1.0

By default Melchert S+1 errors are computed (autoScale -getErrors): for each
optimised parameter, find where the collapse quality S increases to S+1 while
holding the other parameters fixed. Pass --no-errors for point estimates only.

The collapse fit itself is in susc/fss_fit.py.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flexsim.susc.aggregate import aggregate, aggregate_pooled
from flexsim.susc.figures import L_PLOT_STYLE
from flexsim.susc.fss_fit import (
    fit_record,
    fmt_pm,
    print_fit_record,
    optimise_fss,
    s_plus_one_fss_errors,
)


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
    parser.add_argument('--results', default='susceptibility/results',
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
                      help='Skip Melchert S+1 parameter errors (point estimates only)')

    args = parser.parse_args()
    args.results = os.path.abspath(args.results)
    args.outdir = os.path.abspath(args.outdir)
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
            print('  Melchert S+1 errors (autoScale -getErrors):')
            err = s_plus_one_fss_errors(ds_chi, res, x_range, fixed, verbose=True)
        chi_rec = fit_record('chi', res, err, b_is_gamma_nu=True)
        fit_results['chi'] = chi_rec
        print('  Best:')
        print_fit_record(chi_rec)

        x_lbl = (r'$(\varepsilon - \varepsilon^*(L))\,L^{1/\nu}$' if args.peak_shift
                 else r'$(\varepsilon - \varepsilon_c)\,L^{1/\nu}$')
        title = (
            rf'$\chi$ FSS — $\varepsilon_c={fmt_pm(xc, chi_rec["epsilon_c_err_lo"], chi_rec["epsilon_c_err_hi"], 4)}$, '
            rf'$1/\nu={fmt_pm(inv_nu, chi_rec["inv_nu_err_lo"], chi_rec["inv_nu_err_hi"], 3)}$, '
            rf'$\gamma/\nu={fmt_pm(gamma_nu, chi_rec["gamma_nu_err_lo"], chi_rec["gamma_nu_err_hi"], 3)}$'
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
            print('  Melchert S+1 errors (autoScale -getErrors):')
            err = s_plus_one_fss_errors(ds_m, res, x_range, fixed, verbose=True)
        m_rec = fit_record('abs_m', res, err, b_is_gamma_nu=False)
        fit_results['abs_m'] = m_rec
        print('  Best:')
        print_fit_record(m_rec)

        title = (
            rf'$|m|$ FSS — $\varepsilon_c={fmt_pm(xc_m, m_rec["epsilon_c_err_lo"], m_rec["epsilon_c_err_hi"], 4)}$, '
            rf'$1/\nu={fmt_pm(inv_nu_m, m_rec["inv_nu_err_lo"], m_rec["inv_nu_err_hi"], 3)}$, '
            rf'$\beta/\nu={fmt_pm(beta_nu, m_rec["beta_nu_err_lo"], m_rec["beta_nu_err_hi"], 3)}$'
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
