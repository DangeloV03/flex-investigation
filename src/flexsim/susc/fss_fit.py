"""Finite-size-scaling collapse fit (Python 3 port of Melchert's autoScale.py) and its errors."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


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


def _root_bisection(func, x_min: float, x_max: float, epsilon: float = 1e-5) -> float:
    """Bisection root finder (Melchert autoScale rootBisection)."""
    f_min = func(x_min)
    f_max = func(x_max)
    while abs(x_max - x_min) > epsilon:
        x_mid = 0.5 * (x_min + x_max)
        f_mid = func(x_mid)
        if f_min * f_mid <= 0.0:
            x_max, f_max = x_mid, f_mid
        else:
            x_min, f_min = x_mid, f_mid
    return 0.5 * (x_min + x_max)


def _get_brackets(func, mid_val: float, fac: float = 0.01, max_iter: int = 200) -> tuple[list[float], list[float]]:
    """Bracket roots of func on either side of mid_val (Melchert getBrackets)."""
    mid_f = func(mid_val)
    l_val = mid_val * (1.0 - fac)
    l_f = func(l_val)
    n = 0
    while l_f * mid_f > 0.0 and n < max_iter:
        l_val *= 1.0 - fac
        mid_f = l_f
        l_f = func(l_val)
        n += 1
    l_brack = [l_val, mid_val]

    mid_f = func(mid_val)
    r_val = mid_val * (1.0 + fac)
    r_f = func(r_val)
    n = 0
    while r_f * mid_f > 0.0 and n < max_iter:
        r_val *= 1.0 + fac
        mid_f = r_f
        r_f = func(r_val)
        n += 1
    r_brack = [mid_val, r_val]
    return l_brack, r_brack


def s_plus_one_fss_errors(
    dataset: dict[float, np.ndarray],
    best: dict,
    x_range: tuple[float, float],
    fixed: dict[str, float] | None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Melchert S+1 error analysis (autoScale -getErrors).

    For each optimised scaling parameter, bracket where S increases to S_best+1
    while holding the other parameters fixed. Returns asymmetric +/- errors.
    """
    names = ['xc', 'a', 'b']
    fixed = fixed or {}
    scale_par = [best['xc'], best['a'], best['b']]
    best_s = best['S']
    nan = {f'{n}_err_lo': float('nan') for n in names} | {f'{n}_err_hi': float('nan') for n in names}

    out: dict[str, float] = {}
    for par_id, name in enumerate(names):
        if name in fixed:
            out[f'{name}_err_lo'] = 0.0
            out[f'{name}_err_hi'] = 0.0
            continue

        piv = scale_par[par_id]

        def objective(val: float, par_id: int = par_id) -> float:
            trial = list(scale_par)
            trial[par_id] = val
            return fss_quality(trial, dataset, x_range) - (best_s + 1.0)

        try:
            l_brack, r_brack = _get_brackets(objective, piv)
            err_lo = _root_bisection(objective, l_brack[0], l_brack[1])
            err_hi = _root_bisection(objective, r_brack[0], r_brack[1])
            out[f'{name}_err_lo'] = abs(piv - err_lo)
            out[f'{name}_err_hi'] = abs(err_hi - piv)
            if verbose:
                print(
                    f'  S+1 {name}:  -{out[f"{name}_err_lo"]:.6g}  '
                    f'+{out[f"{name}_err_hi"]:.6g}',
                    flush=True,
                )
        except Exception:
            out[f'{name}_err_lo'] = float('nan')
            out[f'{name}_err_hi'] = float('nan')

    return out if out else nan


def fmt_pm(val: float, err_lo: float, err_hi: float, prec: int = 4) -> str:
    if not (np.isfinite(err_lo) and np.isfinite(err_hi)):
        return f'{val:.{prec}f}'
    if err_lo == 0.0 and err_hi == 0.0:
        return f'{val:.{prec}f}'
    if abs(err_lo - err_hi) <= max(10 ** (-prec), 1e-12 * abs(val)):
        return f'{val:.{prec}f} ± {err_hi:.{prec}f}'
    return f'{val:.{prec}f} +{err_hi:.{prec}f}/-{err_lo:.{prec}f}'


def fit_record(
    label: str,
    res: dict,
    err: dict[str, float] | None,
    *,
    b_is_gamma_nu: bool = False,
) -> dict:
    """Normalised fit dict for printing and JSON export."""
    def _errs(prefix: str) -> tuple[float, float]:
        if not err:
            return float('nan'), float('nan')
        return err[f'{prefix}_err_lo'], err[f'{prefix}_err_hi']

    xc_lo, xc_hi = _errs('xc')
    a_lo, a_hi = _errs('a')
    b_lo, b_hi = _errs('b')
    rec: dict = {
        'observable': label,
        'epsilon_c': res['xc'],
        'epsilon_c_err_lo': xc_lo,
        'epsilon_c_err_hi': xc_hi,
        'inv_nu': res['a'],
        'inv_nu_err_lo': a_lo,
        'inv_nu_err_hi': a_hi,
        'S': res['S'],
        'nfev': res['nfev'],
        'success': res['success'],
        'error_method': 'melchert_s_plus_one',
    }
    if b_is_gamma_nu:
        rec['gamma_nu'] = -res['b']
        rec['gamma_nu_err_lo'] = b_lo
        rec['gamma_nu_err_hi'] = b_hi
    else:
        rec['beta_nu'] = res['b']
        rec['beta_nu_err_lo'] = b_lo
        rec['beta_nu_err_hi'] = b_hi
    return rec


def print_fit_record(rec: dict) -> None:
    print(
        f"  εc     = {fmt_pm(rec['epsilon_c'], rec['epsilon_c_err_lo'], rec['epsilon_c_err_hi'], 6)}"
    )
    print(
        f"  1/ν    = {fmt_pm(rec['inv_nu'], rec['inv_nu_err_lo'], rec['inv_nu_err_hi'])}"
    )
    if 'gamma_nu' in rec:
        print(
            f"  γ/ν    = {fmt_pm(rec['gamma_nu'], rec['gamma_nu_err_lo'], rec['gamma_nu_err_hi'])}"
        )
    else:
        print(
            f"  β/ν    = {fmt_pm(rec['beta_nu'], rec['beta_nu_err_lo'], rec['beta_nu_err_hi'])}"
        )
    print(f"  S      = {rec['S']:.4f}  (nfev={rec['nfev']})")
