"""Weighted 4th-order polynomial fit of mu_coex_FITTED vs epsilon.

Usage: python fit_coex.py manage.csv [out.npy]
Saves the coefficients (highest power first, np.polyval order) to out.npy.
Rows with NaN mu or error are skipped; errors are used as 1-sigma weights.
"""
import csv
import sys

import numpy as np

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "mu_coex_fit.npy"

with open(src) as f:
    rows = list(csv.DictReader(f))
x = np.array([float(r["epsilon"]) for r in rows])
y = np.array([float(r["mu_coex_FITTED"]) for r in rows])
dy = np.array([float(r["mu_coex_FITTED_error"]) for r in rows])

ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
x, y, dy = x[ok], y[ok], dy[ok]

# polyfit expects w = 1/sigma; cov="unscaled" treats dy as absolute errors
coef, cov = np.polyfit(x, y, 4, w=1 / dy, cov="unscaled")
np.save(out, coef)

chi2 = np.sum(((y - np.polyval(coef, x)) / dy) ** 2)
print(f"{ok.sum()}/{len(ok)} points used, eps in [{x.min()}, {x.max()}]")
print(f"coef (high->low): {coef}")
print(f"1-sigma errors:   {np.sqrt(np.diag(cov))}")
print(f"chi2/dof = {chi2 / (len(x) - 5):.3g}")
print(f"saved {out}")
