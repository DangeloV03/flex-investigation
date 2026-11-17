"""Write mu_source.csv for the S2A interpolated-mu rerun.

For each bad eps, new mu = linear interpolation between the nearest GOOD S2A eps on each side
(mu read from the original S2A runs). Plus a control eps at its original mu.
Usage (Della, main repo root):  python make_mu_source.py <experiment dir>
"""
import csv
import sys

E = sys.argv[1]
S = "SUSC_RUNS_S2A/_16_16_S1_DF0.0_DMU1.0_K1.0/_{}/susceptibility_data.csv"
PLAN = [(-1.745, -1.755, -1.735), (-1.74, -1.755, -1.735), (-1.685, -1.69, -1.68), (-1.665, -1.67, -1.66)]
CONTROL = -1.735


def mu(e):
    return float(next(csv.DictReader(open(S.format(e))))["mu"])


rows = []
for e, a, b in PLAN:
    ma, mb = mu(a), mu(b)
    new = ma + (e - a) / (b - a) * (mb - ma)
    rows.append(dict(epsilon=e, mu_coex_FITTED=new, mu_original=mu(e),
                     method=f"linear interp between {a} and {b}"))
rows.append(dict(epsilon=CONTROL, mu_coex_FITTED=mu(CONTROL), mu_original=mu(CONTROL),
                 method="control: original mu, new seeds"))
for r in rows:
    r.update(delta_f=0.0, delta_mu=1.0, k=1.0, scheme="homo")
    print(f"eps={r['epsilon']:+.3f}  original={r['mu_original']:.9f}  new={r['mu_coex_FITTED']:.9f}  "
          f"change={r['mu_coex_FITTED'] - r['mu_original']:+.2e}  ({r['method']})")
    r["mu_coex_FITTED"] = f"{r['mu_coex_FITTED']:.10f}"
    r["mu_original"] = f"{r['mu_original']:.10f}"
cols = ["epsilon", "delta_f", "delta_mu", "k", "scheme", "mu_coex_FITTED", "mu_original", "method"]
with open(f"{E}/mu_source.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
