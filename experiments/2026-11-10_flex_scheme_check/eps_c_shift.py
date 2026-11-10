"""Mentor item 2: predicted eps_c for each scheme / dmu (/ k, df) from the single-site eps_eff.

eps_eff = -(h(3) - h(1))/2 at the n=2 coexistence mu; predicted eps_c is where eps_eff hits
-1.7627. Validated on local Ly16/20 BC data (see README): right sign everywhere, magnitude
under-predicted by ~1.5x.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import single_site as ss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default=".")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
rows = []
for scheme in ["homo", "negative_drive", "positive_drive"]:
    for k in [0.1, 1.0, 10.0, 100.0]:
        for df in [-1.0, 0.0, 1.0]:
            for dmu in [-4, -3, -2, -1, 1, 2, 3, 4, 6]:
                rows.append(dict(scheme=scheme, k=k, delta_f=df, delta_mu=dmu,
                                 eps_c_pred=ss.eps_c_pred(df, dmu, k, scheme)))
t = pd.DataFrame(rows)
t["shift_vs_eq"] = t.eps_c_pred - ss.EPS_C_EQ
t.to_csv(os.path.join(a.outdir, "eps_c_predictions.csv"), index=False)
print(t[t.delta_f == 0].pivot_table(index=["scheme", "k"], columns="delta_mu", values="eps_c_pred").round(3).to_string())
print("\nLargest right shifts:")
print(t.sort_values("shift_vs_eq", ascending=False).head(10).round(3).to_string(index=False))
