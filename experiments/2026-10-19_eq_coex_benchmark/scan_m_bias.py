"""Scan SUSC_RUNS_* campaigns for <m> != 0 (sign of being off coexistence).

For each (campaign, L, eps), uses the latest generation only (rows at max prod_time),
and computes the replica-mean <m>, its standard error over replicas, and z = <m>/SE.
If mu were at coexistence, ~0.3% of points would have |z| > 3.

Usage (Della repo root):  python scan_m_bias.py [--out m_bias_scan.csv]
"""
import argparse, csv, glob, math

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", default="m_bias_scan.csv")
args = ap.parse_args()

rows_out = []
for camp in sorted(glob.glob(f"{args.root}/SUSC_RUNS_*")):
    for f in glob.glob(f"{camp}/_*/_*/susceptibility_data.csv"):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        pt = max(float(r["prod_time"]) for r in rows)
        rr = [r for r in rows if float(r["prod_time"]) == pt]
        ms = [float(r["m_mean"]) for r in rr]
        n = len(ms); mean = sum(ms) / n
        se = math.sqrt(sum((x - mean) ** 2 for x in ms) / (n - 1) / n) if n > 1 else float("nan")
        rows_out.append(dict(campaign=camp.split("/")[-1], L=int(rr[0]["L"]),
                             epsilon=float(rr[0]["epsilon"]), mu=float(rr[0]["mu"]), n=n,
                             m_mean=mean, m_se=se, z=mean / se if se > 0 else float("nan"),
                             chi=sum(float(r["chi"]) for r in rr) / n))
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
    w.writeheader(); w.writerows(rows_out)

by = {}
for r in rows_out:
    by.setdefault(r["campaign"], []).append(r)
print(f"{'campaign':28s} points  |z|>3")
for c, v in by.items():
    big = sum(1 for r in v if abs(r["z"]) > 3)
    print(f"{c:28s} {len(v):5d}  {big / len(v):5.0%}")
