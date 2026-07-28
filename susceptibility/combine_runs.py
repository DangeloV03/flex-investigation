"""
combine_runs.py

Combine two (or more) susceptibility results trees into ONE results dir so the
existing analysis scripts pool their replicas together.

Why this works
--------------
find_susceptibility_csvs() globs exactly one level deep:
    <results_dir>/susceptibility_*/susceptibility_data.csv
and every loader (analyze_chi_max_scaling.load_replica_groups,
plot_susceptibility.aggregate / aggregate_pooled) pools replicas by the
(L, epsilon) key read from each CSV -- NOT by directory name -- while looking up
each replica's m_timeseries_{id}.csv *relative to that run dir's own path*.

So to merge two runs we only need both dates' `susceptibility_*` run dirs to sit
directly under the output dir. The two dates have identically named run dirs, so
we de-collide by suffixing each source with a short tag. No id re-tagging is
needed: replica timeseries stay next to their own susceptibility_data.csv, and
the (L, eps) pooling picks up both.

By default we symlink run dirs (no data duplication); use --copy to copy.

Usage
-----
    python combine_runs.py --out 2-5-combined \
        susceptibility/results/exact_2026-07-02 \
        susceptibility/results/exact_2026-07-05

    # then, pointing every analysis at the combined tree:
    python analyze_chi_max_scaling.py --results 2-5-combined \
        --outdir 2-5-combined/plots/chi_max_scaling --loo-only
    python plot_susceptibility.py --results 2-5-combined \
        --outdir 2-5-combined/plots/susceptibility
    python plot_fss.py --results 2-5-combined \
        --outdir 2-5-combined/plots/fss --xc -1.75
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from collections import defaultdict

SUSCEPTIBILITY_DATA_CSV = "susceptibility_data.csv"


def source_tag(path: str) -> str:
    """Short, filesystem-safe tag from a source dir name (e.g. exact_2026-07-02 -> 0702)."""
    base = os.path.basename(os.path.normpath(path))
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", base)
    if m:
        return m.group(2) + m.group(3)           # MMDD
    return re.sub(r"[^0-9A-Za-z]+", "", base)[-8:] or "src"


def parse_L_eps(dirname: str) -> tuple[int | None, float | None]:
    m_L = re.search(r"susceptibility_(\d+)x\d+", dirname)
    L = int(m_L.group(1)) if m_L else None
    m_e = re.search(r"epsilon(m?)(\d+)(?:p(\d+))?", dirname)
    if m_e:
        sign = -1 if m_e.group(1) == "m" else 1
        eps = sign * float(f"{m_e.group(2)}.{m_e.group(3) or '0'}")
    else:
        eps = None
    return L, eps


def count_replicas(run_dir: str) -> int:
    """Data rows in this run dir's CSV (lines - 1), no CSV parsing."""
    csv_path = os.path.join(run_dir, SUSCEPTIBILITY_DATA_CSV)
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine susceptibility result trees for pooled analysis")
    ap.add_argument("--out", required=True, help="Output combined results dir (e.g. 2-5-combined)")
    ap.add_argument("sources", nargs="+", help="Two or more source results dirs to merge")
    ap.add_argument("--copy", action="store_true", help="Copy run dirs instead of symlinking")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing --out dir")
    args = ap.parse_args()

    if args.force and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    out_abs = os.path.abspath(args.out)

    # (L, eps) -> {tag: replicas} for the verification table
    combined: dict[tuple, dict[str, int]] = defaultdict(dict)
    linked_per_source: dict[str, int] = {}

    for src in args.sources:
        if not os.path.isdir(src):
            raise SystemExit(f"Source not found: {src}")
        tag = source_tag(src)
        run_dirs = sorted(
            d for d in glob.glob(os.path.join(src, "susceptibility_*"))
            if os.path.isfile(os.path.join(d, SUSCEPTIBILITY_DATA_CSV))
        )
        if not run_dirs:
            raise SystemExit(f"No susceptibility_*/{SUSCEPTIBILITY_DATA_CSV} under {src}")

        n_linked = 0
        for run_dir in run_dirs:
            base = os.path.basename(os.path.normpath(run_dir))
            dest = os.path.join(out_abs, f"{base}__{tag}")
            if os.path.lexists(dest):
                if not args.force:
                    raise SystemExit(f"Destination exists: {dest} (use --force)")
                if os.path.islink(dest) or os.path.isfile(dest):
                    os.remove(dest)
                else:
                    shutil.rmtree(dest)
            if args.copy:
                shutil.copytree(run_dir, dest)
            else:
                os.symlink(os.path.abspath(run_dir), dest)
            n_linked += 1

            L, eps = parse_L_eps(base)
            if L is not None and eps is not None:
                combined[(L, round(eps, 4))][tag] = count_replicas(run_dir)

        linked_per_source[f"{src}  (tag={tag})"] = n_linked

    # ---- Report ----
    print(f"\nCombined into: {out_abs}  ({'copies' if args.copy else 'symlinks'})")
    for s, n in linked_per_source.items():
        print(f"  {n:>4} run dirs from {s}")

    tags = [source_tag(s) for s in args.sources]
    Ls = sorted({L for (L, _) in combined})
    print(f"\n(L, eps) coverage — replicas per source [{' + '.join(tags)}] = total")
    hdr = f"{'L':>5}  {'eps':>9}  " + "  ".join(f"{t:>7}" for t in tags) + f"  {'total':>7}"
    print(hdr)
    print("-" * len(hdr))
    only_one = 0
    for L in Ls:
        for (l, eps) in sorted(k for k in combined if k[0] == L):
            per = combined[(l, eps)]
            cells = "  ".join(f"{per.get(t, 0):>7}" for t in tags)
            total = sum(per.values())
            flag = "  <-- only one source" if sum(1 for t in tags if per.get(t, 0)) < len(tags) else ""
            if flag:
                only_one += 1
            print(f"{L:>5}  {eps:>9.4f}  {cells}  {total:>7}{flag}")
    if only_one:
        print(f"\nNote: {only_one} (L, eps) points have data from only one run.")
    print(f"\nNow run analyses with  --results {args.out}")


if __name__ == "__main__":
    main()
