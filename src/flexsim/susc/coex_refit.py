"""Refit coex mu(eps) to a smooth function and rerun the eps whose <m> is off zero.

A smart-sweep susc campaign runs each eps at its own coex fit. Noise in those fits
pushes <m> away from 0 (see the S2A study on the experiments branch). The fix:

  fit      fit mu_coex(eps) from a coex manage.csv to a polynomial (degree 1 = linear)
           over the campaign's eps range; write a manage-style CSV the sweep can read
  select   list the eps where any L has |<m>| >= threshold (skipping eps below a cold cut)
  promote  after the reruns converge: move the old eps folders out of the campaign
           into an archive and move the rerun folders in (prints the plan unless --yes)
  compare  <m> vs eps per L, before vs after, with the +-threshold band

`run/coex_refit_rerun.sh` strings these together with the sweep itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime

import numpy as np

from flexsim.susc.layout import (
    find_susc_run_csvs,
    parse_susc_run_dir,
    read_susceptibility_csv,
)

PHYSICS = ("delta_f", "delta_mu", "k", "scheme")


# ---------------------------------------------------------------------------
# Reading campaigns
# ---------------------------------------------------------------------------

def campaign_runs(base: str) -> dict[tuple[int, float], str]:
    """{(L, eps): run_dir} for every run folder with data under a SUSC_RUNS base.

    eps comes from the data rows, not the folder name: folder names keep only 6
    significant figures (-1.568011 is stored as _-1.56801).
    """
    out = {}
    for path in find_susc_run_csvs(base):
        run_dir = os.path.dirname(path)
        L, eps = parse_susc_run_dir(run_dir)
        rows = read_susceptibility_csv(path)
        if rows and rows[0].get("epsilon", "") != "":
            eps = float(rows[0]["epsilon"])
        if L is not None and eps is not None:
            out[(L, round(eps, 6))] = run_dir
    return out


def latest_generation(run_dir: str) -> list[dict]:
    """Rows of the newest generation (max prod_time), as the jump check uses."""
    rows = read_susceptibility_csv(os.path.join(run_dir, "susceptibility_data.csv"))
    if not rows:
        return []
    pt = max(float(r["prod_time"] or 0) for r in rows)
    return [r for r in rows if abs(float(r["prod_time"] or 0) - pt) < 1.0]


def m_table(base: str) -> list[dict]:
    """One row per (L, eps): replica-mean <m>, its standard error, mu, physics."""
    table = []
    for (L, eps), run_dir in sorted(campaign_runs(base).items()):
        rows = latest_generation(run_dir)
        if not rows:
            continue
        ms = np.array([float(r["m_mean"]) for r in rows])
        se = ms.std(ddof=1) / math.sqrt(len(ms)) if len(ms) > 1 else float("nan")
        r0 = rows[0]
        table.append(dict(L=L, epsilon=eps, m_mean=float(ms.mean()), m_se=float(se), n=len(ms),
                          mu=float(r0["mu"]), prod_time=float(r0["prod_time"]),
                          delta_f=float(r0["delta_f"]), delta_mu=float(r0["delta_mu"]),
                          k=float(r0["k"]), scheme=r0["scheme"], run_dir=run_dir))
    return table


def campaign_physics(table: list[dict]) -> dict:
    phys = {(t["delta_f"], t["delta_mu"], t["k"], t["scheme"]) for t in table}
    if len(phys) != 1:
        raise SystemExit(f"campaign mixes physics parameters: {sorted(phys)}")
    return dict(zip(PHYSICS, next(iter(phys))))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def robust_polyfit(eps, mu, err, degree, max_err, clip=4.0, iters=3):
    """Weighted polynomial fit; drops fits with err > max_err and >clip*MAD outliers."""
    eps, mu, err = map(np.asarray, (eps, mu, err))
    keep = np.isfinite(mu) & np.isfinite(err) & (err <= max_err)
    if keep.sum() <= degree + 1:
        raise SystemExit(f"only {keep.sum()} usable coex points for a degree-{degree} fit")
    x0 = float(np.mean(eps))
    coef = resid = None
    for _ in range(max(1, iters)):
        w = 1.0 / np.maximum(err[keep], 5e-5)
        coef = np.polyfit(eps[keep] - x0, mu[keep], degree, w=w)
        resid = mu - np.polyval(coef, eps - x0)
        mad = 1.4826 * np.median(np.abs(resid[keep]))
        new_keep = keep & (np.abs(resid) <= clip * max(mad, 1e-7))
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return coef, x0, resid, keep


def do_fit(args) -> None:
    camp = m_table(args.campaign)
    if not camp:
        raise SystemExit(f"no runs found under {args.campaign}")
    phys = campaign_physics(camp)
    eps_campaign = sorted({t["epsilon"] for t in camp})
    mu_campaign = {t["epsilon"]: t["mu"] for t in camp}
    lo, hi = eps_campaign[0] - 1e-9, eps_campaign[-1] + 1e-9

    eps, mu, err = [], [], []
    with open(args.coex, newline="") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("mu_coex_FITTED") or "").strip()
            if raw.lower() in ("", "nan"):
                continue
            if any(abs(float(r[p]) - float(phys[p])) > 1e-9 for p in ("delta_f", "delta_mu", "k")
                   if r.get(p, "") != ""):
                continue
            e = round(float(r["epsilon"]), 6)
            if lo <= e <= hi:
                eps.append(e)
                mu.append(float(raw))
                err.append(float(r.get("mu_coex_FITTED_error") or "nan"))
    if not eps:
        raise SystemExit(f"{args.coex} has no fitted mu in [{lo:.4f}, {hi:.4f}] for {phys}")

    coef, x0, resid, keep = robust_polyfit(eps, mu, err, args.degree, args.max_fit_err)
    os.makedirs(args.out, exist_ok=True)

    # The sweep reads mu_coex_FITTED from this file, one row per campaign eps.
    rows = []
    for e in eps_campaign:
        fit = float(np.polyval(coef, e - x0))
        rows.append(dict(epsilon=e, **phys, mu_coex_FITTED=f"{fit:.10f}",
                         mu_campaign=f"{mu_campaign[e]:.10f}",
                         shift=f"{fit - mu_campaign[e]:+.3e}"))
    with open(os.path.join(args.out, "fitted_mu.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out, "coex_points.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epsilon", "mu_coex", "mu_coex_error", "residual", "used_in_fit"])
        for row in zip(eps, mu, err, resid, keep):
            w.writerow([row[0], f"{row[1]:.10f}", f"{row[2]:.3e}", f"{row[3]:+.3e}", bool(row[4])])

    used = resid[keep]
    info = dict(degree=args.degree, coefficients_high_to_low=[float(c) for c in coef],
                x0=x0, form=f"mu = polyval(coefficients, eps - x0)", coex=args.coex,
                campaign=args.campaign, physics=phys, eps_range=[eps_campaign[0], eps_campaign[-1]],
                n_points=len(eps), n_used=int(keep.sum()), max_fit_err=args.max_fit_err,
                rms_residual=float(np.sqrt(np.mean(used ** 2))),
                created=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(args.out, "fit.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    _plot_fit(eps, mu, err, resid, keep, coef, x0, args.degree, os.path.join(args.out, "fit.png"))

    print(f"[fit] degree {args.degree} through {keep.sum()}/{len(eps)} coex points "
          f"({args.coex}), rms residual {info['rms_residual']:.1e}")
    print(f"[fit] wrote {args.out}/fitted_mu.csv, fit.json, fit.png")


def _plot_fit(eps, mu, err, resid, keep, coef, x0, degree, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps, mu, err, resid = map(np.asarray, (eps, mu, err, resid))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    xs = np.linspace(eps.min(), eps.max(), 200)
    a1.plot(xs, np.polyval(coef, xs - x0), color="#eb6834", lw=2, label=f"degree-{degree} fit")
    a1.errorbar(eps[keep], mu[keep], yerr=err[keep], fmt="o", ms=4, color="#2a78d6",
                label="coex fit used")
    a1.plot(eps[~keep], mu[~keep], "x", ms=8, color="#52514e", label="coex fit excluded")
    a1.set_ylabel("βμ_coex")
    a1.legend(frameon=False)
    a2.axhline(0, color="#52514e", lw=1)
    a2.errorbar(eps[keep], resid[keep], yerr=err[keep], fmt="o", ms=4, color="#2a78d6")
    a2.plot(eps[~keep], np.clip(resid[~keep], -3e-3, 3e-3), "x", ms=8, color="#52514e")
    a2.set_ylim(-3e-3, 3e-3)
    a2.set_ylabel("coex μ − fit")
    a2.set_xlabel("βε")
    for a in (a1, a2):
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def do_select(args) -> None:
    table = m_table(args.campaign)
    by_eps: dict[float, list[dict]] = {}
    for t in table:
        by_eps.setdefault(t["epsilon"], []).append(t)

    rerun, skipped = [], []
    for eps, ts in sorted(by_eps.items()):
        worst = max(ts, key=lambda t: abs(t["m_mean"]))
        bad_L = [t["L"] for t in ts if abs(t["m_mean"]) >= args.threshold]
        if not bad_L:
            continue
        row = dict(epsilon=eps, worst_L=worst["L"], worst_m=f"{worst['m_mean']:+.4f}",
                   failing_L=" ".join(map(str, sorted(bad_L))))
        (skipped if args.min_eps is not None and eps < args.min_eps - 1e-9 else rerun).append(row)

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("rerun_eps.csv", rerun), ("skipped_cold_eps.csv", skipped)):
        with open(os.path.join(args.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["epsilon", "worst_L", "worst_m", "failing_L"])
            w.writeheader()
            w.writerows(rows)
    print(f"[select] {len(rerun)} eps to rerun (any L with |<m>| >= {args.threshold}); "
          f"{len(skipped)} cold eps < {args.min_eps} skipped")
    for r in rerun:
        print(f"   rerun  eps={r['epsilon']:+.4f}  worst <m>={r['worst_m']} at L={r['worst_L']}  "
              f"failing L: {r['failing_L']}")
    for r in skipped:
        print(f"   skip   eps={r['epsilon']:+.4f}  worst <m>={r['worst_m']} at L={r['worst_L']}")


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------

def plan_promotion(campaign: str, staging: str, archive: str, eps_list: list[float]):
    """List the (old -> archive, new -> campaign) moves; raise if anything is off."""
    old = campaign_runs(campaign)
    new = campaign_runs(staging)
    moves, problems = [], []
    for eps in eps_list:
        eps = round(eps, 6)
        Ls = sorted(L for (L, e) in new if e == eps)
        if not Ls:
            problems.append(f"eps={eps}: no rerun data in {staging}")
        for L in Ls:
            src = new[(L, eps)]
            dst = os.path.join(campaign, *src.split(os.sep)[-2:])
            arch = os.path.join(archive, *src.split(os.sep)[-2:]) if (L, eps) in old else None
            if (L, eps) in old and os.path.normpath(old[(L, eps)]) != os.path.normpath(dst):
                problems.append(f"L={L} eps={eps}: campaign folder {old[(L, eps)]} != {dst}")
            if arch and os.path.exists(arch):
                problems.append(f"archive target already exists: {arch}")
            if not os.access(os.path.dirname(dst) if os.path.isdir(os.path.dirname(dst)) else campaign,
                             os.W_OK):
                problems.append(f"no write permission in {os.path.dirname(dst)}")
            moves.append(dict(L=L, epsilon=eps, new_src=src, campaign_dst=dst, archive_dst=arch,
                              old_mu=_mu_of(old.get((L, eps))), new_mu=_mu_of(src)))
    return moves, problems


def _mu_of(run_dir):
    if not run_dir:
        return ""
    rows = latest_generation(run_dir)
    return rows[0]["mu"] if rows else ""


def do_promote(args) -> None:
    from flexsim.susc.mag_jumps import compute_jump_summary

    eps_list = [float(r["epsilon"]) for r in csv.DictReader(open(args.eps_file))]
    summary = compute_jump_summary(args.staging, threshold=args.threshold)
    failing = summary[~summary["passes"]] if not summary.empty else summary
    if summary.empty or len(failing):
        msg = "no jump data" if summary.empty else f"{len(failing)} (eps, L) pairs below J >= {args.threshold}"
        if not args.force:
            raise SystemExit(f"[promote] staging not converged ({msg}); wait for the chain or pass --force")
        print(f"[promote] WARNING: {msg}; continuing because of --force")

    moves, problems = plan_promotion(args.campaign, args.staging, args.archive, eps_list)
    if problems:
        raise SystemExit("[promote] refusing:\n  " + "\n  ".join(problems))

    print(f"[promote] {len(moves)} run folders for {len(eps_list)} eps")
    for m in moves:
        print(f"   L={m['L']:<4} eps={m['epsilon']:+.4f}  mu {m['old_mu']} -> {m['new_mu']}")
    print(f"   old folders -> {args.archive}")
    if not args.yes:
        print("[promote] dry run: nothing moved. Re-run with --yes to do it.")
        return

    log_path = os.path.join(args.campaign, "promote_log.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "L", "epsilon", "old_mu", "new_mu",
                                           "archived_to", "promoted_from"])
        if new_log:
            w.writeheader()
        for m in moves:
            if m["archive_dst"]:
                os.makedirs(os.path.dirname(m["archive_dst"]), exist_ok=True)
                shutil.move(m["campaign_dst"], m["archive_dst"])
            os.makedirs(os.path.dirname(m["campaign_dst"]), exist_ok=True)
            shutil.move(m["new_src"], m["campaign_dst"])
            w.writerow(dict(time=datetime.now().isoformat(timespec="seconds"), L=m["L"],
                            epsilon=m["epsilon"], old_mu=m["old_mu"], new_mu=m["new_mu"],
                            archived_to=m["archive_dst"] or "", promoted_from=m["new_src"]))
    print(f"[promote] done; log appended to {log_path}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def do_compare(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_set = None
    if args.eps_file:
        eps_set = {round(float(r["epsilon"]), 6) for r in csv.DictReader(open(args.eps_file))}
    before = {(t["L"], t["epsilon"]): t for t in m_table(args.before)}
    after = {(t["L"], t["epsilon"]): t for t in m_table(args.after)}
    if eps_set is not None:
        before = {k: v for k, v in before.items() if k[1] in eps_set}
        after = {k: v for k, v in after.items() if k[1] in eps_set}
    Ls = sorted({L for L, _ in before} | {L for L, _ in after})
    if not Ls:
        raise SystemExit("[compare] nothing to compare")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for key in sorted(set(before) | set(after), key=lambda k: (k[1], k[0])):
        b, a = before.get(key), after.get(key)
        rows.append(dict(L=key[0], epsilon=key[1],
                         m_before=b and f"{b['m_mean']:+.4f}", se_before=b and f"{b['m_se']:.4f}",
                         mu_before=b and b["mu"],
                         m_after=a and f"{a['m_mean']:+.4f}", se_after=a and f"{a['m_se']:.4f}",
                         mu_after=a and a["mu"],
                         after_within=a and abs(a["m_mean"]) < args.threshold))
    with open(os.path.join(args.out, "compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ncol = 3
    nrow = math.ceil(len(Ls) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), sharex=True, squeeze=False)
    for ax, L in zip(axes.ravel(), Ls):
        for data, color, face, label in ((before, "#8a8984", "white", "before"),
                                         (after, "#2a78d6", "#2a78d6", "after")):
            pts = sorted((k[1], v) for k, v in data.items() if k[0] == L)
            if pts:
                ax.errorbar([e for e, _ in pts], [v["m_mean"] for _, v in pts],
                            yerr=[v["m_se"] for _, v in pts], fmt="o", ms=5, color=color,
                            mfc=face, mec=color, capsize=2, label=label)
        ax.axhspan(-args.threshold, args.threshold, color="#e3efe3", lw=0, zorder=0)
        ax.axhline(0, color="#52514e", lw=1)
        n_after = sum(1 for k in after if k[0] == L)
        n_ok = sum(1 for k, v in after.items() if k[0] == L and abs(v["m_mean"]) < args.threshold)
        ax.set_title(f"L = {L}   after: {n_ok}/{n_after} within ±{args.threshold}",
                     loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("βε")
        ax.set_ylabel("⟨m⟩")
    for ax in axes.ravel()[len(Ls):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"⟨m⟩ before vs after the coex refit (green: |⟨m⟩| < {args.threshold})",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "compare.png"), dpi=150)
    plt.close(fig)
    n_ok = sum(1 for r in rows if r["after_within"])
    n_after = sum(1 for r in rows if r["m_after"] is not None)
    print(f"[compare] after: {n_ok}/{n_after} (eps, L) within ±{args.threshold}; "
          f"wrote {args.out}/compare.png, compare.csv")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fit", help="fit mu_coex(eps) and write fitted_mu.csv")
    p.add_argument("--campaign", required=True, help="SUSC_RUNS base whose eps range and physics to use")
    p.add_argument("--coex", required=True, help="coex manage.csv with mu_coex_FITTED(_error)")
    p.add_argument("--degree", type=int, default=1, help="polynomial degree (1 = linear)")
    p.add_argument("--max-fit-err", type=float, default=1e-3,
                   help="drop coex fits with a larger reported error")
    p.add_argument("--out", required=True)
    p.set_defaults(func=do_fit)

    p = sub.add_parser("select", help="list eps where any L has |<m>| >= threshold")
    p.add_argument("--campaign", required=True)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--min-eps", type=float, default=None,
                   help="skip eps below this (e.g. a cold end that never converges)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=do_select)

    p = sub.add_parser("promote", help="swap converged reruns into the campaign")
    p.add_argument("--campaign", required=True)
    p.add_argument("--staging", required=True)
    p.add_argument("--archive", required=True)
    p.add_argument("--eps-file", required=True, help="rerun_eps.csv from select")
    p.add_argument("--threshold", type=float, default=10.0, help="jump threshold J")
    p.add_argument("--yes", action="store_true", help="actually move folders")
    p.add_argument("--force", action="store_true", help="promote even if not converged")
    p.set_defaults(func=do_promote)

    p = sub.add_parser("compare", help="plot <m> before vs after")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--eps-file", default=None)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--out", required=True)
    p.set_defaults(func=do_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
