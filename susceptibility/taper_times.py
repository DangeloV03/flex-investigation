"""
Distance-dependent eq/prod times for susceptibility runs near a critical point.

At ε = ε_c: 7.5× baseline eq, 2.5× baseline prod (homo_dmu1p0 1× baseline).
Multipliers linearly taper to 1× at the edges of the susceptibility ε grid.

Baseline (run_susceptibility_homo_dmu1p0.sh):
  EQ_TIME=100_000, PROD_TIME=1_000_000, chunk_time=PROD/CHUNKS=100.

Usage:
    python susceptibility/taper_times.py --epsilon -1.76
    python susceptibility/taper_times.py --schedule --eps-min -2.0 --eps-max -1.4 --eps-step 0.005
"""

from __future__ import annotations

import argparse
import sys

from generate_samples import frange

# homo_dmu1p0 1× baseline (run_susceptibility_homo_dmu1p0.sh)
BASE_EQ = 100_000.0
BASE_PROD = 1_000_000.0
CHUNK_TIME = 100.0

EPS_CRIT = -1.76
EQ_MULT_MAX = 7.5
PROD_MULT_MAX = 2.5
EQ_MULT_MIN = 1.0
PROD_MULT_MIN = 1.0

EPS_GRID_MIN = -2.0
EPS_GRID_MAX = -1.4


def distance_fraction(eps: float, *, eps_crit: float = EPS_CRIT) -> float:
    """0 at ε_c, 1 at the farther grid edge (linear in |ε − ε_c|)."""
    d = abs(eps - eps_crit)
    d_max = max(eps_crit - EPS_GRID_MIN, EPS_GRID_MAX - eps_crit)
    return min(d / d_max, 1.0)


def multipliers(
    eps: float,
    *,
    eps_crit: float = EPS_CRIT,
    eq_mult_max: float = EQ_MULT_MAX,
    prod_mult_max: float = PROD_MULT_MAX,
    eq_mult_min: float = EQ_MULT_MIN,
    prod_mult_min: float = PROD_MULT_MIN,
) -> tuple[float, float]:
    t = distance_fraction(eps, eps_crit=eps_crit)
    eq_mult = eq_mult_max + t * (eq_mult_min - eq_mult_max)
    prod_mult = prod_mult_max + t * (prod_mult_min - prod_mult_max)
    return eq_mult, prod_mult


def run_times(
    eps: float,
    *,
    eps_crit: float = EPS_CRIT,
    eq_mult_max: float = EQ_MULT_MAX,
    prod_mult_max: float = PROD_MULT_MAX,
    eq_mult_min: float = EQ_MULT_MIN,
    prod_mult_min: float = PROD_MULT_MIN,
) -> tuple[float, float, int, float, float]:
    """Return (eq_time, prod_time, prod_chunks, eq_mult, prod_mult)."""
    eq_mult, prod_mult = multipliers(
        eps,
        eps_crit=eps_crit,
        eq_mult_max=eq_mult_max,
        prod_mult_max=prod_mult_max,
        eq_mult_min=eq_mult_min,
        prod_mult_min=prod_mult_min,
    )
    eq_time = BASE_EQ * eq_mult
    prod_time = BASE_PROD * prod_mult
    prod_chunks = int(round(prod_time / CHUNK_TIME))
    return eq_time, prod_time, prod_chunks, eq_mult, prod_mult


def main() -> int:
    parser = argparse.ArgumentParser(description="ε-dependent susceptibility run times")
    parser.add_argument("--epsilon", type=float, default=None, help="Single ε (prints eq prod chunks)")
    parser.add_argument("--schedule", action="store_true", help="Print table over ε grid")
    parser.add_argument("--eps-min", type=float, default=EPS_GRID_MIN)
    parser.add_argument("--eps-max", type=float, default=EPS_GRID_MAX)
    parser.add_argument("--eps-step", type=float, default=0.005)
    parser.add_argument("--eps-crit", type=float, default=EPS_CRIT)
    parser.add_argument(
        "--format",
        choices=("shell", "line", "table"),
        default="line",
        help="shell: EQ_TIME=… vars; line: three numbers; table: human schedule",
    )
    args = parser.parse_args()

    if args.schedule or args.epsilon is None:
        eps_values = frange(args.eps_min, args.eps_max, args.eps_step)
        print(
            f"# ε_c={args.eps_crit}  eq: {EQ_MULT_MAX}×→{EQ_MULT_MIN}×  "
            f"prod: {PROD_MULT_MAX}×→{PROD_MULT_MIN}×  grid [{args.eps_min}, {args.eps_max}]"
        )
        print(f"{'epsilon':>10}  {'|d|':>6}  {'eq×':>5}  {'prod×':>6}  {'eq_time':>10}  {'prod_time':>12}  chunks")
        for eps in eps_values:
            eq_t, prod_t, chunks, eq_m, prod_m = run_times(eps, eps_crit=args.eps_crit)
            d = abs(eps - args.eps_crit)
            print(
                f"{eps:10.4f}  {d:6.3f}  {eq_m:5.2f}  {prod_m:6.2f}  "
                f"{eq_t:10.0f}  {prod_t:12.0f}  {chunks}"
            )
        return 0

    eq_t, prod_t, chunks, eq_m, prod_m = run_times(args.epsilon, eps_crit=args.eps_crit)
    if args.format == "shell":
        print(f"EQ_TIME={eq_t:g} PROD_TIME={prod_t:g} PROD_CHUNKS={chunks}")
        print(
            f"# eps={args.epsilon} eps_c={args.eps_crit} "
            f"eq_mult={eq_m:.3f} prod_mult={prod_m:.3f}",
            file=sys.stderr,
        )
    elif args.format == "table":
        print(
            f"eps={args.epsilon}  eq×={eq_m:.3f}  prod×={prod_m:.3f}  "
            f"eq={eq_t:.0f}  prod={prod_t:.0f}  chunks={chunks}"
        )
    else:
        print(f"{eq_t:g} {prod_t:g} {chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
