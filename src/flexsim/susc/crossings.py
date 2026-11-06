"""Count barrier crossings -m0 -> +m0 in every m_timeseries_<i>.csv of a folder.

    python src/flexsim/susc/crossings.py outdir [m0]

A time series is in the lower basin when m <= -m0 and in the upper basin when
m >= +m0; points with |m| < m0 are on the barrier and leave the basin label
unchanged. A jump J is counted every time the label switches from lower to upper
(upward crossings only; the reverse direction is not counted). Prints J for every
replica and J_min, J_max, <J> over the folder, and writes them to
outdir/JUMPS.txt (header row, value row, and a third row "OK" if <J> > 10).
m0 defaults to 0.5.
"""

import glob
import os
import sys

import numpy as np

M0 = 0.5  # default; override with the optional second argument


def count_jumps(m, m0):
    jumps, basin = 0, 0  # basin: -1 lower, +1 upper, 0 not yet in either
    for x in m:
        if x <= -m0:
            basin = -1
        elif x >= m0:
            if basin == -1:
                jumps += 1
            basin = 1
    return jumps


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: python {sys.argv[0]} outdir [m0]")
    outdir = sys.argv[1]
    m0 = float(sys.argv[2]) if len(sys.argv) == 3 else M0
    files = glob.glob(os.path.join(outdir, "m_timeseries_*.csv"))
    if not files:
        sys.exit(f"no m_timeseries_*.csv in {outdir}")
    files.sort(key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))

    J = []
    for f in files:
        m = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True)["m"])
        J.append(count_jumps(m, m0))
        print(f"{os.path.basename(f)}: J = {J[-1]}")
    print(f"m0 = {m0}, replicas = {len(J)}: "
          f"J_min = {min(J)}, J_max = {max(J)}, <J> = {np.mean(J):.3f}")

    dst = os.path.join(outdir, "JUMPS.txt")
    with open(dst, "w") as f:
        f.write("J_min J_max J_avg\n")
        f.write(f"{min(J)} {max(J)} {np.mean(J):.3f}\n")
        if np.mean(J) > 10:
            f.write("OK\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
