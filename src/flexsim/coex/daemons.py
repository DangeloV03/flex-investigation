"""Start or stop the coex dispatcher + analyzer in a detached tmux session.

    start           open tmux session "flex-investigation" with two windows,
                    dispatcher (submits jobs) and analyzer (refines mu)
    stop            kill that session (and any stray dispatcher/analyzer)
    stop --slurm    also scancel your flex_sim jobs

Reattach with ``tmux attach -t flex-investigation``; detach with Ctrl-b d.
"""

from __future__ import annotations

import argparse
import getpass
import subprocess
import sys

from flexsim.paths import OUTPUT_ROOT

SESSION = "flex-investigation"

# Non-interactive tmux panes need conda.sh sourced before `conda activate`.
_PANE_SETUP = (
    "module load anaconda3/2024.10 2>/dev/null; "
    'source "$(conda info --base)/etc/profile.d/conda.sh"; '
    "conda activate lattice; "
    'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"; '
    "export PYTHONUNBUFFERED=1"
)

_WINDOWS = {
    "dispatcher": "flexsim.coex.dispatcher",
    "analyzer": "flexsim.coex.analyzer",
}


def _has_session() -> bool:
    return subprocess.run(["tmux", "has-session", "-t", SESSION],
                          capture_output=True).returncode == 0


def start() -> int:
    if _has_session():
        print(f"Session '{SESSION}' already exists. Attach: tmux attach -t {SESSION}")
        return 1
    first = True
    for window, module in _WINDOWS.items():
        if first:
            subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-n", window,
                            "-c", str(OUTPUT_ROOT)], check=True)
            first = False
        else:
            subprocess.run(["tmux", "new-window", "-t", SESSION, "-n", window,
                            "-c", str(OUTPUT_ROOT)], check=True)
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION}:{window}",
                        f"{_PANE_SETUP}; python -u -m {module}", "C-m"], check=True)
    print(f"Started tmux session '{SESSION}' ({', '.join(_WINDOWS)}) in {OUTPUT_ROOT}")
    print(f"  attach: tmux attach -t {SESSION}    detach: Ctrl-b d")
    return 0


def stop(cancel_slurm: bool) -> int:
    if _has_session():
        subprocess.run(["tmux", "kill-session", "-t", SESSION], check=True)
        print(f"Killed tmux session '{SESSION}'")
    else:
        print(f"No tmux session '{SESSION}' (already stopped?)")
    for module in _WINDOWS.values():
        if subprocess.run(["pkill", "-f", f"python.*{module}"]).returncode == 0:
            print(f"Killed stray {module}")
    if cancel_slurm:
        subprocess.run(["scancel", "-u", getpass.getuser(), "-n", "flex_sim"])
        print("Cancelled Slurm flex_sim jobs")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("start", help="start dispatcher + analyzer in tmux")
    sp = sub.add_parser("stop", help="stop them")
    sp.add_argument("--slurm", action="store_true", help="also scancel flex_sim jobs")
    args = p.parse_args()
    return start() if args.action == "start" else stop(args.slurm)


if __name__ == "__main__":
    sys.exit(main())
