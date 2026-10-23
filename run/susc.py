"""Susceptibility campaigns: L x L production runs that measure chi, U4 and <|m|>."""

from flexsim.cli import Command, dispatch

COMMANDS = {
    "sweep": Command("flexsim.susc.smart_sweep", "submit an epsilon sweep; top-ups chain automatically", prefix=["sweep"]),
    "check": Command("flexsim.susc.smart_sweep", "count jumps and submit top-ups by hand", prefix=["check"]),
    "fixed-sweep": Command("flexsim.susc.sweep", "older fixed-length sweep, one job per epsilon (no top-ups)"),
    "run-one": Command("flexsim.susc.runner", "run one (epsilon, L) point directly (debugging)"),
    "refit": Command("flexsim.susc.coex_refit", "fit mu_coex(eps), pick eps with <m> off 0, promote reruns, compare"),
}

if __name__ == "__main__":
    dispatch("susc.py", __doc__, COMMANDS)
