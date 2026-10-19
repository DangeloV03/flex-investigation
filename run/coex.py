"""Coexistence campaigns: generate jobs, run them on Slurm, and find mu_coex."""

from flexsim.cli import Command, dispatch

COMMANDS = {
    "generate": Command("flexsim.coex.generate", "create the (epsilon, delta_mu) job grid and queue it"),
    "generate-susc": Command("flexsim.coex.generate_for_susc", "create slab mu-sweep jobs for the susceptibility study"),
    "start": Command("flexsim.coex.daemons", "start dispatcher + analyzer in tmux", prefix=["start"]),
    "stop": Command("flexsim.coex.daemons", "stop them (--slurm also cancels jobs)", prefix=["stop"]),
    "dispatch": Command("flexsim.coex.dispatcher", "run the job dispatcher in the foreground (--local: no Slurm)"),
    "analyze": Command("flexsim.coex.analyzer", "run the mu_coex analyzer in the foreground"),
    "run-job": Command("flexsim.coex.runner", "run a single job JSON (debugging)"),
}

if __name__ == "__main__":
    dispatch("coex.py", __doc__, COMMANDS)
