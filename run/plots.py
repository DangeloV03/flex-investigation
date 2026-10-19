"""Plots from finished runs."""

from flexsim.cli import Command, dispatch

COMMANDS = {
    "susc": Command("flexsim.susc.plot", "chi, <|m|>, c and U4 vs epsilon; peak chi vs L"),
    "fss": Command("flexsim.susc.plot_fss", "finite-size scaling collapse of chi and <|m|>"),
    "chi-scaling": Command("flexsim.susc.chi_max_scaling", "fit chi_max(L) ~ L^(gamma/nu) with LOO + bootstrap errors"),
    "jumps": Command("flexsim.susc.mag_jumps", "magnetization jump counts per (epsilon, L)"),
    "flex-vs-sim": Command("flexsim.coex.plot_flex_vs_sim", "FLEX vs simulated mu_coex in the (beta eps, beta mu) plane"),
}

if __name__ == "__main__":
    dispatch("plots.py", __doc__, COMMANDS)
