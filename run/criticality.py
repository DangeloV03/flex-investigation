"""Locate the critical point epsilon_c from coex runs (Binder cumulant of the slab order parameter)."""

from flexsim.cli import Command, dispatch

BIMODALITY = "flexsim.criticality.bimodality"

COMMANDS = {
    "phase-diagram": Command(BIMODALITY, "BC_max vs beta*eps for one size, a curve per delta_mu", prefix=["phase-diagram"]),
    "replot-family": Command(BIMODALITY, "replot an existing bc_vs_beta_epsilon.csv with error bars", prefix=["replot-family"]),
    "histograms": Command(BIMODALITY, "P(phi_col) at several epsilon", prefix=["histograms"]),
    "bc-fss": Command(BIMODALITY, "BC_max vs beta*eps for several sizes", prefix=["bc-fss"]),
    "inspect": Command(BIMODALITY, "report epsilon grid coverage per delta_mu", prefix=["inspect"]),
    "scheme-comparison": Command(BIMODALITY, "scheme 1 vs scheme 3 BC_max side by side", prefix=["scheme-comparison"]),
    "L-scaling": Command("flexsim.criticality.plot_L_scaling", "epsilon_c and BC vs L across multi-L coex runs"),
    "distributions": Command("flexsim.criticality.distributions", "first-order line vs critical point from P(m)"),
}

if __name__ == "__main__":
    dispatch("criticality.py", __doc__, COMMANDS)
