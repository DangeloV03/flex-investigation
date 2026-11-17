"""Single-site steady state with n frozen BONDING neighbours, exact lattice-gas rates.

Rates from lattice-gas src/markov_chain/hetero_open_drive_chain.rs (beta = 1):
  E->I  e^{df+mu}          E->B  e^{mu}
  I->E  1                  B->E  e^{n eps}
  I->B  k_n                B->I  k_n e^{df + n eps + dmu}      (dmu > 0 drives B -> I)
  k_n:  homo k;  negative_drive k min(1, e^{-n eps - df - dmu});
        positive_drive k min(1, e^{+n eps - df - dmu})

h(n) = ln P_B / (P_E + P_I)   (log-odds that the site is active)

Coexistence conditions (each gives a mu_coex(eps)):
  n2     h(2) = 0                         == FLEX as used in the pipeline (verified to 1e-12)
  homo   h(2) = 0 with k_n = k            (what the plot would show for a homogeneous drive)
  avg_p  sum_n B(n) P_B(n) = 1/2          n ~ Binomial(4, 1/2)
  avg_h  sum_n B(n) h(n)   = 0

Effective coupling eps_eff = -(h(3) - h(1)) / 2 at mu_coex (n2); equilibrium gives eps_eff = eps.
Predicted eps_c: eps where eps_eff = -2 ln(1 + sqrt 2) = -1.7627.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

EPS_C_EQ = -2 * np.log(1 + np.sqrt(2))
W4 = binom.pmf(range(5), 4, 0.5)


def k_n(n, eps, df, dmu, k, scheme):
    u = n * eps
    if scheme == "homo":
        return k
    if scheme == "negative_drive":
        return k * min(1.0, np.exp(-u - df - dmu))
    if scheme == "positive_drive":
        return k * min(1.0, np.exp(u - df - dmu))
    raise ValueError(scheme)


def h(n, eps, mu, df, dmu, k, scheme):
    kn = k_n(n, eps, df, dmu, k, scheme)
    a, b = np.exp(df + mu), np.exp(mu)               # E->I, E->B
    c, d = 1.0, kn                                    # I->E, I->B
    e, f = np.exp(n * eps), kn * np.exp(df + n * eps + dmu)   # B->E, B->I
    # matrix-tree theorem: weight of a state = sum over spanning trees directed into it
    pE = c * e + c * f + d * e
    pI = a * e + a * f + b * f
    pB = a * d + b * c + b * d
    return np.log(pB / (pE + pI))


def _solve(fun, eps):
    try:
        return brentq(fun, 2 * eps - 8, 2 * eps + 8)
    except ValueError:
        return np.nan


def mu_coex(eps, df, dmu, k, scheme, variant="n2"):
    if variant == "n2":
        return _solve(lambda mu: h(2, eps, mu, df, dmu, k, scheme), eps)
    if variant == "homo":
        return _solve(lambda mu: h(2, eps, mu, df, dmu, k, "homo"), eps)
    if variant == "avg_p":
        return _solve(lambda mu: sum(W4[n] / (1 + np.exp(-h(n, eps, mu, df, dmu, k, scheme)))
                                     for n in range(5)) - 0.5, eps)
    if variant == "avg_h":
        return _solve(lambda mu: sum(W4[n] * h(n, eps, mu, df, dmu, k, scheme) for n in range(5)), eps)
    raise ValueError(variant)


def eps_eff(eps, df, dmu, k, scheme):
    mu = mu_coex(eps, df, dmu, k, scheme)
    return -(h(3, eps, mu, df, dmu, k, scheme) - h(1, eps, mu, df, dmu, k, scheme)) / 2


def eps_c_pred(df, dmu, k, scheme, lo=-3.0, hi=-1.0, n=81):
    """eps where eps_eff crosses the Onsager value; scans for a sign change first."""
    def g(e):
        try:
            return eps_eff(e, df, dmu, k, scheme) - EPS_C_EQ
        except Exception:
            return np.nan
    es = np.linspace(lo, hi, n)
    v = np.array([g(e) for e in es])
    for i in range(n - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
            return brentq(g, es[i], es[i + 1])
    return np.nan
