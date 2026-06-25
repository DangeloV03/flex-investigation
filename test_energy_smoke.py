"""
Smoke test for compute_energy() against hand-computed analytic values.

Lattice: 4x4 (N=16), periodic BC
Params:  beta=1, epsilon=-1.5, mu=-2.0, delta_f=-20.0

Analytic expectations
---------------------
BONDING=2, INERT=1, EMPTY=0

E = (β·ε/2)·Σᵢ_{B} n_bonding_neighbors(i)  −  β·μ·n_B  −  β·(μ+Δf)·n_I

Case 1 — fully BONDING (all 16 sites = BONDING):
  e_interact = 0.5 * 1 * (-1.5) * (16*4)  =  -48.0   [each of 16 sites has 4 bonding neighbors]
  e_chem     = -1 * (-2.0) * 16           =  +32.0
  E_expected =  -48 + 32                  =  -16.0

Case 2 — fully EMPTY (all 16 sites = EMPTY):
  e_interact = 0   [no bonding sites]
  e_chem     = 0   [no bonding, no inert]
  E_expected =  0.0

Case 3 — checkerboard BONDING/EMPTY (8 BONDING, 8 EMPTY):
  Each BONDING site's 4 neighbors are all EMPTY → bonding neighbor count = 0
  e_interact = 0
  e_chem     = -1 * (-2.0) * 8  =  +16.0
  E_expected =  16.0

Case 4 — single BONDING site at (0,0), rest EMPTY:
  That site has 0 bonding neighbors (all neighbors are EMPTY)
  e_interact = 0
  e_chem     = -1 * (-2.0) * 1  =  +2.0
  E_expected =  2.0

Case 5 — two adjacent BONDING sites at (0,0) and (0,1), rest EMPTY:
  Each has exactly 1 bonding neighbor → Σ = 2, with 0.5 prefactor → 1 bond
  e_interact = 0.5 * 1 * (-1.5) * 2  =  -1.5
  e_chem     = -1 * (-2.0) * 2        =  +4.0
  E_expected =  2.5
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from susceptibility_runner import compute_energy, EMPTY, INERT, BONDING

L = 4
N = L * L
beta = 1.0
epsilon = -1.5
mu = -2.0
delta_f = -20.0

cases = []

# Case 1: fully BONDING
s1 = np.full((L, L), BONDING, dtype=np.uint32)
cases.append(("fully BONDING", s1, -16.0))

# Case 2: fully EMPTY
s2 = np.zeros((L, L), dtype=np.uint32)
cases.append(("fully EMPTY", s2, 0.0))

# Case 3: checkerboard BONDING / EMPTY
s3 = np.zeros((L, L), dtype=np.uint32)
for i in range(L):
    for j in range(L):
        if (i + j) % 2 == 0:
            s3[i, j] = BONDING
cases.append(("checkerboard BONDING/EMPTY", s3, 16.0))

# Case 4: single BONDING site
s4 = np.zeros((L, L), dtype=np.uint32)
s4[0, 0] = BONDING
cases.append(("single BONDING site", s4, 2.0))

# Case 5: two adjacent BONDING sites
s5 = np.zeros((L, L), dtype=np.uint32)
s5[0, 0] = BONDING
s5[0, 1] = BONDING
cases.append(("two adjacent BONDING sites", s5, 2.5))

all_passed = True
for name, state, expected in cases:
    got = compute_energy(state, beta, epsilon, mu, delta_f)
    ok = abs(got - expected) < 1e-9
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_passed = False
    print(f"[{status}] {name}: expected={expected:.4f}  got={got:.4f}")

print()
if all_passed:
    print("All cases passed.")
else:
    print("SOME CASES FAILED.")
    sys.exit(1)
