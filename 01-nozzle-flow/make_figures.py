"""
Generate the README figures and validation table for the nozzle solver.

Case: A_in/A* = 3, A_exit/A* = 2, throat at mid-length (same geometry as the
SolidWorks model in solidworks/). Runs non-interactively:

    python make_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nozzle_solver import create_custom_nozzle, solve_nozzle_flow, GAMMA

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

nozzle = create_custom_nozzle(A_inlet=3.0, A_throat=1.0, A_exit=2.0, throat_location=0.5)
r = solve_nozzle_flow(nozzle, N=101)
x, A, M, P, T, mdot = r["x"], r["A"], r["M"], r["P"], r["T"], r["mass_flow"]
P0, T0, it = r["P0"], r["T0"], r["throat_index"]

# --- Figure 1: flow field -------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
ax[0].plot(x, A, color="k"); ax[0].set_ylabel("A / A*")
ax[1].plot(x, M, color="#1f5fa8"); ax[1].axhline(1, ls="--", color="0.5", lw=0.8); ax[1].set_ylabel("Mach number")
ax[2].plot(x, P / P0, color="#b23a2e"); ax[2].set_ylabel("P / P0")
ax[3].plot(x, T / T0, color="#2e7d4f"); ax[3].set_ylabel("T / T0"); ax[3].set_xlabel("x / L")
for a in ax:
    a.axvline(x[it], ls=":", color="0.5", lw=0.8); a.grid(alpha=0.3)
ax[0].set_title("Isentropic flow through a converging-diverging nozzle (A_in/A* = 3, A_e/A* = 2)")
fig.tight_layout(); fig.savefig(FIG / "flow_field.png", dpi=140); plt.close(fig)

# --- Figure 2: mass conservation -------------------------------------------
dev = (mdot - mdot.mean()) / mdot.mean() * 100
fig, a = plt.subplots(figsize=(8, 3.5))
a.plot(x, dev, color="#1f5fa8")
a.set_xlabel("x / L"); a.set_ylabel("mass-flow deviation from mean [%]")
a.set_title(f"Mass conservation check: max |deviation| = {np.abs(dev).max():.2e} %")
a.grid(alpha=0.3); fig.tight_layout(); fig.savefig(FIG / "mass_conservation.png", dpi=140); plt.close(fig)

# --- Validation table -------------------------------------------------------
# Independent check: closed-form choked mass flow, and the area-Mach relation
# evaluated at the solver's own Mach numbers (should return the input area).
R = 287.0
mdot_theory = A[it] * P0 / np.sqrt(T0) * np.sqrt(GAMMA / R) * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
A_back = (1 / M) * ((2 / (GAMMA + 1)) * (1 + (GAMMA - 1) / 2 * M**2)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
print(f"\nmax |mdot deviation|      : {np.abs(dev).max():.3e} %")
print(f"mdot solver / theory      : {mdot.mean():.6f} / {mdot_theory:.6f} kg/s  (err {abs(mdot.mean()/mdot_theory-1)*100:.2e} %)")
print(f"max area-Mach residual    : {np.abs(A_back - A / A[it]).max():.2e}")
print("\n| Station | A/A* | M | P/P0 | T/T0 | P [Pa] | T [K] | V [m/s] |")
print("|---|---|---|---|---|---|---|---|")
for name, i in [("Inlet", 0), ("Throat", it), ("Exit", -1)]:
    print(f"| {name} | {A[i]/A[it]:.3f} | {M[i]:.4f} | {P[i]/P0:.4f} | {T[i]/T0:.4f} | {P[i]:,.0f} | {T[i]:.1f} | {r['V'][i]:.1f} |")
