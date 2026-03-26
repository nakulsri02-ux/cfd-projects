# Quasi-1D Compressible Nozzle Flow Solver

**Author:** Nakul Srivatsan  
**Date:** February/March 2026  
**Language:** Python 3 (used Claude as a coding assistant) · NumPy · SciPy · Matplotlib

---

## Project Scope

This is My first project in my self-directed CFD portfolio built to teach and show the fundementals and basic applications of computational fluid dynamics.

The goal was to build a physics-first solver from scratch, not using any CFD framework or copying a tutorial rather deriving the governing equations and implementing them, then validating against analytical theory, and comparing against known resluts.

The solver computes isentropic compressible flow through a user generated converging-diverging (de Laval) nozzle. Given reservoir conditions and a custom geometry, it outputs the full flow field: Mach number, pressure, temperature, density, and velocity at every point along the nozzle axis. Mass flow conservation is validated to better than 0.0001%.

---

## Physics Background

Physics background from PHY211/222 (Washtenaw Community College), supplemented by self-study of White's *Fluid Mechanics*, 7th Ed. (McGraw-Hill).

### Why a Converging-Diverging Nozzle?

A de Laval nozzle accelerates flow from subsonic to supersonic. This is counterintuitive: in everyday experience (subsonic flow), squeezing a pipe speeds up flow — that's Bernoulli. But at Mach 1, something breaks: the flow can no longer accelerate in a converging section. To go supersonic, the area must *increase* after the throat.

This is a direct consequence of the **area-velocity relation** for compressible flow:

```
dA/A = (M² - 1) · dV/V
```

- When M < 1: dA and dV have opposite signs → area decreases, velocity increases (subsonic intuition)
- When M > 1: dA and dV have the same sign → area must increase for velocity to increase (supersonic)
- When M = 1: dA = 0 → Mach 1 can only occur at a throat (area minimum)

### Governing Equations

All relations assume **isentropic flow** (adiabatic + no friction + no shocks). Stagnation conditions P₀ and T₀ remain constant throughout.

**Area-Mach Relation** (links geometry to Mach number):

```
A/A* = (1/M) · [ (2/(γ+1)) · (1 + (γ-1)/2 · M²) ]^((γ+1)/(2(γ-1)))
```

**Isentropic Pressure:**
```
P/P₀ = (1 + (γ-1)/2 · M²)^(-γ/(γ-1))
```

**Isentropic Temperature:**
```
T/T₀ = 1 / (1 + (γ-1)/2 · M²)
```

**Velocity:**
```
V = M · √(γRT)
```

**Mass Flow Conservation** (must hold at every cross-section):
```
ṁ = ρ · V · A = constant
```

---

## Features

- **User-defined geometry** — specify inlet area, throat area, exit area, and throat location interactively
- **Dual Mach branch solver** — correctly solves subsonic and supersonic branches of the area-Mach relation
- **Full flow field output** — M, P, T, ρ, V at every grid point
- **Mass flow validation** — checks conservation to < 0.1% with deviation plot
- **Interactive terminal interface** — input validation, error messages, re-run loop
- **Professional visualization** — nozzle geometry, velocity distribution, mass flow conservation plots

---

## How to Run

### Requirements
```bash
pip install numpy scipy matplotlib
```

### Run
```bash
python nozzle_solver.py
```

### Example Session
```
============================================================
CUSTOM NOZZLE FLOW SOLVER
============================================================

📐 DEFINE YOUR NOZZLE GEOMETRY
------------------------------------------------------------
Enter nozzle dimensions (or press Enter for defaults):
  Defaults: Inlet=3.0, Throat=1.0, Exit=2.5, Location=0.4

Inlet area [m²] (default 3.0): 
Throat area [m²] (default 1.0): 
Exit area [m²] (default 2.5): 
Throat location [0-1] (default 0.4): 

💨 DEFINE FLOW CONDITIONS
------------------------------------------------------------
Stagnation pressure P0 [Pa] (default 101325): 
Stagnation temperature T0 [K] (default 300): 
Back pressure P_back [Pa] (default 101325): 

🚀 Solving compressible flow...
✅ PASSED: Mass flow is conserved! (Variation < 0.0001%)
```

---

## Code Architecture

```
nozzle_solver.py
│
├── area_mach_relation()           # Root function for Brent solver
├── solve_mach_from_area()         # Solves A/A* → M (subsonic or supersonic)
├── isentropic_pressure()          # P from M via isentropic relation
├── isentropic_temperature()       # T from M via isentropic relation
├── compute_density()              # Ideal gas law: ρ = P/RT
├── compute_velocity()             # V = M · √(γRT)
│
├── create_custom_nozzle()         # Builds user-defined nozzle geometry function
├── solve_nozzle_flow()            # Main solver: loops over grid, calls all above
│
├── plot_nozzle_and_velocity()     # Geometry + velocity plots
├── check_mass_flow_conservation() # Validation plot + pass/fail report
│
└── main block                     # Interactive terminal loop with input validation
```

---

## Sample Output

| Location | M | P (Pa) | T (K) | V (m/s) |
|----------|---|--------|-------|---------|
| Inlet | ~0.18 | ~99,100 | ~299 | ~62 |
| Throat | 1.000 | 53,471 | 250.0 | 316.9 |
| Exit | ~2.20 | ~9,380 | ~152 | ~544 |

Mass flow variation: **< 0.0001%** ✅

---

## Bugs and Lessons Learned

This section documents the real problems encountered during development — what actually broke, why it broke, and what the fix revealed about the underlying physics and numerics.

### Bug 1: `fsolve` Converging to the Wrong Mach Branch

**The problem:** The initial implementation used `scipy.optimize.fsolve` with a fixed initial guess (`M_guess = 0.5` for subsonic, `M_guess = 2.0` for supersonic). For area ratios close to 1.0 near the throat, `fsolve` would occasionally return a solution on the wrong branch — giving a supersonic result in the converging section, or vice versa — with no error or warning.

**Why it happened:** `fsolve` uses Newton-style iteration and is sensitive to the initial guess. Near A/A* ≈ 1, both the subsonic and supersonic roots are numerically close, and the solver can jump branches.

**The fix:** Replaced `fsolve` with `scipy.optimize.brentq`, a bracketed root-finding method. Brent's method is given a guaranteed bracket:
- Subsonic branch: `[1e-6, 1.0 - 1e-6]`
- Supersonic branch: `[1.0 + 1e-6, 10.0]`

Since the area-Mach function is monotonic on each branch within these brackets, `brentq` is guaranteed to find the correct root.

**What I learned:** Numerical solvers don't know physics. `fsolve` will return a mathematically valid root that is physically meaningless if the guess is bad. Bracketed methods are more robust when you can bound the solution — and in isentropic flow, you always can.

---

### Bug 2: Hardcoded Throat Location Breaking the Geometry

**The problem:** The original geometry assumed the throat was always at x = 0.5L. The parametric formula used `2 * x_norm` and `2 * (x_norm - 0.5)` — both hardcoded to split at 50%. When I tried to support user-defined throat locations (e.g., throat at 40%), the converging and diverging sections didn't meet smoothly at the throat, producing a kink in the area profile.

**The fix:** Rewrote `create_custom_nozzle()` with a `throat_location` parameter. Geometry is now computed using normalized coordinates local to each section:
- Converging: `t = x / x_throat` (runs 0→1 over the converging section)
- Diverging: `t = (x - x_throat) / (L - x_throat)` (runs 0→1 over the diverging section)

This guarantees a smooth quadratic profile on both sides regardless of throat position.

**What I learned:** Hardcoded magic numbers hide assumptions. When I generalized the code, those assumptions broke immediately. Parametric design — every geometric feature expressed in terms of named inputs — is far more robust and catches bugs early.

---

### Bug 3: Mass Flow Variation at the Inlet

**The problem:** Early mass flow checks showed ~2% variation concentrated near the inlet. The conservation plot showed ṁ jumping at x=0 before settling to a constant downstream.

**Why it happened:** The area ratio calculation used the user-input throat area as the reference (`A_ratio = A[i] / A_throat_input`). But on a discrete grid, no grid point may land exactly at the true throat minimum. This small mismatch propagated into every Mach number calculation, and the error was most visible at the inlet where Mach numbers are low and small errors in M cause relatively larger errors in ρ and V.

**The fix:** Changed the reference throat area to the **computed minimum of the discrete area array**: `A_throat = np.min(A)`. This ensures the area ratio is always computed relative to the actual grid throat, eliminating the mismatch.

**What I learned:** Conservation laws are the best debugging tool in CFD. If ṁ isn't constant, something is wrong — in the geometry, the solver, or the post-processing. Building the mass flow check early (not as an afterthought) located this bug in minutes rather than hours.

---

## What's Next

- **Project 2:** 2D lid-driven cavity flow — solving incompressible Navier-Stokes from scratch using finite differences (Python)
- **Project 3:** NACA airfoil analysis — RANS turbulence modeling in OpenFOAM

---

## References

- Anderson, J.D. *Modern Compressible Flow*. McGraw-Hill.
- White, F.M. *Fluid Mechanics*. McGraw-Hill.
- Isentropic flow tables verified against: [NASA Glenn Research Center](https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html)
