# CFD Projects

Computational fluid dynamics projects written from scratch in Python. I derive the governing equations by hand first, implement them, and then validate against analytical theory or published benchmark data.

## Projects

| # | Project | Status | Highlights |
|---|---|---|---|
| 1 | [Quasi-1D compressible nozzle flow](01-nozzle-flow) | Complete | Isentropic converging-diverging nozzle solver. Mass flow conserved to 2 × 10⁻¹⁰ %, matches the closed-form choked-flow result and isentropic tables. Geometry also built in SolidWorks and checked against the solver's area ratios. |
| 2 | [2D lid-driven cavity flow](02-cavity-flow) | In progress | Incompressible Navier-Stokes with finite differences. Validation target: Ghia et al. (1982), Re = 100. |

![Nozzle flow field](01-nozzle-flow/figures/flow_field.png)

## Structures: Howe truss bridge FEA study

**Status:** In progress (ME 100 team project, Fall 2026). Write-up will be posted after the project is graded.

A prestressed Howe truss bridge (35 cm span) built from rods, 3D-printed PLA connectors and fishing-line tension members.
- Member forces for every rod from hand method-of-joints and method-of-sections calculations
- Static FEA of each printed connector in SOLIDWORKS Simulation, loaded with the socket forces from the hand analysis
- Hand stress estimate used as a sanity check on the FEA (factor of safety ≈ 4 at the fillets)
- Planned: buckling check on the compression rods, predicted failure load and mode, comparison with the load test

## Approach
- **Physics first.** No CFD frameworks for the core solvers. Each term is derived and discretized by hand before it is coded.
- **Validation built in.** Conservation checks and comparisons to theory are part of each solver, and each README says what the checks do and do not prove.
- **Bugs documented.** Each project records the bugs I hit and how the physics exposed them.

## Background
- White, *Fluid Mechanics* (7th ed.)
- Barba, *CFD Python: 12 Steps to Navier-Stokes*
- Brunton, ME 564/565 lecture series (numerical methods and dynamical systems)
- Math: Calculus I–III, Linear Algebra, Differential Equations
- Engineering core: Statics (current); Dynamics, Mechanics of Materials, Thermodynamics and Fluid Mechanics (upcoming)

## Tools
Python (NumPy, SciPy, Matplotlib) · Jupyter Notebook · SolidWorks and SOLIDWORKS Simulation (FEA) · MATLAB · Excel · Bell Crank (suspension design)

## Contact
Srivatsannakul@gmail.com
