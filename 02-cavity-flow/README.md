# 2D Lid-Driven Cavity Flow Solver

**Status:** In progress. Architecture set up; momentum-step discretization underway. No results yet.

## Goal
Solve the 2D incompressible Navier-Stokes equations in a square cavity with a moving lid, written from scratch in Python with finite differences, and validate against the standard benchmark.

## Plan
- **Method:** finite differences on a uniform grid, following the structure of Barba's *CFD Python* Steps 11–12 (pressure Poisson coupling), with every discretization derived by hand before it is coded
- **Validation:** centerline velocity profiles at Re = 100 against Ghia, Ghia & Shin (1982)
- **Verification:** grid convergence study and a comparison of time-integration schemes (forward Euler through RK4) with measured error and order of accuracy
- **Outputs:** streamlines, velocity and pressure fields, centerline profile comparison plots

## Why this problem
It is the simplest case that needs the full pressure-velocity coupling of incompressible flow, and it has high-quality published benchmark data. The viscous physics also mirrors the analytical Couette and Poiseuille solutions I worked through by hand in White, Ch. 4.10.

## References
- Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48, 387–411.
- Barba, L.A., Forsyth, G.F. *CFD Python: the 12 steps to Navier-Stokes equations.*
- White, F.M. *Fluid Mechanics*, 7th ed.
