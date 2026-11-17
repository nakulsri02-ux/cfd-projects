# Nozzle Flow Solver - Code Plan

## Fixed Inputs (Constants)
- Stagnation pressure: P₀ = 101325 Pa (1 atm)
- Stagnation temperature: T₀ = 300 K
- Gas constant: R = 287 J/(kg·K) for air
- Specific heat ratio: γ = 1.4 for air
- Grid points: N = 100

## Nozzle Geometry
- Inlet area: A_inlet = 3.0 (normalized)
- Throat area: A* = 1.0 (reference)
- Exit area: A_exit = 2.0
- Length: L = 1.0 m
- Shape: Smooth converging-diverging profile

## Variable Input (To explore cases)
- Exit back pressure: P_back
  - High P_back → overexpanded (shock inside)
  - Medium P_back → perfectly expanded (no shock)
  - Low P_back → underexpanded (shock outside)

## Outputs at Each Grid Point x
1. Position: x (0 to L)
2. Area: A(x) [m²]
3. Mach number: M(x) [dimensionless]
4. Pressure: P(x) [Pa]
5. Temperature: T(x) [K]
6. Density: ρ(x) = P / (R × T) [kg/m³]
7. Velocity: V(x) = M × √(γRT) [m/s]
8. Mass flow: ṁ = ρ × V × A [kg/s] (CHECK: should be constant!)

## Solution Steps
1. Define nozzle geometry: A(x) for x in [0, L]
2. Find throat location (where A is minimum)
3. For each x:
   - Calculate area ratio: A/A*
   - Solve area-Mach relation for M(x)
     - M < 1 upstream of throat
     - M = 1 at throat
     - M > 1 downstream of throat
   - Apply isentropic relations:
     - P(x) from P/P₀ = (1 + (γ-1)/2 × M²)^(-γ/(γ-1))
     - T(x) from T/T₀ = (1 + (γ-1)/2 × M²)^(-1)
   - Calculate derived quantities:
     - ρ(x) = P(x) / (R × T(x))
     - V(x) = M(x) × √(γ × R × T(x))
     - ṁ = ρ(x) × V(x) × A(x)
4. (Later) Check for shocks if overexpanded

## Code Functions Needed
1. `nozzle_area(x)` - returns A given position x
2. `solve_mach_from_area(A_ratio, subsonic=True)` - solves area-Mach relation
3. `isentropic_pressure(M, P0, gamma)` - returns P from M
4. `isentropic_temperature(M, T0, gamma)` - returns T from M
5. `compute_density(P, T, R)` - ideal gas law
6. `compute_velocity(M, T, gamma, R)` - velocity from Mach
7. `solve_nozzle_flow(P0, T0, N)` - main solver
8. `plot_results(x, M, P, T, rho, V)` - visualization

## Validation Checks
- Mass flow ṁ should be constant (within 0.1%)
- Mach = 1 exactly at throat
- Properties should be smooth (no jumps)
- Compare to isentropic tables for known cases

## Why N = 100 Grid Points?
- Trade-off between accuracy and speed
- 100 points → ~1 cm resolution for 1 m nozzle
- Runs in <1 second
- Smooth enough for clear visualization
- Can increase to 500 later if needed for shock resolution