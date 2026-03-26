"""
Quasi-1D Compressible Nozzle Flow Solver
Solves isentropic flow through converging-diverging nozzle
Author: Nakul Srivatsan
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================
# CONSTANTS
# ============================================
GAMMA = 1.4          # Specific heat ratio for air
R = 287.0            # Gas constant for air [J/(kg*K)]
P0 = 101325.0        # Stagnation pressure [Pa]
T0 = 300.0           # Stagnation temperature [K]

# Nozzle geometry
L = 1.0              # Nozzle length [m]
A_INLET = 3.0        # Inlet area (normalized)
A_THROAT = 1.0       # Throat area (reference)
A_EXIT = 2.0         # Exit area (normalized)

# Grid
N = 100              # Number of grid points

# ============================================
# FLOW SOLUTION FUNCTIONS
# ============================================

def area_mach_relation(M, A_ratio, gamma=GAMMA):
    """
    Area-Mach relation for isentropic flow.
    Returns 0 when M satisfies A/A* = A_ratio.
    """
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + (gamma - 1.0) / 2.0 * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    A_ratio_calculated = (1.0 / M) * (term1 * term2)**exponent
    return A_ratio_calculated - A_ratio


def solve_mach_from_area(A_ratio, subsonic=True, gamma=GAMMA):
    """
    Solve for Mach number given area ratio A/A*.
    Uses bracketed Brent solver to guarantee correct branch.
    """
    if A_ratio < 1.0:
        raise ValueError(f"Area ratio {A_ratio} < 1.0 is impossible")

    if abs(A_ratio - 1.0) < 1e-6:
        return 1.0

    if subsonic:
        # Subsonic branch: M strictly between 0 and 1
        M_solution = brentq(area_mach_relation, 1e-6, 1.0 - 1e-6,
                            args=(A_ratio, gamma))
    else:
        # Supersonic branch: M strictly between 1 and 10
        M_solution = brentq(area_mach_relation, 1.0 + 1e-6, 10.0,
                            args=(A_ratio, gamma))

    return M_solution


def isentropic_pressure(M, P0=P0, gamma=GAMMA):
    """Calculate static pressure from Mach number."""
    term = 1.0 + (gamma - 1.0) / 2.0 * M**2
    exponent = -gamma / (gamma - 1.0)
    return P0 * term**exponent


def isentropic_temperature(M, T0=T0, gamma=GAMMA):
    """Calculate static temperature from Mach number."""
    term = 1.0 + (gamma - 1.0) / 2.0 * M**2
    return T0 / term


def compute_density(P, T, R=R):
    """Ideal gas law."""
    return P / (R * T)


def compute_velocity(M, T, gamma=GAMMA, R=R):
    """Calculate velocity from Mach number and temperature."""
    a = np.sqrt(gamma * R * T)
    return M * a


# ============================================
# CUSTOM NOZZLE CREATION
# ============================================

def create_custom_nozzle(A_inlet, A_throat, A_exit, throat_location=0.5, L=1.0):
    """
    Create a smooth nozzle geometry with custom parameters.

    Parameters:
    -----------
    A_inlet : float
        Inlet area (normalized)
    A_throat : float
        Throat area (normalized, usually 1.0)
    A_exit : float
        Exit area (normalized)
    throat_location : float
        Location of throat as fraction of L (0 to 1)
    L : float
        Total nozzle length [m]

    Returns:
    --------
    function
        nozzle_area(x) function used by solver
    """
    if A_throat >= A_inlet:
        raise ValueError(f"Throat area ({A_throat}) must be less than inlet ({A_inlet})")
    if A_throat >= A_exit:
        raise ValueError(f"Throat area ({A_throat}) must be less than exit ({A_exit})")
    if not 0.1 <= throat_location <= 0.9:
        raise ValueError(f"Throat location must be between 0.1 and 0.9, got {throat_location}")

    x_throat = throat_location * L

    def nozzle_area(x):
        x = np.asarray(x, dtype=float)
        A = np.zeros_like(x)

        converging = x <= x_throat
        if np.any(converging):
            t = x[converging] / x_throat
            A[converging] = A_inlet - (A_inlet - A_throat) * (t**2)

        diverging = x > x_throat
        if np.any(diverging):
            t = (x[diverging] - x_throat) / (L - x_throat)
            A[diverging] = A_throat + (A_exit - A_throat) * (t**2)

        return A

    return nozzle_area


# ============================================
# MAIN SOLVER
# ============================================

def solve_nozzle_flow(nozzle_func, N=100, L=1.0, P0=101325.0, T0=300.0):
    """
    Solve complete nozzle flow field.

    Parameters:
    -----------
    nozzle_func : function
        Function that returns A given x
    N : int
        Number of grid points
    L : float
        Nozzle length [m]
    P0 : float
        Stagnation pressure [Pa]
    T0 : float
        Stagnation temperature [K]

    Returns:
    --------
    dict
        Results containing x, A, M, P, T, rho, V, mass_flow
    """
    print(f"\nSolving nozzle flow with {N} grid points...")

    x = np.linspace(0, L, N)
    A = nozzle_func(x)

    # Find throat
    throat_idx = np.argmin(A)
    A_throat = np.min(A)
    x_throat = x[throat_idx]

    print(f"Throat detected at x = {x_throat:.3f} m")

    # Solve Mach number at each grid point
    M = np.zeros(N)
    for i in range(N):
        A_ratio = A[i] / A_throat
        if i < throat_idx:
            M[i] = solve_mach_from_area(A_ratio, subsonic=True)
        elif i == throat_idx:
            M[i] = 1.0
        else:
            M[i] = solve_mach_from_area(A_ratio, subsonic=False)

    # Apply isentropic relations (outside loop, operates on full arrays)
    P = isentropic_pressure(M, P0=P0)
    T = isentropic_temperature(M, T0=T0)
    rho = compute_density(P, T)
    V = compute_velocity(M, T)
    mass_flow = rho * V * A

    print(f"Flow solved successfully!")

    return {
        'x': x,
        'A': A,
        'M': M,
        'P': P,
        'T': T,
        'P0': P0,
        'T0': T0,
        'rho': rho,
        'V': V,
        'mass_flow': mass_flow,
        'throat_location': x_throat,
        'throat_index': throat_idx
    }


# ============================================
# PLOTTING FUNCTIONS
# ============================================

def plot_nozzle_and_velocity(results):
    """Plot nozzle geometry and velocity distribution."""
    x = results['x']
    throat_idx = results['throat_index']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Nozzle Flow Analysis', fontsize=14, fontweight='bold')

    # Nozzle geometry
    ax1.plot(x, results['A'], 'b-', linewidth=2.5)
    ax1.axvline(results['throat_location'], color='r', linestyle='--',
                label='Throat', alpha=0.7, linewidth=1.5)
    ax1.scatter([x[0], x[throat_idx], x[-1]],
                [results['A'][0], results['A'][throat_idx], results['A'][-1]],
                color='red', s=100, zorder=5, label='Key points')
    ax1.set_xlabel('Position x [m]', fontsize=11)
    ax1.set_ylabel('Area A [m²]', fontsize=11)
    ax1.set_title('Nozzle Geometry', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Velocity distribution
    ax2.plot(x, results['V'], 'green', linewidth=2.5)
    ax2.axvline(results['throat_location'], color='r', linestyle='--',
                label='Throat (M=1)', alpha=0.7, linewidth=1.5)
    ax2.scatter([x[0], x[throat_idx], x[-1]],
                [results['V'][0], results['V'][throat_idx], results['V'][-1]],
                color='red', s=100, zorder=5)
    ax2.set_xlabel('Position x [m]', fontsize=11)
    ax2.set_ylabel('Velocity V [m/s]', fontsize=11)
    ax2.set_title('Velocity Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print throat conditions
    print("\n" + "="*60)
    print("THROAT CONDITIONS (M = 1)")
    print("="*60)
    print(f"Location:     x = {results['throat_location']:.3f} m")
    print(f"Area:         A = {results['A'][throat_idx]:.3f} m²")
    print(f"Mach:         M = {results['M'][throat_idx]:.4f}")
    print(f"Pressure:     P = {results['P'][throat_idx]/1000:.2f} kPa ({results['P'][throat_idx]/results['P0']:.3f} P₀)")
    print(f"Temperature:  T = {results['T'][throat_idx]:.1f} K ({results['T'][throat_idx]/results['T0']:.3f} T₀)")
    print(f"Density:      ρ = {results['rho'][throat_idx]:.4f} kg/m³")
    print(f"Velocity:     V = {results['V'][throat_idx]:.1f} m/s")
    print(f"Mass flow:    ṁ = {results['mass_flow'][throat_idx]:.5f} kg/s")
    print("="*60)


def check_mass_flow_conservation(results):
    """Verify mass flow conservation throughout nozzle."""
    mass_flow = results['mass_flow']
    x = results['x']

    mean_mass_flow = np.mean(mass_flow)
    std_mass_flow = np.std(mass_flow)
    min_mass_flow = np.min(mass_flow)
    max_mass_flow = np.max(mass_flow)
    variation = (max_mass_flow - min_mass_flow) / mean_mass_flow * 100

    is_conserved = variation < 0.1

    print("\n" + "="*60)
    print("MASS FLOW CONSERVATION CHECK")
    print("="*60)
    print(f"Mean mass flow:     ṁ_avg = {mean_mass_flow:.6f} kg/s")
    print(f"Standard deviation: σ = {std_mass_flow:.2e} kg/s")
    print(f"Minimum:            ṁ_min = {min_mass_flow:.6f} kg/s")
    print(f"Maximum:            ṁ_max = {max_mass_flow:.6f} kg/s")
    print(f"Variation:          {variation:.4f}%")
    print("-"*60)

    if is_conserved:
        print("✅ PASSED: Mass flow is conserved!")
        print("   (Variation < 0.1% indicates excellent conservation)")
    else:
        print("⚠️  WARNING: Mass flow varies more than expected!")
        print("   (This may indicate numerical issues)")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Mass Flow Conservation Analysis', fontsize=14, fontweight='bold')

    ax1.plot(x, mass_flow, 'b-', linewidth=2, label='ṁ(x)')
    ax1.axhline(mean_mass_flow, color='r', linestyle='--',
                linewidth=1.5, label=f'Mean = {mean_mass_flow:.3f} kg/s')
    ax1.axvline(results['throat_location'], color='g', linestyle='--',
                alpha=0.5, label='Throat')
    ax1.set_xlabel('Position x [m]', fontsize=11)
    ax1.set_ylabel('Mass Flow ṁ [kg/s]', fontsize=11)
    ax1.set_title('Mass Flow Distribution', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    deviation_percent = (mass_flow - mean_mass_flow) / mean_mass_flow * 100
    ax2.plot(x, deviation_percent, 'r-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='±0.1% threshold')
    ax2.axhline(-0.1, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(results['throat_location'], color='g', linestyle='--',
                alpha=0.5, label='Throat')
    ax2.set_xlabel('Position x [m]', fontsize=11)
    ax2.set_ylabel('Deviation from Mean [%]', fontsize=11)
    ax2.set_title('Mass Flow Conservation Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return is_conserved


# ============================================
# USER INPUT FUNCTIONS
# ============================================

def get_user_inputs():
    """Get stagnation conditions and back pressure from user."""
    print("\n" + "="*60)
    print("FLOW CONDITIONS")
    print("="*60)

    P0_input = input("\nStagnation pressure P0 [Pa] (default 101325): ").strip()
    P0 = float(P0_input) if P0_input else 101325.0

    T0_input = input("Stagnation temperature T0 [K] (default 300): ").strip()
    T0 = float(T0_input) if T0_input else 300.0

    P_back_input = input("Back pressure P_back [Pa] (default 101325): ").strip()
    P_back = float(P_back_input) if P_back_input else 101325.0

    if T0 <= 0:
        raise ValueError("Temperature must be positive")
    if P0 <= 0:
        raise ValueError("Pressure must be positive")
    if P_back <= 0 or P_back > P0:
        raise ValueError("Back pressure must be between 0 and P0")

    print(f"\n  • Stagnation pressure: {P0:.1f} Pa")
    print(f"  • Stagnation temperature: {T0:.1f} K")
    print(f"  • Back pressure: {P_back:.1f} Pa ({P_back/P0:.3f} P0)")

    return P0, T0, P_back


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("CUSTOM NOZZLE FLOW SOLVER")
    print("="*60)
    print("\nThis solver computes compressible flow through")
    print("a converging-diverging nozzle with YOUR custom geometry.")
    print("="*60)

    while True:
        # STEP 1: NOZZLE GEOMETRY
        print("\n📐 DEFINE YOUR NOZZLE GEOMETRY")
        print("-" * 60)

        while True:
            try:
                print("\nEnter nozzle dimensions (or press Enter for defaults):")
                print("  Defaults: Inlet=3.0, Throat=1.0, Exit=2.5, Location=0.4\n")

                inlet_input = input("Inlet area [m²] (default 3.0): ").strip()
                A_inlet = float(inlet_input) if inlet_input else 3.0

                throat_input = input("Throat area [m²] (default 1.0): ").strip()
                A_throat = float(throat_input) if throat_input else 1.0

                exit_input = input("Exit area [m²] (default 2.5): ").strip()
                A_exit = float(exit_input) if exit_input else 2.5

                location_input = input("Throat location [0-1] (default 0.4): ").strip()
                throat_location = float(location_input) if location_input else 0.4

                if A_throat >= A_inlet:
                    print("\n❌ ERROR: Throat area must be smaller than inlet area!")
                    continue
                if A_throat >= A_exit:
                    print("\n❌ ERROR: Throat area must be smaller than exit area!")
                    continue
                if not 0.1 <= throat_location <= 0.9:
                    print("\n❌ ERROR: Throat location must be between 0.1 and 0.9!")
                    continue
                break

            except ValueError:
                print("\n❌ ERROR: Please enter valid numbers!")
                continue

        print("\n" + "="*60)
        print("YOUR NOZZLE CONFIGURATION:")
        print("="*60)
        print(f"  • Inlet area:      {A_inlet:.2f} m²")
        print(f"  • Throat area:     {A_throat:.2f} m²")
        print(f"  • Exit area:       {A_exit:.2f} m²")
        print(f"  • Throat location: {throat_location:.1%} along length")
        print("="*60)

        # STEP 2: FLOW CONDITIONS
        print("\n💨 DEFINE FLOW CONDITIONS")
        print("-" * 60)

        while True:
            try:
                P0, T0, P_back = get_user_inputs()
                break
            except ValueError as e:
                print(f"\n❌ ERROR: {e}")
                print("   Please try again.\n")
                continue

        # STEP 3: CREATE NOZZLE AND SOLVE
        print("\n🔧 Creating nozzle geometry...")
        my_nozzle = create_custom_nozzle(
            A_inlet=A_inlet,
            A_throat=A_throat,
            A_exit=A_exit,
            throat_location=throat_location
        )

        print("\n🚀 Solving compressible flow...")
        results = solve_nozzle_flow(my_nozzle, N=100, L=1.0, P0=P0, T0=T0)

        # STEP 4: VISUALIZE AND VALIDATE
        print("\n📊 Generating plots...")
        plot_nozzle_and_velocity(results)

        print("\n🔍 Checking mass flow conservation...")
        check_mass_flow_conservation(results)

        # STEP 5: RUN AGAIN?
        print("\n" + "="*60)
        run_again = input("\nRun with different geometry? (y/n): ").strip().lower()
        if run_again != 'y':
            print("\nThank you for using the Nozzle Flow Solver!")
            break
