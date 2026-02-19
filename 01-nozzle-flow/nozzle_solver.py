"""
Quasi-1D Compressible Nozzle Flow Solver
Solves isentropic flow through converging-diverging nozzle
Author: Nakul Srivatsan
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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
# GEOMETRY FUNCTIONS
# ============================================

def nozzle_area(x):
    """
    Define nozzle area as function of position x.
    """
    x_norm = x / L
    
    if np.isscalar(x_norm):
        if x_norm <= 0.5:
            A = A_INLET - (A_INLET - A_THROAT) * (2 * x_norm)**2
        else:
            A = A_THROAT + (A_EXIT - A_THROAT) * (2 * (x_norm - 0.5))**2
    else:
        A = np.zeros_like(x_norm)
        converging = x_norm <= 0.5
        diverging = x_norm > 0.5
        
        A[converging] = A_INLET - (A_INLET - A_THROAT) * (2 * x_norm[converging])**2
        A[diverging] = A_THROAT + (A_EXIT - A_THROAT) * (2 * (x_norm[diverging] - 0.5))**2
    
    return A

# ============================================
# FLOW SOLUTION FUNCTIONS
# ============================================

def area_mach_relation(M, A_ratio, gamma=GAMMA):
    """
    Area-Mach relation for isentropic flow.
    """
    term1 = 2.0 / (gamma + 1.0)
    term2 = 1.0 + (gamma - 1.0) / 2.0 * M**2
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    
    A_ratio_calculated = (1.0 / M) * (term1 * term2)**exponent
    
    return A_ratio_calculated - A_ratio


def solve_mach_from_area(A_ratio, subsonic=True, gamma=GAMMA):
    """
    Solve for Mach number given area ratio A/A*.
    """
    if A_ratio < 1.0:
        raise ValueError(f"Area ratio {A_ratio} < 1.0 is impossible")
    
    if abs(A_ratio - 1.0) < 1e-6:
        return 1.0
    
    if subsonic:
        M_guess = 0.5
    else:
        M_guess = 2.0
    
    M_solution = fsolve(area_mach_relation, M_guess, args=(A_ratio, gamma))
    
    return M_solution[0]


def isentropic_pressure(M, P0=P0, gamma=GAMMA):
    """
    Calculate static pressure from Mach number.
    """
    term = 1.0 + (gamma - 1.0) / 2.0 * M**2
    exponent = -gamma / (gamma - 1.0)
    P = P0 * term**exponent
    return P


def isentropic_temperature(M, T0=T0, gamma=GAMMA):
    """
    Calculate static temperature from Mach number.
    """
    term = 1.0 + (gamma - 1.0) / 2.0 * M**2
    T = T0 / term
    return T


def compute_density(P, T, R=R):
    return P / (R * T)


def compute_velocity(M, T, gamma=GAMMA, R=R):
    """
    Calculate velocity from Mach number and temperature.
    """
    a = np.sqrt(gamma * R * T)
    V = M * a
    return V

# ============================================
# TEST FUNCTIONS
# ============================================

def test_geometry():
    """Test function to visualize nozzle shape"""
    x = np.linspace(0, L, N)
    A = nozzle_area(x)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x, A, 'b-', linewidth=2)
    plt.axhline(A_THROAT, color='r', linestyle='--', label='Throat')
    plt.xlabel('Position x [m]')
    plt.ylabel('Area A [normalized]')
    plt.title('Nozzle Geometry')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Inlet area: {A[0]:.3f}")
    print(f"Throat area: {A[N//2]:.3f}")
    print(f"Exit area: {A[-1]:.3f}")

def test_area_mach():
    """Test area-Mach relation solver"""
    print("\n" + "="*50)
    print("Testing Area-Mach Relation Solver")
    print("="*50)
    
    test_cases = [
        (1.0, "Throat"),
        (1.5, "Converging section (subsonic)"),
        (1.5, "Diverging section (supersonic)"),
        (2.0, "Converging section (subsonic)"),
        (2.0, "Diverging section (supersonic)"),
    ]
    
    for i, (A_ratio, description) in enumerate(test_cases):
        if i == 0:
            M = solve_mach_from_area(A_ratio, subsonic=True)
            print(f"\nA/A* = {A_ratio:.2f} ({description})")
            print(f"  M = {M:.4f}")
        elif "subsonic" in description:
            M = solve_mach_from_area(A_ratio, subsonic=True)
            print(f"\nA/A* = {A_ratio:.2f} ({description})")
            print(f"  M = {M:.4f} (should be < 1)")
        else:
            M = solve_mach_from_area(A_ratio, subsonic=False)
            print(f"\nA/A* = {A_ratio:.2f} ({description})")
            print(f"  M = {M:.4f} (should be > 1)")
    
    print("\n" + "="*50)
    print("Verification against known isentropic flow tables:")
    print("="*50)
    print("A/A* = 2.0 should give:")
    print("  M_subsonic ≈ 0.31")
    print("  M_supersonic ≈ 2.20")


def test_isentropic_relations():
    """Test isentropic relations"""
    print("\n" + "="*50)
    print("Testing Isentropic Relations")
    print("="*50)
    
    M = 1.0
    P = isentropic_pressure(M)
    T = isentropic_temperature(M)
    rho = compute_density(P, T)
    V = compute_velocity(M, T)
    
    print(f"\nAt throat (M = {M:.1f}):")
    print(f"  Pressure:    P = {P:.1f} Pa = {P/P0:.4f} P0")
    print(f"  Temperature: T = {T:.1f} K = {T/T0:.4f} T0")
    print(f"  Density:     ρ = {rho:.4f} kg/m³")
    print(f"  Velocity:    V = {V:.1f} m/s")
    
    M = 2.2
    P = isentropic_pressure(M)
    T = isentropic_temperature(M)
    rho = compute_density(P, T)
    V = compute_velocity(M, T)
    
    print(f"\nAt exit (M = {M:.1f}):")
    print(f"  Pressure:    P = {P:.1f} Pa = {P/P0:.4f} P0")
    print(f"  Temperature: T = {T:.1f} K = {T/T0:.4f} T0")
    print(f"  Density:     ρ = {rho:.4f} kg/m³")
    print(f"  Velocity:    V = {V:.1f} m/s")
    
    print("\n" + "="*50)
    print("Expected at M=1: P/P0 ≈ 0.528, T/T0 ≈ 0.833")
    print("Expected at M=2.2: P/P0 ≈ 0.093, T/T0 ≈ 0.508")
    print("="*50)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("Testing nozzle geometry...")
    test_geometry()
    
    print("\n\nTesting area-Mach solver...")
    test_area_mach()
    
    print("\n\nTesting isentropic relations...")
    test_isentropic_relations()