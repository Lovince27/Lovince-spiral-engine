#!/usr/bin/env python3
"""
final.py

A professional, executable Python script implementing the UltimateLovince model,
comparing it with a standard scientific model, performing self-validation,
and visualizing results.

Model Definition:
    UltimateLovince Model: U_n = (n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)
    Standard Scientific Model: S_n = (E / r^2) * exp(k * n)

Units:
    U_n, S_n: W/m² (energy flux)

Author: AI Assistant (No copyright)
Date: 2025-04-27
"""

import argparse
import math
import sys
import matplotlib.pyplot as plt

def ultimate_lovince_model(n: int, E: float, r: float) -> float:
    """
    Calculate the UltimateLovince State (U_n) based on the custom model.

    Formula:
        U_n = (n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)

    Args:
        n (int): Step number (must be a positive integer)
        E (float): Energy input in Watts (must be non-negative)
        r (float): Distance in meters (must be positive)

    Returns:
        float: U_n in W/m² (energy flux)

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Invalid step number n={n}. Must be a positive integer greater than 0.")
    if E < 0:
        raise ValueError(f"Invalid energy E={E} W. Must be non-negative.")
    if r <= 0:
        raise ValueError(f"Invalid distance r={r} m. Must be positive.")

    quadratic_term = n**2 + n
    cube_term = n**3
    sequence_term = n * (n + 1) / 2
    inverse_square_term = E / (r**2)

    U_n = (quadratic_term + cube_term) * inverse_square_term * sequence_term
    return U_n

def standard_scientific_model(n: int, E: float, r: float, k: float = 0.1) -> float:
    """
    A standard scientific model for comparison: Exponential growth modulated by inverse square law.

    Formula:
        S_n = (E / r^2) * exp(k * n)

    Args:
        n (int): Step number (positive integer)
        E (float): Energy input in Watts (non-negative)
        r (float): Distance in meters (positive)
        k (float): Growth rate constant (default 0.1)

    Returns:
        float: S_n in W/m² (energy flux)

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Invalid step number n={n}. Must be a positive integer greater than 0.")
    if E < 0:
        raise ValueError(f"Invalid energy E={E} W. Must be non-negative.")
    if r <= 0:
        raise ValueError(f"Invalid distance r={r} m. Must be positive.")
    if k < 0:
        raise ValueError(f"Invalid growth rate k={k}. Must be non-negative.")

    return (E / (r**2)) * math.exp(k * n)

def self_check():
    """
    Perform internal self-checks to verify model correctness and consistency.
    """
    print("Performing self-checks for UltimateLovince Model...")

    # Test valid inputs for UltimateLovince
    try:
        val = ultimate_lovince_model(1, 100, 1)
        assert isinstance(val, float), "UltimateLovince output should be float"
        expected = (1**2 + 1 + 1**3) * (100 / 1**2) * (1 * 2 / 2)
        assert abs(val - expected) < 1e-5, f"Expected U_1 = {expected}, got {val}"
        print(f"Self-check passed: ultimate_lovince_model(1, 100, 1) = {val:.2f} W/m²")
    except Exception as e:
        print(f"Self-check failed for UltimateLovince: {e}")
        sys.exit(1)

    # Test valid inputs for Standard Scientific Model
    try:
        val = standard_scientific_model(1, 100, 1)
        assert isinstance(val, float), "Scientific model output should be float"
        expected = (100 / 1**2) * math.exp(0.1 * 1)
        assert abs(val - expected) < 1e-5, f"Expected S_1 = {expected:.2f}, got {val:.2f}"
        print(f"Self-check passed: standard_scientific_model(1, 100, 1) = {val:.2f} W/m²")
    except Exception as e:
        print(f"Self-check failed for Standard Scientific: {e}")
        sys.exit(1)

    # Test invalid inputs for UltimateLovince
    try:
        ultimate_lovince_model(0, 100, 1)
        print("Self-check failed: Did not raise error for n=0 in UltimateLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for n=0: {e}")

    try:
        ultimate_lovince_model(1, -10, 1)
        print("Self-check failed: Did not raise error for negative E in UltimateLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for negative E: {e}")

    try:
        ultimate_lovince_model(1, 100, 0)
        print("Self-check failed: Did not raise error for r=0 in UltimateLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for r=0: {e}")

    print("All self-checks passed.\n")

def plot_models(n_max, E, r_values, log_scale=False, output_file=None):
    """
    Plot UltimateLovince and Standard Scientific models for comparison.

    Args:
        n_max (int): Maximum step number
        E (float): Energy input in Watts
        r_values (list of float): Distances in meters
        log_scale (bool): Whether to use logarithmic scale on y-axis
        output_file (str or None): Path to save the plot image
    """
    plt.figure(figsize=(12, 8))
    n_range = range(1, n_max + 1)
    colors = ['blue', 'green', 'purple', 'orange']  # Cycle through colors for clarity

    for idx, r in enumerate(r_values):
        color = colors[idx % len(colors)]
        U_vals = [ultimate_lovince_model(n, E, r) for n in n_range]
        S_vals = [standard_scientific_model(n, E, r) for n in n_range]

        plt.plot(n_range, U_vals, marker='o', linestyle='-', color=color,
                 label=f'UltimateLovince (r={r} m)', alpha=0.8)
        plt.plot(n_range, S_vals, marker='x', linestyle='--', color=color,
                 label=f'Standard Model (r={r} m)', alpha=0.6)

    plt.title("UltimateLovince Model vs Standard Scientific Model", fontsize=14, pad=15)
    plt.xlabel("Step (n)", fontsize=12)
    plt.ylabel("Intensity (W/m²)", fontsize=12)
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Intensity (W/m²) - Log Scale", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to '{output_file}'")
    else:
        plt.show()

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="UltimateLovince Model Simulator with Scientific Comparison\n"
                    "Model: U_n = (n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)\n"
                    "Comparison: S_n = (E / r^2) * exp(k * n)"
    )
    parser.add_argument('--n_max', type=int, default=10,
                        help='Maximum step number (default: 10)')
    parser.add_argument('--E', type=float, default=100.0,
                        help='Energy input in Watts (default: 100)')
    parser.add_argument('--r', type=float, nargs='+', default=[1.0, 2.0],
                        help='Distance(s) in meters (default: 1.0 2.0)')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use logarithmic scale for y-axis')
    parser.add_argument('--output_file', type=str,
                        help='Save the plot to a file (e.g., plot.png)')
    parser.add_argument('--show_values', action='store_true',
                        help='Print U_n and S_n values to terminal')

    return parser.parse_args()

def main():
    args = parse_args()

    print("\nUltimateLovince Model Simulator")
    print(f"Parameters: n_max={args.n_max}, E={args.E} W, r={args.r} m")
    print(f"Log scale: {'ON' if args.log_scale else 'OFF'}")
    if args.output_file:
        print(f"Output plot file: {args.output_file}")
    print()

    # Run self-checks before proceeding
    self_check()

    # Display values if requested
    if args.show_values:
        print("Step-wise values:\n")
        print(f"{'n':>3} | {'r (m)':>6} | {'U_n (UltimateLovince)':>22} | {'S_n (Standard Model)':>18}")
        print("-" * 60)
        for n in range(1, args.n_max + 1):
            for r in args.r:
                U_n = ultimate_lovince_model(n, args.E, r)
                S_n = standard_scientific_model(n, args.E, r)
                print(f"{n:3d} | {r:6.2f} | {U_n:22.4e} | {S_n:18.4e}")
        print()

    # Plot models
    plot_models(args.n_max, args.E, args.r, args.log_scale, args.output_file)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
final.py - UltimateLovince Model vs. Standard Scientific Model

A professional implementation distinguishing:
1. MATHEMATICAL MODEL (abstract computation)
2. SCIENTIFIC MODEL (physics-based simulation)

Key Components:
- Mathematical Core (Lovince Formula)
- Physics Engine (Inverse-square Law + Exponential Decay)
- Validation & Visualization
"""

import argparse
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

# ====================== MATHEMATICAL CORE ======================
def lovince_sequence(n: int, E: float, r: float) -> float:
    """
    PURE MATHEMATICAL MODEL: Polynomial growth with combinatorial terms.
    Formula: Uₙ = (n² + n + n³) × (E/r²) × (n(n+1)/2)
    
    Args:
        n: Step number (positive integer)
        E: Arbitrary energy scalar (mathematical only)
        r: Arbitrary distance scalar (mathematical only)
    
    Returns:
        Computed value (unitless in pure math context)
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be positive integer")
    
    polynomial_term = n**2 + n + n**3
    combinatorial_term = n * (n + 1) / 2
    return polynomial_term * (E / r**2) * combinatorial_term

# ====================== SCIENTIFIC MODEL ======================
def scientific_energy_flux(E: float, r: float, k: float = 0.1) -> float:
    """
    PHYSICS-BASED MODEL: Exponential decay with inverse-square law.
    Formula: S = (E/r²) × exp(-k×r) 
    
    Args:
        E: Energy in Joules (physically meaningful)
        r: Distance in meters (physically meaningful)
        k: Absorption coefficient (1/meter)
    
    Returns:
        Energy flux in W/m² (scientifically valid units)
    """
    if E < 0 or r <= 0 or k < 0:
        raise ValueError("Physical parameters must be positive")
    
    return (E / r**2) * math.exp(-k * r)

# ====================== VALIDATION ======================
def validate_models():
    """Test cases showing math vs. science behavior"""
    print("\n=== Validation Results ===")
    
    # Mathematical validation
    math_test = lovince_sequence(3, 100, 1)
    print(f"Math Model (n=3): {math_test:.2f} (arbitrary units)")
    
    # Scientific validation
    physics_test = scientific_energy_flux(100, 3)
    print(f"Physics Model (r=3m): {physics_test:.2e} W/m²")

# ====================== VISUALIZATION ======================
def plot_comparison(max_n: int = 10, E: float = 100, r: float = 1):
    """Comparative plot between abstract math and physical science"""
    n_values = range(1, max_n + 1)
    
    # Mathematical curve
    math_values = [lovince_sequence(n, E, r) for n in n_values]
    
    # Scientific curve (converted to comparable scale)
    physics_values = [scientific_energy_flux(E, r) * n for n in n_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, math_values, 'o-', label='Mathematical Model (Lovince)')
    plt.plot(n_values, physics_values, 's--', label='Scientific Model (Physics)')
    
    plt.title("Abstract Math vs. Physical Science", pad=20)
    plt.xlabel("Step (n) / Scaled Distance")
    plt.ylabel("Relative Intensity")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.show()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare abstract mathematical modeling vs. physical scientific principles"
    )
    parser.add_argument('--max_n', type=int, default=10, 
                       help='Maximum step number for math model')
    parser.add_argument('--energy', type=float, default=100,
                       help='Energy scalar (Joules for physics)')
    parser.add_argument('--distance', type=float, default=1,
                       help='Distance scalar (meters for physics)')
    
    args = parser.parse_args()
    
    # Run validation tests
    validate_models()
    
    # Generate comparative plot
    plot_comparison(
        max_n=args.max_n,
        E=args.energy,
        r=args.distance
    )


#!/usr/bin/env python3
"""
reality_model.py - Where Mathematical Abstraction Meets Physical Reality

Key Components:
1. MATHEMATICAL CORE: Lovince sequence (abstract growth pattern)
2. PHYSICS ENGINE: Inverse-square law + Quantum scaling
3. REALITY BRIDGE: Combines both with dimensional analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# ====================== MATHEMATICAL CORE ======================
def lovince_math(n: int) -> float:
    """Pure abstract growth formula: (n³ + n² + n) × n(n+1)/2"""
    return (n**3 + n**2 + n) * (n * (n + 1)) / 2

# ====================== PHYSICS ENGINE ======================
def photon_flux(E: float, r: float) -> float:
    """Physical light propagation: E/(4πr²)"""
    return E / (4 * np.pi * r**2)

# ====================== REALITY BRIDGE ======================
def reality_model(n: int, E: float, r: float) -> Tuple[float, float, float]:
    """
    Combines abstract math with physical reality:
    1. Math term → Dimensionless growth factor
    2. Physics term → Energy flux (W/m²)
    3. Reality term → Mathematically scaled physics
    """
    math_term = lovince_math(n)
    physics_term = photon_flux(E, r)
    
    # Bridge equation: [math] × [physics] with dimensional correction
    reality_term = (math_term / 1e10) * physics_term  # 1e10 scales to real-world values
    
    return math_term, physics_term, reality_term

# ====================== VISUALIZATION ======================
def plot_reality(max_n: int = 10, E: float = 100, r: float = 1):
    """Triple plot showing the interaction"""
    n_values = np.arange(1, max_n + 1)
    results = [reality_model(n, E, r) for n in n_values]
    
    math, physics, reality = zip(*results)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Math Plot
    ax1.plot(n_values, math, 'bo-')
    ax1.set_title("Abstract Mathematical Growth")
    ax1.set_ylabel("Lovince Sequence Value")
    ax1.grid(True)
    
    # Physics Plot
    ax2.plot(n_values, physics, 'rs--')
    ax2.set_title("Physical Photon Flux")
    ax2.set_ylabel("Energy (W/m²)")
    ax2.grid(True)
    
    # Reality Plot
    ax3.plot(n_values, reality, 'gD-.')
    ax3.set_title("Reality: Math × Physics")
    ax3.set_xlabel("Step (n) / Distance Scale")
    ax3.set_ylabel("Scaled Reality Metric")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Sample Output
    print("n | Math | Physics (W/m²) | Reality")
    print("----------------------------------")
    for n in range(1, 6):
        m, p, r = reality_model(n, E=100, r=1)
        print(f"{n} | {m:.1f} | {p:.2e} | {r:.2e}")
    
    # Generate Interactive Plot
    plot_reality(max_n=15, E=1000, r=0.5)