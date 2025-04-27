#!/usr/bin/env python3
"""
final.py - The Fundamental Equation of Reality (Improved Version)

Concept:
1. Reality Equation: 0 (void) × ∞ (quantum) = All possibilities
   - Reinterpreted as a probabilistic superposition of states rather than NaN.
2. Recursive Unfolding: Reality emerges from zero through iterative possibilities.
3. Quantum Manifestation: The Big Bang collapses the equation into observable phenomena.
4. Model Integration: ZeroQuantumLovince Model adapted to reflect this paradigm.

Model Definition:
    ZeroQuantumLovince Model: ZQ_n = [(n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)] * reality(n) * (1 + big_bang_factor)
    Where:
        - n: Step number (dimensionless)
        - E: Energy input in Watts (W)
        - r: Distance in meters (m)
        - big_bang_factor: Quantum fluctuation from QuantumUniverse

Units:
    ZQ_n: W/m² (energy flux modulated by quantum effects)

Author: AI Assistant
Date: 2025-04-27 (Improved Version)
"""

import argparse
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# === CORE EQUATION ===
def reality(n: int, max_depth: int = 10) -> float:
    """
    Recursive function representing the unfolding of reality from 0 × ∞.

    Args:
        n (int): Step number (non-negative)
        max_depth (int): Maximum recursion depth to prevent overflow (default: 10)

    Returns:
        float: Probability-like value representing reality's state

    Raises:
        ValueError: If n is negative or max_depth is invalid.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Invalid step number n={n}. Must be non-negative.")
    if not isinstance(max_depth, int) or max_depth <= 0:
        raise ValueError(f"Invalid max_depth={max_depth}. Must be positive.")

    if n == 0:
        return 0.0  # Void state
    if n > max_depth:
        return 1.0 / max_depth  # Bound the recursion
    # To avoid division by zero, handle reality(n-1) == 0
    prev = reality(n - 1, max_depth)
    if prev == 0:
        return 1.0  # Avoid division by zero by returning 1
    return 1.0 / prev

# === QUANTUM MANIFESTATION ===
class QuantumUniverse:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.void = 0.0
        self.infinity_potential = np.random.uniform(1e6, 1e9)  # Simulate infinite range

    def big_bang(self) -> float:
        """
        Collapse the reality equation (0 × ∞) into observable existence.

        Returns a random value from the void-infinity superposition.

        Returns:
            float: Simulated energy or state value
        """
        fluctuation = np.random.uniform(0, 1)
        return self.void * self.infinity_potential * fluctuation  # Scaled random outcome (always zero here)

# === ZEROQUANTUM LOVINCE MODEL ===
def zero_quantum_lovince_model(n: int, E: float, r: float, qu: QuantumUniverse) -> float:
    """
    Calculate the ZeroQuantumLovince State (ZQ_n) integrating reality and quantum uncertainty.

    Formula:
        ZQ_n = [(n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)] * reality(n) * (1 + big_bang_factor)

    Args:
        n (int): Step number (must be a positive integer)
        E (float): Energy input in Watts (must be non-negative)
        r (float): Distance in meters (must be positive)
        qu (QuantumUniverse): QuantumUniverse instance for quantum fluctuations

    Returns:
        float: ZQ_n in W/m² (energy flux with quantum and reality modulation)

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
    reality_factor = reality(n)
    big_bang_factor = qu.big_bang()

    base_value = (quadratic_term + cube_term) * inverse_square_term * sequence_term
    ZQ_n = base_value * reality_factor * (1 + big_bang_factor)
    return max(0, ZQ_n)  # Ensure non-negative output

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
    print("Performing self-checks for ZeroQuantumLovince Model...")

    # Test reality function
    try:
        val = reality(0)
        assert val == 0.0, f"Expected reality(0) = 0, got {val}"
        print(f"Self-check passed: reality(0) = {val}")
    except Exception as e:
        print(f"Self-check failed for reality: {e}")
        sys.exit(1)

    # Test ZeroQuantumLovince
    try:
        qu = QuantumUniverse(seed=42)
        val = zero_quantum_lovince_model(1, 100, 1, qu)
        assert isinstance(val, float), "ZeroQuantumLovince output should be float"
        assert val >= 0, f"Expected ZQ_1 non-negative, got {val}"
        print(f"Self-check passed: zero_quantum_lovince_model(1, 100, 1) = {val:.2f} W/m²")
    except Exception as e:
        print(f"Self-check failed for ZeroQuantumLovince: {e}")
        sys.exit(1)

    # Test Standard Scientific Model
    try:
        val = standard_scientific_model(1, 100, 1)
        assert isinstance(val, float), "Scientific model output should be float"
        expected = (100 / 1**2) * math.exp(0.1 * 1)
        assert abs(val - expected) < 1e-5, f"Expected S_1 = {expected:.2f}, got {val:.2f}"
        print(f"Self-check passed: standard_scientific_model(1, 100, 1) = {val:.2f} W/m²")
    except Exception as e:
        print(f"Self-check failed for Standard Scientific: {e}")
        sys.exit(1)

    # Test invalid inputs
    try:
        zero_quantum_lovince_model(0, 100, 1, qu)
        print("Self-check failed: Did not raise error for n=0 in ZeroQuantumLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for n=0: {e}")

    try:
        zero_quantum_lovince_model(1, -10, 1, qu)
        print("Self-check failed: Did not raise error for negative E in ZeroQuantumLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for negative E: {e}")

    try:
        zero_quantum_lovince_model(1, 100, 0, qu)
        print("Self-check failed: Did not raise error for r=0 in ZeroQuantumLovince")
        sys.exit(1)
    except ValueError as e:
        print(f"Self-check passed: Correctly raised error for r=0: {e}")

    print("All self-checks passed.\n")

def plot_models(n_max, E, r_values, qu, log_scale=False, output_file=None):
    """
    Plot ZeroQuantumLovince and Standard Scientific models for comparison.

    Args:
        n_max (int): Maximum step number
        E (float): Energy input in Watts
        r_values (list of float): Distances in meters
        qu (QuantumUniverse): QuantumUniverse instance for quantum fluctuations
        log_scale (bool): Whether to use logarithmic scale on y-axis
        output_file (str or None): Path to save the plot image
    """
    plt.figure(figsize=(12, 8))
    n_range = range(1, n_max + 1)
    colors = ['blue', 'green', 'purple', 'orange']  # Cycle through colors

    for idx, r in enumerate(r_values):
        color = colors[idx % len(colors)]
        ZQ_vals = [zero_quantum_lovince_model(n, E, r, qu) for n in n_range]
        S_vals = [standard_scientific_model(n, E, r) for n in n_range]

        plt.plot(n_range, ZQ_vals, marker='o', linestyle='-', color=color,
                 label=f'ZeroQuantumLovince (r={r} m)', alpha=0.8)
        plt.plot(n_range, S_vals, marker='x', linestyle='--', color=color,
                 label=f'Standard Model (r={r} m)', alpha=0.6)

    plt.title("ZeroQuantumLovince Model vs Standard Scientific Model", fontsize=16, pad=15)
    plt.xlabel("Step (n)", fontsize=14)
    plt.ylabel("Intensity (W/m²)", fontsize=14)
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Intensity (W/m²) - Log Scale", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend on right

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
        description="ZeroQuantumLovince Model Simulator with Scientific Comparison\n"
                    "Model: ZQ_n = [(n^2 + n + n^3) * (E / r^2) * (n(n+1) / 2)] * reality(n) * (1 + big_bang_factor)\n"
                    "Comparison: S_n = (E / r^2) * exp(k * n)\n\n"
                    "Example usage:\n"
                    "  python final.py --n_max 15 --E 150 --r 1.0 3.0 --log_scale --output_file plot.png --seed 42",
        formatter_class=argparse.RawTextHelpFormatter
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
                        help='Print ZQ_n and S_n values to terminal')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility (default: None)')

    return parser.parse_args()

def main():
    args = parse_args()

    print("\nZeroQuantumLovince Model Simulator - The Fundamental Equation of Reality")
    print(f"Parameters: n_max={args.n_max}, E={args.E} W, r={args.r} m")
    print(f"Log scale: {'ON' if args.log_scale else 'OFF'}")
    if args.output_file:
        print(f"Output plot file: {args.output_file}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # Initialize QuantumUniverse with seed for reproducibility
    qu = QuantumUniverse(seed=args.seed)

    # Run self-checks before proceeding
    self_check()

    # Display Reality Equation
    print("REALITY EQUATION:")
    print(f"0 × ∞ = {qu.big_bang():.2e} (Simulated infinite potential, always zero here)\n")

    print("MATHEMATICAL RECURSION:")
    for n in range(3):
        val = reality(n)
        print(f"reality({n}) = {val:.4f}")

    # Display values if requested
    if args.show_values:
        print("\nStep-wise values:\n")
        print(f"{'n':>3} | {'r (m)':>6} | {'ZQ_n (ZeroQuantumLovince)':>25} | {'S_n (Standard Model)':>20}")
        print("-" * 70)
        for n in range(1, args.n_max + 1):
            for r in args.r:
                ZQ_n = zero_quantum_lovince_model(n, args.E, r, qu)
                S_n = standard_scientific_model(n, args.E, r)
                print(f"{n:3d} | {r:6.2f} | {ZQ_n:25.4e} | {S_n:20.4e}")
        print()

    # Plot models
    plot_models(args.n_max, args.E, args.r, qu, args.log_scale, args.output_file)

if __name__ == "__main__":
    main()
