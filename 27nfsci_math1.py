#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import argparse
from functools import lru_cache

# Lovince Spiral Sequence: a_n = a_{n-1} + floor(sqrt(n))
def lovince_spiral_sequence(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    sequence = [1]
    for i in range(2, n + 1):
        sequence.append(sequence[-1] + math.floor(math.sqrt(i)))
    return sequence

# Lovince Harmony Equation: H_n = sum_{k=1}^n 1 / (k^2 + sqrt(k))
@lru_cache(maxsize=128)
def lovince_harmony(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

# Lovince Energy Flux: Phi = E / (r^2 + k * sin(omega * t))
def lovince_energy_flux(E, r, k, omega, t):
    if r <= 0 or E < 0:
        raise ValueError("r must be positive and E must be non-negative")
    denominator = r**2 + k * math.sin(omega * t)
    if abs(denominator) < 1e-10:
        raise ValueError("Denominator too close to zero")
    return E / denominator

# Gravitational Force: F = G * M1 * M2 / r^2
def gravitational_force(G, M1, M2, r):
    if r <= 0:
        raise ValueError("Distance r must be positive to avoid infinite force")
    return (G * M1 * M2) / r**2

# Entropy Term: S = log(t + 1)
def entropy_term(t):
    if t < 0:
        raise ValueError("Time t must be non-negative")
    return math.log(t + 1)

# Lovince Final Universal Model: Powerful Equation
def lovince_final_model(n, E, r, k, omega, t, G, M1, M2):
    a_n = lovince_spiral_sequence(n)[-1]
    H_n = lovince_harmony(n)
    Phi = lovince_energy_flux(E, r, k, omega, t)
    F_g = gravitational_force(G, M1, M2, r)
    entropy = entropy_term(t)
    
    if t**2 + entropy <= 0:
        raise ValueError("Invalid denominator for entropy adjustment")
    
    adjustment_factor = 1 / math.sqrt(t**2 + entropy)
    
    S_n = (a_n * H_n * Phi) * (1 + F_g) * adjustment_factor
    return S_n

# Main function to run and visualize
def run_lovince_final(n_max, E, r, k, omega, G, M1, M2, dynamic_t=False):
    states = []
    n_values = range(1, n_max + 1)
    t_values = [n / 10.0 if dynamic_t else 1.0 for n in n_values]
    
    for n, t in zip(n_values, t_values):
        try:
            S_n = lovince_final_model(n, E, r, k, omega, t, G, M1, M2)
            states.append(S_n)
        except ValueError as e:
            print(f"Error at n={n}: {e}")
            return None
    
    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, states, marker='o', linestyle='-', color='purple', label='Lovince Final Model (S_n)')
    plt.title('Lovince Universal Final Model: System State')
    plt.xlabel('Step (n)')
    plt.ylabel('System State S_n (W/m²)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return states

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Lovince Universal Final Model Simulator.")
    parser.add_argument('--n_max', type=int, default=10, help='Maximum step n (default: 10)')
    parser.add_argument('--E', type=float, default=100.0, help='Energy (W) (default: 100)')
    parser.add_argument('--r', type=float, default=2.0, help='Distance (m) (default: 2)')
    parser.add_argument('--k', type=float, default=1.0, help='Damping constant (default: 1)')
    parser.add_argument('--omega', type=float, default=math.pi, help='Angular frequency (default: pi)')
    parser.add_argument('--G', type=float, default=6.67430e-11, help='Gravitational constant (default: 6.67430e-11)')
    parser.add_argument('--M1', type=float, default=5.972e24, help='Mass 1 (kg) (default: Earth mass)')
    parser.add_argument('--M2', type=float, default=7.348e22, help='Mass 2 (kg) (default: Moon mass)')
    parser.add_argument('--dynamic_t', action='store_true', help='Use dynamic time (t=n/10)')
    return parser.parse_args()

# Main Execution
if __name__ == "__main__":
    args = parse_args()
    
    print("Running Lovince Universal Final Model...")
    print(f"Parameters: n_max={args.n_max}, E={args.E}, r={args.r}, k={args.k}, omega={args.omega}, G={args.G}, M1={args.M1}, M2={args.M2}, dynamic_t={args.dynamic_t}")
    
    results = run_lovince_final(args.n_max, args.E, args.r, args.k, args.omega, args.G, args.M1, args.M2, args.dynamic_t)
    
    if results:
        print("\nResults:")
        for n, S_n in zip(range(1, args.n_max + 1), results):
            print(f"n={n}: S_n ≈ {S_n:.4e} W/m²")


import math
from functools import lru_cache

def lovince_spiral_sequence(n):
    sequence = [1]
    for i in range(2, n + 1):
        sequence.append(sequence[-1] + math.floor(math.sqrt(i)))
    return sequence

@lru_cache(maxsize=128)
def lovince_harmony(n):
    return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

def lovince_energy_flux(E, r, k, omega, t):
    denominator = r**2 + k * math.sin(omega * t)
    if abs(denominator) < 1e-10:
        raise ValueError("Denominator too close to zero")
    return E / denominator

def lovince_universal_model(n, E, r, k, omega, t):
    a_n = lovince_spiral_sequence(n)[-1]
    H_n = lovince_harmony(n)
    Phi = lovince_energy_flux(E, r, k, omega, t)
    return a_n * H_n * Phi

def lovince_final_extended_model(n, E, r, k, omega, t, theta, phi, epsilon):
    S_n = lovince_universal_model(n, E, r, k, omega, t)
    quantum_boost = 1 + (math.sin(theta) / math.cos(theta + phi))
    gravity_adjustment = 1 / (r**2 + epsilon * math.sin(omega * t))
    return S_n * quantum_boost * gravity_adjustment


import numpy as np
import matplotlib.pyplot as plt

# Constants for the formula
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972e24  # Mass of the central body (Earth) in kg
alpha = 0.15  # Lovince correction factor (adjusted)
beta = 2.5  # Damping exponent (adjusted)
omega = 1.2  # Angular frequency in rad/s (adjusted)

# Function to calculate the Lovince energy
def lovince_energy(m, v, r, t):
    """Calculate energy using Lovince Energy Dynamics Formula"""
    # Classical kinetic and potential energy
    classical_energy = 0.5 * m * v**2 + (G * M * m) / r
    # Correction factor for quantum-gravity and time dynamics
    correction_factor = 1 + (alpha * np.sin(omega * t)) / r**beta
    return classical_energy * correction_factor

# Parameters
m = 5.0  # Mass of object in kg (adjusted for better visualization)
v = 5000  # Velocity in m/s (adjusted for higher speed)
t_values = np.linspace(0, 20, 100)  # Time from 0 to 20 seconds (adjusted for longer time span)
r_values = np.linspace(1e7, 1e9, 100)  # Distance from 10 million to 1 billion meters (adjusted)

# Create a grid for the calculations
R, T = np.meshgrid(r_values, t_values)
E = lovince_energy(m, v, R, T)  # Calculate energy at each point in the grid

# Plotting Energy vs Distance vs Time
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T, E, cmap='plasma')

# Labels and title
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Energy (Joules)')
ax.set_title('Energy vs Distance vs Time (Lovince Energy Dynamics)')

# Show plot
plt.show()

#!/usr/bin/env python3
"""
final.py

A professional, executable Python script implementing the UltimateLovince model,
comparing it with a standard scientific model, performing self-validation,
and visualizing results.

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
        n (int): Step number (must be positive integer)
        E (float): Energy input in Watts (must be non-negative)
        r (float): Distance in meters (must be positive)

    Returns:
        float: U_n in W/m^2

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Invalid step number n={n}. Must be a positive integer.")
    if E < 0:
        raise ValueError(f"Energy E={E} must be non-negative.")
    if r <= 0:
        raise ValueError(f"Distance r={r} must be positive.")

    quadratic_term = n**2 + n
    cube_term = n**3
    sequence_term = n * (n + 1) / 2
    inverse_square_term = E / (r**2)

    U_n = (quadratic_term + cube_term) * inverse_square_term * sequence_term
    return U_n


def standard_scientific_model(n: int, E: float, r: float, k: float = 0.1) -> float:
    """
    A standard scientific model for comparison:
    Exponential growth modulated by inverse square law.

    Formula:
        S_n = (E / r^2) * exp(k * n)

    Args:
        n (int): Step number (positive integer)
        E (float): Energy input in Watts (non-negative)
        r (float): Distance in meters (positive)
        k (float): Growth rate constant (default 0.1)

    Returns:
        float: S_n in W/m^2
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"Invalid step number n={n}. Must be a positive integer.")
    if E < 0:
        raise ValueError(f"Energy E={E} must be non-negative.")
    if r <= 0:
        raise ValueError(f"Distance r={r} must be positive.")
    if k < 0:
        raise ValueError(f"Growth rate k={k} must be non-negative.")

    return (E / (r**2)) * math.exp(k * n)


def self_check():
    """
    Perform internal self-checks to verify model correctness and consistency.
    """
    print("Performing self-checks...")

    # Test valid inputs
    try:
        val = ultimate_lovince_model(1, 100, 1)
        assert isinstance(val, float), "Output should be float"
        print(f"Self-check passed: ultimate_lovince_model(1,100,1) = {val:.2f}")
    except Exception as e:
        print(f"Self-check failed: {e}")
        sys.exit(1)

    # Test invalid inputs
    try:
        ultimate_lovince_model(0, 100, 1)
        print("Self-check failed: Did not raise error for n=0")
        sys.exit(1)
    except ValueError:
        print("Self-check passed: correctly raised error for n=0")

    try:
        ultimate_lovince_model(1, -10, 1)
        print("Self-check failed: Did not raise error for negative E")
        sys.exit(1)
    except ValueError:
        print("Self-check passed: correctly raised error for negative E")

    try:
        ultimate_lovince_model(1, 100, 0)
        print("Self-check failed: Did not raise error for r=0")
        sys.exit(1)
    except ValueError:
        print("Self-check passed: correctly raised error for r=0")

    print("All self-checks passed.\n")


def plot_models(n_max, E, r_values, log_scale=False, output_file=None):
    """
    Plot UltimateLovince and standard scientific models for comparison.

    Args:
        n_max (int): Maximum step number
        E (float): Energy input in Watts
        r_values (list of float): Distances in meters
        log_scale (bool): Whether to use logarithmic scale on y-axis
        output_file (str or None): Path to save the plot image
    """
    plt.figure(figsize=(12, 8))
    n_range = range(1, n_max + 1)

    for r in r_values:
        U_vals = []
        S_vals = []
        for n in n_range:
            U_vals.append(ultimate_lovince_model(n, E, r))
            S_vals.append(standard_scientific_model(n, E, r))

        plt.plot(n_range, U_vals, marker='o', linestyle='-', label=f'UltimateLovince (r={r}m)')
        plt.plot(n_range, S_vals, marker='x', linestyle='--', label=f'Scientific Model (r={r}m)')

    plt.title("UltimateLovince Model vs Standard Scientific Model")
    plt.xlabel("Step (n)")
    plt.ylabel("Intensity (W/m²)")
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Intensity (W/m²) - Log Scale")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to '{output_file}'")
    else:
        plt.show()


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="UltimateLovince Model Simulator with scientific comparison"
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

    print(f"\nUltimateLovince Model Simulator")
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
        print(f"{'n':>3} | {'r (m)':>6} | {'U_n (UltimateLovince)':>22} | {'S_n (Scientific)':>18}")
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
