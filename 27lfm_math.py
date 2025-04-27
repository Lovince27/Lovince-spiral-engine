import numpy as np
import sympy as sp

# Define the function based on Lovince's formula
def lovince_formula(x, t, r, theta, a, b, dx_dt):
    # First term: e^(ix) / (x^2 + r^2)
    first_term = (np.exp(1j * x) / (x**2 + r**2))
    
    # Second term: sin(theta) / x
    second_term = np.sin(theta) / x
    
    # Third term: (pi * r^2) / 2
    third_term = (np.pi * r**2) / 2
    
    # Fourth term: dx/dt * integral of cos(theta) from 0 to x
    # We assume cos(theta) does not depend on x, so we can integrate just cos(theta) w.r.t x
    integral_cos = np.cos(theta) * x  # Simple integral of cos(theta)
    fourth_term = dx_dt * integral_cos
    
    # Fifth term: (a^2 + b^2) / 2
    fifth_term = (a**2 + b**2) / 2
    
    # Combining all terms to compute the final result
    result = first_term * second_term + third_term + fourth_term + fifth_term
    
    return result

# Example values for x, t, r, theta, a, b, and dx/dt
x = 2.0  # Example value for x
t = 1.0  # Example value for t (not used directly in formula)
r = 3.0  # Example value for r
theta = np.pi / 4  # Example value for theta (45 degrees)
a = 4.0  # Example value for a
b = 5.0  # Example value for b
dx_dt = 2.0  # Example value for dx/dt

# Calculate the Lovince formula result
result = lovince_formula(x, t, r, theta, a, b, dx_dt)

print("Result of Lovince's formula:", result)

import numpy as np
import math

# Function for Lovince formula
def lovince_formula(x, t, r, theta, a, b, m1, m2):
    """
    Lovince formula:
    L(x, t, r, theta, a, b) = (e^(ix) / (x^2 + r^2)) * (sin(theta) / x) + (pi * r^2 / 2)
                               + (dx/dt) * (cos(theta) * x) + (a^2 + b^2) / 2
                               + (m1 * m2 / r^2) * sin(theta)

    Parameters:
    - x: position (distance)
    - t: time (used for dynamic changes, derivative part)
    - r: radius or another relevant geometric parameter
    - theta: angle in radians
    - a, b: variables for geometric relations
    - m1, m2: masses for gravitational interaction

    Returns:
    - Result of the Lovince formula
    """
    
    # First term: e^(ix) / (x^2 + r^2)
    first_term = (np.exp(1j * x) / (x**2 + r**2))

    # Second term: (sin(theta) / x)
    second_term = np.sin(theta) / x

    # Third term: (pi * r^2 / 2)
    third_term = (np.pi * r**2) / 2

    # Fourth term: (dx/dt) * (cos(theta) * x)
    dx_dt = (x - t)  # assuming a simple linear change in position (velocity approximation)
    fourth_term = dx_dt * np.cos(theta) * x

    # Fifth term: (a^2 + b^2) / 2
    fifth_term = (a**2 + b**2) / 2

    # Sixth term: (m1 * m2 / r^2) * sin(theta)
    sixth_term = (m1 * m2 / r**2) * np.sin(theta)

    # Total result of the Lovince formula
    result = first_term * second_term + third_term + fourth_term + fifth_term + sixth_term
    
    return result

# Example usage
x = 2  # Position
t = 1  # Time (used for velocity approximation)
r = 3  # Radius or another relevant parameter
theta = np.pi / 4  # Angle (45 degrees in radians)
a = 1  # Variable a
b = 2  # Variable b
m1 = 5  # Mass m1
m2 = 10  # Mass m2

# Calculate the Lovince formula result
result = lovince_formula(x, t, r, theta, a, b, m1, m2)

# Output the result
print(f"Lovince formula result: {result}")

#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import argparse
from functools import lru_cache

# Lovince Spiral Sequence: a_n = a_{n-1} + floor(sqrt(n))
def lovince_spiral_sequence(n):
    """Compute Lovince Spiral Sequence up to n terms."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    sequence = [1]  # a_1 = 1
    for i in range(2, n + 1):
        sequence.append(sequence[-1] + math.floor(math.sqrt(i)))
    return sequence

# Lovince Harmony Equation: H_n = sum_{k=1}^n 1 / (k^2 + sqrt(k))
@lru_cache(maxsize=128)
def lovince_harmony(n):
    """Compute Lovince Harmony sum for n terms (cached for performance)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

# Lovince Energy Flux: Phi = E / (r^2 + k * sin(omega * t))
def lovince_energy_flux(E, r, k, omega, t):
    """Compute Lovince Energy Flux at time t."""
    if r <= 0 or E < 0:
        raise ValueError("E must be non-negative, r must be positive")
    denominator = r**2 + k * math.sin(omega * t)
    if abs(denominator) < 1e-10:
        raise ValueError("Denominator too close to zero")
    return E / denominator

# Lovince Universal Model: S_n = a_n * H_n * Phi(t, r)
def lovince_universal_model(n, E, r, k, omega, t):
    """Compute Lovince Universal Model state at step n."""
    a_n = lovince_spiral_sequence(n)[-1]
    H_n = lovince_harmony(n)
    Phi = lovince_energy_flux(E, r, k, omega, t)
    return a_n * H_n * Phi

# Main function to run and visualize the model
def run_lovince_model(n_max, E, r, k, omega, dynamic_t=False):
    """Run Lovince Universal Model and plot results."""
    states = []
    fluxes = []
    n_values = range(1, n_max + 1)
    t_values = [n / 10.0 if dynamic_t else 1.0 for n in n_values]  # Dynamic or fixed time
    
    for n, t in zip(n_values, t_values):
        try:
            S_n = lovince_universal_model(n, E, r, k, omega, t)
            Phi = lovince_energy_flux(E, r, k, omega, t)
            states.append(S_n)
            fluxes.append(Phi)
        except ValueError as e:
            print(f"Error at n={n}: {e}")
            return None
    
    # Visualization: Dual plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot S_n vs n
    ax1.plot(n_values, states, marker='o', linestyle='-', color='b', label='S_n')
    ax1.set_title('Lovince Universal Model: System State')
    ax1.set_xlabel('Step (n)')
    ax1.set_ylabel('S_n (W/m²)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Phi vs t
    ax2.plot(t_values, fluxes, marker='s', linestyle='--', color='r', label='Phi')
    ax2.set_title('Lovince Energy Flux')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phi (W/m²)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return states, t_values, fluxes

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Lovince Universal Model: A powerful dynamic system simulator.")
    parser.add_argument('--n_max', type=int, default=10, help='Maximum step (default: 10)')
    parser.add_argument('--E', type=float, default=100.0, help='Energy in watts (default: 100)')
    parser.add_argument('--r', type=float, default=2.0, help='Distance in meters (default: 2)')
    parser.add_argument('--k', type=float, default=1.0, help='Damping constant (default: 1)')
    parser.add_argument('--omega', type=float, default=math.pi, help='Angular frequency (default: pi)')
    parser.add_argument('--dynamic_t', action='store_true', help='Use dynamic time (t=n/10) instead of t=1')
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    
    print("Running Lovince Universal Model...")
    print(f"Parameters: n_max={args.n_max}, E={args.E}, r={args.r}, k={args.k}, omega={args.omega}, dynamic_t={args.dynamic_t}")
    
    results = run_lovince_model(args.n_max, args.E, args.r, args.k, args.omega, args.dynamic_t)
    
    if results:
        states, t_values, fluxes = results
        print("\nResults:")
        for n, t, S_n, Phi in zip(range(1, args.n_max + 1), t_values, states, fluxes):
            print(f"n={n}, t={t:.2f}s: S_n ≈ {S_n:.2f} W/m², Phi ≈ {Phi:.2f} W/m²")


import math

# Lovince Spiral Sequence
def lovince_spiral_sequence(n):
    sequence = [1]  # a_1 = 1
    for i in range(2, n + 1):
        sequence.append(sequence[-1] + math.floor(math.sqrt(i)))
    return sequence

# Lovince Harmony
def lovince_harmony(n):
    return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

# Lovince Energy Flux
def lovince_energy_flux(E, r, k, omega, t):
    return E / (r**2 + k * math.sin(omega * t))

# Gravitational Force Contribution
def gravitational_force(G, M1, M2, r):
    return (G * M1 * M2) / r**2

# Entropy Function
def entropy(t):
    return math.log(t + 1)

# Core Lovince Universal Model Function
def lovince_universal_model(n, E, r, k, omega, t, G, M1, M2):
    a_n = lovince_spiral_sequence(n)[-1]
    H_n = lovince_harmony(n)
    Phi = lovince_energy_flux(E, r, k, omega, t)
    grav_force = gravitational_force(G, M1, M2, r)
    entropy_factor = 1 / math.sqrt(t**2 + entropy(t))
    
    S_n = (a_n * H_n * Phi) * (1 + grav_force) * entropy_factor
    return S_n

# Test the function with sample values
n = 10
E = 100
r = 5
k = 0.5
omega = math.pi
t = 2
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M1 = 1.0e24  # Mass of object 1 (kg)
M2 = 1.0e24  # Mass of object 2 (kg)

S_n = lovince_universal_model(n, E, r, k, omega, t, G, M1, M2)
print(f"Lovince Universal Model S_n = {S_n:.5f}")