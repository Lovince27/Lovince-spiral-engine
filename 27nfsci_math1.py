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