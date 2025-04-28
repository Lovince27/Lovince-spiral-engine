import numpy as np

# Constants
h = 6.626e-34  # Planck's constant (J·s)
k = 1.0  # Damping constant (arbitrary value, adjust as needed)
omega = 1.2  # Angular frequency (rad/s)
alpha = 0.15  # Lovince correction factor
beta = 2.5  # Damping exponent (arbitrary value)
t = np.linspace(0, 20, 1000)  # Time array (0 to 20 seconds)

# Photon and Biophoton Frequencies (arbitrary values, adjust based on context)
nu_photon = 5e14  # Frequency of photon (Hz)
nu_biophoton = 1e13  # Frequency of biophoton (Hz)

# Spiral Sequence (a_n)
def lovince_spiral_sequence(n):
    a_n = np.zeros(n)
    a_n[0] = 1  # Starting value (a1)
    for i in range(1, n):
        a_n[i] = a_n[i-1] + np.floor(np.sqrt(i))  # a_n = a_{n-1} + floor(sqrt(n))
    return a_n

# Harmony Equation (H_n)
def lovince_harmony_equation(n):
    H_n = np.zeros(n)
    for k in range(1, n+1):
        H_n[k-1] = np.sum(1 / (k**2 + np.sqrt(k)))  # H_n = sum(1/(k^2 + sqrt(k)))
    return H_n

# Energy Flux (Photon + Biophoton)
def energy_flux(r, t, nu_photon, nu_biophoton):
    combined_energy_flux = (h * (nu_photon + nu_biophoton)) / (r**2 + k * np.sin(omega * t))
    return combined_energy_flux

# Quantum-Gravity Correction (Q(t, θ, φ, ε))
def quantum_gravity_correction(t, theta, phi, epsilon):
    return epsilon * np.cos(theta) * np.sin(phi) + np.sqrt(epsilon) / (1 + np.cos(theta + phi))

# Lovince Universal Model (S_n)
def lovince_universal_model(n, r, t, theta, phi, epsilon):
    a_n = lovince_spiral_sequence(n)
    H_n = lovince_harmony_equation(n)
    
    S_n = np.zeros(n)
    for i in range(n):
        flux = energy_flux(r[i], t, nu_photon, nu_biophoton)
        Q = quantum_gravity_correction(t, theta, phi, epsilon)
        S_n[i] = a_n[i] * H_n[i] * flux * Q  # Final system state
    
    return S_n

# Example Parameters
n = 100  # Number of steps (n can be adjusted)
r = np.linspace(1e7, 1e9, n)  # Distance range (10 million to 1 billion meters)
theta = np.pi / 4  # Example angle (45 degrees)
phi = np.pi / 3  # Example angle (60 degrees)
epsilon = 0.1  # Example epsilon for quantum-gravity correction

# Compute the Lovince Universal Model
S_n = lovince_universal_model(n, r, t, theta, phi, epsilon)

# Plotting the result (optional)
import matplotlib.pyplot as plt

plt.plot(r, S_n)
plt.xlabel('Distance (r) in meters')
plt.ylabel('System State (S_n)')
plt.title('Lovince Universal Model: System State vs Distance')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 6.626e-34  # Planck's constant (J·s)
k_damping = 1.0  # Damping constant (rename to avoid conflict)
omega = 1.2  # Angular frequency (rad/s)
alpha = 0.15  # Lovince correction factor (unused in current code)
beta = 2.5  # Damping exponent (unused in current code)

# Time array (0 to 20 seconds)
t = np.linspace(0, 20, 1000)

# Photon and Biophoton Frequencies (Hz)
nu_photon = 5e14
nu_biophoton = 1e13

# Spiral Sequence (a_n)
def lovince_spiral_sequence(n):
    a_n = np.zeros(n)
    a_n[0] = 1
    for i in range(1, n):
        a_n[i] = a_n[i-1] + np.floor(np.sqrt(i))
    return a_n

# Harmony Equation (H_n) - cumulative sums
def lovince_harmony_equation(n):
    H_n = np.zeros(n)
    for i in range(1, n+1):
        k_vals = np.arange(1, i+1)
        H_n[i-1] = np.sum(1 / (k_vals**2 + np.sqrt(k_vals)))
    return H_n

# Energy Flux at a specific time point (scalar)
def energy_flux(r, time, nu_photon, nu_biophoton):
    denominator = r**2 + k_damping * np.sin(omega * time)
    # Avoid division by zero or negative denominator
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return (h * (nu_photon + nu_biophoton)) / denominator

# Quantum-Gravity Correction (no time dependence)
def quantum_gravity_correction(theta, phi, epsilon):
    return epsilon * np.cos(theta) * np.sin(phi) + np.sqrt(epsilon) / (1 + np.cos(theta + phi))

# Lovince Universal Model
def lovince_universal_model(n, r, time_point, theta, phi, epsilon):
    a_n = lovince_spiral_sequence(n)
    H_n = lovince_harmony_equation(n)
    Q = quantum_gravity_correction(theta, phi, epsilon)  # scalar

    S_n = np.zeros(n)
    for i in range(n):
        flux = energy_flux(r[i], time_point, nu_photon, nu_biophoton)
        S_n[i] = a_n[i] * H_n[i] * flux * Q
    return S_n

# Parameters
n = 100
r = np.linspace(1e7, 1e9, n)
theta = np.pi / 4
phi = np.pi / 3
epsilon = 0.1

# Choose a specific time point to evaluate energy flux
time_point = 10  # seconds

# Compute the model
S_n = lovince_universal_model(n, r, time_point, theta, phi, epsilon)

# Plot
plt.plot(r, S_n)
plt.xlabel('Distance (r) in meters')
plt.ylabel('System State (S_n)')
plt.title(f'Lovince Universal Model at t={time_point}s')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
h = 6.626e-34  # Planck's constant (J·s)
k_damping = 1.0  # Damping constant (rename to avoid conflict)
omega = 1.2  # Angular frequency (rad/s)
alpha = 0.15  # Lovince correction factor (unused in current code)
beta = 2.5  # Damping exponent (unused in current code)

# Time array (0 to 20 seconds)
t = np.linspace(0, 20, 1000)

# Photon and Biophoton Frequencies (Hz)
nu_photon = 5e14
nu_biophoton = 1e13

# Spiral Sequence (a_n) - vectorized implementation
def lovince_spiral_sequence(n):
    indices = np.arange(n)
    floor_sqrt = np.floor(np.sqrt(indices))
    # The first term is 1, subsequent terms add floor(sqrt(i))
    a_n = 1 + np.cumsum(floor_sqrt[1:])
    return np.concatenate(([1], a_n))

# Harmony Equation (H_n) - vectorized implementation
def lovince_harmony_equation(n):
    k_vals = np.arange(1, n+1)
    denominators = k_vals**2 + np.sqrt(k_vals)
    H_n = np.cumsum(1 / denominators)
    return H_n

# Energy Flux at a specific time point (vectorized for r)
def energy_flux(r, time, nu_photon, nu_biophoton):
    denominator = r**2 + k_damping * np.sin(omega * time)
    # Avoid division by zero or negative denominator
    denominator = np.maximum(denominator, 1e-10)
    return (h * (nu_photon + nu_biophoton)) / denominator

# Quantum-Gravity Correction (no time dependence)
def quantum_gravity_correction(theta, phi, epsilon):
    return epsilon * np.cos(theta) * np.sin(phi) + np.sqrt(epsilon) / (1 + np.cos(theta + phi))

# Lovince Universal Model (vectorized where possible)
def lovince_universal_model(n, r, time_point, theta, phi, epsilon):
    a_n = lovince_spiral_sequence(n)
    H_n = lovince_harmony_equation(n)
    Q = quantum_gravity_correction(theta, phi, epsilon)  # scalar
    flux = energy_flux(r, time_point, nu_photon, nu_biophoton)
    S_n = a_n * H_n * flux * Q
    return S_n

# Parameters
n = 100
r = np.linspace(1e7, 1e9, n)
theta = np.pi / 4
phi = np.pi / 3
epsilon = 0.1

# Choose a specific time point to evaluate energy flux
time_point = 10  # seconds

# Compute the model
S_n = lovince_universal_model(n, r, time_point, theta, phi, epsilon)

# Plot with enhanced visualization
plt.figure(figsize=(10, 6))
plt.plot(r, S_n, label='System State Sₙ', color='blue')
plt.xlabel('Distance (r) [m]', fontsize=12)
plt.ylabel('System State (Sₙ)', fontsize=12)
plt.title(f'Lovince Universal Model at t={time_point}s', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Use logarithmic scale if values span many orders of magnitude
if np.max(S_n) / np.min(S_n) > 1e3:
    plt.yscale('log')
    plt.ylabel('System State (Sₙ) [log scale]')

plt.show()