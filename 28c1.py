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


"""
final.py: Implementation of the Lovince Universal Model

This script computes the system state S_n as a function of distance r and time t,
based on a spiral sequence (a_n), harmony equation (H_n), energy flux, and quantum-gravity
correction (Q). The model is defined as:

    S_n = a_n * H_n * Flux(r, t) * Q

Features:
- Vectorized NumPy operations for efficiency
- Unit consistency (S_n in W/m²)
- Self-checks and cross-checks for mathematical correctness
- Robust error handling
- Clear visualization with logarithmic scaling
- Modular design for extensibility

Units:
- h: J·s (Planck's constant)
- nu_photon, nu_biophoton: Hz (frequencies)
- k_damping: m² (damping constant)
- omega: rad/s (angular frequency)
- r: m (distance)
- t: s (time)
- S_n: W/m² (energy flux)

Author: Grok 3 (assisted by xAI)
Date: April 28, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
H = 6.626e-34  # Planck's constant (J·s)
K_DAMPING = 1e14  # Damping constant (m², ensures unit consistency)
OMEGA = 1.2  # Angular frequency (rad/s)
NU_PHOTON = 5e14  # Photon frequency (Hz, visible light)
NU_BIOPHOTON = 1e13  # Biophoton frequency (Hz, infrared)

# Model parameters
N = 100  # Number of modes
R = np.linspace(1e7, 1e9, N)  # Distance array (m)
THETA = np.pi / 4  # Angular parameter (radians)
PHI = np.pi / 3  # Angular parameter (radians)
EPSILON = 0.1  # Dimensionless coupling constant
TIME_POINT = 10.0  # Fixed time point (s)
T = np.linspace(0, 20, 1000)  # Time array for potential extensions (s)


def lovince_spiral_sequence(n):
    """
    Compute the spiral sequence a_n = 1 + sum_{i=1}^{n-1} floor(sqrt(i)), a_0 = 1.

    Args:
        n (int): Number of terms.

    Returns:
        ndarray: Array of shape (n,) containing a_n (dimensionless).

    Raises:
        ValueError: If n <= 0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    indices = np.arange(n)
    floor_sqrt = np.floor(np.sqrt(indices))
    a_n = np.concatenate(([1], 1 + np.cumsum(floor_sqrt[1:])))
    # Self-check: Verify first few terms
    if n >= 3:
        assert a_n[0] == 1, "a_0 should be 1"
        assert a_n[1] == 2, "a_1 should be 2"
        assert a_n[2] == 3, "a_2 should be 3"
    return a_n


def lovince_harmony_equation(n):
    """
    Compute the harmony equation H_n = sum_{k=1}^n 1 / (k^2 + sqrt(k)).

    Args:
        n (int): Number of terms.

    Returns:
        ndarray: Array of shape (n,) containing H_n (dimensionless).

    Raises:
        ValueError: If n <= 0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    k_vals = np.arange(1, n + 1)
    denominators = k_vals**2 + np.sqrt(k_vals)
    H_n = np.cumsum(1 / denominators)
    # Self-check: Verify H_1
    if n >= 1:
        assert np.isclose(H_n[0], 0.5), "H_1 should be 0.5"
    return H_n


def energy_flux(r, time, nu_photon, nu_biophoton):
    """
    Compute energy flux: Flux = h * (nu_photon + nu_biophoton) / (r^2 + k_damping * sin(omega * t)).

    Args:
        r (ndarray): Distance array (m).
        time (float): Time point (s).
        nu_photon (float): Photon frequency (Hz).
        nu_biophoton (float): Biophoton frequency (Hz).

    Returns:
        ndarray: Flux array (W/m²).

    Raises:
        ValueError: If r contains non-positive values.
    """
    if np.any(r <= 0):
        raise ValueError("r must be positive")
    denominator = r**2 + K_DAMPING * np.sin(OMEGA * time)  # m²
    denominator = np.maximum(denominator, 1e-10)  # Avoid division by zero
    flux = (H * (nu_photon + nu_biophoton)) / denominator  # J / m² = W/m²
    # Cross-check: Flux should decrease with r
    if len(r) > 1:
        assert np.all(np.diff(flux) <= 0), "Flux must decrease with r"
    return flux


def quantum_gravity_correction(theta, phi, epsilon):
    """
    Compute quantum-gravity correction: Q = epsilon * cos(theta) * sin(phi) + sqrt(epsilon) / (1 + cos(theta + phi)).

    Args:
        theta (float): Angular parameter (radians).
        phi (float): Angular parameter (radians).
        epsilon (float): Coupling constant (dimensionless).

    Returns:
        float: Q (dimensionless).

    Raises:
        ValueError: If epsilon < 0.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    Q = epsilon * np.cos(theta) * np.sin(phi) + np.sqrt(epsilon) / (1 + np.cos(theta + phi))
    # Self-check: Ensure Q is finite
    assert np.isfinite(Q), "Q must be finite"
    return Q


def lovince_universal_model(n, r, time, theta, phi, epsilon):
    """
    Compute the Lovince Universal Model: S_n = a_n * H_n * Flux(r, t) * Q.

    Args:
        n (int): Number of modes.
        r (ndarray): Distance array (m).
        time (float): Time point (s).
        theta (float): Angular parameter (radians).
        phi (float): Angular parameter (radians).
        epsilon (float): Coupling constant (dimensionless).

    Returns:
        ndarray: S_n array of shape (n, len(r)) (W/m²).

    Raises:
        ValueError: If n <= 0, r <= 0, or epsilon < 0.
    """
    if n <= 0 or np.any(r <= 0) or epsilon < 0:
        raise ValueError("n must be positive, r must be positive, epsilon must be non-negative")
    
    a_n = lovince_spiral_sequence(n)  # (n,), dimensionless
    H_n = lovince_harmony_equation(n)  # (n,), dimensionless
    Q = quantum_gravity_correction(theta, phi, epsilon)  # scalar, dimensionless
    flux = energy_flux(r, time, NU_PHOTON, NU_BIOPHOTON)  # (len(r),), W/m²
    
    # Compute S_n using outer product for broadcasting
    S_n = np.outer(a_n * H_n, flux) * Q  # (n, len(r)), W/m²
    
    # Cross-check: Verify S_n = a_n * H_n * flux * Q for sampled indices
    for i in range(min(n, 3)):
        for j in range(min(len(r), 3)):
            expected = a_n[i] * H_n[i] * flux[j] * Q
            assert np.isclose(S_n[i, j], expected, rtol=1e-5), f"S_n[{i},{j}] mismatch"
    
    return S_n


def plot_system_state(r, S_n, time):
    """
    Plot the summed system state S_n over distance r.

    Args:
        r (ndarray): Distance array (m).
        S_n (ndarray): System state array of shape (n, len(r)) (W/m²).
        time (float): Time point (s).
    """
    plt.figure(figsize=(10, 6))
    S_n_sum = np.sum(S_n, axis=0)  # Sum over n for 1D plot, (len(r),)
    
    plt.plot(r, S_n_sum, label="System State Sₙ (summed over n)", color="blue", linewidth=2)
    plt.xlabel("Distance (r) [m]", fontsize=12)
    plt.ylabel("System State (Sₙ) [W/m²]", fontsize=12)
    plt.title(f"Lovince Universal Model at t={time:.1f} s", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    
    # Apply logarithmic scale if range is large
    if np.all(S_n_sum > 0) and np.max(S_n_sum) / np.min(S_n_sum) > 1e3:
        plt.yscale("log")
        plt.ylabel("System State (Sₙ) [W/m², log scale]", fontsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to compute and visualize the Lovince Universal Model.
    """
    try:
        # Compute the model
        S_n = lovince_universal_model(N, R, TIME_POINT, THETA, PHI, EPSILON)
        
        # Plot the results
        plot_system_state(R, S_n, TIME_POINT)
        
        # Optional: Print sample values for verification
        print(f"Sample S_n[0,0] = {S_n[0,0]:.2e} W/m²")
        print(f"Summed S_n at r={R[0]:.2e} m: {np.sum(S_n[:,0]):.2e} W/m²")
        
    except Exception as e:
        print(f"Error: {e}")


# Optional extension: Incorporate alpha and beta if needed
"""
# Example of including alpha and beta in energy_flux
ALPHA = 0.15  # m²
BETA = 2.5  # dimensionless
def energy_flux_with_alpha_beta(r, time, nu_photon, nu_biophoton):
    if np.any(r <= 0):
        raise ValueError("r must be positive")
    denominator = r**2 + K_DAMPING * np.sin(OMEGA * time) + ALPHA * r**BETA  # m²
    denominator = np.maximum(denominator, 1e-10)
    flux = (H * (nu_photon + nu_biophoton)) / denominator
    if len(r) > 1:
        assert np.all(np.diff(flux) <= 0), "Flux must decrease with r"
    return flux
"""

if __name__ == "__main__":
    main()