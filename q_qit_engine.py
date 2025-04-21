import numpy as np
import cmath

# Constants
phi = (1 + np.sqrt(5)) / 2      # Golden ratio
pi = np.pi                      # Pi
c = 299792458                   # Speed of light (m/s)
h = 6.626e-34                   # Planck's constant (J·s)
ħ = 1.055e-34                   # Reduced Planck’s constant
lovince_magnitude = 40.5
E0 = ħ * lovince_magnitude
beta = 0.8                      # Biophoton biological factor
ν = 6e14                        # Sample frequency in Hz

def energy_sequence(n):
    """Quantum golden energy formula"""
    return phi**n * pi**(3*n - 1) * E0

def biophoton_energy(n):
    return energy_sequence(n) * h * ν * beta

def master_sequence(n, theta_n):
    decay = 9 * (1/3)**n * c
    amp = decay * phi**n * pi**(3*n - 1)
    phase = cmath.exp(-1j * n * pi / phi) * cmath.exp(1j * theta_n)
    return amp * phase

def quantum_state(n):
    A_n = (1/phi**n) * (1/3)**n
    theta_n = (2 * pi * n) / phi
    ket_n = f"|{n}⟩"
    return f"{A_n:.5e}·e^(i·{theta_n:.3f})·{ket_n}"

# Example Usage
if __name__ == "__main__":
    for n in range(1, 6):
        print(f"n={n}")
        print("Zₙ:", master_sequence(n, (2 * pi * n) / phi))
        print("Eₙ:", energy_sequence(n), "J")
        print("ψₙ:", quantum_state(n))
        print("Biophoton E:", biophoton_energy(n), "J")
        print("-" * 40)