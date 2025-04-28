import numpy as np

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
h = 6.62607015e-34  # Planck's constant in m^2 kg / s
Lambda = 1.105e-52  # Cosmological constant in m^-2 (an approximate value)
c = 3.0e8  # Speed of light in m/s
k = 1.380649e-23  # Boltzmann constant in J/K
# Sample values for temperature (T) and mass (m)
T = 300  # Temperature in Kelvin
m = 5.0  # Mass in kg

# Formula to calculate S
S = (G * h * Lambda / c**4) * (k * T / m**2)

# Output the result
print(f"The system state (S) is: {S} J·K·kg^(-2)·m^2")