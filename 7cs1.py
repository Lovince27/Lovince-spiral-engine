import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 3e8          # Speed of light (m/s)
rho = 1000       # Mass-energy density (kg/m^3)
pi = np.pi

# Functions

# Einstein Tensor term calculation
def tensor_term(t):
    # Tensor term based on simplified model (t^2 dependency)
    constant = 2.07e-40  # Precomputed constant for tensor term
    return constant * t**2

# Lovince formula (simplified)
def lovince_formula(x, t, r, theta, dx_dt, a, b):
    # First term: Complex exponential and angular sine function
    term1 = (np.exp(1j * x) * np.sin(theta)) / (x * (x**2 + r**2))
    
    # Second term: Pi * r^2 / 2 (geometric term)
    term2 = (pi * r**2) / 2
    
    # Third term: dx/dt * x * cos(theta) (kinematic term)
    term3 = dx_dt * x * np.cos(theta)
    
    # Fourth term: a^2 + b^2 / 2 (energy term)
    term4 = (a**2 + b**2) / 2
    
    # Einstein tensor term
    term5 = tensor_term(t)
    
    # Summing all terms to get the final value
    return term1 + term2 + term3 + term4 + term5

# Parameters
x_values = np.linspace(1, 100, 200)  # x ranging from 1 to 100 (avoid division by zero)
t_values = np.linspace(0, 10, 200)   # Time ranging from 0 to 10 seconds
r = 50                               # Example radius (meters)
theta = np.pi / 4                    # Angle (radians)
dx_dt = 10                            # Example velocity (m/s)
a = 1                                 # Example value for a
b = 1                                 # Example value for b

# Compute Lovince formula for each x and t combination
X, T = np.meshgrid(x_values, t_values)
Z = np.zeros_like(X, dtype=complex)

for i in range(len(t_values)):
    for j in range(len(x_values)):
        Z[i, j] = lovince_formula(X[i, j], T[i], r, theta, dx_dt, a, b)

# Plotting
plt.figure(figsize=(10, 6))
plt.contourf(X, T, np.real(Z), 20, cmap='viridis')
plt.colorbar(label='Real Part of L(x, t, r, θ)')
plt.title('Visualization of Lovince Formula with Einstein Tensor Term')
plt.xlabel('x (Position)')
plt.ylabel('t (Time)')
plt.show()