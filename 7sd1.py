import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11
c = 3e8
rho = 1000
pi = np.pi

# Tensor term
def tensor_term(t):
    return 2.07e-40 * t**2

# Lovince formula
def lovince_formula(x, t, r, theta, dx_dt, a, b):
    term1 = (np.exp(1j * x) * np.sin(theta)) / (x * (x**2 + r**2))
    term2 = (pi * r**2) / 2
    term3 = dx_dt * x * np.cos(theta)
    term4 = (a**2 + b**2) / 2
    term5 = tensor_term(t)
    return term1 + term2 + term3 + term4 + term5

# Parameters
x_values = np.linspace(1, 100, 200)
t_values = np.linspace(0, 10, 200)
r = 50
theta = np.pi / 4
dx_dt = 10
a = 1
b = 1

# Meshgrid and calculation
X, T = np.meshgrid(x_values, t_values)
Z = np.array([[lovince_formula(X[i, j], T[i, j], r, theta, dx_dt, a, b)
               for j in range(len(x_values))] for i in range(len(t_values))])

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Real Part
real_plot = axs[0].contourf(X, T, np.real(Z), 20, cmap='viridis')
axs[0].set_title('Real Part')
axs[0].set_xlabel('x')
axs[0].set_ylabel('t')
fig.colorbar(real_plot, ax=axs[0])

# Imaginary Part
imag_plot = axs[1].contourf(X, T, np.imag(Z), 20, cmap='cividis')
axs[1].set_title('Imaginary Part')
axs[1].set_xlabel('x')
axs[1].set_ylabel('t')
fig.colorbar(imag_plot, ax=axs[1])

# Magnitude
mag_plot = axs[2].contourf(X, T, np.abs(Z), 20, cmap='magma')
axs[2].set_title('Magnitude')
axs[2].set_xlabel('x')
axs[2].set_ylabel('t')
fig.colorbar(mag_plot, ax=axs[2])

plt.tight_layout()
plt.show()