import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Spiral Sequence
def lovince_spiral_sequence(n):
    seq = [1]
    for i in range(2, n+1):
        seq.append(seq[-1] + int(np.sqrt(i)))
    return seq[-1]

# Harmony Equation
def lovince_harmony_equation(n):
    return sum(1 / (k**2 + np.sqrt(k)) for k in range(1, n+1))

# Energy Flux
def energy_flux(E, r, k, omega, t):
    return E / (r**2 + k * np.sin(omega * t))

# Quantum Gravity Correction
def Q(t, theta, phi, epsilon):
    return epsilon * np.cos(theta) * np.sin(phi) + (np.sqrt(epsilon) / (1 + np.cos(theta + phi)))

# Lovince Dual Reflection Constant
def lovince_LL_constant(psi_forward, psi_reflected):
    numerator = np.abs(np.vdot(psi_forward, psi_reflected))
    denominator = np.vdot(psi_forward, psi_forward).real
    LL = numerator / denominator
    return LL


# Dynamic Quantum States
def psi1_func(t, r):
    return np.array([np.exp(1j*t), np.sin(t + r*1e-6), np.cos(t)])

def psi2_func(t, r):
    return np.array([np.exp(-1j*t), np.cos(t - r*1e-6), np.sin(t)])

# Grid for t and r
t_vals = np.linspace(0, 20, 100)
r_vals = np.linspace(1e6, 1e7, 100)
T, R = np.meshgrid(t_vals, r_vals)
S_vals = np.zeros_like(T)
LL_vals = np.zeros_like(T)

# Fixed parameters
n = 10
E = 1000
k = 0.15
omega = 1.2
theta = np.pi/4
phi = np.pi/6
epsilon = 0.01

# Compute Sₙ and LL over grid
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        t = T[i, j]
        r = R[i, j]
        psi1 = psi1_func(t, r)
        psi2 = psi2_func(t, r)
        a_n = lovince_spiral_sequence(n)
        H_n = lovince_harmony_equation(n)
        Phi = energy_flux(E, r, k, omega, t)
        Q_val = Q(t, theta, phi, epsilon)
        LL = lovince_LL_constant(psi1, psi2)
        S = a_n * H_n * Phi * Q_val * LL
        S_vals[i, j] = S
        LL_vals[i, j] = LL

fig = plt.figure(figsize=(14, 6))

# Plot Sₙ
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(T, R, S_vals, cmap='viridis')
ax1.set_title("Lovince Universal State Sₙ(t, r)")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Distance (r)")
ax1.set_zlabel("Sₙ")

# Plot LL
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(T, R, LL_vals, cmap='plasma')
ax2.set_title("Lovince Reflection Constant LL(t, r)")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Distance (r)")
ax2.set_zlabel("LL")

plt.tight_layout()
plt.show()