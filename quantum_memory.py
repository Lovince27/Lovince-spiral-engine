import numpy as np
import matplotlib.pyplot as plt

# Constants
phi = (1 + np.sqrt(5)) / 2
pi = np.pi
hbar = 1.055e-34
log2 = np.log(2)

# Physical Parameters
E = 1.0
F = 1.0
ma = 1.0
mc2 = 1.0
alpha = 1.0
S = 256
gamma = 0.05  # memory decay rate

# Time and Quantum Levels
t_values = np.linspace(0, 50, 200)     # Time array
n = 5                                  # Fixed quantum level to visualize
memory = np.cos(0.2 * t_values)        # Oscillatory memory function

# Ψ_n(t) computation
psi_values = (
    (E * (hbar**alpha) * ma * (phi**n) * (pi**(3*n - 1)) * np.log(S)) /
    (F * mc2 * log2)
) * memory * np.exp(-gamma * t_values)

# Plotting Ψ_n(t)
plt.figure(figsize=(10, 6))
plt.plot(t_values, psi_values, color='darkgreen')
plt.title(f"Ψₙ(t) with Memory Decay (n = {n})")
plt.xlabel("Time (t)")
plt.ylabel("Ψₙ(t)")
plt.grid(True)
plt.tight_layout()
plt.show()