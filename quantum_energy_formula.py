import numpy as np
import matplotlib.pyplot as plt

# Constants
phi = (1 + np.sqrt(5)) / 2         # Golden Ratio
pi = np.pi
hbar = 1.055e-34                   # Reduced Planck’s constant (J·s)
log2 = np.log(2)

# Physical Parameters (can be adjusted)
E = 1.0                            # Energy input
F = 1.0                            # Force
ma = 1.0                           # Active mass
mc2 = 1.0                          # Mass-energy equivalence
alpha = 1.0                        # Quantum uncertainty exponent
S = 256                            # Number of quantum states (entropy)

# Quantum levels
n_values = np.arange(0, 10, 1)
psi_values = []

# Compute Ψ(t) for each level n
for n in n_values:
    psi = (E * (hbar**alpha) * ma * (phi**n) * (pi**(3*n - 1)) * np.log(S)) / (F * mc2 * log2)
    psi_values.append(psi)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_values, psi_values, marker='o', linestyle='-', color='purple')
plt.title("Ψ(t) vs n — Quantum Energy Field Evolution")
plt.xlabel("Quantum Level (n)")
plt.ylabel("Ψ(t)")
plt.grid(True)
plt.tight_layout()
plt.show()