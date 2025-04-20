import numpy as np

# Define constants
c = 1
phi = 1.618  # Example: golden ratio
n_values = np.arange(1, 10)

# Compute Z_n
Z_n = (9 * c * (phi**n_values) * (np.pi ** (3 * n_values - 1)) / (3**n_values)) * np.exp(-1j * (n_values * np.pi / phi))

# Print results
for n, z in zip(n_values, Z_n):
    print(f"Z_{n} = {z}")

import numpy as np
import matplotlib.pyplot as plt

# Define constants (replace with your values)
c = 1
phi = 1.618

# Check inputs
if phi == 0:
    raise ValueError("phi cannot be zero")

# Enable overflow warnings
np.seterr(all='warn')

# Compute Z_n
n_values = np.arange(1, 10)
Z_n = (9 * c * (phi**n_values) * (np.pi ** (3 * n_values - 1)) / (3**n_values)) * np.exp(-1j * (n_values * np.pi / phi))

# Check for non-finite values
if not np.all(np.isfinite(Z_n)):
    print("Warning: Non-finite values in Z_n")

# Print results
for n, z in zip(n_values, Z_n):
    print(f"Z_{n} = {z:.4e}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(n_values, np.abs(Z_n), label='|Z_n|')
plt.plot(n_values, np.real(Z_n), label='Re(Z_n)')
plt.plot(n_values, np.imag(Z_n), label='Im(Z_n)')
plt.xlabel('n')
plt.ylabel('Value')
plt.title('Z_n Sequence')
plt.legend()
plt.grid(True)
plt.show()