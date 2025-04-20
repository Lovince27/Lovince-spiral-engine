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