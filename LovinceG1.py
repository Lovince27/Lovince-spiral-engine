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

import numpy as np
import matplotlib.pyplot as plt

def compute_Z_n(n_values, c=3e8, phi=1.618033988749895):
    """
    Compute the complex sequence Z_n for given n values.
    
    Parameters:
    - n_values: Array of integers (e.g., np.arange(1, 10)).
    - c: Speed of light (default: 3e8 m/s).
    - phi: Golden ratio (default: 1.618033988749895).
    
    Returns:
    - Z_n: Complex-valued array of Z_n values.
    """
    # Input validation
    if phi == 0:
        raise ValueError("phi cannot be zero")
    if not np.all(np.isfinite(n_values)):
        raise ValueError("n_values must be finite")
    
    # Enable overflow warnings
    np.seterr(all='warn')
    
    # Compute magnitude in logarithmic form to avoid overflow
    log_magnitude = (
        np.log(9 * c) +
        n_values * np.log(phi) +
        (3 * n_values - 1) * np.log(np.pi) -
        n_values * np.log(3)
    )
    
    # Compute phase
    phase = -n_values * np.pi / phi
    
    # Combine magnitude and phase
    Z_n = np.exp(log_magnitude) * np.exp(1j * phase)
    
    # Check for non-finite values
    if not np.all(np.isfinite(Z_n)):
        print("Warning: Non-finite values detected in Z_n")
    
    return Z_n

def plot_Z_n(n_values, Z_n):
    """
    Plot magnitude, real, and imaginary parts of Z_n.
    
    Parameters:
    - n_values: Array of n values.
    - Z_n: Complex-valued Z_n array.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, np.abs(Z_n), label='|Z_n|', marker='o')
    plt.plot(n_values, np.real(Z_n), label='Re(Z_n)', marker='s')
    plt.plot(n_values, np.imag(Z_n), label='Im(Z_n)', marker='^')
    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title('Complex Sequence Z_n')
    plt.yscale('symlog')  # Symlog for large values
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Define parameters
    c = 3e8  # Speed of light (m/s)
    phi = 1.618033988749895  # Golden ratio
    n_values = np.arange(1, 10)  # n from 1 to 9
    
    # Compute Z_n
    Z_n = compute_Z_n(n_values, c, phi)
    
    # Print results
    print("Computed Z_n values:")
    for n, z in zip(n_values, Z_n):
        print(f"Z_{n} = {z:.4e}")
    
    # Plot results
    plot_Z_n(n_values, Z_n)

if __name__ == "__main__":
    main()