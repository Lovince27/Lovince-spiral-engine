import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_Z_n(n_values, c=3e8, phi=1.618033988749895):
    """
    Compute the complex sequence Z_n using the custom formula:
    Z_n = (9 * c * phi^n * pi^(3n-1) / 3^n) * exp(-i * n * pi / phi)
    
    Parameters:
    - n_values: Array of integers (e.g., np.arange(1, 1001)).
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

def normalize_Z_n(Z_n):
    """
    Normalize Z_n to unit magnitude for AI compatibility.
    
    Parameters:
    - Z_n: Complex-valued array.
    
    Returns:
    - normalized_Z_n: Normalized complex array (|Z_n| = 1).
    """
    magnitudes = np.abs(Z_n)
    return Z_n / magnitudes

def save_dataset(n_values, Z_n, filename='Z_n_dataset.csv'):
    """
    Save normalized Z_n as a dataset for AI training.
    
    Parameters:
    - n_values: Array of n values.
    - Z_n: Complex-valued Z_n array.
    - filename: Output CSV file name.
    """
    normalized_Z_n = normalize_Z_n(Z_n)
    data = {
        'n': n_values,
        'Real_Zn': np.real(normalized_Z_n),
        'Imag_Zn': np.imag(normalized_Z_n),
        'Magnitude_Zn': np.abs(Z_n)
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to '{filename}'")

def plot_Z_n(n_values, Z_n, save_plot=False):
    """
    Plot Z_n: magnitude, real/imaginary parts, and complex plane trajectory.
    
    Parameters:
    - n_values: Array of n values.
    - Z_n: Complex-valued Z_n array.
    - save_plot: Whether to save the plot as a file.
    """
    plt.figure(figsize=(14, 5))
    
    # Plot magnitude and components
    plt.subplot(1, 2, 1)
    plt.plot(n_values, np.abs(Z_n), label='|Z_n| (Magnitude)', marker='o', linewidth=2)
    plt.plot(n_values, np.real(Z_n), label='Real Part', marker='s', linewidth=2)
    plt.plot(n_values, np.imag(Z_n), label='Imaginary Part', marker='^', linewidth=2)
    plt.xlabel('n', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Z_n Sequence Components', fontsize=14)
    plt.yscale('symlog')  # Symmetric log scale for large values
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Plot complex plane trajectory
    plt.subplot(1, 2, 2)
    plt.plot(np.real(Z_n), np.imag(Z_n), 'o-', label='Z_n Trajectory')
    plt.xlabel('Real Part', fontsize=12)
    plt.ylabel('Imaginary Part', fontsize=12)
    plt.title('Z_n in Complex Plane', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    if save_plot:
        plt.savefig('Z_n_AI_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to 'Z_n_AI_plot.png'")
    plt.show()

def main():
    # Parameters
    c = 3e8  # Speed of light (m/s)
    phi = 1.618033988749895  # Golden ratio
    n_values = np.arange(1, 1001)  # Large range for AI dataset
    
    # Compute Z_n
    Z_n = compute_Z_n(n_values, c, phi)
    
    # Print first few results
    print("First 9 Z_n values from your formula:")
    for n, z in zip(n_values[:9], Z_n[:9]):
        print(f"Z_{n} = {z:.4e} (Magnitude: {np.abs(z):.4e})")
    
    # Save dataset for AI
    save_dataset(n_values, Z_n)
    
    # Plot results (first 9 for clarity)
    plot_Z_n(n_values[:9], Z_n[:9], save_plot=True)

if __name__ == "__main__":
    main()