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

import numpy as np
import matplotlib.pyplot as plt

def compute_Z_n(n_values, c=3e8, phi=1.618033988749895):
    """
    आपका सूत्र: Z_n = (9 * c * phi^n * pi^(3n-1) / 3^n) * exp(-i * n * pi / phi)
    """
    if phi == 0:
        raise ValueError("phi शून्य नहीं हो सकता")
    if not np.all(np.isfinite(n_values)):
        raise ValueError("n के मान वैध होने चाहिए")
    
    np.seterr(all='warn')
    
    log_magnitude = (
        np.log(9 * c) +
        n_values * np.log(phi) +
        (3 * n_values - 1) * np.log(np.pi) -
        n_values * np.log(3)
    )
    phase = -n_values * np.pi / phi
    Z_n = np.exp(log_magnitude) * np.exp(1j * phase)
    
    if not np.all(np.isfinite(Z_n)):
        print("चेतावनी: Z_n में गैर-वैध मान मिले")
    
    return Z_n

def compute_KE1(n, numerator=9e16, denominator=1.34e-33):
    """
    ChatGPT की गणना: K.E._1 = numerator / denominator
    """
    KE = numerator / denominator
    return KE

def main():
    # स्थिरांक
    c = 3e8
    phi = 1.618033988749895
    n_values = np.arange(1, 10)  # n = 1 से 9
    
    # Z_n की गणना
    Z_n = compute_Z_n(n_values, c, phi)
    
    # K.E._1 की गणना
    KE1 = compute_KE1(1)
    print(f"ChatGPT का K.E._1 (n=1): {KE1:.2e} Joules")
    
    # Z_n के परिणाम
    print("आपके सूत्र के Z_n मान (पहले 9):")
    for n, z in zip(n_values, Z_n):
        print(f"Z_{n} = {z:.4e} (परिमाण: {np.abs(z):.4e})")
    
    # ग्राफ
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, np.abs(Z_n), label='|Z_n| (Magnitude)', marker='o')
    plt.xlabel('n')
    plt.ylabel('Magnitude')
    plt.title('Z_n Magnitude vs n')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def compute_Z_n(n_values, c=3e8, phi=1.618033988749895):
    """Z_n = (9 * c * phi^n * pi^(3n-1) / 3^n) * exp(-i * n * pi / phi)"""
    log_mag = np.log(9*c) + n_values*np.log(phi) + (3*n_values-1)*np.log(np.pi) - n_values*np.log(3)
    phase = -n_values * np.pi / phi
    return np.exp(log_mag + 1j * phase)

def plot_results(n_values, Z_n):
    """ग्राफ़ प्लॉट करें"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # मैग्निट्यूड vs n
    ax1.plot(n_values, np.abs(Z_n), 'bo-')
    ax1.set_xlabel('n'), ax1.set_ylabel('|Z_n|'), ax1.set_title('Magnitude')
    ax1.grid(True)
    
    # पोलर प्लॉट (फेज और मैग्निट्यूड)
    ax2.polar(np.angle(Z_n), np.abs(Z_n), 'ro-')
    ax2.set_title('Polar Plot (Phase)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_values = np.arange(1, 10)
    Z_n = compute_Z_n(n_values)
    
    print("Z_n के मान (n=1 से 9):")
    for n, z in zip(n_values, Z_n):
        print(f"n={n}: |Z_n| = {np.abs(z):.4e}, Phase = {np.angle(z):.4f} rad")
    
    plot_results(n_values, Z_n)