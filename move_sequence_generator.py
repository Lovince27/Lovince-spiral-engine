import numpy as np
import matplotlib.pyplot as plt

# Constants
phi = (1 + np.sqrt(5)) / 2  # Golden Ratio

# Generate Custom Fibonacci-like Sequence: 1, 3, 4, 7, 11, 18, 29...
def generate_sequence(n_terms):
    sequence = [1, 3]
    while len(sequence) < n_terms:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

# Compute Move Sequence
def compute_move_sequence(sequence):
    M = []
    for n, T_n in enumerate(sequence, start=1):
        phi_term = phi ** T_n
        theta_n = 2 * np.pi * n / phi
        harmonic_sum = sum(1 / (i + 1) for i in range(n))
        complex_move = phi_term * harmonic_sum * T_n * np.exp(1j * theta_n)
        M.append(complex_move)
    return M

# Plot the Complex Move Sequence
def plot_sequence(M):
    real = [z.real for z in M]
    imag = [z.imag for z in M]

    plt.figure(figsize=(10, 6))
    plt.plot(real, imag, marker='o', linestyle='-', color='darkblue')
    plt.title("Quantum-Energy Move Sequence (Modified Fibonacci)", fontsize=14)
    plt.xlabel("Real Axis", fontsize=12)
    plt.ylabel("Imaginary Axis", fontsize=12)
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    terms = 10
    sequence = generate_sequence(terms)
    move_sequence = compute_move_sequence(sequence)
    plot_sequence(move_sequence)