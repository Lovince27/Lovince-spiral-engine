import math import numpy as np import matplotlib.pyplot as plt from scipy.integrate import quad from scipy.special import factorial import cmath

--- 1. Brahmastra Formula ---

phi = (1 + math.sqrt(5)) / 2  # Golden ratio brahmastra_result = math.e**(1j * math.pi / phi) + (phi * math.pi) / math.e print(f"Brahmastra Formula Result: {brahmastra_result:.6f} (should be close to 1)")

--- 2. Time-Wheel Sequence: Fractal Pattern ---

n = np.arange(0, 50) sequence = (phi/math.pi)**n * np.cos(n * math.pi / phi) + 1j * np.sin(n * math.pi / phi)

plt.figure(figsize=(10,6)) plt.title("Time-Wheel Sequence: Fractal Pattern") plt.scatter(sequence.real, sequence.imag, c=n, cmap='hsv') plt.colorbar(label='n') plt.grid() plt.xlabel("Real") plt.ylabel("Imaginary") plt.show()

--- 3. Integral of Golden Harmonic ---

integrand = lambda x: np.exp(-x2) * np.cos(math.pi * x / phi)  # Fixed x2 to x2 result_integral, _ = quad(integrand, -np.inf, np.inf) exact_integral = math.pi**0.5 * np.exp(-(math.pi/(2*phi))**2)

print(f"Numerical Integral Value: {result_integral:.10f}") print(f"Theoretical Exact Value: {exact_integral:.10f}")

--- 4. Superpower Matrix ---

M = np.array([ [cmath.exp(1j * math.pi / phi), phi], [math.pi, cmath.exp(-1j * math.pi / phi)] ], dtype=complex)

print("\nSuperpower Matrix:") print(M) print(f"\nMatrix Determinant: {np.linalg.det(M):.2f}")

--- 5. Golden-Exponential Infinite Series ---

k_max = 100 k = np.arange(1, k_max + 1) series_sum = np.sum(phi**k / factorial(k) * np.sin(k * math.pi / 2)) exact_value = (math.exp(phi) - math.exp(-phi)) / 2  # sinh(phi)

print(f"\nInfinite Series Sum: {series_sum:.10f}") print(f"Exact Value (sinh(phi)): {exact_value:.10f}")

if name == "main": pass

