import cmath
import math

def compute_zn(fn, fn_minus_1):
    # θ = tan⁻¹(F_{n-1}/F_n)
    theta = math.atan(fn_minus_1 / fn)
    # Z_n = F_n · e^(iθ)
    zn = fn * cmath.exp(1j * theta)
    return zn

# Lucas sequence example (first few terms)
lucas_seq = [1, 3, 4, 7, 11, 18, 29]

# Spiral generation
print("Lovince Spiral (Z_n) values:")
for n in range(1, len(lucas_seq)):
    fn = lucas_seq[n]
    fn_1 = lucas_seq[n - 1]
    zn = compute_zn(fn, fn_1)
    print(f"Z_{n+1} = {zn.real:.3f} + {zn.imag:.3f}i")

import numpy as np
import matplotlib.pyplot as plt

# Lovince Formula
magnitude = 5
angle_deg = 30.9
angle_rad = np.deg2rad(angle_deg)
z = magnitude * (np.cos(angle_rad) + 1j * np.sin(angle_rad))

# Plot on Argand Plane
plt.figure(figsize=(6, 6))
plt.plot([0, z.real], [0, z.imag], 'ro-', label='z ≈ 5e^(i30.9°)')
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Lovince Formula Visualization')
plt.legend()
plt.show()

print(f"Lovince Formula: z = {z:.3f}")
