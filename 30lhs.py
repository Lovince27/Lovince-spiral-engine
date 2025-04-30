import math

# Constants
pi = math.pi
phi = (1 + math.sqrt(5)) / 2
e = math.e
h = 6.62607015e-34  # Planck constant in J·s
nu = 5e14  # Frequency in Hz (visible light)

# Sequence definitions
A1 = pi + phi
A2 = pi * (1 + phi)
A3 = phi * (1 + pi)
A4 = pi**phi + phi**pi
A5 = (pi * phi) / (pi + phi)
A6 = pi**2 + phi**2
A7 = (pi + phi)**2
A8 = math.exp(pi - phi)
A9 = (pi / phi) + (phi / pi)
A10 = ((h * nu) / (pi * e) + phi**2) * (pi + phi)

# Store in list
sequence = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]

# Print values
print("Lovince Harmony Sequence:")
for i, val in enumerate(sequence, start=1):
    print(f"A{i} = {val:.10f}")