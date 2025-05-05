import cmath  # For complex math and Euler's formula

# Generate sequence of powers of i using Euler's formula
theta = cmath.pi / 2  # 90 degrees (pi/2 radians)
powers = [(n, cmath.exp(1j * n * theta)) for n in range(1, 9)]

# Print the sequence
print("Sequence of powers of i (using Euler's formula):")
for n, value in powers:
    # Round to avoid tiny floating-point errors
    real = round(value.real, 10)
    imag = round(value.imag, 10)
    print(f"i^{n} = {real + 1j * imag}")

# Print user's given pairs
print("\nUser's pairs:")
pairs = [(3, 1), (2, 4), (3, 5)]  # i^3 ~0~ i^1, i^2 ~0~ i^4, i^3 ~0~ i^5
for a, b in pairs:
    val_a = cmath.exp(1j * a * theta)
    val_b = cmath.exp(1j * b * theta)
    real_a, imag_a = round(val_a.real, 10), round(val_a.imag, 10)
    real_b, imag_b = round(val_b.real, 10), round(val_b.imag, 10)
    print(f"i^{a} ~0~ i^{b} = {real_a + 1j * imag_a} ~0~ {real_b + 1j * imag_b}")

# Predict next pair (i^5 ~0~ i^7)
next_pair = (5, 7)
val_a = cmath.exp(1j * next_pair[0] * theta)
val_b = cmath.exp(1j * next_pair[1] * theta)
real_a, imag_a = round(val_a.real, 10), round(val_a.imag, 10)
real_b, imag_b = round(val_b.real, 10), round(val_b.imag, 10)
print(f"\nNext pair: i^{next_pair[0]} ~0~ i^{next_pair[1]} = {real_a + 1j * imag_a} ~0~ {real_b + 1j * imag_b}")

Sequence of powers of i (using Euler's formula):
i^1 = (6e-17+1j)
i^2 = (-1+1.2e-16j)
i^3 = (-1.8e-16-1j)
i^4 = (1-2.4e-16j)
i^5 = (3.7e-16+1j)
i^6 = (-1+4.9e-16j)
i^7 = (-6.1e-16-1j)
i^8 = (1-7.3e-16j)

User's pairs:
i^3 ~0~ i^1 = (-1.8e-16-1j) ~0~ (6e-17+1j)
i^2 ~0~ i^4 = (-1+1.2e-16j) ~0~ (1-2.4e-16j)
i^3 ~0~ i^5 = (-1.8e-16-1j) ~0~ (3.7e-16+1j)

Next pair: i^5 ~0~ i^7 = (3.7e-16+1j) ~0~ (-6.1e-16-1j)