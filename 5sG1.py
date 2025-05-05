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


# 1.py
import cmath
import matplotlib.pyplot as plt

# Euler's formula parameters
theta = cmath.pi / 2  # 90 degrees in radians

def i_power(n):
    """Return i^n using Euler's formula, rounded to avoid floating-point errors."""
    val = cmath.exp(1j * n * theta)
    return complex(round(val.real, 10), round(val.imag, 10))

def equivalent_powers(a, b):
    """Return True if i^a and i^b are equivalent (modulo 4)."""
    return (a - b) % 4 == 0

# 1. Print the sequence of i^n for n = 0 to 7
print("Sequence of powers of i (using Euler's formula):")
for n in range(8):
    val = i_power(n)
    print(f"i^{n} = {val}")

# 2. Check and print equivalence of some user pairs
print("\nEquivalence checks (modulo 4):")
pairs = [(3, 1), (2, 4), (3, 5), (5, 7)]
for a, b in pairs:
    val_a = i_power(a)
    val_b = i_power(b)
    eq = equivalent_powers(a, b)
    print(f"i^{a} = {val_a}, i^{b} = {val_b} -> Equivalent: {eq}")

# 3. Print summary table for i^n mod 4
print("\nSummary Table (i^n, n = 0 to 3):")
table = {0: 1, 1: 1j, 2: -1, 3: -1j}
for n in range(4):
    print(f"i^{n} = {table[n]}")

# 4. Plot the powers of i on the complex plane
points = [i_power(n) for n in range(8)]
x = [p.real for p in points]
y = [p.imag for p in points]

plt.figure(figsize=(5,5))
plt.scatter(x, y, color='blue')
for n, (xr, yr) in enumerate(zip(x, y)):
    plt.text(xr, yr, f'i^{n}', fontsize=12, ha='right', va='bottom')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title('Powers of i on the Complex Plane')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)
plt.axis('equal')
plt.show()
