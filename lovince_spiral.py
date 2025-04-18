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
