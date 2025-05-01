import numpy as np

# Spiral Sequence (a_n)
def lovince_spiral_sequence(n):
    seq = [1]
    for i in range(2, n+1):
        seq.append(seq[-1] + int(np.sqrt(i)))
    return seq[-1]

# Harmony Equation (H_n)
def lovince_harmony_equation(n):
    return sum(1 / (k**2 + np.sqrt(k)) for k in range(1, n+1))

# Energy Flux (Φ)
def energy_flux(E, r, k, omega, t):
    return E / (r**2 + k * np.sin(omega * t))

# Quantum Gravity Correction (Q)
def Q(t, theta, phi, epsilon):
    return epsilon * np.cos(theta) * np.sin(phi) + (np.sqrt(epsilon) / (1 + np.cos(theta + phi)))

# Lovince LL Constant
def lovince_LL_constant(psi_forward, psi_reflected):
    numerator = np.abs(np.vdot(psi_forward, psi_reflected))
    denominator = np.vdot(psi_forward, psi_forward).real
    LL = numerator / denominator
    return LL

# Full Universal Model
def lovince_universal_model(n, E, r, k, omega, t, theta, phi, epsilon, psi1, psi2):
    a_n = lovince_spiral_sequence(n)
    H_n = lovince_harmony_equation(n)
    Phi = energy_flux(E, r, k, omega, t)
    Q_val = Q(t, theta, phi, epsilon)
    LL = lovince_LL_constant(psi1, psi2)
    S_n = a_n * H_n * Phi * Q_val * LL
    return S_n


# Sample quantum states
psi1 = np.array([1+0j, 1j, 0])
psi2 = np.array([0, 1j, 1+0j])

S = lovince_universal_model(
    n=10,
    E=1000,
    r=1e6,
    k=0.15,
    omega=1.2,
    t=5,
    theta=np.pi/4,
    phi=np.pi/6,
    epsilon=0.01,
    psi1=psi1,
    psi2=psi2
)

print(f"Lovince Universal State (Sₙ) = {S:.6e}")