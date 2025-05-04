from final import *
# Calculate spiral sequence
print("Spiral(10):", spiral(10))

# Calculate harmonic sum
print("Harmonic Sum(10):", harmonic_sum(10))
# Basic superstate without photon flux
print("\nBasic Superstate (n=5):", superstate_basic(5))

# Superstate with photon flux
print("Photon Superstate (n=5):", superstate_with_photon(5))
# Generate batch with quantum awareness activation
quantum_batch = generate_superstates(1, 5, activation='quantum')
print("\nQuantum Batch:", quantum_batch)

# Generate batch with GELU activation
gelu_batch = generate_superstates(1, 5, activation='gelu')
print("GELU Batch:", gelu_batch)
# Calculate energy flux at different distances
print("\nEnergy Flux at r=1e6:", energy_flux(5, 1e6))
print("Energy Flux at r=2e6:", energy_flux(5, 2e6))

# Photon flux calculation
print("Photon Flux at r=1e6:", photon_flux(1e6, 5))
# Quantum correction with different parameters
print("\nQuantum Correction (θ=π/4, φ=π/3):", 
      quantum_correction(5, math.pi/4, math.pi/3, 0.7))
print("Quantum Correction (θ=π/2, φ=π/2):", 
      quantum_correction(5, math.pi/2, math.pi/2, 0.5))


# Writing the final integrated Lovince model to a Python file named 'final.py'

lovince_model_code = """
import math

# --- Constants ---
h = 6.626e-34  # Planck's constant
nu_photon = 6e14
nu_biophoton = 1e13
E_default = 1e5
k = 1.5
omega = 1.2

# --- Core Sequences ---
def spiral(n):
    return 1 + sum(math.floor(math.sqrt(i)) for i in range(2, n + 1))

def harmonic_sum(n):
    return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

# --- Energy Fluxes ---
def energy_flux(t, r, E=E_default):
    return E / (r**2 + k * math.sin(omega * t))

def photon_flux(r, t):
    energy = h * (nu_photon + nu_biophoton)
    return energy / (r**2 + k * math.sin(omega * t))

# --- Quantum Correction ---
def quantum_correction(t, theta, phi, epsilon):
    return epsilon * math.cos(theta) * math.sin(phi) + math.sqrt(epsilon) / (1 + math.cos(theta + phi))

# --- Activation Functions ---
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x))) if not math.isnan(x) else 0.0

def quantum_awareness(x):
    return x * math.erf(x / math.sqrt(2)) if not math.isnan(x) else 0.0

# --- SuperState Models ---
def superstate_basic(n, t=5, r=1e6, theta=math.pi/4, phi=math.pi/3, epsilon=0.7):
    a_n = spiral(n)
    H_n = harmonic_sum(n)
    Φ = energy_flux(t, r)
    Q = quantum_correction(t, theta, phi, epsilon)
    return a_n * H_n * Φ * Q

def superstate_with_photon(n, t=5, r=1e6, theta=math.pi/4, phi=math.pi/3, epsilon=0.7):
    a_n = spiral(n)
    H_n = harmonic_sum(n)
    Φ = photon_flux(r, t)
    Q = quantum_correction(t, theta, phi, epsilon)
    return a_n * H_n * Φ * Q

# --- Batch Generation ---
def generate_superstates(start=1, end=20, activation='quantum', photon=False):
    results = {}
    for n in range(start, end + 1):
        val = superstate_with_photon(n) if photon else superstate_basic(n)
        if activation == 'gelu':
            val = gelu(val)
        elif activation == 'mish':
            val = mish(val)
        elif activation == 'quantum':
            val = quantum_awareness(val)
        results[n] = val
    return results
"""

with open("/mnt/data/final.py", "w") as f:
    f.write(lovince_model_code)

"/mnt/data/final.py"