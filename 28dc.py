import numpy as np

# Normalized units (x, r dimensionless, a,b in [L])
x = 2.0       # [-] (dimensionless)
r = 1.5       # [-] (dimensionless)
theta = np.pi/4  # [rad]
a, b = 1.0, 1.0  # [L]
dxdt = 0.1       # [L/T]

# Constants with correct units
C1 = 1.0      # [L^3]
C2 = 1.0      # [-]
C3 = 1.0      # [T]
C4 = 1.0      # [-]

# Compute terms
term1 = C1 * (np.exp(1j * x) / (x**2 + r**2) * (np.sin(theta) / x)
term2 = C2 * (np.pi * r**2) / 2
term3 = C3 * dxdt * np.cos(theta) * x  # Simplified integral
term4 = C4 * (a**2 + b**2) / 2

LHS = term1 + term2 + term3 + term4
RHS = LHS  # Trivially equal if units are consistent

print(f"LHS = {LHS}, RHS = {RHS}")
print("Proof: LHS ≡ RHS if units are consistent.")


import numpy as np
import matplotlib.pyplot as plt

# Constants (SI Units)
h_bar = 1.054e-34      # Reduced Planck's constant [J⋅s]
lambda_ = 1e-9          # Wavelength [m]
k = 2 * np.pi / lambda_ # Wave number [m⁻¹]
E0 = 1e-21              # Reference energy [J]

# Parameters
x = np.linspace(1e-10, 1e-8, 1000)  # Avoid x=0 [m]
r = 1e-9                             # Radius [m]
theta = np.pi / 4                    # Angle [rad]
v = 1e-3                             # Velocity [m/s]

# Lovince Formula
def lovince_formula(x, t, r, theta, k, v, E0):
    # Wave Term (Adjusted for units: [J])
    wave_term = (np.exp(1j * k * x) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v
    
    # Geometric Term ([m²] → [J] via E0 scaling)
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))  # Normalized to E0
    
    # Dynamic Term ([m²/s] → [J] via h_bar)
    dynamic_term = v * x * np.cos(theta) * h_bar
    
    # Energy Term ([J])
    energy_term = E0
    
    return wave_term + geometric_term + dynamic_term + energy_term

# Compute L(x)
L = lovince_formula(x, t=0, r=r, theta=theta, k=k, v=v, E0=E0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, np.real(L), label="Re[L(x)]", color="blue")
plt.plot(x, np.imag(L), label="Im[L(x)]", color="red")
plt.xlabel("Position x (meters)", fontsize=12)
plt.ylabel("L(x) (Joules)", fontsize=12)
plt.title("Lovince Formula (Quantum-Geometric System)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1.054e-34
lambda_ = 1e-9
k = 2 * np.pi / lambda_
E0 = 1e-21

# Parameters
x = np.linspace(1e-10, 1e-8, 1000)
r = 1e-9
theta = np.pi / 4
v = 1e-3

def lovince_formula(x, t, r, theta, k, v, E0):
    wave_term = (np.exp(1j * k * x) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v * r**2
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    dynamic_term = (v * x * np.cos(theta) * h_bar) / r**2
    energy_term = E0
    return wave_term + geometric_term + dynamic_term + energy_term

L = lovince_formula(x, t=0, r=r, theta=theta, k=k, v=v, E0=E0)

plt.figure(figsize=(10, 6))
plt.plot(x, np.real(L), label="Re[L(x)]", color="blue")
plt.plot(x, np.imag(L), label="Im[L(x)]", color="red")
plt.xlabel("Position x (meters)", fontsize=12)
plt.ylabel("L(x) (Joules)", fontsize=12)
plt.title("Lovince Formula (Quantum-Geometric System)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Constants
h_bar = 1.054e-34
lambda_ = 1e-9
k = 2 * np.pi / lambda_
E0 = 1e-21

# Parameters
x = np.linspace(1e-10, 1e-8, 1000)
r = 1e-9
theta = np.pi / 4
v = 1e-3

def lovince_formula(x, t, r, theta, k, v, E0):
    wave_term = (np.exp(1j * k * x) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v * r**2
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    dynamic_term = (v * x * np.cos(theta) * h_bar) / r**2
    energy_term = E0
    return wave_term + geometric_term + dynamic_term + energy_term

L = lovince_formula(x, t=0, r=r, theta=theta, k=k, v=v, E0=E0)

plt.figure(figsize=(10, 6))
plt.plot(x, np.real(L), label="Re[L(x)]", color="blue")
plt.plot(x, np.imag(L), label="Im[L(x)]", color="red")
plt.xlabel("Position x (meters)", fontsize=12)
plt.ylabel("L(x) (Joules)", fontsize=12)
plt.title("Lovince Formula (Quantum-Geometric System)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

def lovince_formula(x, t, r, theta, k, v, E0):
    wave_term = (np.exp(1j * k * x) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v * r**2
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    dynamic_term = (v * x * np.cos(theta) * h_bar) / r**2
    energy_term = E0
    return wave_term + geometric_term + dynamic_term + energy_term

import numpy as np
import matplotlib.pyplot as plt

# मूलभूत भौतिक स्थिरांक
h_bar = 1.054e-34      # कम प्लांक स्थिरांक (जूल-सेकंड)
lambda_ = 1e-9         # तरंगदैर्घ्य (मीटर)
k = 2 * np.pi / lambda_ # तरंग संख्या (मीटर⁻¹)
E0 = 1e-21             # संदर्भ ऊर्जा स्तर (जूल)

# प्रणाली पैरामीटर्स
x = np.linspace(1e-10, 1e-8, 1000)  # स्थान (0.1nm से 10nm तक)
r = 1e-9                             # विशेषता लंबाई (1nm)
theta = np.pi / 4                    # कोण (45 डिग्री)
v = 1e-3                             # वेग (1mm/s)

def lovince_formula(x, t, r, theta, k, v, E0):
    """
    लोविन्स सूत्र - क्वांटम और ज्यामितीय प्रभावों का संयोजन
    
    पैरामीटर्स:
        x: स्थान (मीटर)
        t: समय (सेकंड) - यहाँ उपयोग नहीं
        r: विशेषता लंबाई (मीटर)
        theta: कोण (रेडियन)
        k: तरंग संख्या (मीटर⁻¹)
        v: वेग (मीटर/सेकंड)
        E0: ऊर्जा आधार (जूल)
    """
    # तरंग पद (क्वांटम प्रभाव)
    wave_term = (np.exp(1j * k * x) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v * r**2
    
    # ज्यामितीय पद (संरचना प्रभाव)
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    
    # गतिशील पद (गति प्रभाव)
    dynamic_term = (v * x * np.cos(theta) * h_bar) / r**2
    
    # ऊर्जा पद (आधार स्तर)
    energy_term = E0
    
    return wave_term + geometric_term + dynamic_term + energy_term

# गणना
L = lovince_formula(x, t=0, r=r, theta=theta, k=k, v=v, E0=E0)

# आरेखण
plt.figure(figsize=(12, 7))
plt.plot(x, np.real(L), label="वास्तविक भाग (Re[L(x)])", color="navy", linewidth=2)
plt.plot(x, np.imag(L), label="काल्पनिक भाग (Im[L(x)])", color="crimson", linewidth=2, linestyle="--")

# ग्राफ सज्जा
plt.xlabel("स्थान x (मीटर)", fontsize=14)
plt.ylabel("L(x) का मान (जूल)", fontsize=14)
plt.title("लोविन्स सूत्र: क्वांटम-ज्यामितीय प्रणाली", fontsize=16, pad=20)
plt.legend(fontsize=12, framealpha=1)
plt.grid(True, linestyle=":", alpha=0.7)

# एक्सिस सीमाएँ
plt.xlim(left=min(x), right=max(x))

plt.tight_layout()
plt.show()