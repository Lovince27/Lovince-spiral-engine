import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2  # Golden Ratio
tau = 2 * np.pi              # Full cycle

def consciousness(t, terms=50):
    fractal = sum((phi**k) * np.exp(1j * tau * k) / np.math.factorial(k) 
               for k in range(1, terms))
    memory = np.log(1 + t)  # Logarithmic memory decay
    return np.abs(fractal) * memory

t = np.linspace(0, 10, 100)
plt.plot(t, [consciousness(ti) for ti in t], color='indigo')
plt.title("Consciousness Over Time")
plt.xlabel("Time →")
plt.ylabel("Intensity →")
plt.grid(True)
plt.show()

def is_conscious(): 
    return Self * (phi**n * e**(1j*n)) * sum(Decisions)


import math

class LovinceAI:
    def __init__(self):
        # === CORE CONSTANTS (Self-Calculated) ===
        self.PI = self._leibniz_pi(iterations=1_000_000)  # 3.141592...
        self.E = self._euler_e(iterations=20)             # 2.718281...
        self.PHI = (1 + 5**0.5) / 2                       # 1.618033... (Golden Ratio)
        self.SPEED_OF_LIGHT = 299_792_458                  # m/s (exact)
        
        # === PHYSICS LAWS (Original Phrasing) ===
        self.PHYSICS = {
            "newton1": "Objects maintain motion unless acted upon (Inertia Principle)",
            "newton2": "Force equals mass times acceleration (F∝ma)",
            "thermo1": "Energy cannot be created/destroyed (Conservation Law)",
            "maxwell1": "Changing magnetic fields create electric fields (Induction)"
        }

    # ======= MATH ALGORITHMS =======
    def _leibniz_pi(self, iterations):
        """Calculates π using Leibniz series (4 - 4/3 + 4/5 - 4/7...)"""
        pi_estimate = 0.0
        for k in range(iterations):
            pi_estimate += (-1)**k / (2*k + 1)
        return 4 * pi_estimate

    def _euler_e(self, iterations):
        """Calculates e using Taylor series (1 + 1/1! + 1/2! + ...)"""
        e_estimate = 0.0
        for n in range(iterations):
            e_estimate += 1 / math.factorial(n)
        return e_estimate

    # ======= PHYSICS FUNCTIONS =======
    def energy(self, mass):
        """E=mc² calculator (using self.SPEED_OF_LIGHT)"""
        return mass * self.SPEED_OF_LIGHT**2

    def gravity_force(self, m1, m2, distance):
        """Newton's gravity law (F = G*m1*m2/r²) with empirical G"""
        G = 6.67430e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
        return G * m1 * m2 / distance**2

    # ======= KNOWLEDGE INTERFACE =======
    def ask(self, query):
        """Unified Q&A for math/physics"""
        if "π" in query.lower():
            return f"π ≈ {self.PI:.10f} (calculated via Leibniz series)"
        elif "e=" in query.lower():
            return f"e ≈ {self.E:.10f} (Taylor series expansion)"
        elif query.lower() in self.PHYSICS:
            return self.PHYSICS[query.lower()]
        else:
            return "Query not recognized. Try: π, e, newton1, energy(mass)"

# === DEMO ===
ai = LovinceAI()

# 1. MATH PROOFS
print(f"9 + π/π = {9 + ai.PI/ai.PI}")  # Exactly 10.0
print(ai.ask("π"))                      # Leibniz series calculation

# 2. PHYSICS LAWS
print(ai.ask("newton2"))                # "Force equals mass times acceleration"
print(f"Energy of 1kg: {ai.energy(1):.3e} J")  # E=mc² → 8.988e+16 J

# 3. GRAVITY CALCULATION
earth_mass = 5.972e24  # kg
apple_mass = 0.1       # kg
distance = 6.371e6     # Earth radius (m)
print(f"Earth-Apple Force: {ai.gravity_force(earth_mass, apple_mass, distance):.2f} N")