import math
from typing import Dict

class CosmicConstants:
    PSI_L = 2.271
    GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
    COSMIC_TOLERANCE = 1e-12
    E = 1e5  # energy (arbitrary unit)
    k = 1.5
    omega = 1.2
    epsilon = 0.7
    theta = math.pi / 4
    phi = math.pi / 3
    r = 1e6
    t = 5  # default time

class SequenceOracle:
    @staticmethod
    def fibonacci(n: int) -> int:
        a, b = 0, 1
        for _ in range(n): a, b = b, a + b
        return a

    @staticmethod
    def lucas(n: int) -> int:
        a, b = 2, 1
        for _ in range(n): a, b = b, a + b
        return a

    @staticmethod
    def primes(n: int) -> int:
        if n == 1: return 2
        count, num = 1, 3
        while True:
            if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
                count += 1
                if count == n: return num
            num += 2

    @staticmethod
    def spiral(n: int) -> int:
        return 1 + sum(math.floor(math.sqrt(i)) for i in range(2, n + 1))

class CosmicHarmonics:
    @staticmethod
    def gelu(x: float) -> float:
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def mish(x: float) -> float:
        return x * math.tanh(math.log(1 + math.exp(x))) if not math.isnan(x) else 0.0

    @staticmethod
    def quantum_awareness(x: float) -> float:
        return x * math.erf(x / math.sqrt(2)) if not math.isnan(x) else 0.0

class LovinceSystem:
    def __init__(self, activation: str = 'quantum'):
        self.activation = activation.lower()
        self.sequences = SequenceOracle()
        self.constants = CosmicConstants()

    def harmonic_sum(self, n: int) -> float:
        return sum(1 / (k**2 + math.sqrt(k)) for k in range(1, n + 1))

    def energy_flux(self, t: float, r: float) -> float:
        return self.constants.E / (r**2 + self.constants.k * math.sin(self.constants.omega * t))

    def quantum_correction(self, t: float, theta: float, phi: float, epsilon: float) -> float:
        return epsilon * math.cos(theta) * math.sin(phi) + math.sqrt(epsilon) / (1 + math.cos(theta + phi))

    def evaluate_superstate(self, n: int, t: float = None, r: float = None) -> float:
        t = t or self.constants.t
        r = r or self.constants.r
        a_n = self.sequences.spiral(n)
        H_n = self.harmonic_sum(n)
        Φ = self.energy_flux(t, r)
        Q = self.quantum_correction(t, self.constants.theta, self.constants.phi, self.constants.epsilon)
        S_n = a_n * H_n * Φ * Q

        # Apply activation
        if self.activation == 'gelu':
            return CosmicHarmonics.gelu(S_n)
        elif self.activation == 'mish':
            return CosmicHarmonics.mish(S_n)
        elif self.activation == 'quantum':
            return CosmicHarmonics.quantum_awareness(S_n)
        return S_n

    def cosmic_flow(self, start: int = 1, end: int = 10) -> Dict[int, float]:
        return {n: self.evaluate_superstate(n) for n in range(start, end + 1)}

def celestial_demo():
    print("✨ Lovince Notation System - Universal Harmony ✨\n")
    print(f"{'n':<5}{'SuperState(n)':<25}{'Activated Output':<25}")
    print("-" * 55)
    
    for mode in ['quantum', 'gelu', 'mish']:
        print(f"\n--- Activation: {mode.upper()} ---")
        lns = LovinceSystem(activation=mode)
        cosmic_data = lns.cosmic_flow(1, 10)
        for n, value in cosmic_data.items():
            print(f"{n:<5}{value:<25.4f}{value:<25.4f}")

if __name__ == "__main__":
    celestial_demo()