from decimal import Decimal, getcontext
import math
from typing import List, Optional, Tuple, Union
from scipy.constants import h, hbar, m_e  # Physical constants


class QuantumNumberAnalyzer:
    """
    Analyze irrational numbers, Collatz sequences, Schrödinger wave functions,
    and compute quantum energies for various systems.
    """

    def __init__(self, precision: int = 100, digit_limit: int = 50):
        """Initialize with decimal precision and digit extraction limit."""
        getcontext().prec = precision
        self.digit_limit = digit_limit

        # Mathematical and physical constants with high precision
        self.constants = {
            "φ": (Decimal(1) + Decimal(5).sqrt()) / 2,
            "√2": Decimal(2).sqrt(),
            "√5": Decimal(5).sqrt(),
            "π": Decimal(str(math.pi)),
            "e": Decimal(str(math.e)),
            "h": Decimal(str(h)),      # Planck's constant (J·s)
            "ħ": Decimal(str(hbar)),   # Reduced Planck's constant (J·s)
            "m_e": Decimal(str(m_e)),  # Electron mass (kg)
        }

        self.collatz_start = 27

        # Parameters for quantum systems
        self.quantum_system_params = {
            "particle_in_box": {
                "n": 1,
                "L": Decimal("1E-9"),  # 1 nm box width
                "m": self.constants["m_e"],
            },
            "harmonic_oscillator": {
                "n": 1,
                "ω": Decimal("1E15"),  # Angular frequency in rad/s
            },
            "hydrogen_atom": {
                "n": 1,
            },
        }

    def calculate_quantum_energies(self) -> dict:
        """
        Calculate energies for particle in a box, harmonic oscillator,
        and hydrogen atom (in eV).
        """
        energies = {}

        # Particle in a box energy: E = n^2 h^2 / (8 m L^2)
        n = self.quantum_system_params["particle_in_box"]["n"]
        L = self.quantum_system_params["particle_in_box"]["L"]
        m = self.quantum_system_params["particle_in_box"]["m"]
        h = self.constants["h"]
        E_box = (Decimal(n) ** 2 * h ** 2) / (8 * m * L ** 2)
        energies["particle_in_box"] = E_box

        # Quantum harmonic oscillator energy: E = (n + 1/2) ħ ω
        n = self.quantum_system_params["harmonic_oscillator"]["n"]
        ω = self.quantum_system_params["harmonic_oscillator"]["ω"]
        ħ = self.constants["ħ"]
        E_ho = (Decimal(n) + Decimal("0.5")) * ħ * ω
        energies["harmonic_oscillator"] = E_ho

        # Hydrogen atom ground state energy (in eV)
        n = self.quantum_system_params["hydrogen_atom"]["n"]
        E_h = Decimal("-13.6") / (Decimal(n) ** 2)
        energies["hydrogen_atom"] = E_h

        return energies

    def collatz_sequence(self) -> List[int]:
        """Generate Collatz sequence starting from self.collatz_start."""
        sequence = []
        current = self.collatz_start
        while current != 1:
            sequence.append(current)
            current = current // 2 if current % 2 == 0 else 3 * current + 1
        sequence.append(1)
        return sequence

    def schrodinger_wave(self, points: int = 10) -> List[Decimal]:
        """
        Compute wave function values for a particle in an infinite potential well.

        :param points: Number of points to calculate.
        :return: List of absolute wave function values.
        """
        n = self.quantum_system_params["particle_in_box"]["n"]
        L = self.quantum_system_params["particle_in_box"]["L"]
        psi_values = []

        for i in range(1, points + 1):
            x = Decimal(i) / points * L
            sin_arg = n * Decimal(str(math.pi)) * x / L
            psi = (Decimal(2) / L).sqrt() * Decimal(str(math.sin(float(sin_arg))))
            psi_values.append(abs(psi))

        return psi_values

    def extract_digits(
        self,
        number: Optional[Decimal] = None,
        integers: Optional[List[int]] = None,
        decimals: Optional[List[Decimal]] = None,
    ) -> str:
        """
        Extract digits from a number's fractional part, list of integers, or list of decimals.

        :return: String of digits limited by self.digit_limit.
        """
        if number is not None:
            parts = str(number).split(".")
            return parts[1][: self.digit_limit] if len(parts) > 1 else ""

        if integers is not None:
            digits = "".join(str(n) for n in integers)
            return digits[: self.digit_limit]

        if decimals is not None:
            digits = "".join(str(d).split(".")[-1] for d in decimals)
            return digits[: self.digit_limit]

        return ""

    def blend_digits(self, digit_sequences: List[str]) -> str:
        """
        Interleave digits from multiple sequences.

        :param digit_sequences: List of digit strings.
        :return: Blended digit string.
        """
        if not digit_sequences or any(not seq for seq in digit_sequences):
            return ""

        max_length = max(len(seq) for seq in digit_sequences)
        blended = []

        for i in range(max_length):
            for seq in digit_sequences:
                if i < len(seq):
                    blended.append(seq[i])

        return "".join(blended)

    def check_pattern(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """
        Detect repeating patterns in a digit sequence.

        :return: Tuple (pattern_found, pattern)
        """
        seq_len = len(sequence)
        if seq_len < 6:
            return False, None

        max_pattern_len = min(10, seq_len // 2)

        for length in range(3, max_pattern_len + 1):
            for start in range(seq_len - 2 * length + 1):
                if sequence[start : start + length] == sequence[start + length : start + 2 * length]:
                    return True, sequence[start : start + length]

        return False, None

    def compute_product(self, psi_values: List[Decimal]) -> Decimal:
        """
        Compute product of constants multiplied by the norm of the wave function.

        :param psi_values: Wave function values.
        :return: Decimal product.
        """
        product = Decimal(1)
        for val in self.constants.values():
            product *= val

        norm_sq = sum(psi ** 2 for psi in psi_values) / Decimal(len(psi_values))
        norm = norm_sq.sqrt()

        return product * norm

    def quantum_sequence(self, blended_digits: str) -> List[Decimal]:
        """
        Generate a quantum-inspired sequence from blended digits.

        :param blended_digits: Digit string.
        :return: List of Decimals.
        """
        if not blended_digits:
            return [Decimal(0)] * 5

        seed_str = blended_digits[:20].ljust(20, "0")
        seed = Decimal(f"0.{seed_str}")

        total = sum(int(d) for d in blended_digits[: self.digit_limit] if d.isdigit()) or 1

        probs = [Decimal(int(d)) / total for d in blended_digits[:5] if d.isdigit()]

        return [seed * p for p in probs]

    def analyze(self) -> None:
        """Perform integrated analysis and print results."""
        print("=== Quantum Number Analysis ===")

        # Quantum energies
        energies = self.calculate_quantum_energies()
        print("\nQuantum Energy Calculations:")
        print(f"Particle in a Box (n=1): {energies['particle_in_box']:.6e} J")
        print(f"Harmonic Oscillator (n=1): {energies['harmonic_oscillator']:.6e} J")
        print(f"Hydrogen Atom (n=1): {energies['hydrogen_atom']} eV")

        # Extract digits from constants
        digit_sequences = []
        print("\nExtracted Digits:")
        for name, value in self.constants.items():
            digits = self.extract_digits(number=value)
            digit_sequences.append(digits)
            print(f"{name}: {digits[:30]}{'...' if len(digits) > 30 else ''}")

        # Collatz sequence digits
        collatz_seq = self.collatz_sequence()
        collatz_digits = self.extract_digits(integers=collatz_seq)
        digit_sequences.append(collatz_digits)
        print(f"Collatz (start={self.collatz_start}): {collatz_digits[:30]}{'...' if len(collatz_digits) > 30 else ''}")

        # Schrödinger wave digits
        psi_values = self.schrodinger_wave()
        psi_digits = self.extract_digits(decimals=psi_values)
        digit_sequences.append(psi_digits)
        print(f"Schrödinger Wave: {psi_digits[:30]}{'...' if len(psi_digits) > 30 else ''}")

        # Blend digits
        blended = self.blend_digits(digit_sequences)
        print(f"\nBlended Sequence (first {self.digit_limit} digits): {blended[:self.digit_limit]}")

        # Pattern check in blended sequence
        has_pattern, pattern = self.check_pattern(blended)
        print(f"Pattern check in blended sequence: {'⚠️ Found pattern: ' + pattern if has_pattern else '✅ No repeating pattern found.'}")

        # Compute product and pattern check
        product = self.compute_product(psi_values)
        product_digits = self.extract_digits(number=product)
        print(f"\nProduct of constants and wave function norm: {product:.6e}")
        print(f"Product digits (first {self.digit_limit}): {product_digits[:self.digit_limit]}")
        has_pattern, pattern = self.check_pattern(product_digits)
        print(f"Pattern check in product digits: {'⚠️ Found pattern: ' + pattern if has_pattern else '✅ No repeating pattern found.'}")

        # Quantum-inspired sequence
        q_seq = self.quantum_sequence(blended)
        print("\nQuantum-Inspired Sequence:")
        print("[")
        for val in q_seq:
            print(f"  {val:.6f}")
        print("]")


def main() -> None:
    analyzer = QuantumNumberAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
