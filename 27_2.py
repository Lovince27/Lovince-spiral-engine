from decimal import Decimal, getcontext
import math
from typing import List, Optional, Tuple


class QuantumNumberAnalyzer:
    """Analyze irrational numbers, Collatz sequences, Schrödinger waves, quantum energies, and Consciousness Bridge Metric."""

    def __init__(self, precision: int = 100, digit_limit: int = 50):
        """Initialize with decimal precision and digit extraction limit."""
        getcontext().prec = precision
        self.digit_limit = digit_limit

        self.constants = {
            "φ": (Decimal(1) + Decimal(5).sqrt()) / 2,
            "√2": Decimal(2).sqrt(),
            "√5": Decimal(5).sqrt(),
            "π": Decimal(str(math.pi)),
            "e": Decimal(str(math.e)),
            "h": Decimal("6.62607015E-34"),    # Planck's constant (J·s)
            "ħ": Decimal("1.054571817E-34"),   # Reduced Planck's constant (J·s)
            "m_e": Decimal("9.1093837E-31"),   # Electron mass (kg)
        }

        self.collatz_start = 27

        self.quantum_system_params = {
            "particle_in_box": {"n": 1, "L": Decimal("1E-9"), "m": self.constants["m_e"]},
            "harmonic_oscillator": {"n": 1, "ω": Decimal("1E15")},
            "hydrogen_atom": {"n": 1},
        }

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
        """Compute wave function values for a particle in an infinite potential well."""
        n = self.quantum_system_params["particle_in_box"]["n"]
        L = self.quantum_system_params["particle_in_box"]["L"]
        psi_values = []
        for i in range(1, points + 1):
            x = Decimal(i) / points * L
            sin_arg = n * Decimal(str(math.pi)) * x / L
            psi = (Decimal(2) / L).sqrt() * Decimal(str(math.sin(float(sin_arg))))
            psi_values.append(abs(psi))
        return psi_values

    def calculate_quantum_energies(self) -> dict:
        """Calculate energies for particle in a box, harmonic oscillator, and hydrogen atom."""
        energies = {}
        # Particle in a box: E = n^2 h^2 / (8 m L^2)
        n = self.quantum_system_params["particle_in_box"]["n"]
        L = self.quantum_system_params["particle_in_box"]["L"]
        m = self.quantum_system_params["particle_in_box"]["m"]
        h = self.constants["h"]
        energies["particle_in_box"] = (Decimal(n) ** 2 * h ** 2) / (8 * m * L ** 2)

        # Harmonic oscillator: E = (n + 1/2) ħ ω
        n = self.quantum_system_params["harmonic_oscillator"]["n"]
        ω = self.quantum_system_params["harmonic_oscillator"]["ω"]
        ħ = self.constants["ħ"]
        energies["harmonic_oscillator"] = (Decimal(n) + Decimal("0.5")) * ħ * ω

        # Hydrogen atom: E = -13.6 / n^2 eV, converted to joules (1 eV = 1.602176634E-19 J)
        n = self.quantum_system_params["hydrogen_atom"]["n"]
        E_h_eV = Decimal("-13.6") / (Decimal(n) ** 2)
        energies["hydrogen_atom"] = E_h_eV * Decimal("1.602176634E-19")

        return energies

    def extract_digits(
        self,
        number: Optional[Decimal] = None,
        integers: Optional[List[int]] = None,
        decimals: Optional[List[Decimal]] = None,
    ) -> str:
        """Extract digits from a number, integer list, or decimal list."""
        if number is not None:
            parts = str(number).split(".")
            return parts[1][: self.digit_limit].rstrip("0") if len(parts) > 1 else ""
        if integers is not None:
            return "".join(str(n) for n in integers)[: self.digit_limit]
        if decimals is not None:
            return "".join(str(d).split(".")[-1] for d in decimals)[: self.digit_limit]
        return ""

    def blend_digits(self, digit_sequences: List[str]) -> str:
        """Interleave digits from multiple sequences."""
        if not digit_sequences or any(not seq for seq in digit_sequences):
            return ""
        max_length = max(len(seq) for seq in digit_sequences)
        return "".join(
            seq[i] for i in range(max_length) for seq in digit_sequences if i < len(seq)
        )

    def check_pattern(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """Detect repeating patterns in a sequence."""
        if len(sequence) < 6:
            return False, None
        for length in range(3, min(11, len(sequence) // 2 + 1)):
            for start in range(len(sequence) - 2 * length + 1):
                if sequence[start : start + length] == sequence[start + length : start + 2 * length]:
                    return True, sequence[start : start + length]
        return False, None

    def compute_product(self, psi_values: List[Decimal]) -> Decimal:
        """Compute product of constants and wave function norm."""
        product = Decimal(1)
        for val in self.constants.values():
            product *= val
        norm = (sum(psi ** 2 for psi in psi_values) / Decimal(len(psi_values))).sqrt()
        return product * norm

    def quantum_sequence(self, blended_digits: str) -> List[Decimal]:
        """Generate a quantum-inspired sequence from blended digits."""
        if not blended_digits:
            return [Decimal(0)] * 5
        seed = Decimal(f"0.{blended_digits[:20].ljust(20, '0')}")
        total = sum(int(d) for d in blended_digits[: self.digit_limit] if d.isdigit()) or 1
        probs = [Decimal(int(d)) / total for d in blended_digits[:5] if d.isdigit()]
        return [seed * p for p in probs] + [Decimal(0)] * (5 - len(probs))

    def digit_distribution(self, sequence: str) -> dict:
        """Calculate frequency of digits (0-9) in a sequence."""
        dist = {str(i): 0 for i in range(10)}
        for d in sequence:
            if d.isdigit():
                dist[d] += 1
        return dist

    def sequence_entropy(self, sequence: str) -> Decimal:
        """Calculate Shannon entropy of digit distribution in a sequence."""
        dist = self.digit_distribution(sequence)
        total = sum(dist.values())
        if total == 0:
            return Decimal(0)
        entropy = Decimal(0)
        for count in dist.values():
            if count > 0:
                p = Decimal(count) / total
                entropy -= p * Decimal(str(math.log2(float(p))))
        return entropy

    def consciousness_bridge_metric(
        self,
        blended_sequence: str,
        energies: dict,
        collatz_seq: List[int],
        psi_values: List[Decimal],
    ) -> Decimal:
        """Compute Consciousness Bridge Metric (CBM)."""
        H = self.sequence_entropy(blended_sequence)
        E_box = energies["particle_in_box"]
        E_ho = energies["harmonic_oscillator"]
        E_H = abs(energies["hydrogen_atom"])
        energy_ratio = (E_box + E_ho) / E_H if E_H != 0 else Decimal(1)
        L_collatz = Decimal(len(collatz_seq))
        N_psi = (sum(psi ** 2 for psi in psi_values) / Decimal(len(psi_values))).sqrt()
        cbm = H * energy_ratio * Decimal(str(math.log(float(L_collatz)))) * N_psi
        return cbm if cbm.is_finite() else Decimal(0)

    def validate(
        self, digit_sequences: List[str], blended: str, product: Decimal, cbm: Decimal
    ) -> bool:
        """Validate digit sequences, blended output, product, and CBM."""
        if any(len(seq) > self.digit_limit for seq in digit_sequences):
            return False
        expected_len = sum(len(seq) for seq in digit_sequences)
        if len(blended) > expected_len:
            return False
        if not product.is_finite() or product <= 0:
            return False
        if not cbm.is_finite():
            return False
        return True

    def analyze(self) -> None:
        """Perform integrated analysis and print results."""
        print("=== Quantum Number Analysis ===")

        # Quantum energies
        energies = self.calculate_quantum_energies()
        print("\nQuantum Energies:")
        print(f"Particle in Box (n=1): {energies['particle_in_box']:.2e} J")
        print(f"Harmonic Oscillator (n=1): {energies['harmonic_oscillator']:.2e} J")
        print(f"Hydrogen Atom (n=1): {energies['hydrogen_atom']:.2e} J (-13.6 eV)")

        # Extract digits
        digit_sequences = []
        print("\nExtracted Digits:")
        for name, value in self.constants.items():
            digits = self.extract_digits(number=value)
            digit_sequences.append(digits)
            print(f"{name}: {digits[:30]}{'...' if len(digits) > 30 else ''}")

        collatz_seq = self.collatz_sequence()
        collatz_digits = self.extract_digits(integers=collatz_seq)
        digit_sequences.append(collatz_digits)
        print(f"Collatz (start={self.collatz_start}): {collatz_digits[:30]}{'...' if len(collatz_digits) > 30 else ''}")

        psi_values = self.schrodinger_wave()
        psi_digits = self.extract_digits(decimals=psi_values)
        digit_sequences.append(psi_digits)
        print(f"Schrödinger Wave: {psi_digits[:30]}{'...' if len(psi_digits) > 30 else ''}")

        # Blend digits
        blended = self.blend_digits(digit_sequences)
        print(f"\nBlended Sequence (first {self.digit_limit}): {blended[:self.digit_limit]}")
        has_pattern, pattern = self.check_pattern(blended)
        print(f"Pattern: {'⚠️ Found: ' + pattern if has_pattern else '✅ None'}")

        # Digit distribution and entropy
        dist = self.digit_distribution(blended)
        entropy = self.sequence_entropy(blended)
        print("\nDigit Distribution in Blended Sequence:")
        print(", ".join(f"{k}: {v}" for k, v in sorted(dist.items())))
        print(f"Entropy: {entropy:.4f} bits")

        # Product
        product = self.compute_product(psi_values)
        product_digits = self.extract_digits(number=product)
        print(f"\nProduct: {product:.2e}")
        print(f"Digits: {product_digits[:self.digit_limit]}")
        has_pattern, pattern = self.check_pattern(product_digits)
        print(f"Pattern: {'⚠️ Found: ' + pattern if has_pattern else '✅ None'}")

        # Consciousness Bridge Metric
        cbm = self.consciousness_bridge_metric(blended, energies, collatz_seq, psi_values)
        print(f"\nConsciousness Bridge Metric: {cbm:.2e}")

        # Quantum-inspired sequence
        q_seq = self.quantum_sequence(blended)
        print("\nQuantum-Inspired Sequence:")
        print("[")
        for val in q_seq:
            print(f"  {val:.6f}")
        print("]")

        # Validation
        valid = self.validate(digit_sequences, blended, product, cbm)
        print(f"\nValidation: {'✅ Passed' if valid else '⚠️ Failed'}")


def main() -> None:
    analyzer = QuantumNumberAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
