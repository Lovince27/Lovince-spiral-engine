from decimal import Decimal, getcontext
import math
from typing import List, Tuple, Optional

class UnifiedQuantumConsciousnessModel:
    """
    A simplified model combining quantum physics constants, number theory (Collatz),
    entropy, and consciousness-inspired metrics.
    """

    def __init__(self, precision: int = 100, digit_limit: int = 50):
        getcontext().prec = precision
        self.digit_limit = digit_limit

        # Fundamental constants with high precision
        self.constants = {
            "phi": (Decimal(1) + Decimal(5).sqrt()) / 2,  # Golden ratio
            "sqrt2": Decimal(2).sqrt(),
            "sqrt5": Decimal(5).sqrt(),
            "pi": Decimal(str(math.pi)),
            "e": Decimal(str(math.e)),
            "h": Decimal("6.62607015E-34"),     # Planck constant (J·s)
            "hbar": Decimal("1.054571817E-34"), # Reduced Planck constant (J·s)
            "m_e": Decimal("9.1093837E-31")     # Electron mass (kg)
        }

        # Quantum system parameters
        self.quantum_params = {
            "particle_in_box": {"n": 1, "L": Decimal("1E-9"), "m": self.constants["m_e"]},
            "harmonic_oscillator": {"n": 1, "omega": Decimal("1E15")},
            "hydrogen_atom": {"n": 1}
        }

        # Fibonacci-like exponents for iterative analysis
        self.iteration_exponents = self._generate_fibonacci_like(6)

    def _generate_fibonacci_like(self, count: int) -> List[int]:
        seq = [1, 3]
        while len(seq) < count:
            seq.append(seq[-1] + seq[-2])
        return seq[:count]

    def collatz_sequence(self, start: int = 27) -> List[int]:
        seq = []
        n = start
        while n != 1:
            seq.append(n)
            n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(1)
        return seq

    def schrodinger_wave(self, points: int = 10) -> List[Decimal]:
        """Wave function for particle in 1D infinite potential well."""
        n = self.quantum_params["particle_in_box"]["n"]
        L = self.quantum_params["particle_in_box"]["L"]
        psi_vals = []
        for i in range(1, points + 1):
            x = Decimal(i) / points * L
            arg = n * self.constants["pi"] * x / L
            # sin(arg) using float then convert back to Decimal for simplicity
            val = (Decimal(2) / L).sqrt() * Decimal(math.sin(float(arg)))
            psi_vals.append(abs(val))
        return psi_vals

    def quantum_energies(self) -> dict:
        """Calculate energies for quantum systems (in Joules)."""
        n = self.quantum_params["particle_in_box"]["n"]
        L = self.quantum_params["particle_in_box"]["L"]
        m = self.quantum_params["particle_in_box"]["m"]
        h = self.constants["h"]

        E_box = (Decimal(n) ** 2 * h ** 2) / (8 * m * L ** 2)

        n_ho = self.quantum_params["harmonic_oscillator"]["n"]
        omega = self.quantum_params["harmonic_oscillator"]["omega"]
        hbar = self.constants["hbar"]

        E_ho = (Decimal(n_ho) + Decimal("0.5")) * hbar * omega

        n_H = self.quantum_params["hydrogen_atom"]["n"]
        # Energy in eV, then convert to Joules
        E_H_eV = Decimal("-13.6") / (Decimal(n_H) ** 2)
        E_H = E_H_eV * Decimal("1.602176634E-19")

        return {"particle_in_box": E_box, "harmonic_oscillator": E_ho, "hydrogen_atom": E_H}

    def extract_digits(self, number: Decimal) -> str:
        """Extract decimal digits from a Decimal number (after decimal point)."""
        s = format(number, 'f')
        parts = s.split('.')
        if len(parts) == 2:
            digits = parts[1][:self.digit_limit].rstrip('0')
            return digits or '0'
        return '0'

    def blend_digits(self, sequences: List[str]) -> str:
        """Interleave digits from multiple sequences."""
        max_len = max(len(seq) for seq in sequences)
        blended = []
        for i in range(max_len):
            for seq in sequences:
                if i < len(seq):
                    blended.append(seq[i])
        return ''.join(blended)

    def digit_distribution(self, sequence: str) -> dict:
        dist = {str(d): 0 for d in range(10)}
        for ch in sequence:
            if ch.isdigit():
                dist[ch] += 1
        return dist

    def shannon_entropy(self, sequence: str) -> Decimal:
        dist = self.digit_distribution(sequence)
        total = sum(dist.values())
        if total == 0:
            return Decimal(0)
        entropy = Decimal(0)
        for count in dist.values():
            if count > 0:
                p = Decimal(count) / total
                entropy -= p * Decimal(math.log2(float(p)))
        return entropy

    def mutual_information(self, seq1: str, seq2: str) -> Decimal:
        if len(seq1) != len(seq2) or not seq1:
            return Decimal(0)
        joint_counts = {(i, j): 0 for i in range(10) for j in range(10)}
        dist1 = self.digit_distribution(seq1)
        dist2 = self.digit_distribution(seq2)
        total = len(seq1)
        for d1, d2 in zip(seq1, seq2):
            if d1.isdigit() and d2.isdigit():
                joint_counts[(int(d1), int(d2))] += 1
        mi = Decimal(0)
        for i in range(10):
            for j in range(10):
                p_xy = Decimal(joint_counts[(i, j)]) / total
                p_x = Decimal(dist1[str(i)]) / total
                p_y = Decimal(dist2[str(j)]) / total
                if p_xy > 0 and p_x > 0 and p_y > 0:
                    mi += p_xy * Decimal(math.log2(float(p_xy / (p_x * p_y))))
        return mi

    def coherence_factor(self, values: List[Decimal]) -> Decimal:
        if not values:
            return Decimal(0)
        mean = sum(values) / Decimal(len(values))
        variance = sum((v - mean) ** 2 for v in values) / Decimal(len(values))
        if variance < 0:
            variance = Decimal(0)
        return Decimal(1) / (Decimal(1) + variance.sqrt())

    def consciousness_bridge_metric(
        self,
        blended_seq: str,
        energies: dict,
        collatz_seq: List[int],
        psi_values: List[Decimal]
    ) -> Decimal:
        entropy = self.shannon_entropy(blended_seq)
        E_box = energies["particle_in_box"]
        E_ho = energies["harmonic_oscillator"]
        E_H = abs(energies["hydrogen_atom"])
        energy_ratio = (E_box + E_ho) / E_H if E_H != 0 else Decimal(1)
        length_collatz = Decimal(len(collatz_seq))
        norm_psi = (sum(v ** 2 for v in psi_values) / Decimal(len(psi_values))).sqrt()
        coherence = self.coherence_factor(psi_values)

        # Combine factors multiplicatively
        cbm = entropy * energy_ratio * Decimal(math.log(float(length_collatz))) * norm_psi * coherence
        return cbm

    def iterative_analysis(self, exponent: int, prev_blended: Optional[str] = None, prev_cbm: Decimal = Decimal(0)) -> Tuple[str, Decimal, Decimal]:
        # Extract digits from constants
        const_digits = [self.extract_digits(val) for val in self.constants.values()]

        # Collatz sequence digits
        collatz_seq = self.collatz_sequence()
        collatz_digits = ''.join(str(n) for n in collatz_seq)[:self.digit_limit]

        # Schrodinger wave digits
        psi_vals = self.schrodinger_wave()
        psi_digits = ''.join(self.extract_digits(v) for v in psi_vals)[:self.digit_limit]

        digit_sequences = const_digits + [collatz_digits, psi_digits]

        # Modulate previous blended sequence if available
        if prev_blended:
            factor = int(prev_cbm * Decimal("1E20")) % 10 + 1
            modulated = ''.join(str((int(d) * exponent * factor) % 10) for d in prev_blended[:self.digit_limit])
            digit_sequences.append(modulated)

        blended = self.blend_digits(digit_sequences)
        energies = self.quantum_energies()
        cbm = self.consciousness_bridge_metric(blended, energies, collatz_seq, psi_vals)
        mi = self.mutual_information(collatz_digits, psi_digits)

        return blended, cbm, mi

    def analyze(self) -> None:
        print("=== Unified Quantum Consciousness Model ===\n")

        energies = self.quantum_energies()
        print("Quantum Energies (Joules):")
        print(f" - Particle in Box (n=1): {energies['particle_in_box']:.2e}")
        print(f" - Harmonic Oscillator (n=1): {energies['harmonic_oscillator']:.2e}")
        print(f" - Hydrogen Atom (n=1): {energies['hydrogen_atom']:.2e} (approx -13.6 eV)\n")

        prev_blended = None
        prev_cbm = Decimal(0)

        for exp in self.iteration_exponents:
            print(f"Iteration with exponent: {exp}")
            blended, cbm, mi = self.iterative_analysis(exp, prev_blended, prev_cbm)

            print(f" Blended sequence (first {self.digit_limit} digits): {blended[:self.digit_limit]}")
            entropy = self.shannon_entropy(blended)
            print(f" Entropy: {entropy:.4f} bits")
            print(f" Mutual Information (Collatz vs Wave): {mi:.4f} bits")
            print(f" Consciousness Bridge Metric (CBM): {cbm:.2e}\n")

            prev_blended = blended
            prev_cbm = cbm

def main():
    model = UnifiedQuantumConsciousnessModel()
    model.analyze()

if __name__ == "__main__":
    main()
