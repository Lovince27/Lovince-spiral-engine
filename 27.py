from decimal import Decimal, getcontext
import math

class QuantumNumberAnalyzer:
    """Analyze irrational numbers, Collatz sequences, and Schrödinger wave functions."""
    
    def __init__(self, precision=100, digit_limit=50):
        """Initialize with precision and digit extraction limit."""
        getcontext().prec = precision
        self.digit_limit = digit_limit
        self.constants = {
            "φ": (1 + Decimal(5).sqrt()) / 2,
            "√2": Decimal(2).sqrt(),
            "√5": Decimal(5).sqrt(),
            "π": Decimal(str(math.pi)),
            "e": Decimal(str(math.e))
        }
        self.collatz_start = 27  # From user's query
        self.schrodinger_params = {
            "n": 1,  # Quantum number
            "L": Decimal("1E-9"),  # Well width (1 nm)
        }

    def collatz_sequence(self):
        """Generate Collatz sequence for the starting number."""
        sequence = []
        current = self.collatz_start
        while current != 1:
            sequence.append(current)
            current = current // 2 if current % 2 == 0 else 3 * current + 1
        sequence.append(1)
        return sequence

    def schrodinger_wave(self):
        """Compute wave function values for a particle in an infinite potential well."""
        n = self.schrodinger_params["n"]
        L = self.schrodinger_params["L"]
        points = 10
        psi_values = []
        for i in range(1, points + 1):
            x = Decimal(i) / points * L
            sin_arg = n * Decimal(str(math.pi)) * x / L
            psi = (Decimal(2) / L).sqrt() * Decimal(str(math.sin(float(sin_arg))))
            psi_values.append(abs(psi))
        return psi_values

    def extract_digits(self, number=None, integers=None, decimals=None):
        """Extract digits from a number, integer list, or decimal list."""
        if number is not None:
            return str(number).split('.')[-1][:self.digit_limit].rstrip('0')
        if integers is not None:
            return ''.join(str(n) for n in integers)[:self.digit_limit]
        if decimals is not None:
            return ''.join(str(d).split('.')[-1] for d in decimals)[:self.digit_limit]
        return ""

    def blend_digits(self, digit_sequences):
        """Interleave digits from multiple sequences."""
        if not digit_sequences or not all(digit_sequences):
            return ""
        max_len = max(len(seq) for seq in digit_sequences)
        blended = []
        for i in range(max_len):
            for seq in digit_sequences:
                if i < len(seq):
                    blended.append(seq[i])
        return ''.join(blended)

    def check_pattern(self, sequence):
        """Check for repeating patterns in a sequence."""
        if len(sequence) < 6:
            return False, None
        for length in range(3, min(11, len(sequence) // 2 + 1)):
            for start in range(len(sequence) - 2 * length + 1):
                if sequence[start:start + length] == sequence[start + length:start + 2 * length]:
                    return True, sequence[start:start + length]
        return False, None

    def compute_product(self, psi_values):
        """Compute product of constants and wave function norm."""
        product = Decimal(1)
        for value in self.constants.values():
            product *= value
        norm = sum(psi ** 2 for psi in psi_values) / len(psi_values)
        return product * norm.sqrt()

    def quantum_sequence(self, blended_digits):
        """Generate a quantum-inspired sequence from blended digits."""
        if not blended_digits:
            return [Decimal(0)] * 5
        # Use first 20 digits as a seed
        seed = Decimal(f"0.{blended_digits[:20]}")
        # Normalize to mimic probability density
        total = sum(int(d) for d in blended_digits[:self.digit_limit])
        if total == 0:
            total = 1
        probs = [Decimal(int(d)) / total for d in blended_digits[:5]]
        # Scale by seed to create a sequence
        return [seed * p for p in probs]

    def analyze(self):
        """Perform integrated analysis and display results."""
        print("=== Quantum Number Analysis ===")
        
        # Extract digits
        digit_sequences = []
        print("\nExtracted Digits:")
        for name, value in self.constants.items():
            digits = self.extract_digits(number=value)
            digit_sequences.append(digits)
            print(f"{name}: {digits[:30]}...")

        collatz_seq = self.collatz_sequence()
        collatz_digits = self.extract_digits(integers=collatz_seq)
        digit_sequences.append(collatz_digits)
        print(f"Collatz (I={self.collatz_start}): {collatz_digits[:30]}...")

        psi_values = self.schrodinger_wave()
        psi_digits = self.extract_digits(decimals=psi_values)
        digit_sequences.append(psi_digits)
        print(f"Schrödinger Wave: {psi_digits[:30]}...")

        # Blend digits
        blended = self.blend_digits(digit_sequences)
        print(f"\nBlended Sequence (first 50): {blended[:50]}")
        has_pattern, pattern = self.check_pattern(blended)
        print(f"Pattern: {'ablacklist = {'⚠️ Found: ' + pattern if has_pattern else '✅ None'}")

        # Compute product
        product = self.compute_product(psi_values)
        product_digits = self.extract_digits(number=product)
        print(f"\nProduct: {product:.6f}")
        print(f"Digits: {product_digits[:50]}")
        has_pattern, pattern = self.check_pattern(product_digits)
        print(f"Pattern: {'⚠️ Found: ' + pattern if has_pattern else '✅ None'}")

        # Quantum-inspired sequence
        print(f"\nQuantum-Inspired Sequence:")
        q_seq = self.quantum_sequence(blended)
        print("[")
        for val in q_seq:
            print(f"  {val:.6f}")
        print("]")

def main():
    """Run the analysis."""
    analyzer = QuantumNumberAnalyzer()
    analyzer.analyze()

if __name__ == "__main__":
    main()


