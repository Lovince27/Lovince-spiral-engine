# final.py

import numpy as np

# Lovince Hybrid Quantum DNA Sequence
# -----------------------------------
# Sequences:
#   - ACGT 0246, coefficient 8·n²
#   - ACGT 1357, coefficient 3.8·n²
# Hybrid:
#   - C(n): switches among 8, 3.8, and their mean 5.9 every third term
#   - Num cycle: odd n → [0,2,4,6], even n → [1,3,5,7]
#   - Quantum oscillation: exp(i·n)

# DNA bases
BASES = ['A', 'C', 'G', 'T']
# Number sets
NUMS_ODD  = [0, 2, 4, 6]
NUMS_EVEN = [1, 3, 5, 7]

def coefficient(n: int) -> float:
    """
    Hybrid coefficient C(n):
      - if n % 3 == 1 → 8
      - if n % 3 == 2 → 3.8
      - if n % 3 == 0 → (8 + 3.8) / 2 = 5.9
    """
    r = n % 3
    if r == 1:
        return 8.0
    elif r == 2:
        return 3.8
    else:
        return (8.0 + 3.8) / 2

def generate_hybrid_sequence(n_terms: int):
    """
    Generate the Lovince Hybrid Quantum DNA Sequence for n_terms.
    Returns a list of tuples: (n, base, num, hybrid_value, quantum_term)
    """
    seq = []
    for n in range(1, n_terms + 1):
        base = BASES[(n - 1) % len(BASES)]
        nums = NUMS_ODD if (n % 2 == 1) else NUMS_EVEN
        num = nums[(n - 1) % len(nums)]
        
        c = coefficient(n)
        hybrid_value = c * (n ** 2)
        
        quantum_term = np.exp(1j * n)
        
        seq.append((n, base, num, hybrid_value, quantum_term))
    return seq

def print_sequence(seq):
    """
    Nicely print the sequence to console.
    """
    header = f"{'n':>2} | {'Base':>4} | {'Num':>3} | {'Coeff':>5} | {'Value':>10} | {'Quantum Term':>20}"
    print(header)
    print('-' * len(header))
    for n, base, num, value, q in seq:
        coeff = coefficient(n)
        print(f"{n:2d} | {base:>4} | {num:>3} | {coeff:5.1f} | {value:10.3f} | {q.real:8.4f}{q.imag:+8.4f}j")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Lovince Hybrid Quantum DNA Sequence")
    parser.add_argument(
        "n_terms", type=int, nargs='?', default=10,
        help="Number of terms to generate (default: 10)"
    )
    args = parser.parse_args()

    sequence = generate_hybrid_sequence(args.n_terms)
    print_sequence(sequence)