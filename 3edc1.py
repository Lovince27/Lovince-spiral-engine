import numpy as np import random import cmath

DNA base numeric encoding

dna_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

def encode_dna(seq): return [dna_map[base] for base in seq]

def xor_sequences(seq1, seq2): return [a ^ b for a, b in zip(seq1, seq2)]

def chaotic_sequence(n_terms): ACGT_seq = encode_dna("ACGT" * (n_terms // 4 + 1))[:n_terms] TGCA_seq = encode_dna("TGCA" * (n_terms // 4 + 1))[:n_terms]

xor_seq = xor_sequences(ACGT_seq, TGCA_seq)
const_product = 1234 * 1357

sequence = []
prev_1 = prev_2 = 0

for n in range(1, n_terms + 1):
    non_linear = (8 / 3) * (n ** 2)
    noise = random.uniform(-5, 5)
    euler_flip = (-1) ** n

    delta_n = abs(prev_1 - (prev_2 if n > 2 else 0)) % np.pi

    term = (xor_seq[n - 1] + const_product + non_linear + noise) * euler_flip + delta_n

    prev_2 = prev_1
    prev_1 = term

    sequence.append(term)

return sequence

Example usage

if name == "main": terms = 50 chaotic_seq = chaotic_sequence(terms) for i, val in enumerate(chaotic_seq, 1): print(f"C_{i} = {val}")

