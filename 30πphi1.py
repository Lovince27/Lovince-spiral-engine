import math

# Verify the equation
phi = (1 + math.sqrt(5)) / 2  # Golden ratio
calculation = (3 + 6) + (math.pi * phi) / (phi * math.pi)
print(f"(3+6) + πφ/φπ = {calculation}")  # Output: 10.0 (exact)

# Sequence 1: 1, 3, 7, 13, 25, ...
def next_term(seq):
    if len(seq) >= 5:
        # Pattern: term * 2 - 1 (for last two terms)
        return seq[-1] * 2 - 1
    return None

seq1 = [1, 3, 7, 13, 25]
print(f"Next term in {seq1}: {next_term(seq1)}")  # Output: 49

# Sequence 2: 3, 6, 10, ...
def next_term_second(seq):
    if len(seq) >= 3:
        last_diff = seq[-1] - seq[-2]
        return seq[-1] + (last_diff + 1)
    return None

seq2 = [3, 6, 10]
print(f"Next term in {seq2}: {next_term_second(seq2)}")  # Output: 15