import cmath

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5)+1, 2):
        if n % i == 0:
            return False
    return True

def next_prime(n):
    n += 1
    while not is_prime(n):
        n += 1
    return n

def lucas_numbers(limit):
    lucas = [2, 1]
    while len(lucas) < limit:
        lucas.append(lucas[-1] + lucas[-2])
    return lucas

def lucas_primes(limit):
    lucas = lucas_numbers(50)
    primes = [x for x in lucas if is_prime(x)]
    return primes[:limit]

def i_cycle(n):
    # Returns i^n using Euler's formula
    theta = cmath.pi / 2
    val = cmath.exp(1j * n * theta)
    # Round to avoid floating point errors
    return complex(round(val.real, 10), round(val.imag, 10))

def lukas_plusprimee_with_i_cycle(terms):
    lucas_pr = lucas_primes(terms)
    seq = []
    for idx, lp in enumerate(lucas_pr):
        p = next_prime(lp)
        val = lp + p
        # Chaotic twist: On every 3rd step, multiply by previous Lucas prime
        if idx > 1 and (idx+1) % 3 == 0:
            val = val * lucas_pr[idx-1]
        # Multiply by i^n
        i_val = i_cycle(idx)
        final_val = val * i_val
        seq.append(final_val)
    return seq

# Generate and print first 10 terms
sequence = lukas_plusprimee_with_i_cycle(10)
print("Lukas PlusPrimee Chaotic Sequence with i Cycle (first 10 terms):")
for idx, val in enumerate(sequence):
    print(f"Term {idx+1}: {val}")
