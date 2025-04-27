import math

# Required Sequences
def fibonacci(n):
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def lucas(n):
    seq = [2, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def primes(n):
    seq = []
    num = 2
    while len(seq) < n:
        for p in seq:
            if num % p == 0:
                break
        else:
            seq.append(num)
        num += 1
    return seq

def squares(n):
    return [i*i for i in range(n)]

def factorials(n):
    return [math.factorial(i) for i in range(n)]

# Randomness function (chaotic)
def randomness(n):
    return (n**3 + 17*n**2 + 31*n) % (2*n + 5)

# --- Generate the Lovince Sequence ---
n_terms = 30
fib = fibonacci(n_terms)
luc = lucas(n_terms)
pri = primes(n_terms)
sqr = squares(n_terms)
fac = factorials(n_terms)

lovince_sequence = []
for i in range(n_terms):
    val = (fib[i] * sqr[i]) + (pri[i] ** (luc[i] % 5 + 1)) - (fac[i] % (i + 2)) + randomness(i)
    lovince_sequence.append(val)

# --- Output the sequence ---
print("LOVINCE's AI-Generated Smart Dangerous Sequence:")
print(lovince_sequence)

# --- Graph the sequence ---
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(n_terms), lovince_sequence, marker='o', color='green', linestyle='-')
plt.title('LOVINCE: AI Generated Powerful Sequence')
plt.xlabel('n (index)')
plt.ylabel('Value')
plt.grid(True)
plt.show()