import hashlib, math

def is_prime(n):
    return n > 1 and all(n % i for i in range(2, int(math.sqrt(n)) + 1))

def npcds_key(secret, length=100):
    # Seed from SHA3-256 of secret
    seed = int(hashlib.sha3_256(secret.encode()).hexdigest(), 16) / 16**64
    r, x = 3.99, seed
    key = []
    
    for i in range(1, length + 1):
        x = r * x * (1 - x)
        chaotic_val = int(x * 1e16) % 256
        term = (chaotic_val % 4) + 1  # DNA base (1-4)
        
        if i % 5 == 0:  # Prime mutation
            p = chaotic_val + i
            while not is_prime(p): p += 1
            term = (p % 4) + 1
        
        # Finalize with SHA3
        term = (term + int(hashlib.sha3_256(f"{x}{i}".encode()).hexdigest()[:8], 16)) % 10
        key.append(term)
    
    return key

secret = "My@Secr3t!2024"
key = npcds_key(secret)
print("NPCDS Key:", key)


import hashlib
import math

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def npcds_key(secret, length=100):
    # Seed from SHA3-256 of secret
    h = hashlib.sha3_256(secret.encode()).hexdigest()
    seed = int(h, 16) / float(2**256)  # better precision normalization
    r, x = 3.99, seed
    key = []

    for i in range(1, length + 1):
        x = r * x * (1 - x)
        chaotic_val = int(x * 1e16) % 256
        term = (chaotic_val % 4) + 1  # DNA base (1-4)

        if i % 5 == 0:  # Prime mutation
            p = chaotic_val + i
            while not is_prime(p):
                p += 1
            term = (p % 4) + 1

        # Finalize with SHA3
        h2 = hashlib.sha3_256(f"{x}{i}".encode()).hexdigest()
        term = (term + int(h2[:8], 16)) % 10
        key.append(term)

    return key

secret = "My@Secr3t!2024"
key = npcds_key(secret)
print("NPCDS Key:", key)
