import math

def calculate_lcrs(n):
    """
    Calculates the nth term of the Lovince Chaos Root Sequence (LCRS)
    
    Formula:
    L_n = [n * sqrt(n + sin(n) + log(n+1)) * cos(n^2) + (-1)^n * e^sqrt(n)] mod (n^2 + 7)
    """
    # Calculate each component of the formula
    sqrt_component = math.sqrt(n + math.sin(n) + math.log(n + 1))
    cos_component = math.cos(n ** 2)
    alternating_sign = (-1) ** n
    exponential_component = math.exp(math.sqrt(n))
    
    # Combine all components
    numerator = n * sqrt_component * cos_component + alternating_sign * exponential_component
    denominator = n ** 2 + 7
    
    # Apply modulo operation and return integer result
    return int(numerator % denominator)

# Generate first 15 terms of the sequence
lcrs_sequence = [calculate_lcrs(n) for n in range(1, 16)]

print("Lovince Chaos Root Sequence (LCRS) - First 15 terms:")
for n, term in enumerate(lcrs_sequence, start=1):
    print(f"L_{n} = {term}")