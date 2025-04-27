# final.py

import math
from decimal import Decimal, getcontext

# Set decimal precision (digits after decimal)
getcontext().prec = 150  # increased precision for product calculation

# Define constants with high precision
phi = (1 + Decimal(5).sqrt()) / 2           # Golden Ratio φ
sqrt2 = Decimal(2).sqrt()                    # Square root of 2
e = Decimal(str(math.e))                     # Euler's number e (limited precision)
pi = Decimal(str(math.pi))                   # Pi π (limited precision)

def get_decimal_digits(number, digits=50):
    """
    Extract the first 'digits' decimal digits of a Decimal number.
    """
    s = str(number)
    if '.' in s:
        _, decimal_part = s.split('.')
        decimal_part = decimal_part[:digits]
    else:
        decimal_part = '0' * digits
    return decimal_part

def has_repeating_pattern(s, min_len=3, max_len=10):
    """
    Basic check for repeating patterns in string s.
    Checks for any substring of length between min_len and max_len
    that repeats consecutively at least twice.
    Returns True if pattern found, else False.
    """
    for length in range(min_len, max_len + 1):
        for start in range(len(s) - 2 * length + 1):
            pattern = s[start:start + length]
            next_seq = s[start + length:start + 2 * length]
            if pattern == next_seq:
                return True, pattern
    return False, None

def display_constants():
    """
    Display constants and their decimal digits, and check for repeating patterns.
    """
    constants = {
        "Golden Ratio (φ)": phi,
        "Square Root of 2 (√2)": sqrt2,
        "Euler's Number (e)": e,
        "Pi (π)": pi
    }
    print("Irrational Constants and their first 50 decimal digits:\n")
    for name, value in constants.items():
        decimals = get_decimal_digits(value, 50)
        repeating, pattern = has_repeating_pattern(decimals)
        print(f"{name}: {value}")
        print(f"Decimal digits: {decimals}")
        if repeating:
            print(f"⚠️ Repeating pattern detected: '{pattern}'")
        else:
            print("✅ No repeating pattern detected.")
        print()

def combined_product():
    """
    Calculate product of φ, √2, e, π and display with decimal digits and pattern check.
    """
    product = phi * sqrt2 * e * pi
    decimals = get_decimal_digits(product, 50)
    repeating, pattern = has_repeating_pattern(decimals)
    print(f"Product of φ × √2 × e × π:\n{product}")
    print(f"Decimal digits: {decimals}")
    if repeating:
        print(f"⚠️ Repeating pattern detected in product decimals: '{pattern}'")
    else:
        print("✅ No repeating pattern detected in product decimals.")
    print()

if __name__ == "__main__":
    display_constants()
    combined_product()


# advanced_irrational_blend.py

from decimal import Decimal, getcontext
import math

# Set precision high enough to extract many digits
getcontext().prec = 110

# List of quadratic irrationals to blend (square roots of primes)
primes = [2, 3, 5, 7, 11]

def get_sqrt_decimal(prime, digits=100):
    """
    Compute the square root of a prime number with high precision,
    return the decimal digits as a string (without the integer part).
    """
    root = Decimal(prime).sqrt()
    s = str(root)
    if '.' in s:
        _, dec_part = s.split('.')
        return dec_part[:digits]
    else:
        return '0' * digits

def blend_digits(digit_lists):
    """
    Blend digits from multiple decimal digit strings by interleaving them.
    """
    blended = []
    max_len = max(len(d) for d in digit_lists)
    for i in range(max_len):
        for digits in digit_lists:
            if i < len(digits):
                blended.append(digits[i])
    return ''.join(blended)

def main():
    digits_per_constant = 100  # Number of digits to extract from each sqrt
    digit_lists = []

    print("Extracting decimal digits from square roots of primes:")
    for p in primes:
        dec_digits = get_sqrt_decimal(p, digits_per_constant)
        digit_lists.append(dec_digits)
        print(f"√{p} decimal digits (first {digits_per_constant}): {dec_digits[:50]}...")

    blended_sequence = blend_digits(digit_lists)
    print("\nBlended decimal sequence (first 500 digits):")
    print(blended_sequence[:500])

    # Optional: simple check for repeating patterns
    def has_repeating_pattern(s, min_len=3, max_len=10):
        for length in range(min_len, max_len + 1):
            for start in range(len(s) - 2*length + 1):
                if s[start:start+length] == s[start+length:start+2*length]:
                    return True, s[start:start+length]
        return False, None

    repeating, pattern = has_repeating_pattern(blended_sequence)
    if repeating:
        print(f"\n⚠️ Repeating pattern detected: '{pattern}'")
    else:
        print("\n✅ No repeating pattern detected in blended sequence.")

if __name__ == "__main__":
    main()

