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
