def lovince_chaotic(n):
    steps = 0
    print(f"Start: n = {n}")
    
    while n != 1:
        # Apply the chaotic rules
        if n % 3 == 0:
            n = n // 3
        elif n % 2 == 0:
            n = n // 2
        elif n % 4 == 1:
            n = 3 * n + 1
        else:
            n = n - 1
        
        # Fallback if n exceeds 100
        if n > 100:
            n = n // 2
        
        steps += 1
        print(f"Step {steps}: n = {n}")
        
    print("Reached 1!")

# Test with n = 29
lovince_chaotic(29)

def lovince_chaotic_engine(n, verbose=True):
    steps = 0
    path = [n]

    while n != 1:
        if n % 3 == 0:
            n = n // 3
        elif n % 2 == 0:
            n = n // 2
        elif n % 4 == 1:
            n = (n * n + 1) // 2
        else:
            n -= 1

        path.append(n)
        steps += 1

    if verbose:
        print(f"Converged in {steps} steps.")
        print("Path:", path)

    return path

# Example usage
if __name__ == "__main__":
    number = int(input("Enter a positive integer to test LCCE: "))
    lovince_chaotic_engine(number)