import numpy as np

def Gamma(k):
    """
    Computes the Tri-Cyclic Complex Balance equation.
    Args:
        k (int): Input value in {-1, 0, 1}.
    Returns:
        complex: Result of Γ(k).
    """
    zeta = { -1: -1, 0: 0, 1: 1 }.get(k, 0)
    return np.exp(1j * np.pi * k) * np.sin(np.pi * k / 2) + zeta

# Test the function
for k in [-1, 0, 1]:
    print(f"Γ({k}) = {Gamma(k)}")