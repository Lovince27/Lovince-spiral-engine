import math
import cmath

# --- Constants and Definitions ---
phi = (1 + 5 ** 0.5) / 2              # Golden ratio φ
pi = math.pi                          # π
ħ = 1.055e-34                         # Reduced Planck's constant
h = 6.626e-34                         # Planck's constant
c = 3e8                               # Speed of light in m/s
ν = 6e14                              # Light frequency
β = 0.8                               # Biophoton biological factor
lovince_magnitude = 40.5             # |Lovince|
lovince = lovince_magnitude * cmath.exp(-1j * pi / 4)  # Complex Lovince constant
E0 = ħ * lovince_magnitude           # Base energy

# --- Logic Functions ---
def compute_Z(n, theta_n):
    """Quantum-spiral position in the complex plane"""
    magnitude = (9 * (1/3)**n * c) * phi**n * pi**(3*n - 1)
    phase = cmath.exp(-1j * n * pi / phi) * cmath.exp(1j * theta_n)
    return lovince * magnitude * phase

def compute_energy(n):
    """Quantum-photonic-biophoton energy level"""
    base = phi**n * pi**(3*n - 1) * E0 * h * ν
    biophoton = base * β
    return base + biophoton

def compute_quantum_state(n):
    """Quantum state with amplitude decay and evolving phase"""
    A_n = (1 / phi**n) * (1 / 3**n)
    theta_n = (2 * pi * n) / phi + pi / phi
    return A_n, theta_n

def compute_velocity(n, energy):
    """Inverse relation between energy and velocity"""
    return c / (energy ** 0.5)

# --- Gate-based Interactive System ---
def gate_interaction(user_input):
    """Gate logic for responding to user input"""
    if "energy" in user_input.lower():
        n = int(input("Enter the quantum state n: "))
        energy = compute_energy(n)
        print(f"The energy at state n={n} is: {energy:.3e} J")
    
    elif "quantum state" in user_input.lower():
        n = int(input("Enter the quantum state n: "))
        A_n, theta_n = compute_quantum_state(n)
        print(f"The quantum state |ψₙ⟩ is: {A_n:.5f} · e^(i·{theta_n:.3f}) · |{n}⟩")
    
    elif "spiral" in user_input.lower():
        n = int(input("Enter the quantum spiral state n: "))
        theta_n = float(input("Enter the phase shift θₙ (in radians): "))
        Z_n = compute_Z(n, theta_n)
        print(f"The complex quantum spiral Zₙ at state n={n} is: {Z_n}")
    
    elif "exit" in user_input.lower():
        print("Exiting the Quantum-Gate AI system. Stay enlightened.")
        return False
    
    else:
        print("I'm sorry, I didn't understand. Please ask about energy, quantum states, or spirals.")
    
    return True

# --- Main Gate Loop ---
if __name__ == "__main__":
    print("Welcome to the Quantum-Gate AI System! Ask about energy, quantum states, or spirals.")
    print("Type 'exit' to exit the system.")
    
    while True:
        user_input = input("You: ")
        if not gate_interaction(user_input):
            break