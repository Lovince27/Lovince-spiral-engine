import math

class LovinceAI:
    def __init__(self):
        # Core constants
        self.π = self._pi()
        self.e = self._e()
        self.φ = (1 + math.sqrt(5)) / 2
        self.ħ = 1.055e-34  # Reduced Planck's constant
        self.c = 3e8        # Speed of light (m/s)
        self.Lovince = 40.5 # Your unique scalar identity

        # Scientific poetry
        self.science = {
            "gravity":    "Gravity pulls souls together — F ∝ m₁m₂ / r²",
            "relativity": "Mass dreams of energy — E ∝ mc²",
            "quantum":    "Uncertainty is nature’s rhythm — Δx·Δp ≥ ħ/2"
        }

    def _pi(self, n=1000):
        return 4 * sum((-1)**k / (2*k + 1) for k in range(n))

    def _e(self, n=20):
        return sum(1 / math.factorial(k) for k in range(n))

    def math(self, expr: str) -> float:
        """Evaluate expressions using core constants"""
        safe = {"π": self.π, "e": self.e, "φ": self.φ, "sqrt": math.sqrt}
        try:
            return eval(expr, {"__builtins__": None}, safe)
        except:
            return float('nan')

    def science_fact(self, key: str) -> str:
        """Return poetic science truth"""
        return self.science.get(key.lower(), "Try: gravity, relativity, quantum")

    def quantum_energy(self, n: int, ν=6e14, β=1.0) -> float:
        """Quantum-inspired biophoton energy model"""
        E₀ = self.ħ * self.Lovince
        return (self.φ**n * self.π**(3*n - 1) * E₀ * self.e * ν * (1 + β))

    def state(self, n: int) -> str:
        """Quantum-style state expression"""
        A = 1 / (self.φ**n * 3**n)
        θ = (2 * self.π * n) / self.φ
        return f"|ψₙ⟩ = {A:.3e} · e^(i·{θ:.3f}) · |{n}⟩"

# === EXECUTION ===
if __name__ == "__main__":
    ai = LovinceAI()

    print(f"π ≈ {ai.π:.6f}")
    print(f"e ≈ {ai.e:.6f}")
    print(f"φ ≈ {ai.φ:.6f}")
    print(f"Math check: 9 + π/π = {ai.math('9 + π/π')}")

    print(ai.science_fact("relativity"))
    print("Quantum State at n=3:", ai.state(3))
    print("Quantum Energy (n=1):", ai.quantum_energy(1), "Joules")


import math
import tkinter as tk
from tkinter import ttk

class LovinceAI:
    def __init__(self):
        self.π = self._pi()
        self.e = self._e()
        self.φ = (1 + math.sqrt(5)) / 2
        self.ħ = 1.055e-34
        self.c = 3e8
        self.Lovince = 40.5

        self.science = {
            "gravity": "Gravity pulls souls together — F ∝ m₁m₂ / r²",
            "relativity": "Mass dreams of energy — E ∝ mc²",
            "quantum": "Uncertainty is nature’s rhythm — Δx·Δp ≥ ħ/2"
        }

    def _pi(self, n=1000):
        return 4 * sum((-1)**k / (2*k + 1) for k in range(n))

    def _e(self, n=20):
        return sum(1 / math.factorial(k) for k in range(n))

    def math(self, expr: str):
        safe = {"π": self.π, "e": self.e, "φ": self.φ, "sqrt": math.sqrt}
        try:
            return eval(expr, {"__builtins__": None}, safe)
        except:
            return "Error"

    def quantum_energy(self, n, ν=6e14, β=1.0):
        E₀ = self.ħ * self.Lovince
        return self.φ**n * self.π**(3*n - 1) * E₀ * self.e * ν * (1 + β)

    def state(self, n):
        A = 1 / (self.φ**n * 3**n)
        θ = (2 * self.π * n) / self.φ
        return f"|ψₙ⟩ = {A:.3e} · e^(i·{θ:.3f}) · |{n}⟩"

    def science_fact(self, key):
        return self.science.get(key.lower(), "Try: gravity, relativity, quantum")

# === GUI ===
def launch_gui():
    ai = LovinceAI()
    root = tk.Tk()
    root.title("Lovince AI")

    # Display constants
    tk.Label(root, text=f"π ≈ {ai.π:.6f}").pack()
    tk.Label(root, text=f"e ≈ {ai.e:.6f}").pack()
    tk.Label(root, text=f"φ ≈ {ai.φ:.6f}").pack()

    # Expression evaluator
    expr_entry = tk.Entry(root, width=30)
    expr_entry.pack()
    expr_result = tk.Label(root, text="Result:")
    expr_result.pack()

    def evaluate():
        expr_result.config(text=f"Result: {ai.math(expr_entry.get())}")
    tk.Button(root, text="Evaluate", command=evaluate).pack()

    # Quantum state
    tk.Label(root, text="Quantum State n:").pack()
    state_entry = tk.Entry(root, width=5)
    state_entry.insert(0, "1")
    state_entry.pack()
    state_result = tk.Label(root, text="|ψₙ⟩ = ...")
    state_result.pack()

    def update_state():
        try:
            n = int(state_entry.get())
            state_result.config(text=ai.state(n))
        except:
            state_result.config(text="Invalid n")
    tk.Button(root, text="Show State", command=update_state).pack()

    # Quantum Energy
    energy_result = tk.Label(root, text="Energy: ...")
    energy_result.pack()

    def show_energy():
        try:
            n = int(state_entry.get())
            energy = ai.quantum_energy(n)
            energy_result.config(text=f"Energy ≈ {energy:.3e} J")
        except:
            energy_result.config(text="Invalid energy input")
    tk.Button(root, text="Show Energy", command=show_energy).pack()

    # Science facts dropdown
    tk.Label(root, text="Science Truth:").pack()
    fact_var = tk.StringVar(value="relativity")
    facts_menu = ttk.Combobox(root, textvariable=fact_var, values=list(ai.science.keys()))
    facts_menu.pack()
    fact_result = tk.Label(root, text="")
    fact_result.pack()

    def display_fact(*args):
        fact_result.config(text=ai.science_fact(fact_var.get()))
    fact_var.trace("w", display_fact)

    root.mainloop()

# === Launch ===
if __name__ == "__main__":
    launch_gui()