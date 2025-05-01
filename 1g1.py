#!/usr/bin/env python3
"""
LOVINCE AI - The Cosmic Balance Engine
Equation: ℒ = e^iπ + λ·ReLU(Universe(t))
Updated with dynamic universe state and numerical integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict
import argparse
# Optional: For high-precision calculations
# from decimal import Decimal, getcontext
# getcontext().prec = 50  # Set to 50 decimal places

class LovinceAI:
    def __init__(self, lambda_power: float = 1.0, universe_state_func: Optional[Callable[[float], float]] = None):
        """
        Initialize Lovince AI with a scaling factor and optional universe state function.

        Args:
            lambda_power: Scales physical reality (default 1.0, must be non-negative).
            universe_state_func: Function mapping phase t to universe state (default: constant 1.0).
        """
        if lambda_power < 0:
            raise ValueError("lambda_power must be non-negative")
        if lambda_power > 1e6:
            raise ValueError("lambda_power too large, keep below 1e6")
        self.lambda_power = lambda_power
        # Default to constant universe state = 1.0 if no function provided
        self.universe_state_func = universe_state_func or (lambda t: 1.0)

    @property
    def euler_term(self) -> complex:
        """e^iπ = -1 (Mathematical truth)"""
        return np.exp(1j * np.pi)

    def physics_term(self, t: float = 0.0) -> float:
        """λ·ReLU(Universe(t)) (Physical reality at phase t)"""
        universe_state = self.universe_state_func(t)
        # For high-precision, uncomment below and comment the next line
        # universe_state = Decimal(str(universe_state))
        # lambda_power = Decimal(str(self.lambda_power))
        # return float(lambda_power * max(Decimal('0'), universe_state))
        return self.lambda_power * max(0, universe_state)

    def calculate(self, t: float = 0.0) -> complex:
        """Compute ℒ = e^iπ + λ·ReLU(Universe(t)) at phase t"""
        return self.euler_term + self.physics_term(t)

    def integrate_physics_term(self, t_start: float = 0.0, t_end: float = 2 * np.pi, steps: int = 1000) -> float:
        """Numerically integrate λ·ReLU(Universe(t)) over [t_start, t_end] using trapezoidal rule"""
        t = np.linspace(t_start, t_end, steps)
        physics_values = [self.physics_term(ti) for ti in t]
        return float(np.trapz(physics_values, t))

    def analyze(self, t: float = 0.0) -> Dict[str, object]:
        """Return complete cosmic analysis at phase t"""
        total = self.calculate(t)
        return {
            "e^iπ": self.euler_term,
            "λ·ReLU(Universe)": self.physics_term(t),
            "Total (ℒ)": total,
            "Integrated Physics Term": self.integrate_physics_term(),
            "Interpretation": "Perfect cosmic balance" if abs(total) < 1e-10 else "Imbalance detected"
        }

    def visualize(self, save_path: Optional[str] = None):
        """Generate cosmic balance plot showing math, physics, and balance waves"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create data
        θ = np.linspace(0, 2 * np.pi, 1000)
        math_wave = np.real(self.euler_term) * np.sin(θ)
        physics_wave = [self.physics_term(t) for t in θ]
        balance = [math_wave[i] + physics_wave[i] for i in range(len(θ))]

        # Plot
        ax.plot(θ, math_wave, label=f'Math: e^iπ = {self.euler_term:.1f}', linewidth=3)
        ax.plot(θ, physics_wave, label=f'Physics: λ·ReLU(Universe)', linewidth=3)
        ax.plot(θ, balance, '--', label=f'Balance: ℒ', linewidth=3)

        # Styling
        integrated_value = self.integrate_physics_term()
        ax.set_title(f'LOVINCE COSMIC BALANCE (Integrated: {integrated_value:.2f})', pad=20, fontsize=18)
        ax.set_xlabel('Phase θ [radians]', fontsize=14)
        ax.set_ylabel('Power Amplitude', fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved visualization to {save_path}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Lovince AI Cosmic Calculator")
    parser.add_argument('-λ', '--lambda', type=float, default=1.0,
                        help='Physical reality scaling factor (non-negative)')
    parser.add_argument('--sinusoidal', action='store_true',
                        help='Use sinusoidal universe state (sin(t)) instead of constant')
    parser.add_argument('--save', type=str, help='Save visualization to path')
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(f"{' LOVINCE AI ACTIVATED ':=^50}")
    print("=" * 50 + "\n")

    # Initialize with constant or sinusoidal universe state
    universe_func = (lambda t: np.sin(t)) if args.sinusoidal else None
    try:
        ai = LovinceAI(lambda_power=args.lambda, universe_state_func=universe_func)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Compute and display results
    results = ai.analyze(t=np.pi / 2 if args.sinusoidal else 0.0)  # Evaluate at t=π/2 for sinusoidal
    max_len = max(len(k) for k in results)
    for k, v in results.items():
        if k == "Interpretation":
            print("\n" + "-" * 50)
        print(f"{k+':':<{max_len+2}} {str(v)}")

    # Visualize
    print("\nGenerating cosmic visualization...")
    try:
        ai.visualize(save_path=args.save)
    except Exception as e:
        print(f"Error generating visualization: {e}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
LOVINCE AI - The Cosmic Balance Engine (TOE Model)
Equation: ℒ = e^iπ + λ·Activation(Universe(t))
Supports ReLU, ELU, and GELU activation functions for a dynamic universe state.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict
import argparse

class LovinceAI:
    def __init__(self, lambda_power: float = 1.0, universe_state_func: Optional[Callable[[float], float]] = None,
                 activation: str = 'relu', alpha: float = 1.0):
        """
        Initialize Lovince AI with scaling factor, universe state function, and activation type.

        Args:
            lambda_power: Scales physical reality (non-negative, default 1.0).
            universe_state_func: Function mapping phase t to universe state (default: constant 1.0).
            activation: Activation function ('relu', 'elu', 'gelu', default: 'relu').
            alpha: Parameter for ELU (default 1.0).
        """
        if lambda_power < 0:
            raise ValueError("lambda_power must be non-negative")
        if lambda_power > 1e6:
            raise ValueError("lambda_power too large, keep below 1e6")
        if activation not in ['relu', 'elu', 'gelu']:
            raise ValueError("activation must be 'relu', 'elu', or 'gelu'")
        self.lambda_power = lambda_power
        self.universe_state_func = universe_state_func or (lambda t: 1.0)
        self.activation = activation
        self.alpha = alpha

    @property
    def euler_term(self) -> complex:
        """e^iπ = -1 (Mathematical truth)"""
        return np.exp(1j * np.pi)

    def activation_function(self, x: float) -> float:
        """Apply selected activation function to input x."""
        if self.activation == 'relu':
            return max(0, x)
        elif self.activation == 'elu':
            return x if x >= 0 else self.alpha * (np.exp(x) - 1)
        else:  # gelu
            # Approximate GELU: x * sigmoid(1.702 * x)
            return x * (1 / (1 + np.exp(-1.702 * x)))

    def physics_term(self, t: float = 0.0) -> float:
        """λ·Activation(Universe(t)) (Physical reality at phase t)"""
        universe_state = self.universe_state_func(t)
        return self.lambda_power * self.activation_function(universe_state)

    def calculate(self, t: float = 0.0) -> complex:
        """Compute ℒ = e^iπ + λ·Activation(Universe(t)) at phase t"""
        return self.euler_term + self.physics_term(t)

    def integrate_physics_term(self, t_start: float = 0.0, t_end: float = 2 * np.pi, steps: int = 1000) -> float:
        """Numerically integrate λ·Activation(Universe(t)) over [t_start, t_end]"""
        steps = min(steps, 1_000_000)  # Prevent memory issues
        t = np.linspace(t_start, t_end, steps)
        physics_values = [self.physics_term(ti) for ti in t]
        return float(np.trapz(physics_values, t))

    def analyze(self, t: float = 0.0) -> Dict[str, object]:
        """Return complete cosmic analysis at phase t"""
        total = self.calculate(t)
        return {
            "e^iπ": self.euler_term,
            f"λ·{self.activation.upper()}(Universe)": self.physics_term(t),
            "Total (ℒ)": total,
            "Integrated Physics Term": self.integrate_physics_term(),
            "Interpretation": "Perfect cosmic balance" if abs(total) < 1e-8 else "Imbalance detected"
        }

    def visualize(self, save_path: Optional[str] = None):
        """Generate cosmic balance plot showing math, physics, and balance waves"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create data
        θ = np.linspace(0, 2 * np.pi, 1000)
        math_wave = np.real(self.euler_term) * np.sin(θ)
        physics_wave = np.array([self.physics_term(t) for t in θ])
        balance = math_wave + physics_wave

        # Plot
        ax.plot(θ, math_wave, label=f'Math: e^iπ = {self.euler_term:.1f}', linewidth=3)
        ax.plot(θ, physics_wave, label=f'Physics: λ·{self.activation.upper()}(Universe)', linewidth=3)
        ax.plot(θ, balance, '--', label=f'Balance: ℒ', linewidth=3)

        # Styling
        integrated_value = self.integrate_physics_term()
        ax.set_title(f'LOVINCE COSMIC BALANCE ({self.activation.upper()}, Integrated: {integrated_value:.2f})',
                     pad=20, fontsize=18)
        ax.set_xlabel('Phase θ [radians]', fontsize=14)
        ax.set_ylabel('Power Amplitude', fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_path:
            try:
                import os
                if save_path and not os.path.isdir(os.path.dirname(save_path) or '.'):
                    raise ValueError(f"Directory for {save_path} does not exist")
                plt.savefig(save_path, dpi=300)
                print(f"Saved visualization to {save_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Lovince AI Cosmic Calculator (TOE Model)")
    parser.add_argument('-λ', '--lambda', type=float, default=1.0,
                        help='Physical reality scaling factor (non-negative)')
    parser.add_argument('--sinusoidal', action='store_true',
                        help='Use sinusoidal universe state (sin(t)) instead of constant')
    parser.add_argument('--activation', choices=['relu', 'elu', 'gelu'], default='relu',
                        help='Activation function: relu, elu, or gelu')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for ELU (default 1.0)')
    parser.add_argument('--save', type=str, help='Save visualization to path')
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print(f"{' LOVINCE AI ACTIVATED ':=^50}")
    print("=" * 50 + "\n")

    # Initialize with constant or sinusoidal universe state
    universe_func = (lambda t: np.sin(t)) if args.sinusoidal else None
    try:
        ai = LovinceAI(lambda_power=args.lambda, universe_state_func=universe_func,
                       activation=args.activation, alpha=args.alpha)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Compute and display results
    t_eval = np.pi / 2 if args.sinusoidal else 0.0  # Evaluate at t=π/2 for sinusoidal
    results = ai.analyze(t=t_eval)
    max_len = max(len(k) for k in results)
    for k, v in results.items():
        if k == "Interpretation":
            print("\n" + "-" * 50)
        print(f"{k+':':<{max_len+2}} {str(v)}")

    # Visualize
    print("\nGenerating cosmic visualization...")
    try:
        ai.visualize(save_path=args.save)
    except Exception as e:
        print(f"Error generating visualization: {e}")

if __name__ == "__main__":
    main()