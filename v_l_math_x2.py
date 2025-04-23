import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.animation as animation
from typing import Tuple, Dict
import sys

class QuantumPro:
    """A pro-level quantum computing demo with entanglement and visualizations."""
    
    def __init__(self, num_qubits: int = 2, shots: int = 1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit = QuantumCircuit(num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.phase = 0.0
        self.fig = plt.figure(figsize=(12, 4))
        plt.ion()  # Interactive mode for smooth visuals

    def create_entangled_state(self) -> None:
        """Set up an entangled state with Hadamard and CNOT."""
        self.circuit.h(0)  # Superposition on qubit 0
        self.circuit.cx(0, 1)  # Entangle qubits: (|00⟩ + |11⟩)/√2
        print("\n🎉 Entangled State Created:")
        self._show_state()

    def apply_phase_gate(self, phase_degrees: float) -> None:
        """Apply a phase gate to qubit 1."""
        self.phase = np.radians(phase_degrees)
        self.circuit.p(self.phase, 1)
        print(f"\n🌟 Applied Phase Gate ({phase_degrees}°):")
        self._show_state()

    def apply_oracle(self, target: str = "11") -> None:
        """Apply a phase oracle to mark a target state."""
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        self.circuit.cz(0, 1)  # Phase flip for |11⟩
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        print(f"\n🔍 Oracle Applied (Target: |{target}⟩):")
        self._show_state()

    def add_interference(self) -> None:
        """Add a Hadamard gate for interference effects."""
        self.circuit.h(0)
        print("\n✨ Interference Added:")
        self._show_state()

    def measure(self) -> Dict[str, int]:
        """Measure the circuit and return counts."""
        self.circuit.measure_all()
        result = execute(self.circuit, self.simulator, shots=self.shots).result()
        return result.get_counts()

    def _show_state(self) -> None:
        """Display the current statevector in LaTeX."""
        state = Statevector(self.circuit)
        display(state.draw('latex'))

    def animate_bloch(self, frames: int = 50) -> animation.FuncAnimation:
        """Animate the phase gate's effect on the Bloch sphere."""
        ax = self.fig.add_subplot(131, projection='3d')

        def update(frame: int) -> None:
            ax.clear()
            temp_circuit = QuantumCircuit(self.num_qubits)
            temp_circuit.h(0)
            temp_circuit.cx(0, 1)
            temp_circuit.p(self.phase * frame / frames, 1)
            state = Statevector(temp_circuit)
            plot_bloch_multivector(state, ax=ax)
            ax.set_title(f"Phase: {np.degrees(self.phase * frame / frames):.0f}°")

        print("\n🎥 Animating Bloch Sphere...")
        return animation.FuncAnimation(self.fig, update, frames=frames, interval=50)

    def plot_results(self, counts: Dict[str, int]) -> None:
        """Plot measurement histogram and probability curve."""
        # Histogram
        ax2 = self.fig.add_subplot(132)
        plot_histogram(counts, ax=ax2)
        ax2.set_title("Measurement Results")

        # Probability vs. Phase
        ax3 = self.fig.add_subplot(133)
        phases = np.linspace(-np.pi, np.pi, 50)
        probs_00, probs_11 = [], []
        for angle in phases:
            temp_circuit = QuantumCircuit(self.num_qubits)
            temp_circuit.h(0)
            temp_circuit.cx(0, 1)
            temp_circuit.p(angle, 1)
            temp_circuit.h(0)
            probs = Statevector(temp_circuit).probabilities()
            probs_00.append(probs[0])  # |00⟩
            probs_11.append(probs[3])  # |11⟩
        ax3.plot(np.degrees(phases), probs_00, label="|00⟩", color="blue")
        ax3.plot(np.degrees(phases), probs_11, label="|11⟩", color="orange")
        ax3.set_xlabel("Phase Angle (degrees)")
        ax3.set_ylabel("Probability")
        ax3.set_title("Probability vs. Phase")
        ax3.legend()

    def run(self) -> None:
        """Run the quantum demo with user input."""
        print("🚀 QuantumPro: A Pythonic Quantum Adventure")
        try:
            phase_degrees = float(input("Enter phase angle in degrees (e.g., -45): ") or -45)
        except ValueError:
            phase_degrees = -45
            print("Using default phase: -45°")

        # Build and run the circuit
        self.create_entangled_state()
        self.apply_phase_gate(phase_degrees)
        self.apply_oracle()
        self.add_interference()
        counts = self.measure()

        # Visualize
        ani = self.animate_bloch()
        self.plot_results(counts)
        plt.tight_layout()

        print("\n📊 Visualizations Ready! Press Ctrl+C to exit.")
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close('all')
            print("\n🛑 QuantumPro Shutdown Gracefully")
            sys.exit(0)

if __name__ == "__main__":
    quantum = QuantumPro()
    quantum.run()