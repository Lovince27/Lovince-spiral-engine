import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.animation as animation
from tkinter import Tk, Label, Entry, Button, StringVar
from typing import Dict, Optional
import sys
import logging

# Setup logging for error handling
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class QuantumPro:
    """A professional quantum computing demo with entanglement and GUI."""
    
    def __init__(self, num_qubits: int = 2, shots: int = 1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit = QuantumCircuit(num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.phase = 0.0
        self.fig = plt.figure(figsize=(14, 4))
        plt.ion()  # Interactive mode
        self.root = None  # Tkinter root for GUI
        self.state_label = None

    def create_entangled_state(self) -> None:
        """Create an entangled state with Hadamard and CNOT."""
        self.circuit.h(0)
        self.circuit.cx(0, 1)  # (|00⟩ + |11⟩)/√2
        logging.info("Entangled state created")
        self._update_state_display()

    def apply_phase_gate(self, phase_degrees: float) -> None:
        """Apply a phase gate to qubit 1."""
        try:
            self.phase = np.radians(phase_degrees)
            self.circuit.p(self.phase, 1)
            logging.info(f"Applied phase gate ({phase_degrees}°)")
            self._update_state_display()
        except ValueError as e:
            logging.error(f"Invalid phase angle: {e}")

    def apply_oracle(self, target: str = "11") -> None:
        """Apply a phase oracle for a target state."""
        if len(target) != self.num_qubits:
            logging.error(f"Target state must have {self.num_qubits} bits")
            return
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        self.circuit.cz(0, 1)
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        logging.info(f"Oracle applied for |{target}⟩")
        self._update_state_display()

    def add_interference(self) -> None:
        """Add Hadamard for interference."""
        self.circuit.h(0)
        logging.info("Interference added")
        self._update_state_display()

    def measure(self) -> Optional[Dict[str, int]]:
        """Measure the circuit and return counts."""
        try:
            self.circuit.measure_all()
            result = execute(self.circuit, self.simulator, shots=self.shots).result()
            return result.get_counts()
        except Exception as e:
            logging.error(f"Measurement failed: {e}")
            return None

    def _update_state_display(self) -> None:
        """Update GUI state display with current statevector."""
        if self.state_label:
            state = Statevector(self.circuit)
            self.state_label.set(f"State: {state.draw('text')}")

    def animate_bloch(self, frames: int = 50) -> animation.FuncAnimation:
        """Animate phase gate effect on Bloch sphere."""
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

        logging.info("Animating Bloch sphere")
        return animation.FuncAnimation(self.fig, update, frames=frames, interval=50)

    def plot_results(self, counts: Optional[Dict[str, int]]) -> None:
        """Plot histogram and probability curve."""
        # Histogram
        ax2 = self.fig.add_subplot(132)
        if counts:
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
            probs_00.append(probs[0])
            probs_11.append(probs[3])
        ax3.plot(np.degrees(phases), probs_00, label="|00⟩", color="blue")
        ax3.plot(np.degrees(phases), probs_11, label="|11⟩", color="orange")
        ax3.set_xlabel("Phase Angle (degrees)")
        ax3.set_ylabel("Probability")
        ax3.set_title("Probability vs. Phase")
        ax3.legend()
        plt.tight_layout()

    def run_gui(self) -> None:
        """Run the quantum demo with a Tkinter GUI."""
        self.root = Tk()
        self.root.title("QuantumPro: Quantum Computing Demo")
        self.state_label = StringVar()
        self.state_label.set("State: Initialize circuit to see state")

        # GUI elements
        Label(self.root, text="Enter Phase Angle (degrees):").pack(pady=5)
        phase_entry = Entry(self.root)
        phase_entry.pack(pady=5)
        phase_entry.insert(0, "-45")

        Label(self.root, textvariable=self.state_label, wraplength=400).pack(pady=10)

        def run_circuit():
            try:
                phase_degrees = float(phase_entry.get())
                self.circuit = QuantumCircuit(self.num_qubits)  # Reset circuit
                self.create_entangled_state()
                self.apply_phase_gate(phase_degrees)
                self.apply_oracle()
                self.add_interference()
                counts = self.measure()
                plt.clf()
                self.animate_bloch()
                self.plot_results(counts)
                plt.draw()
            except Exception as e:
                logging.error(f"GUI execution failed: {e}")

        Button(self.root, text="Run Quantum Circuit", command=run_circuit).pack(pady=5)
        Button(self.root, text="Exit", command=self._exit).pack(pady=5)

        logging.info("Starting GUI")
        self.root.mainloop()

    def _exit(self) -> None:
        """Cleanly exit the application."""
        plt.close('all')
        if self.root:
            self.root.quit()
        logging.info("QuantumPro shutdown gracefully")
        sys.exit(0)

if __name__ == "__main__":
    try:
        quantum = QuantumPro()
        quantum.run_gui()
    except KeyboardInterrupt:
        quantum._exit()