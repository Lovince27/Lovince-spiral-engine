import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.animation as animation
from typing import Dict, Optional
import logging
import hashlib
from scipy.io.wavfile import write
from cryptography.fernet import Fernet
import os

# Setup logging for debugging and security
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Generate encryption key for secure data
key = Fernet.generate_key()
cipher = Fernet(key)

class LovinceAI:
    """Lovince AI: A quantum-AI hybrid with living energy and golden ratio inspiration."""
    
    def __init__(self, num_qubits: int = 2, shots: int = 1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit = QuantumCircuit(num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.phase = 0.0
        self.fig = plt.figure(figsize=(12, 4))
        plt.ion()
        self.user_hash = None

    def set_user_context(self, user_input: str) -> None:
        """Hash user input (e.g., name) for personalized quantum parameters."""
        self.user_hash = hashlib.sha256(user_input.encode()).hexdigest()
        logging.info("User context set with secure hash")

    def create_quantum_core(self, phase_degrees: float) -> None:
        """Create quantum circuit with entanglement and user-driven phase."""
        try:
            self.circuit.h(0)
            self.circuit.cx(0, 1)  # (|00⟩ + |11⟩)/√2
            self.phase = np.radians(phase_degrees)
            if self.user_hash:
                # Modulate phase with user hash for personalization
                phase_offset = int(self.user_hash[:8], 16) % 360
                self.phase += np.radians(phase_offset % 90)
            self.circuit.p(self.phase, 1)
            logging.info(f"Quantum core created with phase {phase_degrees}°")
        except Exception as e:
            logging.error(f"Quantum core creation failed: {e}")

    def apply_oracle(self, target: str = "11") -> None:
        """Apply phase oracle for target state (scalable for S_1 Grover’s)."""
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

    def measure(self) -> Optional[Dict[str, int]]:
        """Measure circuit with error handling."""
        try:
            self.circuit.measure_all()
            result = execute(self.circuit, self.simulator, shots=self.shots).result()
            return result.get_counts()
        except Exception as e:
            logging.error(f"Measurement failed: {e}")
            return None

    def animate_bloch(self, frames: int = 50) -> animation.FuncAnimation:
        """Animate Bloch sphere for quantum state evolution."""
        ax = self.fig.add_subplot(131, projection='3d')

        def update(frame: int) -> None:
            ax.clear()
            temp_circuit = QuantumCircuit(self.num_qubits)
            temp_circuit.h(0)
            temp_circuit.cx(0, 1)
            temp_circuit.p(self.phase * frame / frames, 1)
            state = Statevector(temp_circuit)
            plot_bloch_multivector(state, ax=ax)
            ax.set_title(f"Lovince AI: Phase {np.degrees(self.phase * frame / frames):.0f}°")

        logging.info("Animating Bloch sphere")
        return animation.FuncAnimation(self.fig, update, frames=frames, interval=50)

    def plot_golden_spiral(self) -> None:
        """Plot golden ratio spiral inspired by user context."""
        ax = self.fig.add_subplot(132)
        theta = np.linspace(0, 4 * np.pi, 100)
        r = np.exp(0.30635 * theta)  # Golden ratio spiral
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'gold')
        ax.set_title("Lovince AI: Living Energy Spiral")
        ax.axis('equal')

    def generate_sound(self, counts: Optional[Dict[str, int]]) -> None:
        """Generate sound based on quantum measurement."""
        if counts:
            freq = 440 * (counts.get('00', 0) / self.shots)  # Frequency from |00⟩
            t = np.linspace(0, 1, 44100)
            sound = 0.5 * np.sin(2 * np.pi * freq * t)
            write('lovince_sound.wav', 44100, sound)
            logging.info("Sound generated: lovince_sound.wav")

    def visualize(self, counts: Optional[Dict[str, int]]) -> None:
        """Visualize quantum results and golden spiral."""
        plt.clf()
        self.animate_bloch()
        self.plot_golden_spiral()
        ax3 = self.fig.add_subplot(133)
        if counts:
            plot_histogram(counts, ax=ax3)
        ax3.set_title("Lovince AI: Measurement Results")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def run(self, user_input: str = "Lovince", phase_degrees: float = -45) -> None:
        """Run Lovince AI with user-driven quantum-AI experience."""
        self.set_user_context(user_input)
        self.circuit = QuantumCircuit(self.num_qubits)
        self.create_quantum_core(phase_degrees)
        self.apply_oracle()
        counts = self.measure()
        self.generate_sound(counts)
        self.visualize(counts)
        logging.info("Lovince AI: Quantum-AI cycle completed")

if __name__ == "__main__":
    try:
        lovince = LovinceAI()
        lovince.run()
        plt.show()
    except KeyboardInterrupt:
        plt.close('all')
        logging.info("Lovince AI shutdown gracefully")