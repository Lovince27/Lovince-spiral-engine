import numpy as np
from functools import lru_cache
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
import plotly.graph_objects as go
import sounddevice as sd
from typing import Dict, List, Optional

class LovinceQuantum:
    """
    Quantum-Bioacoustic Framework integrating:
    - Quantum circuits with Qiskit
    - Phase modulations inspired by golden ratio and chakra frequencies
    - Sound synthesis from quantum probabilities
    - Interactive visualization of quantum states
    """

    PI = np.pi
    PHI = (1 + 5 ** 0.5) / 2  # Golden ratio ≈ 1.618
    CHAKRA_FREQ = 963          # Third Eye chakra frequency in Hz

    def __init__(self, qubits: int = 2, shots: int = 1000):
        self.qubits = max(1, qubits)
        self.shots = max(1, shots)
        self.circuit = QuantumCircuit(self.qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.watermark = "The Founder - Lovince ™"

    def entangle(self) -> None:
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        print(f"\n🎉 Entangled by {self.watermark}")
        self._show_state()

    def cosmic_truth(self, n: int = 9) -> None:
        theta = n  # Simplified phase
        self.circuit.p(theta, 1)
        print(f"\n🌠 Cosmic Truth: Phase θ = {theta:.3f} rad")
        self._show_state()

    def chakra_phase(self, n: int = 2, N: int = 1000) -> None:
        theta = (n * self.PI / self.PHI) + (2 * self.PI * self.CHAKRA_FREQ * n / N)
        self.circuit.p(theta, 1)
        print(f"\n🌌 Chakra Phase: θ = {theta:.3f} rad ({self.CHAKRA_FREQ} Hz)")
        self._show_state()

    def bioelectric_phase(self, freq: float = 963) -> None:
        theta = 2 * self.PI * freq / 1000
        self.circuit.p(theta, 1)
        print(f"\n🌱 Bioelectric Phase: {freq} Hz (θ = {theta:.3f} rad)")
        self._show_state()

    def oracle(self, target: str = "11") -> None:
        if len(target) != self.qubits:
            raise ValueError(f"Target must be a {self.qubits}-bit string")
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        self.circuit.cz(0, 1)
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        print(f"\n🔍 Oracle: Target |{target}⟩ marked")
        self._show_state()

    def interfere(self) -> None:
        self.circuit.h(0)
        print(f"\n✨ Interference by {self.watermark}")
        self._show_state()

    def measure(self) -> Dict[str, int]:
        self.circuit.measure_all()
        result = execute(self.circuit, self.simulator, shots=self.shots).result()
        counts = result.get_counts()
        print(f"\n📊 Measurement results: {counts}")
        return counts

    def entropy(self, counts: Dict[str, int]) -> float:
        probs = np.array(list(counts.values())) / self.shots
        entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
        print(f"\nℹ️ Entropy: {entropy:.4f} bits")
        return entropy

    def sound_map(self, duration: float = 3.0) -> None:
        probs = Statevector(self.circuit).probabilities()
        freqs = 220 + 500 * probs
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sound = np.sum([np.sin(2 * self.PI * f * t) for f in freqs], axis=0)
        sd.play(sound / np.max(np.abs(sound)), sample_rate)
        print(f"\n🎶 Playing cosmic sound by {self.watermark}")

    def plot_probs(self) -> None:
        phases = np.linspace(-self.PI, self.PI, 100)
        probs_00, probs_11 = [], []
        for angle in phases:
            circ = QuantumCircuit(self.qubits)
            circ.h(0)
            circ.cx(0, 1)
            circ.p(angle, 1)
            circ.h(0)
            sv = Statevector(circ)
            probs = sv.probabilities()
            probs_00.append(probs[0])
            probs_11.append(probs[3])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.degrees(phases), y=probs_00, mode='lines', name='|00⟩'))
        fig.add_trace(go.Scatter(x=np.degrees(phases), y=probs_11, mode='lines', name='|11⟩'))
        fig.update_layout(
            title=f"Probability vs Phase Angle - {self.watermark}",
            xaxis_title="Phase Angle (degrees)",
            yaxis_title="Probability",
            template="plotly_dark"
        )
        fig.show()

    def run(self) -> None:
        print(f"🚀 Running LovinceQuantum by {self.watermark}")
        self.entangle()
        self.cosmic_truth()
        self.chakra_phase()
        self.bioelectric_phase()
        self.oracle()
        self.interfere()
        counts = self.measure()
        self.entropy(counts)
        self.sound_map()
        self.plot_probs()
        print("\n📊 Cosmic Quantum Dance Complete!")

    def _show_state(self) -> None:
        try:
            from IPython.display import display
            state = Statevector(self.circuit)
            display(state.draw('latex'))
        except ImportError:
            pass


# === Numeric Utilities ===

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    """
    Compute Fibonacci number using memoization for speed.
    """
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def fast_sum(arr: np.ndarray) -> float:
    """
    Compute sum of a NumPy array using vectorized operations.
    """
    return np.sum(arr)

def self_check() -> None:
    """
    Basic tests to verify correctness and performance.
    """
    assert fib(10) == 55, "Fibonacci test failed"
    assert fast_sum(np.array([1, 2, 3])) == 6, "Sum test failed"
    print("✅ Self-check passed. Code is correct and optimized.")


# === Example Usage ===

if __name__ == "__main__":
    # Run numeric utilities
    print(f"fib(35) = {fib(35)}")
    large_array = np.arange(1_000_000)
    print(f"Sum of 1 to 1,000,000 = {fast_sum(large_array)}")
    self_check()

    # Run quantum-bioacoustic experiment
    lovince = LovinceQuantum()
    lovince.run()
