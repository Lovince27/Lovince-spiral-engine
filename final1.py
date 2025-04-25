#!/usr/bin/env python3
"""Quantum Spiral Dynamics: Golden Ratio Spirals meets Quantum Entanglement."""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.animation as animation
from typing import Tuple, Dict, Literal
from dataclasses import dataclass
import sys

# ======================== CONFIGURATION ========================
@dataclass(frozen=True)
class DemoConfig:
    """Configuration for quantum and spiral demo with validated parameters."""
    # Spiral Parameters
    phi: float = (1 + np.sqrt(5)) / 2  # Golden Ratio (φ ≈ 1.6180339887)
    amplitude: float = 40.5            # Base amplitude (Lovince Constant)
    theta: float = np.pi / 4           # Angular phase (45°)
    max_iterations: int = 10           # Maximum spiral progression
    spiral_resolution: int = 2000      # Points for spiral smoothness
    # Quantum Parameters
    num_qubits: int = 2                # Number of qubits
    shots: int = 1000                  # Number of measurement shots
    phase_degrees: float = -45.0       # Default phase angle
    animation_frames: int = 50         # Frames for animations

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.phi > 1, f"Golden ratio must be > 1, got {self.phi}"
        assert self.amplitude > 0, f"Amplitude must be positive, got {self.amplitude}"
        assert 0 < self.theta < 2 * np.pi, f"Theta must be in (0, 2π), got {self.theta}"
        assert 0 < self.max_iterations <= 100, f"Max iterations must be in (0, 100], got {self.max_iterations}"
        assert 100 <= self.spiral_resolution <= 10000, f"Resolution must be in [100, 10000], got {self.spiral_resolution}"
        assert self.num_qubits >= 2, f"Number of qubits must be >= 2, got {self.num_qubits}"
        assert 1 <= self.shots <= 10000, f"Shots must be in [1, 10000], got {self.shots}"
        assert -360 <= self.phase_degrees <= 360, f"Phase degrees must be in [-360, 360], got {self.phase_degrees}"
        assert 10 <= self.animation_frames <= 100, f"Animation frames must be in [10, 100], got {self.animation_frames}"

# ======================== SPIRAL GENERATOR ========================
class SpiralGenerator:
    """Generates golden ratio spiral coordinates with vectorized operations."""
    
    def __init__(self, config: DemoConfig) -> None:
        """Initialize with validated configuration."""
        self.config = config
        self._validate_numerical_stability()

    def _validate_numerical_stability(self) -> None:
        """Cross-check for numerical overflow/underflow."""
        max_radius = self.config.amplitude * (self.config.phi ** self.config.max_iterations)
        min_radius = self.config.amplitude * (self.config.phi ** -self.config.max_iterations)
        assert max_radius < 1e308, f"Overflow risk: max radius {max_radius} too large"
        assert min_radius > 1e-308, f"Underflow risk: min radius {min_radius} too small"

    def generate(self, direction: Literal["creation", "collapse"]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spiral coordinates.

        Args:
            direction: 'creation' (φ^n) or 'collapse' (φ^-n).

        Returns:
            Tuple of (x, y) numpy arrays.

        Raises:
            ValueError: If direction is invalid.
        """
        if direction not in {"creation", "collapse"}:
            raise ValueError(f"Invalid direction: {direction}. Must be 'creation' or 'collapse'.")

        n = np.linspace(0, self.config.max_iterations, self.config.spiral_resolution)
        power = n if direction == "creation" else -n
        radius = self.config.amplitude * np.power(self.config.phi, power)
        angle = -n * self.config.theta

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Mathematical proof: Verify golden ratio progression
        expected_ratio = np.power(self.config.phi, power)
        assert np.allclose(radius / self.config.amplitude, expected_ratio, rtol=1e-5), \
            f"Golden ratio progression failed for {direction} spiral"

        return x, y

# ======================== QUANTUM CIRCUIT ========================
class QuantumCircuitDemo:
    """Manages quantum circuit with entanglement and visualizations."""
    
    def __init__(self, config: DemoConfig) -> None:
        """Initialize quantum circuit."""
        self.config = config
        self.circuit = QuantumCircuit(config.num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.phase = np.radians(config.phase_degrees)

    def create_entangled_state(self) -> None:
        """Create entangled state: (|00⟩ + |11⟩)/√2."""
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        print("\n🎉 Entangled State Created:")
        self._show_state()

    def apply_phase_gate(self) -> None:
        """Apply phase gate to qubit 1."""
        self.circuit.p(self.phase, 1)
        print(f"\n🌟 Applied Phase Gate ({self.config.phase_degrees}°):")
        self._show_state()

    def apply_oracle(self, target: str = "11") -> None:
        """Apply phase oracle to mark target state."""
        if len(target) != self.config.num_qubits or not all(b in "01" for b in target):
            raise ValueError(f"Invalid target state: {target}. Must be {self.config.num_qubits}-bit binary.")
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        self.circuit.cz(0, 1)
        for i, bit in enumerate(target):
            if bit == "0":
                self.circuit.x(i)
        print(f"\n🔍 Oracle Applied (Target: |{target}⟩):")
        self._show_state()

    def add_interference(self) -> None:
        """Add Hadamard for interference."""
        self.circuit.h(0)
        print("\n✨ Interference Added:")
        self._show_state()

    def measure(self) -> Dict[str, int]:
        """Measure circuit and return counts."""
        self.circuit.measure_all()
        result = execute(self.circuit, self.simulator, shots=self.config.shots).result()
        return result.get_counts()

    def _show_state(self) -> None:
        """Display statevector in LaTeX."""
        state = Statevector(self.circuit)
        display(state.draw('latex'))

# ======================== VISUALIZER ========================
class DemoVisualizer:
    """Visualizes spirals and quantum states."""
    
    def __init__(self, config: DemoConfig) -> None:
        """Initialize visualization."""
        self.config = config
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        plt.style.use('dark_background')

    def plot_spirals(self, creation_coords: Tuple[np.ndarray, np.ndarray],
                    collapse_coords: Tuple[np.ndarray, np.ndarray]) -> None:
        """Plot golden ratio spirals."""
        ax1 = self.fig.add_subplot(231)
        ax2 = self.fig.add_subplot(232)
        
        ax1.plot(*collapse_coords, color='cyan', linewidth=1.5, label='Collapse (φ^-n)')
        ax1.set_title('Quantum Collapse', color='white')
        ax1.grid(color='gray', alpha=0.2)
        ax1.legend(facecolor='black', edgecolor='yellow', labelcolor='white')

        ax2.plot(*creation_coords, color='magenta', linewidth=1.5, label='Creation (φ^n)')
        ax2.set_title('Cosmic Creation', color='white')
        ax2.grid(color='gray', alpha=0.2)
        ax2.legend(facecolor='black', edgecolor='yellow', labelcolor='white')

        max_radius = self.config.amplitude * (self.config.phi ** self.config.max_iterations)
        for ax in (ax1, ax2):
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('yellow')
        ax2.set_xlim(-max_radius * 1.1, max_radius * 1.1)
        ax2.set_ylim(-max_radius * 1.1, max_radius * 1.1)
        ax1.set_xlim(-self.config.amplitude * 1.1, self.config.amplitude * 1.1)
        ax1.set_ylim(-self.config.amplitude * 1.1, self.config.amplitude * 1.1)

    def animate_bloch(self, circuit_demo: QuantumCircuitDemo) -> animation.FuncAnimation:
        """Animate Bloch sphere for phase gate."""
        ax = self.fig.add_subplot(233, projection='3d')

        def update(frame: int) -> None:
            ax.clear()
            temp_circuit = QuantumCircuit(self.config.num_qubits)
            temp_circuit.h(0)
            temp_circuit.cx(0, 1)
            temp_circuit.p(self.config.phase_degrees * frame / self.config.animation_frames, 1)
            state = Statevector(temp_circuit)
            plot_bloch_multivector(state, ax=ax)
            ax.set_title(f"Phase: {self.config.phase_degrees * frame / self.config.animation_frames:.0f}°", color='white')

        print("\n🎥 Animating Bloch Sphere...")
        return animation.FuncAnimation(self.fig, update, frames=self.config.animation_frames, interval=50)

    def plot_quantum_results(self, counts: Dict[str, int]) -> None:
        """Plot quantum measurement histogram and probability curve."""
        ax4 = self.fig.add_subplot(235)
        plot_histogram(counts, ax=ax4)
        ax4.set_title("Measurement Results", color='white')
        ax4.set_facecolor('black')

        ax5 = self.fig.add_subplot(236)
        phases = np.linspace(-np.pi, np.pi, 50)
        probs_00, probs_11 = [], []
        for angle in phases:
            temp_circuit = QuantumCircuit(self.config.num_qubits)
            temp_circuit.h(0)
            temp_circuit.cx(0, 1)
            temp_circuit.p(angle, 1)
            temp_circuit.h(0)
            probs = Statevector(temp_circuit).probabilities()
            probs_00.append(probs[0])  # |00⟩
            probs_11.append(probs[3])  # |11⟩
        ax5.plot(np.degrees(phases), probs_00, label="|00⟩", color="blue")
        ax5.plot(np.degrees(phases), probs_11, label="|11⟩", color="orange")
        ax5.set_xlabel("Phase Angle (degrees)", color='white')
        ax5.set_ylabel("Probability", color='white')
        ax5.set_title("Probability vs. Phase", color='white')
        ax5.legend(facecolor='black', edgecolor='yellow', labelcolor='white')
        ax5.set_facecolor('black')
        ax5.grid(color='gray', alpha=0.2)

# ======================== MAIN DEMO ========================
class QuantumSpiralDemo:
    """Combines golden ratio spirals and quantum circuit demo."""
    
    def __init__(self) -> None:
        """Initialize demo with user-configurable settings."""
        try:
            phase_degrees = float(input("Enter phase angle in degrees (e.g., -45): ") or -45)
        except ValueError:
            phase_degrees = -45
            print("Using default phase: -45°")
        self.config = DemoConfig(phase_degrees=phase_degrees)
        self.spiral_generator = SpiralGenerator(self.config)
        self.quantum_circuit = QuantumCircuitDemo(self.config)
        self.visualizer = DemoVisualizer(self.config)

    def run(self) -> None:
        """Execute the combined demo."""
        print("🚀 Quantum Spiral Dynamics: A Pythonic Adventure")
        
        # Generate spirals
        creation_coords = self.spiral_generator.generate("creation")
        collapse_coords = self.spiral_generator.generate("collapse")
        
        # Run quantum circuit
        self.quantum_circuit.create_entangled_state()
        self.quantum_circuit.apply_phase_gate()
        self.quantum_circuit.apply_oracle()
        self.quantum_circuit.add_interference()
        counts = self.quantum_circuit.measure()
        
        # Visualize
        self.visualizer.plot_spirals(creation_coords, collapse_coords)
        ani = self.visualizer.animate_bloch(self.quantum_circuit)
        self.visualizer.plot_quantum_results(counts)
        self.visualizer.fig.suptitle("Quantum Spiral Dynamics", fontsize=16, color='white', y=0.95)
        plt.tight_layout()

        print("\n📊 Visualizations Ready! Press Ctrl+C to exit.")
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close('all')
            print("\n🛑 Demo Shutdown Gracefully")
            sys.exit(0)

# ======================== EXECUTION ========================
if __name__ == "__main__":
    try:
        demo = QuantumSpiralDemo()
        demo.run()
    except Exception as e:
        print(f"Cosmic Anomaly Detected: {str(e)}")
        sys.exit(1)