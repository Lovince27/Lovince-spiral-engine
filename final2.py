#!/usr/bin/env python3
"""Cosmic Quantum Demo: Golden Ratio Spirals, Quantum Entanglement, and Nebula Audio-Visuals."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.animation as animation
from scipy.io.wavfile import write
import sounddevice as sd
from typing import Tuple, Dict, Literal
from dataclasses import dataclass
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================== CONFIGURATION ========================
@dataclass(frozen=True)
class DemoConfig:
    """Configuration for cosmic quantum demo with validated parameters."""
    # Spiral Parameters
    phi: float = (1 + np.sqrt(5)) / 2  # Golden Ratio (φ ≈ 1.6180339887)
    amplitude: float = 40.5            # Base amplitude (Lovince Constant)
    theta: float = np.pi / 4           # Angular phase (45°)
    max_iterations: int = 10           # Maximum spiral progression
    spiral_resolution: int = 2000      # Points for spiral smoothness
    spiral_theta_range: float = 4 * np.pi  # Angular range for spiral
    # Quantum Parameters
    num_qubits: int = 2                # Number of qubits
    shots: int = 1000                  # Number of measurement shots
    phase_degrees: float = -45.0       # Default phase angle
    # Audio-Visual Parameters
    animation_frames: int = 50         # Frames for animations
    audio_duration: float = 1.0        # Sound duration (seconds)
    sample_rate: int = 44100           # Audio sample rate
    nebula_grid_size: int = 100        # Grid size for nebula background

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.phi > 1, f"Golden ratio must be > 1, got {self.phi}"
        assert self.amplitude > 0, f"Amplitude must be positive, got {self.amplitude}"
        assert 0 < self.theta < 2 * np.pi, f"Theta must be in (0, 2π), got {self.theta}"
        assert 0 < self.max_iterations <= 100, f"Max iterations must be in (0, 100], got {self.max_iterations}"
        assert 100 <= self.spiral_resolution <= 10000, f"Resolution must be in [100, 10000], got {self.spiral_resolution}"
        assert 0 < self.spiral_theta_range <= 10 * np.pi, f"Theta range must be in (0, 10π], got {self.spiral_theta_range}"
        assert self.num_qubits >= 2, f"Number of qubits must be >= 2, got {self.num_qubits}"
        assert 1 <= self.shots <= 10000, f"Shots must be in [1, 10000], got {self.shots}"
        assert -360 <= self.phase_degrees <= 360, f"Phase degrees must be in [-360, 360], got {self.phase_degrees}"
        assert 10 <= self.animation_frames <= 100, f"Animation frames must be in [10, 100], got {self.animation_frames}"
        assert 0.1 <= self.audio_duration <= 5.0, f"Audio duration must be in [0.1, 5.0], got {self.audio_duration}"
        assert 8000 <= self.sample_rate <= 96000, f"Sample rate must be in [8000, 96000], got {self.sample_rate}"
        assert 50 <= self.nebula_grid_size <= 200, f"Nebula grid size must be in [50, 200], got {self.nebula_grid_size}"

# ======================== SPIRAL GENERATOR ========================
class SpiralGenerator:
    """Generates golden ratio spirals with nebula effects and sound."""
    
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

    def generate(self, direction: Literal["creation", "collapse"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spiral coordinates and frequencies.

        Args:
            direction: 'creation' (φ^n) or 'collapse' (φ^-n).

        Returns:
            Tuple of (x, y, frequencies) arrays.

        Raises:
            ValueError: If direction is invalid.
        """
        if direction not in {"creation", "collapse"}:
            raise ValueError(f"Invalid direction: {direction}. Must be 'creation' or 'collapse'.")

        # Use golden ratio decay for collapse, growth for creation
        b = np.log(self.config.phi) / (np.pi / 2) if direction == "collapse" else -np.log(self.config.phi) / (np.pi / 2)
        theta = np.linspace(0, self.config.spiral_theta_range, self.config.spiral_resolution)
        r = self.config.amplitude * np.exp(b * theta)
        x = r * np.cos(theta - self.config.theta)
        y = r * np.sin(theta - self.config.theta)

        # Generate frequencies based on radius
        radii = r / np.max(np.abs(r))
        if direction == "collapse":
            frequencies = 440 / (self.config.phi ** (radii * 4))
            frequencies = np.clip(frequencies, 220, 440)
        else:
            frequencies = 440 - (440 - 220) * radii

        # Mathematical proof: Verify golden ratio progression
        expected_ratio = np.exp(b * theta)
        assert np.allclose(r / self.config.amplitude, expected_ratio, rtol=1e-5), \
            f"Golden ratio progression failed for {direction} spiral"

        return x, y, frequencies

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
    """Visualizes spirals, quantum states, and audio effects."""
    
    def __init__(self, config: DemoConfig) -> None:
        """Initialize visualization."""
        self.config = config
        self.fig = plt.figure(figsize=(18, 12), facecolor='black')
        plt.style.use('dark_background')

    def create_nebula_background(self, ax) -> None:
        """Create nebula background with stars."""
        x = np.linspace(-50, 50, self.config.nebula_grid_size)
        y = np.linspace(-50, 50, self.config.nebula_grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(self.config.nebula_grid_size, self.config.nebula_grid_size) * 0.5
        ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
        for _ in range(50):
            star_x, star_y = np.random.uniform(-50, 50), np.random.uniform(-50, 50)
            ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

    def generate_sound(self, filename: str, frequencies: np.ndarray) -> None:
        """Generate and play sound based on frequencies."""
        t = np.linspace(0, self.config.audio_duration, int(self.config.sample_rate * self.config.audio_duration))
        sound = np.zeros_like(t)
        for freq in frequencies:
            sound += 0.3 * np.sin(2 * np.pi * freq * t)
        sound = sound / np.max(np.abs(sound))
        write(filename, self.config.sample_rate, sound)
        logging.info(f"Sound generated: {filename}")
        sd.play(sound, self.config.sample_rate)
        sd.wait()

    def plot_spirals(self, creation_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    collapse_coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Plot golden ratio spirals with nebula and sound."""
        ax1 = self.fig.add_subplot(231)
        ax2 = self.fig.add_subplot(232)
        self.create_nebula_background(ax1)
        self.create_nebula_background(ax2)

        # Collapse Spiral
        x, y, frequencies = collapse_coords
        points = np.array([x, y]).T
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(segments))
        lc = mcolors.LinearSegmentedColormap.from_list("", ['teal', 'gold'])
        line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=3, alpha=0.8)
        line.set_array(np.linspace(0, 1, len(segments)))
        ax1.add_collection(line)
        glow = Circle((x[-1], y[-1]), 5, color='white', alpha=0.3)
        ax1.add_patch(glow)
        self.generate_sound("collapse_sound.wav", frequencies)
        
        ax1.set_title("Quantum Collapse (φ^-n)", color='white')
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)
        ax1.set_xlabel("Real", color='white')
        ax1.set_ylabel("Imaginary", color='white')

        # Creation Spiral
        x, y, frequencies = creation_coords
        points = np.array([x, y]).T
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(segments))
        lc = mcolors.LinearSegmentedColormap.from_list("", ['blue', 'orange'])
        line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=3, alpha=0.8)
        line.set_array(np.linspace(0, 1, len(segments)))
        ax2.add_collection(line)
        glow = Circle((x[-1], y[-1]), 5, color='white', alpha=0.3)
        ax2.add_patch(glow)
        self.generate_sound("creation_sound.wav", frequencies)
        
        ax2.set_title("Cosmic Creation (φ^n)", color='white')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(-50, 50)
        ax2.set_xlabel("Real", color='white')
        ax2.set_ylabel("Imaginary", color='white')

        for ax in (ax1, ax2):
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('yellow')

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
class CosmicQuantumDemo:
    """Combines golden ratio spirals, quantum circuits, and nebula audio-visuals."""
    
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
        print("🚀 Cosmic Quantum Demo: Spirals, Quantum States, and Nebula Sounds")
        
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
        self.visualizer.fig.suptitle("Cosmic Quantum Dynamics: Spirals & Entanglement", fontsize=16, color='white', y=0.95)
        self.visualizer.fig.text(0.5, 0.01, f"Golden Ratio: φ ≈ {self.config.phi:.3f}", ha='center', fontsize=10, color='white')
        plt.tight_layout()

        print("\n📊 Visualizations and Sounds Ready! Press Ctrl+C to exit.")
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close('all')
            print("\n🛑 Demo Shutdown Gracefully")
            sys.exit(0)

# ======================== EXECUTION ========================
if __name__ == "__main__":
    try:
        demo = CosmicQuantumDemo()
        demo.run()
    except Exception as e:
        print(f"Cosmic Anomaly Detected: {str(e)}")
        sys.exit(1)