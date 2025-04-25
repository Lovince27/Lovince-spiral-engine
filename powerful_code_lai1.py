#!/usr/bin/env python3
"""Lovince AI: Advanced Golden Ratio Spirals, Quantum Circuits, ML, QML, and Cosmic Audio-Visuals."""
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
from typing import Tuple, Dict, Literal, Optional
from dataclasses import dataclass
import logging
import sys
import tensorflow as tf
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZZFeatureMap
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import time
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======================== CONFIGURATION ========================
@dataclass(frozen=True)
class LovinceConfig:
    """Configuration for Lovince AI with rigorous validation."""
    phi: float = (1 + np.sqrt(5)) / 2  # Golden Ratio
    amplitude: float = 40.5            # Base amplitude
    theta: float = np.pi / 4           # Angular phase
    max_iterations: int = 10           # Spiral progression
    spiral_resolution: int = 2000      # Spiral points
    spiral_theta_range: float = 4 * np.pi
    num_qubits: int = 2                # Quantum qubits
    shots: int = 1000                  # Measurement shots
    phase_degrees: float = -45.0       # Phase angle
    animation_frames: int = 50         # Animation frames
    audio_duration: float = 1.0        # Sound duration
    sample_rate: int = 44100           # Audio sample rate
    nebula_grid_size: int = 100        # Nebula grid
    ml_epochs: int = 50                # ML training epochs
    qml_iterations: int = 100          # QML training iterations
    gui_update_interval: float = 0.1    # GUI refresh rate (seconds)

    def __post_init__(self) -> None:
        """Validate configuration with self-checks."""
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
        assert 10 <= self.ml_epochs <= 1000, f"ML epochs must be in [10, 1000], got {self.ml_epochs}"
        assert 10 <= self.qml_iterations <= 500, f"QML iterations must be in [10, 500], got {self.qml_iterations}"
        assert 0.01 <= self.gui_update_interval <= 1.0, f"GUI update interval must be in [0.01, 1.0], got {self.gui_update_interval}"

# ======================== ML MODEL ========================
class SpiralMLModel:
    """Advanced ML model for spiral pattern prediction."""
    
    def __init__(self, config: LovinceConfig) -> None:
        """Initialize ML model with validation."""
        self.config = config
        self.model = self._build_model()
        self._validate_model()

    def _build_model(self) -> tf.keras.Model:
        """Build a deep neural network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(3,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def _validate_model(self) -> None:
        """Self-check for model integrity."""
        assert self.model.input_shape == (None, 3), f"Expected input shape (None, 3), got {self.model.input_shape}"
        assert self.model.output_shape == (None, 1), f"Expected output shape (None, 1), got {self.model.output_shape}"

    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic spiral data with noise."""
        theta = np.linspace(0, self.config.spiral_theta_range, 1000)
        b = np.log(self.config.phi) / (np.pi / 2)
        r_creation = self.config.amplitude * np.exp(-b * theta) + np.random.normal(0, 0.1, theta.size)
        r_collapse = self.config.amplitude * np.exp(b * theta) + np.random.normal(0, 0.1, theta.size)
        X = np.vstack((theta, np.ones_like(theta), np.sin(theta))).T  # Features: theta, constant, sin(theta)
        y = np.concatenate([r_creation, r_collapse])
        assert not np.any(np.isnan(X)) and not np.any(np.isnan(y)), "NaN values detected in training data"
        return X, y

    def train(self) -> float:
        """Train the ML model with performance monitoring."""
        X, y = self.generate_training_data()
        history = self.model.fit(X, y, epochs=self.config.ml_epochs, validation_split=0.2, verbose=0)
        final_loss = history.history['loss'][-1]
        logging.info(f"ML model trained. Final loss: {final_loss:.4f}")
        # Cross-check: Ensure loss is reasonable
        assert final_loss < 10.0, f"ML training failed: high loss {final_loss}"
        return final_loss

    def predict(self, theta: np.ndarray) -> np.ndarray:
        """Predict spiral radii with validation."""
        X = np.vstack((theta, np.ones_like(theta), np.sin(theta))).T
        predictions = self.model.predict(X, verbose=0).flatten()
        assert not np.any(np.isnan(predictions)), "NaN values in ML predictions"
        assert np.all(predictions >= 0), "Negative radii predicted by ML model"
        return predictions

# ======================== QML MODEL ========================
class QuantumMLModel:
    """Advanced QML model for quantum state classification."""
    
    def __init__(self, config: LovinceConfig) -> None:
        """Initialize QML model."""
        self.config = config
        self.feature_map = ZZFeatureMap(self.config.num_qubits, reps=2)
        self.quantum_kernel = QuantumKernel(feature_map=self.feature_map, quantum_instance=Aer.get_backend('statevector_simulator'))
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel, max_iter=self.config.qml_iterations)

    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic quantum state data."""
        angles = np.linspace(-np.pi, np.pi, 100)
        X = np.array([[angle, np.cos(angle)] for angle in angles])  # Features: angle, cos(angle)
        y = np.array([1 if angle > 0 else 0 for angle in angles])  # Labels: positive/negative phase
        assert X.shape[0] == y.shape[0], f"Mismatch in QML data: X={X.shape}, y={y.shape}"
        return X, y

    def train(self) -> float:
        """Train the QML model with performance monitoring."""
        X, y = self.generate_training_data()
        start_time = time.time()
        self.qsvc.fit(X, y)
        elapsed_time = time.time() - start_time
        score = self.qsvc.score(X, y)
        logging.info(f"QML model trained. Accuracy: {score:.4f}, Time: {elapsed_time:.2f}s")
        # Cross-check: Ensure reasonable accuracy
        assert score > 0.5, f"QML training failed: low accuracy {score}"
        return score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantum state classes."""
        predictions = self.qsvc.predict(X)
        assert not np.any(np.isnan(predictions)), "NaN values in QML predictions"
        return predictions

# ======================== SPIRAL GENERATOR ========================
class SpiralGenerator:
    """Generates golden ratio spirals with ML integration."""
    
    def __init__(self, config: LovinceConfig, ml_model: SpiralMLModel) -> None:
        """Initialize with ML model."""
        self.config = config
        self.ml_model = ml_model
        self._validate_numerical_stability()

    def _validate_numerical_stability(self) -> None:
        """Cross-check for numerical stability."""
        max_radius = self.config.amplitude * (self.config.phi ** self.config.max_iterations)
        min_radius = self.config.amplitude * (self.config.phi ** -self.config.max_iterations)
        assert max_radius < 1e308, f"Overflow risk: max radius {max_radius}"
        assert min_radius > 1e-308, f"Underflow risk: min radius {min_radius}"

    def generate(self, direction: Literal["creation", "collapse"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate spiral coordinates and frequencies."""
        if direction not in {"creation", "collapse"}:
            raise ValueError(f"Invalid direction: {direction}")

        b = np.log(self.config.phi) / (np.pi / 2) if direction == "collapse" else -np.log(self.config.phi) / (np.pi / 2)
        theta = np.linspace(0, self.config.spiral_theta_range, self.config.spiral_resolution)
        r_true = self.config.amplitude * np.exp(b * theta)
        r = self.ml_model.predict(theta) if direction == "creation" else r_true
        # Cross-check: Compare ML predictions with true radii
        if direction == "creation":
            mse = mean_squared_error(r_true, r)
            assert mse < 1.0, f"ML predictions deviate too much: MSE={mse}"
            logging.info(f"ML prediction MSE for creation spiral: {mse:.4f}")
        x = r * np.cos(theta - self.config.theta)
        y = r * np.sin(theta - self.config.theta)

        radii = r / np.max(np.abs(r))
        frequencies = 440 / (self.config.phi ** (radii * 4)) if direction == "collapse" else 440 - (440 - 220) * radii
        frequencies = np.clip(frequencies, 220, 440)

        assert np.allclose(r / self.config.amplitude, np.exp(b * theta), rtol=1e-5, atol=1e-5), \
            f"Golden ratio progression failed for {direction} spiral"
        return x, y, frequencies

# ======================== QUANTUM CIRCUIT ========================
class QuantumCircuitDemo:
    """Manages quantum circuit with QML integration."""
    
    def __init__(self, config: LovinceConfig, qml_model: QuantumMLModel) -> None:
        """Initialize quantum circuit."""
        self.config = config
        self.qml_model = qml_model
        self.circuit = QuantumCircuit(config.num_qubits)
        self.simulator = Aer.get_backend('qasm_simulator')
        self.phase = np.radians(config.phase_degrees)

    def create_entangled_state(self) -> None:
        """Create entangled state."""
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        print("\n🎉 Entangled State Created:")
        self._show_state()

    def apply_phase_gate(self) -> None:
        """Apply phase gate."""
        self.circuit.p(self.phase, 1)
        print(f"\n🌟 Applied Phase Gate ({self.config.phase_degrees}°):")
        self._show_state()

    def apply_oracle(self, target: str = "11") -> None:
        """Apply phase oracle."""
        if len(target) != self.config.num_qubits or not all(b in "01" for b in target):
            raise ValueError(f"Invalid target state: {target}")
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
        """Measure circuit."""
        self.circuit.measure_all()
        result = execute(self.circuit, self.simulator, shots=self.config.shots).result()
        counts = result.get_counts()
        assert sum(counts.values()) == self.config.shots, f"Measurement shots mismatch: {sum(counts.values())} != {self.config.shots}"
        return counts

    def classify_state(self) -> None:
        """Classify quantum state using QML."""
        state = Statevector(self.circuit)
        X = np.array([[self.phase, np.cos(self.phase)]])
        prediction = self.qml_model.predict(X)
        print(f"\n🧠 QML Prediction: State class = {'Positive' if prediction[0] == 1 else 'Negative'}")

    def _show_state(self) -> None:
        """Display statevector."""
        state = Statevector(self.circuit)
        display(state.draw('latex'))

# ======================== VISUALIZER ========================
class DemoVisualizer:
    """Visualizes spirals, quantum states, and audio."""
    
    def __init__(self, config: LovinceConfig) -> None:
        """Initialize visualization."""
        self.config = config
        self.fig = plt.figure(figsize=(18, 12), facecolor='black')
        plt.style.use('dark_background')

    def create_nebula_background(self, ax) -> None:
        """Create dynamic nebula background."""
        x = np.linspace(-50, 50, self.config.nebula_grid_size)
        y = np.linspace(-50, 50, self.config.nebula_grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/5 + np.random.uniform(-0.5, 0.5)) + np.cos(Y/5 + np.random.uniform(-0.5, 0.5)) + np.random.rand(self.config.nebula_grid_size, self.config.nebula_grid_size) * 0.5
        ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
        for _ in range(50):
            star_x, star_y = np.random.uniform(-50, 50), np.random.uniform(-50, 50)
            ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

    def generate_sound(self, filename: str, frequencies: np.ndarray) -> None:
        """Generate AI-inspired cosmic sound."""
        t = np.linspace(0, self.config.audio_duration, int(self.config.sample_rate * self.config.audio_duration))
        sound = np.zeros_like(t)
        for freq in frequencies:
            sound += 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t)  # Add decay for cosmic effect
        sound = sound / np.max(np.abs(sound))
        write(filename, self.config.sample_rate, sound)
        logging.info(f"Sound generated: {filename}")
        sd.play(sound, self.config.sample_rate)
        sd.wait()

    def plot_spirals(self, creation_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    collapse_coords: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Plot spirals with nebula and sound."""
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
        
        ax2.set_title("Cosmic Creation (φ^n, ML-Driven)", color='white')
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
        """Animate Bloch sphere."""
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
        """Plot quantum results."""
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
            probs_00.append(probs[0])
            probs_11.append(probs[3])
        ax5.plot(np.degrees(phases), probs_00, label="|00⟩", color="blue")
        ax5.plot(np.degrees(phases), probs_11, label="|11⟩", color="orange")
        ax5.set_xlabel("Phase Angle (degrees)", color='white')
        ax5.set_ylabel("Probability", color='white')
        ax5.set_title("Probability vs. Phase", color='white')
        ax5.legend(facecolor='black', edgecolor='yellow', labelcolor='white')
        ax5.set_facecolor('black')
        ax5.grid(color='gray', alpha=0.2)

# ======================== GUI ========================
class LovinceGUI:
    """Interactive GUI for Lovince AI."""
    
    def __init__(self, config: LovinceConfig, demo: 'LovinceAIDemo') -> None:
        """Initialize GUI."""
        self.config = config
        self.demo = demo
        self.root = tk.Tk()
        self.root.title("Lovince AI: Cosmic Quantum Dynamics")
        self.root.geometry("400x300")
        self.root.configure(bg='black')
        self.running = False

        # GUI Elements
        self.label_phase = tk.Label(self.root, text="Phase Angle (°):", fg='white', bg='black')
        self.label_phase.pack(pady=5)
        self.entry_phase = ttk.Entry(self.root)
        self.entry_phase.insert(0, str(self.config.phase_degrees))
        self.entry_phase.pack(pady=5)

        self.label_amplitude = tk.Label(self.root, text="Spiral Amplitude:", fg='white', bg='black')
        self.label_amplitude.pack(pady=5)
        self.entry_amplitude = ttk.Entry(self.root)
        self.entry_amplitude.insert(0, str(self.config.amplitude))
        self.entry_amplitude.pack(pady=5)

        self.start_button = ttk.Button(self.root, text="Start Demo", command=self.start_demo)
        self.start_button.pack(pady=10)
        self.stop_button = ttk.Button(self.root, text="Stop Demo", command=self.stop_demo, state='disabled')
        self.stop_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Idle", fg='yellow', bg='black')
        self.status_label.pack(pady=10)

    def start_demo(self) -> None:
        """Start the demo in a separate thread."""
        try:
            phase = float(self.entry_phase.get())
            amplitude = float(self.entry_amplitude.get())
            if not (-360 <= phase <= 360):
                raise ValueError("Phase must be in [-360, 360]")
            if not (0 < amplitude <= 1000):
                raise ValueError("Amplitude must be in (0, 1000]")
            self.running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Status: Running")
            # Update config dynamically
            object.__setattr__(self.config, 'phase_degrees', phase)
            object.__setattr__(self.config, 'amplitude', amplitude)
            # Run demo in thread
            self.demo_thread = Thread(target=self.demo.run)
            self.demo_thread.start()
            self.root.after(int(self.config.gui_update_interval * 1000), self.check_demo_status)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Status: Error")

    def stop_demo(self) -> None:
        """Stop the demo."""
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Stopped")
        plt.close('all')

    def check_demo_status(self) -> None:
        """Check if demo is running."""
        if self.running and self.demo_thread.is_alive():
            self.root.after(int(self.config.gui_update_interval * 1000), self.check_demo_status)
        else:
            self.stop_demo()

    def run(self) -> None:
        """Run the GUI."""
        self.root.mainloop()

# ======================== MAIN DEMO ========================
class LovinceAIDemo:
    """Advanced Lovince AI prototype with ML, QML, and GUI."""
    
    def __init__(self) -> None:
        """Initialize demo."""
        self.config = LovinceConfig()
        self.ml_model = SpiralMLModel(self.config)
        self.qml_model = QuantumMLModel(self.config)
        self.spiral_generator = SpiralGenerator(self.config, self.ml_model)
        self.quantum_circuit = QuantumCircuitDemo(self.config, self.qml_model)
        self.visualizer = DemoVisualizer(self.config)
        self.gui = LovinceGUI(self.config, self)

    def run(self) -> None:
        """Execute the demo."""
        print("🚀 Lovince AI: Advanced Cosmic Quantum Demo")
        
        # Train ML and QML models
        ml_loss = self.ml_model.train()
        qml_accuracy = self.qml_model.train()
        
        # Generate spirals
        creation_coords = self.spiral_generator.generate("creation")
        collapse_coords = self.spiral_generator.generate("collapse")
        
        # Run quantum circuit
        self.quantum_circuit.create_entangled_state()
        self.quantum_circuit.apply_phase_gate()
        self.quantum_circuit.apply_oracle()
        self.quantum_circuit.add_interference()
        self.quantum_circuit.classify_state()
        counts = self.quantum_circuit.measure()
        
        # Visualize
        self.visualizer.plot_spirals(creation_coords, collapse_coords)
        ani = self.visualizer.animate_bloch(self.quantum_circuit)
        self.visualizer.plot_quantum_results(counts)
        self.visualizer.fig.suptitle(f"Lovince AI: ML Loss={ml_loss:.4f}, QML Acc={qml_accuracy:.2f}", fontsize=16, color='white', y=0.95)
        self.visualizer.fig.text(0.5, 0.01, f"Golden Ratio: φ ≈ {self.config.phi:.3f}", ha='center', fontsize=10, color='white')
        plt.tight_layout()

        print("\n📊 Visualizations and Sounds Ready!")
        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close('all')
            print("\n🛑 Demo Shutdown Gracefully")
            sys.exit(0)

# ======================== EXECUTION ========================
if __name__ == "__main__":
    try:
        demo = LovinceAIDemo()
        demo.gui.run()
    except Exception as e:
        print(f"Cosmic Anomaly Detected: {str(e)}")
        sys.exit(1)