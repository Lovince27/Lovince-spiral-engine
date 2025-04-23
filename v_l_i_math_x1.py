import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# ======================
# 1. Complex Number Calculation
# ======================
magnitude = 40.5
phase = -np.pi/4  # -45 degrees
complex_num = magnitude * np.exp(1j * phase)
print(f"Complex Number: {complex_num:.3f}")  # 28.640 - 28.640i

# ======================
# 2. Quantum Circuit with Phase Gate
# ======================
qc = QuantumCircuit(1)

# Initialize superposition
qc.h(0)  
print("\nInitial State:")
display(Statevector(qc).draw('latex'))

# Apply phase gate (e^{-iπ/4})
qc.p(phase, 0)  
print("\nState After Phase Gate:")
display(Statevector(qc).draw('latex'))

# Visualize on Bloch Sphere
print("\nBloch Sphere Representation:")
display(plot_bloch_multivector(Statevector(qc)))

# ======================
# 3. Measurement (Collapse)
# ======================
qc.measure_all()

# Simulate 1000 shots
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

print("\nMeasurement Results (Collapsed States):")
display(plot_histogram(counts))

# ======================
# 4. Advanced: Verify Phase Impact
# ======================
def state_probabilities(phase_angle):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.p(phase_angle, 0)
    return Statevector(qc).probabilities()

print("\nProbability Comparison:")
phases = [-np.pi/4, 0, np.pi/4]  # -45°, 0°, +45°
for angle in phases:
    probs = state_probabilities(angle)
    print(f"Phase {np.degrees(angle):.0f}° → |0⟩: {probs[0]:.2%} | |1⟩: {probs[1]:.2%}")

plt.show()

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display

# ======================
# 1. Setup
# ======================
plt.ion()  # Interactive mode for animations
fig = plt.figure(figsize=(12, 4))

# ======================
# 2. Complex Number and User Input
# ======================
magnitude = 40.5
default_phase = -np.pi/4  # -45 degrees
complex_num = magnitude * np.exp(1j * default_phase)
print(f"Complex Number: {complex_num:.3f}")  # 28.640 - 28.640i

# Interactive phase input
try:
    phase_input = float(input("Enter phase angle in degrees (default -45): ") or -45)
    phase = np.radians(phase_input)
except ValueError:
    phase = default_phase
    print("Using default phase: -45°")

# ======================
# 3. 2-Qubit Quantum Circuit with Entanglement
# ======================
qc = QuantumCircuit(2)

# Create entanglement
qc.h(0)
qc.cx(0, 1)  # Entangle qubits: (|00⟩ + |11⟩)/√2
print("\nEntangled State:")
display(Statevector(qc).draw('latex'))

# Apply phase gate to second qubit
qc.p(phase, 1)
print("\nState After Phase Gate:")
display(Statevector(qc).draw('latex'))

# Apply second Hadamard to show interference
qc.h(0)
print("\nState After Interference:")
display(Statevector(qc).draw('latex'))

# ======================
# 4. Phase Oracle (Simple Quantum Algorithm)
# ======================
def apply_phase_oracle(qc, target_state="11"):
    """Apply a phase oracle marking a target state."""
    for i, bit in enumerate(target_state):
        if bit == "0":
            qc.x(i)
    qc.cz(0, 1)  # Apply phase to |11⟩
    for i, bit in enumerate(target_state):
        if bit == "0":
            qc.x(i)

print("\nApplying Phase Oracle for |11⟩:")
apply_phase_oracle(qc, "11")
display(Statevector(qc).draw('latex'))

# ======================
# 5. Bloch Sphere Animation
# ======================
def animate_bloch_sphere(frames=50):
    """Animate phase gate application."""
    ax = fig.add_subplot(131, projection='3d')
    
    def update(frame):
        ax.clear()
        temp_qc = QuantumCircuit(2)
        temp_qc.h(0)
        temp_qc.cx(0, 1)
        temp_qc.p(phase * frame / frames, 1)
        state = Statevector(temp_qc)
        plot_bloch_multivector(state, ax=ax)
        ax.set_title(f"Phase: {np.degrees(phase * frame / frames):.0f}°")
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)
    return ani

print("\nAnimating Bloch Sphere...")
ani = animate_bloch_sphere()

# ======================
# 6. Measurement
# ======================
qc.measure_all()
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

ax2 = fig.add_subplot(132)
plot_histogram(counts, ax=ax2)
ax2.set_title("Measurement Results")

# ======================
# 7. Advanced: Phase Impact on Probabilities
# ======================
def state_probabilities(phase_angle):
    """Compute probabilities after phase gate and interference."""
    temp_qc = QuantumCircuit(2)
    temp_qc.h(0)
    temp_qc.cx(0, 1)
    temp_qc.p(phase_angle, 1)
    temp_qc.h(0)
    return Statevector(temp_qc).probabilities()

ax3 = fig.add_subplot(133)
phases = np.linspace(-np.pi, np.pi, 50)
probs_00 = []
probs_11 = []
for angle in phases:
    probs = state_probabilities(angle)
    probs_00.append(probs[0])  # |00⟩
    probs_11.append(probs[3])  # |11⟩
ax3.plot(np.degrees(phases), probs_00, label="|00⟩")
ax3.plot(np.degrees(phases), probs_11, label="|11⟩")
ax3.set_xlabel("Phase Angle (degrees)")
ax3.set_ylabel("Probability")
ax3.set_title("Probability vs. Phase")
ax3.legend()

# ======================
# 8. Finalize
# ======================
plt.tight_layout()
plt.show()