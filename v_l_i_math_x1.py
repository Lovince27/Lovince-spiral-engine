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